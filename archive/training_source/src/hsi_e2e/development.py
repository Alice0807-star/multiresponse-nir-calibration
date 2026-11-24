"""Export fit/reward data for development without exposing evaluation records.

The full source NPZ is read only at this one export boundary. Downstream
development runners should open development_only.npz, never the full source.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _development_rows(split: dict, name: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Access only the named development partition, not any other split member."""
    group = split[name]
    raw = group["row_indices_0based"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name}: expected a nonempty list of row indices")
    if any(isinstance(i, bool) or not isinstance(i, int) for i in raw):
        raise ValueError(f"{name}: row indices must be integers, excluding booleans")
    if any(i < 0 or i >= n_rows for i in raw) or len(raw) != len(set(raw)):
        raise ValueError(f"{name}: duplicate or out-of-range row indices")
    declared_ids = np.asarray(group["sample_ids"])
    if declared_ids.ndim != 1 or len(declared_ids) != len(raw):
        raise ValueError(f"{name}: sample IDs must align with row indices")
    return np.asarray(raw, dtype=np.int64), declared_ids


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Fixed ZIP timestamps and member order permit immutable hash checking."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.save(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def _ensure_unchanged_or_absent(path: Path, expected: bytes) -> None:
    if path.exists() and _file_sha256(path) != hashlib.sha256(expected).hexdigest():
        raise FileExistsError(f"Refusing to overwrite an artifact with a different hash: {path}")


def _create_if_absent(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        # A concurrent writer is safe only if it wrote exactly the same artifact.
        _ensure_unchanged_or_absent(path, content)


def export_development(source_npz, original_split_json, output_dir) -> dict:
    """Export untransformed fit/reward arrays and an immutable source manifest.

    Only fit and reward_validation entries are consumed from the split record.
    The compressed source arrays necessarily load as whole arrays, but only the
    two development slices are checked, transformed, hashed as arrays or saved.
    Original source/split hashes are opaque provenance checks, not label analysis.
    Existing output bytes must match exactly; no different artifact is replaced.
    """
    source = Path(source_npz).resolve(strict=True)
    split_path = Path(original_split_json).resolve(strict=True)
    destination = Path(output_dir).resolve()
    development_path = destination / "development_only.npz"
    manifest_path = destination / "development_manifest.json"
    if development_path == source or manifest_path == split_path:
        raise ValueError("Development output must not replace a source artifact")

    source_hash = _file_sha256(source)
    split_bytes = split_path.read_bytes()
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    split = json.loads(split_bytes.decode("utf-8-sig"))
    with np.load(source, allow_pickle=False) as archive:
        X_source = archive["X"]
        y_source = archive["y"]
        ids_source = archive["ids"]
        targets = archive["target_names"]
        if (X_source.ndim != 2 or y_source.ndim != 2 or ids_source.ndim != 1
                or len(X_source) != len(y_source) or len(X_source) != len(ids_source)
                or X_source.shape[1] == 0 or y_source.shape[1] == 0):
            raise ValueError("Source sample/feature/target dimensions are inconsistent")
        if targets.ndim != 1 or len(targets) != y_source.shape[1] or targets.dtype.kind not in "US":
            raise ValueError("Source target names must be a string per target column")
        target_names = np.asarray(targets, dtype=str)
        if len(set(target_names.tolist())) != len(target_names) or np.any(target_names == ""):
            raise ValueError("Target names must be distinct and nonempty")

        fit_rows, declared_fit_ids = _development_rows(split, "fit", len(X_source))
        reward_rows, declared_reward_ids = _development_rows(split, "reward_validation", len(X_source))
        if np.intersect1d(fit_rows, reward_rows).size:
            raise ValueError("Fit and reward partitions overlap in row indices")
        fit_ids = np.array(ids_source[fit_rows], copy=True)
        reward_ids = np.array(ids_source[reward_rows], copy=True)
        if not np.array_equal(fit_ids, declared_fit_ids) or not np.array_equal(reward_ids, declared_reward_ids):
            raise ValueError("Split sample IDs do not match source row indices")
        for name, values in (("fit", fit_ids), ("reward", reward_ids)):
            if values.dtype.hasobject or len(np.unique(values)) != len(values):
                raise ValueError(f"{name}: IDs must be distinct and pickle-free")
        if np.intersect1d(fit_ids, reward_ids).size:
            raise ValueError("Fit and reward partitions overlap in sample IDs")

        arrays = {
            "X_fit": np.array(X_source[fit_rows], dtype=np.float64, order="C", copy=True),
            "y_fit": np.array(y_source[fit_rows], dtype=np.float64, order="C", copy=True),
            "fit_ids": fit_ids,
            "X_reward": np.array(X_source[reward_rows], dtype=np.float64, order="C", copy=True),
            "y_reward": np.array(y_source[reward_rows], dtype=np.float64, order="C", copy=True),
            "reward_ids": reward_ids,
            "target_names": target_names,
        }
    for name in ("X_fit", "y_fit", "X_reward", "y_reward"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Development array contains nonfinite values: {name}")
    for name in ("fit_ids", "reward_ids"):
        if arrays[name].dtype.kind in "fc" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"Development sample IDs contain nonfinite values: {name}")

    content = _npz_bytes(arrays)
    manifest = {
        "schema_version": 1,
        "source_npz": str(source),
        "source_npz_sha256": source_hash,
        "original_split_json": str(split_path),
        "original_split_sha256": split_hash,
        "development_npz": str(development_path),
        "development_npz_sha256": hashlib.sha256(content).hexdigest(),
        "partition_definitions": {
            "fit": {"source_split_key": "fit", "n_rows": len(fit_rows), "row_indices_0based": fit_rows.tolist()},
            "reward": {"source_split_key": "reward_validation", "n_rows": len(reward_rows), "row_indices_0based": reward_rows.tolist()},
        },
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "target_names": target_names.tolist(),
        "preprocessing": {
            "export": "Original values retained; X/y cast to float64. No scaling, imputation, feature selection, target construction, unit conversion or row exclusion.",
            "attention": "Fit X/y scaling only within the attention internal-training partition; internal early-stop rows are transformed using those fitted scales.",
            "reward": "Target means/standard deviations estimated on fit only, ddof=0; reward rows transformed with fit scales. Selected-column PLSR has scale=True.",
            "downstream_boundary": "Development runners load this export only; they must not reopen the full source or original split record.",
        },
        "evaluation_status": "Development selection scores only, not independent generalization estimates.",
        "historical_exposure": "The original dataset was already used in historical exploration and the first internal end-to-end verification; this export does not create previously unseen data.",
        "excluded_evaluation_partition": "No evaluation-partition arrays, row indices or sample IDs are included. Its split member and label values were not used in export validation or calculations.",
        "integrity": "Fixed ZIP timestamps/member order. Existing NPZ and manifest must match expected SHA256; different artifacts are never overwritten.",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Validate both paths before creating either file, so a conflicting manifest
    # cannot leave a newly-created archive beside inconsistent existing metadata.
    _ensure_unchanged_or_absent(development_path, content)
    _ensure_unchanged_or_absent(manifest_path, manifest_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    _create_if_absent(development_path, content)
    _create_if_absent(manifest_path, manifest_bytes)
    return manifest
