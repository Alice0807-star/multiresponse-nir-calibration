"""Training-only task hints for the band-selection research prototype.

``train_attention`` accepts only the fit partition.  Its internal early-stop
partition is distinct from the reward-validation partition used by a selector.
The returned hints are frozen dataset-level quantities, not final predictions.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class MultiTaskAttention(nn.Module):
    """Task gates followed by a shared encoder and task-token attention.

    ``forward`` receives already standardized X, and returns standardized task
    predictions of shape (N, T) and head-averaged relations of shape (N, T, T).
    Scaling parameters are saved separately by ``train_attention``.
    """

    def __init__(self, n_bands: int, n_tasks: int) -> None:
        super().__init__()
        if n_bands < 1 or n_tasks < 1:
            raise ValueError("n_bands and n_tasks must be positive")
        self.n_bands = int(n_bands)
        self.n_tasks = int(n_tasks)
        self.gate_logits = nn.Parameter(torch.empty(n_tasks, n_bands))
        self.task_tokens = nn.Parameter(torch.empty(n_tasks, 32))
        nn.init.normal_(self.gate_logits, mean=0.0, std=0.01)
        nn.init.normal_(self.task_tokens, mean=0.0, std=0.02)
        self.encoder = nn.Sequential(nn.Linear(n_bands, 32), nn.GELU())
        self.attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(32)
        self.heads = nn.ModuleList(nn.Linear(32, 1) for _ in range(n_tasks))

    def gates(self) -> torch.Tensor:
        return torch.softmax(self.gate_logits, dim=-1)

    def forward(self, X_standardized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if X_standardized.ndim != 2 or X_standardized.shape[1] != self.n_bands:
            raise ValueError("X_standardized must have shape (N, n_bands)")
        task_input = (
            X_standardized[:, None, :]
            * self.gates()[None, :, :]
            * self.n_bands
        )
        encoded = self.encoder(task_input) + self.task_tokens[None, :, :]
        attended, relations = self.attention(
            encoded, encoded, encoded,
            need_weights=True, average_attn_weights=True,
        )
        hidden = self.norm(encoded + attended)
        predictions = torch.cat(
            [head(hidden[:, task, :]) for task, head in enumerate(self.heads)],
            dim=1,
        )
        return predictions, relations


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_attention_optimizer(
    model: MultiTaskAttention, regularization: str
) -> torch.optim.Adam:
    """Keep coupled Adam decay after clipping, as in the original prototype.

    Q/K/V are slices of the same leaf Parameters.  For exclude_qk the packed
    parameters have optimizer decay zero; _add_packed_v_decay adds only their
    V decay after clipping, before step.  All other parameters retain the
    original Adam-internal decay and its original position after clipping.
    """
    if regularization not in {"all", "exclude_qk", "none"}:
        raise ValueError("regularization must be all, exclude_qk, or none")
    if regularization == "all":
        return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    if regularization == "none":
        return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    packed = [model.attention.in_proj_weight, model.attention.in_proj_bias]
    packed_ids = {id(parameter) for parameter in packed}
    other = [parameter for parameter in model.parameters() if id(parameter) not in packed_ids]
    return torch.optim.Adam([
        {"params": other, "weight_decay": 1e-4},
        {"params": packed, "weight_decay": 0.0},
    ], lr=0.001)


def _add_packed_v_decay(model: MultiTaskAttention, regularization: str) -> None:
    """Apply only V's coupled L2 after clip, before Adam.step (exclude_qk)."""
    if regularization != "exclude_qk":
        return
    width = model.attention.embed_dim
    with torch.no_grad():
        for parameter in (model.attention.in_proj_weight, model.attention.in_proj_bias):
            # Adam also skips decay for parameters without a data gradient.
            if parameter.grad is not None:
                parameter.grad[2 * width:3 * width].add_(
                    parameter[2 * width:3 * width], alpha=1e-4
                )


def _qkv_gradient_diagnostics(
    model: MultiTaskAttention, regularization: str
) -> dict[str, float]:
    """Read QKV gradients after loss.backward, before clipping or any L2.

    The explicit gate-entropy term has no Q/K/V dependence, so these gradients
    equal the standardized prediction-MSE gradients for these parameters.
    """
    result: dict[str, float] = {}
    width = model.attention.embed_dim
    weight, bias = model.attention.in_proj_weight, model.attention.in_proj_bias
    with torch.no_grad():
        for index, label in enumerate(("q", "k", "v")):
            section = slice(index * width, (index + 1) * width)
            gradient_squares = torch.zeros((), device=weight.device)
            parameter_squares = weight[section].square().sum() + bias[section].square().sum()
            for parameter in (weight, bias):
                if parameter.grad is not None:
                    gradient_squares += parameter.grad[section].square().sum()
            coefficient = 1e-4 if regularization == "all" or (
                regularization == "exclude_qk" and label == "v"
            ) else 0.0
            result[f"{label}_gradient_preclip_l2_mean"] = float(gradient_squares.sqrt().item())
            result[f"{label}_decay_term_l2_mean"] = float(parameter_squares.sqrt().item()) * coefficient
    return result


def _attention_diagnostics(
    model: MultiTaskAttention,
    X_internal_training: torch.Tensor,
    *,
    epoch: int,
    regularization: str,
    gradient_means: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Deterministic forward-only diagnostics on fixed internal-training rows.

    Manual per-head QK avoids conflating cross-head averaging with collapse.
    This function consumes no random numbers and leaves model mode unchanged.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            gates = model.gates()
            encoded = model.encoder(
                X_internal_training[:, None, :] * gates[None, :, :] * model.n_bands
            ) + model.task_tokens[None, :, :]
            width = model.attention.embed_dim
            heads = model.attention.num_heads
            head_width = width // heads
            weight, bias = model.attention.in_proj_weight, model.attention.in_proj_bias
            q = F.linear(encoded, weight[:width], bias[:width])
            k = F.linear(encoded, weight[width:2 * width], bias[width:2 * width])
            shape = (len(encoded), model.n_tasks, heads, head_width)
            q, k = q.reshape(shape).transpose(1, 2), k.reshape(shape).transpose(1, 2)
            logits = (q @ k.transpose(-2, -1)) / np.sqrt(head_width)
            per_head = torch.softmax(logits, dim=-1)
            averaged = per_head.mean(dim=1)
            uniform = 1.0 / model.n_tasks
            entropy = -(gates * gates.clamp_min(1e-12).log()).sum(dim=1)
            normalized_entropy = entropy / np.log(model.n_bands) if model.n_bands > 1 else torch.zeros_like(entropy)
            result: dict[str, Any] = {
                "epoch": epoch,
                "gate_normalized_entropy_mean": float(normalized_entropy.mean().item()),
                "relation_mean_max_abs_from_uniform": float((averaged.mean(dim=0) - uniform).abs().max().item()),
                "relation_sample_max_abs_from_uniform": float((averaged - uniform).abs().max().item()),
                "relation_per_head_max_abs_from_uniform": float((per_head - uniform).abs().max().item()),
                "relation_between_heads_max_abs": float((per_head.max(dim=1).values - per_head.min(dim=1).values).max().item()),
                "qk_logit_abs_max": float(logits.abs().max().item()),
                "qk_logit_within_row_span_max": float((logits.max(dim=-1).values - logits.min(dim=-1).values).max().item()),
                "relation_mean": averaged.mean(dim=0).cpu().tolist(),
            }
            for task, value in enumerate(normalized_entropy.cpu().tolist()):
                result[f"gate_normalized_entropy_task_{task}"] = float(value)
            for index, label in enumerate(("q", "k", "v")):
                section = slice(index * width, (index + 1) * width)
                result[f"{label}_weight_l2"] = float(weight[section].norm().item())
                result[f"{label}_bias_l2"] = float(bias[section].norm().item())
                result[f"{label}_gradient_preclip_l2_mean"] = (
                    gradient_means[f"{label}_gradient_preclip_l2_mean"] if gradient_means is not None else None
                )
                result[f"{label}_decay_term_l2_mean"] = (
                    gradient_means[f"{label}_decay_term_l2_mean"] if gradient_means is not None else None
                )
                result[f"{label}_effective_decay_coefficient"] = (
                    1e-4 if regularization == "all" or (
                        regularization == "exclude_qk" and label == "v"
                    ) else 0.0
                )
                # Short aliases used by the experiment-level summary.  These
                # are projection-weight norms, matching the original audit.
                result[f"{label}_norm"] = result[f"{label}_weight_l2"]
            result["R_max_abs_from_uniform"] = result["relation_mean_max_abs_from_uniform"]
    finally:
        model.train(was_training)
    return result


def train_attention(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 17,
    epochs: int = 60,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    regularization: str = "all",
) -> dict[str, Any]:
    """Learn frozen A (T x B) and R (T x T) from fit data only.

    An internal 15% split with random_state=9090 supplies the early-stop MSE.
    X and y scaling is fit exclusively on the other 85%.  Zero/near-zero
    standard deviations (<=1e-12, ddof=0) are replaced with 1.  The patience is
    10 epochs; the loss is standardized MSE plus 1e-4 times mean gate entropy.
    ``epochs`` is a maximum, and does not promise convergence.

    regularization='all' preserves the original Adam coupled L2.  'none'
    disables it everywhere; 'exclude_qk' disables it only for packed Q/K
    weights and biases while retaining the original decay for V and all other
    parameters.  Fixed training-only diagnostics do not select checkpoints;
    early stopping remains the internal-validation prediction MSE.

    If output_dir is provided it must already exist, and must not contain an
    attention checkpoint.  Artifacts contain CPU tensors and primitive types
    so torch.load(..., map_location='cpu', weights_only=True) can load them.
    """
    X_array = np.asarray(X, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if X_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError("X and y must be two-dimensional arrays")
    if X_array.shape[0] != y_array.shape[0] or X_array.shape[0] < 3:
        raise ValueError("X/y must have equal row counts and at least 3 rows")
    if X_array.shape[1] < 1 or y_array.shape[1] < 1:
        raise ValueError("X and y must contain at least one column")
    if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
        raise ValueError("X and y must contain only finite values")
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if regularization not in {"all", "exclude_qk", "none"}:
        raise ValueError("regularization must be all, exclude_qk, or none")
    seed = int(seed)
    epochs = int(epochs)
    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("This prototype supports CPU and CUDA devices")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    artifact_dir = Path(output_dir) if output_dir is not None else None
    checkpoint_path = (
        artifact_dir / "attention_checkpoint.pt" if artifact_dir is not None else None
    )
    if artifact_dir is not None:
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"output_dir must already exist: {artifact_dir}")
        if checkpoint_path.exists():
            raise FileExistsError(f"Attention checkpoint already exists: {checkpoint_path}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    _synchronize(torch_device)
    started = time.perf_counter()

    all_indices = np.arange(X_array.shape[0])
    train_indices, early_stop_indices = train_test_split(
        all_indices, test_size=0.15, random_state=9090, shuffle=True
    )
    X_mean = X_array[train_indices].mean(axis=0)
    X_raw_std = X_array[train_indices].std(axis=0, ddof=0)
    y_mean = y_array[train_indices].mean(axis=0)
    y_raw_std = y_array[train_indices].std(axis=0, ddof=0)
    X_scale = np.where(X_raw_std > 1e-12, X_raw_std, 1.0)
    y_scale = np.where(y_raw_std > 1e-12, y_raw_std, 1.0)
    X_scaled = ((X_array - X_mean) / X_scale).astype(np.float32)
    y_scaled = ((y_array - y_mean) / y_scale).astype(np.float32)
    if not np.isfinite(X_scaled).all() or not np.isfinite(y_scaled).all():
        raise ValueError("Standardized input cannot be represented as finite float32")

    train_dataset = TensorDataset(
        torch.from_numpy(X_scaled[train_indices]),
        torch.from_numpy(y_scaled[train_indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        generator=generator, num_workers=0,
    )
    validation_X = torch.from_numpy(X_scaled[early_stop_indices]).to(torch_device)
    validation_y = torch.from_numpy(y_scaled[early_stop_indices]).to(torch_device)
    model = MultiTaskAttention(X_array.shape[1], y_array.shape[1]).to(torch_device)
    optimizer = _make_attention_optimizer(model, regularization)
    initial_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_digest.update(name.encode("utf-8"))
        initial_digest.update(value.detach().cpu().numpy().tobytes())
    initial_state_sha256 = initial_digest.hexdigest()
    diagnostic_indices = train_indices[:min(32, len(train_indices))]
    diagnostic_X = torch.from_numpy(X_scaled[diagnostic_indices]).to(torch_device)
    diagnostic_history = [_attention_diagnostics(
        model, diagnostic_X, epoch=0, regularization=regularization
    )]
    criterion = nn.MSELoss()
    best_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_used = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = mse_sum = entropy_sum = 0.0
        gradient_totals: dict[str, float] = {}
        batches = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(batch_X)
            mse = criterion(predictions, batch_y)
            gates = model.gates()
            entropy = -(gates * gates.clamp_min(1e-12).log()).sum(dim=1).mean()
            loss = mse + 1e-4 * entropy
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite attention loss at epoch {epoch}")
            loss.backward()
            gradients = _qkv_gradient_diagnostics(model, regularization)
            for name, value in gradients.items():
                gradient_totals[name] = gradient_totals.get(name, 0.0) + value
            batches += 1
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            _add_packed_v_decay(model, regularization)
            optimizer.step()
            count = batch_X.shape[0]
            loss_sum += float(loss.detach().item()) * count
            mse_sum += float(mse.detach().item()) * count
            entropy_sum += float(entropy.detach().item()) * count

        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(validation_X)
            validation_mse = float(criterion(validation_prediction, validation_y).item())
        if not np.isfinite(validation_mse):
            raise FloatingPointError(f"Non-finite early-stop MSE at epoch {epoch}")
        diagnostic_history.append(_attention_diagnostics(
            model, diagnostic_X, epoch=epoch, regularization=regularization,
            gradient_means={name: value / batches for name, value in gradient_totals.items()},
        ))
        improved = validation_mse < best_mse
        if improved:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            patience_used = 0
        else:
            patience_used += 1
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train_dataset),
            "train_mse": mse_sum / len(train_dataset),
            "gate_entropy": entropy_sum / len(train_dataset),
            "early_stop_mse": validation_mse,
            "is_best": bool(improved),
        })
        if patience_used >= 10:
            break

    if best_state is None:
        raise RuntimeError("Attention training did not produce a checkpoint")
    best_checkpoint_diagnostics = copy.deepcopy(diagnostic_history[best_epoch])
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        A = model.gates().detach().cpu().numpy().copy()
        relation_sum = np.zeros((y_array.shape[1], y_array.shape[1]), dtype=np.float64)
        for start in range(0, len(X_scaled), 64):
            batch_X = torch.from_numpy(X_scaled[start:start + 64]).to(torch_device)
            _, relations = model(batch_X)
            relation_sum += relations.detach().cpu().numpy().astype(np.float64).sum(axis=0)
        R = relation_sum / len(X_scaled)
    if not np.isfinite(A).all() or not np.isfinite(R).all():
        raise FloatingPointError("Attention hints contain non-finite values")
    if (
        (A < 0).any() or (R < 0).any()
        or not np.allclose(A.sum(axis=1), 1.0, atol=1e-6)
        or not np.allclose(R.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise FloatingPointError("Attention hints are not row-stochastic")
    _synchronize(torch_device)
    train_seconds = float(time.perf_counter() - started)

    scaling = {
        "X_mean": X_mean.tolist(), "X_scale": X_scale.tolist(),
        "y_mean": y_mean.tolist(), "y_scale": y_scale.tolist(),
        "ddof": 0, "scale_floor": 1e-12,
        "X_constant_columns_zero_based": np.flatnonzero(X_raw_std <= 1e-12).tolist(),
        "y_constant_columns_zero_based": np.flatnonzero(y_raw_std <= 1e-12).tolist(),
    }
    metadata = {
        "seed": seed,
        "device": str(torch_device),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "n_samples": len(X_array),
        "n_bands": X_array.shape[1],
        "n_tasks": y_array.shape[1],
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_early_stop_mse": best_mse,
        "train_seconds": train_seconds,
        "timing_scope": "split, scaling, model training, training-only diagnostics, best-state restore, and A/R extraction; artifact I/O excluded",
        "regularization": regularization,
        "initial_state_sha256": initial_state_sha256,
        "internal_split": {
            "random_state": 9090, "test_size": 0.15,
            "train_indices_zero_based": train_indices.tolist(),
            "early_stop_indices_zero_based": early_stop_indices.tolist(),
        },
        "hyperparameters": {
            "hidden_dim": 32, "n_heads": 4, "dropout": 0.0,
            "batch_size": 64, "patience": 10, "learning_rate": 0.001,
            "weight_decay": 0.0 if regularization == "none" else 1e-4, "gradient_clip_norm": 5.0,
            "gate_entropy_coefficient": 1e-4, "early_stop_metric": "standardized MSE",
            "gate_logits_init_std": 0.01, "task_tokens_init_std": 0.02,
        },
        "scaling": scaling,
        "regularization_policy": {
            "mode": regularization,
            "optimizer": "Adam, coupled L2 (not AdamW)",
            "all_mode": "original single parameter group, weight_decay=1e-4",
            "none_mode": "original single parameter group, weight_decay=0",
            "exclude_qk_mode": "in_proj_weight and in_proj_bias are leaf parameters with optimizer weight_decay=0; remaining parameters have optimizer weight_decay=1e-4; V rows (last third) alone receive manual grad += 1e-4 * parameter after global gradient clipping, before optimizer.step",
            "update_order": "training loss backward -> read preclip QKV data gradients -> global clip(max_norm=5) -> manual packed V decay for exclude_qk -> Adam step with group-specific coupled decay",
            "packed_QK_rows": [0, 64], "packed_V_rows": [64, 96],
            "row_interval_convention": "zero-based, half-open; embed_dim=32",
        },
        "diagnostic_scope": {
            "sample_source": "first at most 32 rows of internal training indices in deterministic split order",
            "fit_row_indices_zero_based": diagnostic_indices.tolist(),
            "contains_early_stop_rows": False,
            "contains_reward_validation_or_test_rows": False,
            "random_sampling_used": False,
            "relation_computation": "manual per-head scaled QK softmax, then head/sample mean; dropout=0",
            "summary_aliases": "q_norm/k_norm/v_norm are projection-weight norms; R_max_abs_from_uniform is the sample-averaged R deviation on the fixed internal-training diagnostic rows",
            "gradient_norms": "arithmetic mean across minibatch updates of QKV combined weight+bias L2 norms before clipping and before coupled L2; explicit gate entropy has no QKV dependence",
            "decay_term_norms": "arithmetic mean across updates of coefficient * combined weight+bias L2 norm; coefficient zero for excluded parameters",
            "epoch_zero_gradient_fields": "null, as no backward/update has occurred",
            "relation_nonuniformity_is_success_criterion": False,
        },
        "diagnostic_history": diagnostic_history,
        "best_checkpoint_diagnostics": best_checkpoint_diagnostics,
        "torch_num_threads": torch.get_num_threads(),
        "R_extraction_scope": "all supplied fit samples, eval mode, mean across samples and heads",
    }
    if artifact_dir is not None:
        checkpoint = {
            "state_dict": best_state,
            "metadata": copy.deepcopy(metadata),
            "history": history,
            "diagnostic_history": diagnostic_history,
            "best_checkpoint_diagnostics": best_checkpoint_diagnostics,
            "A": torch.from_numpy(A.copy()),
            "R": torch.from_numpy(R.copy()),
        }
        torch.save(checkpoint, checkpoint_path)
        (artifact_dir / "attention_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        with (artifact_dir / "attention_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        with (artifact_dir / "diagnostic_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diagnostic_history[0]))
            writer.writeheader()
            writer.writerows({
                name: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for name, value in row.items()
            } for row in diagnostic_history)
        np.save(artifact_dir / "attention_A.npy", A)
        np.save(artifact_dir / "attention_R.npy", R)

    return {
        "A": A,
        "R": R,
        "train_seconds": train_seconds,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_early_stop_mse": best_mse,
        "history": history,
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
        "metadata": metadata,
        "regularization": regularization,
        "diagnostic_history": diagnostic_history,
        "best_checkpoint_diagnostics": best_checkpoint_diagnostics,
    }
