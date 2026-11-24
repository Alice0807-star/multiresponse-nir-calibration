"""Attention-assisted Double-DQN search for a shared spectral-column subset.

Only the evaluator's fit/reward-validation interface is accepted here.  Test
features and labels have no place in this module.  Column indices are zero-based
internally; logged action and subset column numbers are one-based.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SubsetEvaluator(Protocol):
    def evaluate(self, subset: list[int] | tuple[int, ...]) -> float: ...

    def snapshot(self) -> dict[str, Any]: ...


def _validate_context(A: np.ndarray, R: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if A.ndim != 2 or not A.shape[0] or not A.shape[1]:
        raise ValueError("A must have shape (n_targets, n_bands).")
    if R.shape != (A.shape[0], A.shape[0]):
        raise ValueError("R must have shape (n_targets, n_targets).")
    if not 1 <= k <= A.shape[1]:
        raise ValueError("k must be between 1 and n_bands.")
    if not np.isfinite(A).all() or not np.isfinite(R).all():
        raise ValueError("A and R must be finite.")
    if (A < 0).any() or (R < 0).any():
        raise ValueError("A and R must be nonnegative attention weights.")
    if not np.allclose(A.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Each row of A must sum to one.")
    if not np.allclose(R.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Each row of R must sum to one.")
    return A, R


def context_for(mask: np.ndarray, A: np.ndarray, R: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Construct the frozen-attention dynamic context from a binary mask.

    Coverage is an attention-mass heuristic, not physical information coverage.
    The final state has B+B+T+T*T+1 float32 entries.
    """
    A, R = _validate_context(A, R, k)
    mask = np.asarray(mask)
    if mask.shape != (A.shape[1],) or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must be a binary n_bands vector.")
    mask_f = mask.astype(np.float64, copy=False)
    if mask_f.sum() > k:
        raise ValueError("mask already contains more than k columns.")
    reference = np.partition(A, A.shape[1] - k, axis=1)[:, -k:].sum(axis=1)
    coverage = np.clip((A @ mask_f) / reference, 0.0, 1.0)
    demand = np.maximum(0.05, 1.0 - coverage)
    mixed_demand = 0.5 * demand + 0.5 * (R @ demand)
    hint = mixed_demand @ A
    span = float(hint.max() - hint.min())
    if span > 1e-12:
        hint = (hint - hint.min()) / span
    else:
        hint = np.zeros_like(hint)
    state = np.concatenate(
        [mask_f, hint, coverage, R.ravel(), np.array([mask_f.sum() / k])]
    ).astype(np.float32)
    return {
        "state": state,
        "hint": hint.astype(np.float32),
        "coverage": coverage.astype(np.float32),
    }


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_bands: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, n_bands))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


@dataclass
class _Transition:
    state: np.ndarray
    hint: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_hint: np.ndarray
    next_mask: np.ndarray
    done: bool


def _learning_step(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    transitions: list[_Transition],
    device: torch.device,
    q_prior: float,
) -> float:
    states = torch.as_tensor(np.stack([t.state for t in transitions]), device=device)
    hints = torch.as_tensor(np.stack([t.hint for t in transitions]), device=device)
    actions = torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=device)
    rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), device=device)
    next_hints = torch.as_tensor(np.stack([t.next_hint for t in transitions]), device=device)
    next_masks = torch.as_tensor(np.stack([t.next_mask for t in transitions]), dtype=torch.bool, device=device)
    done = torch.as_tensor([t.done for t in transitions], dtype=torch.bool, device=device)

    q_values = (online(states) + q_prior * hints).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        # Terminal states never use a bootstrap target, including k == n_bands.
        # Computing only nonterminal rows also avoids all-masked argmax and 0*inf.
        bootstrap = torch.zeros_like(rewards)
        active = ~done
        if bool(active.any()):
            active_masks = next_masks[active]
            if bool(active_masks.all(dim=1).any()):
                raise RuntimeError("A nonterminal transition has no available action.")
            next_online = online(next_states[active]) + q_prior * next_hints[active]
            next_online = next_online.masked_fill(active_masks, -torch.inf)
            next_actions = next_online.argmax(dim=1)
            next_target = target(next_states[active]) + q_prior * next_hints[active]
            bootstrap[active] = next_target.gather(1, next_actions[:, None]).squeeze(1)
        targets = rewards + bootstrap  # gamma = 1.0, fixed-length subset episodes.

    loss = F.smooth_l1_loss(q_values, targets)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite Double-DQN training loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach().cpu())


def select_dqn(
    evaluator: SubsetEvaluator,
    *,
    n_bands: int,
    k: int,
    A: np.ndarray,
    R: np.ndarray,
    seed: int = 17,
    episodes: int = 48,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    q_prior: float = 0.05,
    use_context: bool = True,
) -> dict[str, Any]:
    """Search k-column subsets using reward-validation NMSE improvements.

    The result called ``best_subset`` is the best complete subset encountered
    at an episode end or in the final greedy rollout.  ``greedy_subset`` is
    retained separately to distinguish search quality from final-policy quality.
    The local NumPy generator controls exploration and replay sampling.
    """
    A, R = _validate_context(A, R, k)
    if A.shape[1] != n_bands:
        raise ValueError("n_bands does not match A.")
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if not np.isfinite(q_prior):
        raise ValueError("q_prior must be finite.")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    started = perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    initial_mask = np.zeros(n_bands, dtype=bool)
    state_dim = 2 * n_bands + A.shape[0] + A.shape[0] ** 2 + 1
    online = QNetwork(state_dim, n_bands).to(torch_device)
    target = QNetwork(state_dim, n_bands).to(torch_device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    replay: deque[_Transition] = deque(maxlen=4000)
    batch_size = 32
    environment_steps = 0
    gradient_updates = 0
    trace: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    best_error = float("inf")
    best_subset: list[int] | None = None
    best_source: str | None = None
    complete_subsets: set[tuple[int, ...]] = set()

    # The hints used in the fixed Q prior are separate from the network input.
    # This permits a genuine prior-only ablation without erasing its hints.
    def make_context(mask: np.ndarray) -> dict[str, np.ndarray]:
        context = context_for(mask, A, R, k)
        if not use_context:
            state = np.zeros(state_dim, dtype=np.float32)
            state[:n_bands] = mask
            state[-1] = mask.sum() / k
            context["state"] = state
        return context

    def score(subset: list[int] | tuple[int, ...]) -> float:
        value = float(evaluator.evaluate(subset))
        if not np.isfinite(value):
            raise FloatingPointError("Evaluator returned non-finite reward-validation NMSE.")
        return value

    def choose_action(mask: np.ndarray, context: dict[str, np.ndarray], epsilon: float) -> int:
        available = np.flatnonzero(~mask)
        if available.size == 0:
            raise RuntimeError("No action is available.")
        if rng.random() < epsilon:
            return int(rng.choice(available))
        with torch.no_grad():
            state_tensor = torch.as_tensor(context["state"], device=torch_device).unsqueeze(0)
            hint_tensor = torch.as_tensor(context["hint"], device=torch_device)
            values = online(state_tensor).squeeze(0) + q_prior * hint_tensor
            values = values.masked_fill(torch.as_tensor(mask, device=torch_device), -torch.inf)
            action = int(values.argmax().cpu())
        if mask[action]:
            raise RuntimeError("The action mask failed to exclude a selected band.")
        return action

    def counter_fields() -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name, value in evaluator.snapshot().items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, str, bool)) or value is None:
                fields[f"evaluator_{name}"] = value
        return fields

    empty_error = score(())
    for episode in range(episodes):
        epsilon = 1.0 - 0.9 * episode / max(episodes - 1, 1)
        mask = initial_mask.copy()
        context = make_context(mask)
        current_error = empty_error
        episode_return = 0.0
        action_order: list[int] = []
        for step in range(k):
            action = choose_action(mask, context, epsilon)
            next_mask = mask.copy()
            next_mask[action] = True
            subset = np.flatnonzero(next_mask).tolist()
            next_error = score(subset)
            reward = current_error - next_error
            done = step + 1 == k
            next_context = make_context(next_mask)
            replay.append(
                _Transition(
                    state=context["state"].copy(),
                    hint=context["hint"].copy(),
                    action=action,
                    reward=reward,
                    next_state=next_context["state"].copy(),
                    next_hint=next_context["hint"].copy(),
                    next_mask=next_mask.copy(),
                    done=done,
                )
            )
            environment_steps += 1
            loss: float | None = None
            if len(replay) >= batch_size:
                sampled = rng.choice(len(replay), size=batch_size, replace=False)
                loss = _learning_step(
                    online, target, optimizer, [replay[int(i)] for i in sampled], torch_device, q_prior
                )
                gradient_updates += 1
            if environment_steps % 64 == 0:
                target.load_state_dict(online.state_dict())
            action_order.append(action + 1)
            episode_return += reward
            trace.append(
                {
                    "phase": "training",
                    "episode": episode + 1,
                    "step": step + 1,
                    "environment_step": environment_steps,
                    "action_1based": action + 1,
                    "selected_count": len(subset),
                    "subset_indices_1based": json.dumps([i + 1 for i in subset]),
                    "reward": reward,
                    "validation_nmse": next_error,
                    "epsilon": epsilon,
                    "done": done,
                    "training_loss": loss,
                    "elapsed_seconds": perf_counter() - started,
                    **counter_fields(),
                }
            )
            mask, context, current_error = next_mask, next_context, next_error

        # gamma=1 makes the complete episode return telescope exactly up to
        # floating-point summation.  Failing this is a conclusion-changing bug.
        if not np.isclose(episode_return, empty_error - current_error, atol=1e-9, rtol=1e-9):
            raise AssertionError("Episode rewards did not telescope to the final improvement.")
        complete = tuple(np.flatnonzero(mask).tolist())
        complete_subsets.add(complete)
        if current_error < best_error:
            best_error, best_subset = current_error, list(complete)
            best_source = f"episode_{episode + 1}"
        episode_summaries.append(
            {
                "episode": episode + 1,
                "epsilon": epsilon,
                "return": episode_return,
                "validation_nmse": current_error,
                "best_complete_validation_nmse": best_error,
                "subset_indices_0based": list(complete),
                "action_order_1based": action_order,
                "elapsed_seconds": perf_counter() - started,
                **counter_fields(),
            }
        )
        if (episode + 1) % 8 == 0 or episode + 1 == episodes:
            print(
                f"Double-DQN episode {episode + 1}/{episodes}: "
                f"val_NMSE={current_error:.6f}, best={best_error:.6f}, epsilon={epsilon:.3f}",
                flush=True,
            )

    online.eval()
    mask = initial_mask.copy()
    current_error = empty_error
    greedy_action_order: list[int] = []
    for step in range(k):
        context = make_context(mask)
        action = choose_action(mask, context, epsilon=0.0)
        mask[action] = True
        subset = np.flatnonzero(mask).tolist()
        next_error = score(subset)
        greedy_action_order.append(action + 1)
        trace.append(
            {
                "phase": "final_greedy",
                "episode": episodes + 1,
                "step": step + 1,
                "environment_step": environment_steps,
                "action_1based": action + 1,
                "selected_count": len(subset),
                "subset_indices_1based": json.dumps([i + 1 for i in subset]),
                "reward": current_error - next_error,
                "validation_nmse": next_error,
                "epsilon": 0.0,
                "done": step + 1 == k,
                "training_loss": None,
                "elapsed_seconds": perf_counter() - started,
                **counter_fields(),
            }
        )
        current_error = next_error
    greedy_subset = np.flatnonzero(mask).tolist()
    complete_subsets.add(tuple(greedy_subset))
    if current_error < best_error:
        best_error, best_subset = current_error, greedy_subset.copy()
        best_source = "final_greedy"
    assert best_subset is not None
    if len(best_subset) != k or len(set(best_subset)) != k:
        raise AssertionError("The best complete subset must contain exactly k unique columns.")
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    search_seconds = perf_counter() - started

    summary: dict[str, Any] = {
        "best_subset": best_subset,
        "greedy_subset": greedy_subset,
        "best_validation_nmse": best_error,
        "greedy_validation_nmse": current_error,
        "empty_validation_nmse": empty_error,
        "best_source": best_source,
        "greedy_action_order_1based": greedy_action_order,
        "search_seconds": search_seconds,
        "episodes_completed": episodes,
        "environment_steps": environment_steps,
        "gradient_updates": gradient_updates,
        "replay_size": len(replay),
        "complete_candidate_count": episodes + 1,
        "unique_complete_candidate_count": len(complete_subsets),
        "seed": seed,
        "device": str(torch_device),
        "use_context": use_context,
        "q_prior": q_prior,
        "state_dim": state_dim,
        "configuration": {
            "hidden_units": 128,
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "replay_capacity": 4000,
            "target_update_environment_steps": 64,
            "gamma": 1.0,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_schedule": "linear_by_episode_including_endpoints",
            "gradient_clip_norm": 5.0,
            "loss": "SmoothL1Loss",
            "rng": "numpy.default_rng; shared exploration and replay generator",
            "search_seconds_excludes_checkpoint_serialization": True,
        },
        "evaluator_counters": evaluator.snapshot(),
        "episode_summaries": episode_summaries,
    }
    checkpoint_path: str | None = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = output_path / "dqn_checkpoint.pt"
        summary_file = output_path / "dqn_summary.json"
        if checkpoint.exists() or summary_file.exists():
            raise FileExistsError("DQN outputs already exist; use a new run directory.")
        torch.save(
            {
                "online_state_dict": {name: value.detach().cpu().clone() for name, value in online.state_dict().items()},
                "target_state_dict": {name: value.detach().cpu().clone() for name, value in target.state_dict().items()},
                "configuration": summary["configuration"],
                "state_dim": state_dim,
                "n_bands": n_bands,
                "k": k,
                "seed": seed,
                "A": torch.as_tensor(A.copy()),
                "R": torch.as_tensor(R.copy()),
                "use_context": use_context,
                "q_prior": q_prior,
                "replay_summary": {
                    "size": len(replay),
                    "terminal_transitions": sum(t.done for t in replay),
                    "gradient_updates": gradient_updates,
                    "environment_steps": environment_steps,
                },
                "resume_supported": False,
            },
            checkpoint,
        )
        checkpoint_path = str(checkpoint.resolve())
        summary["checkpoint_path"] = checkpoint_path
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        summary["checkpoint_path"] = None
    summary["trace"] = trace
    return summary
