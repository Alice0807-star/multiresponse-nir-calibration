"""Summaries of the predeclared development-only attention/RL diagnostic.

This module reads only the diagnostic's explicitly named CSV/JSON files.  It
does not load the dataset, test predictions, or the historical e2e results.
"""
from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ORDER = [
    "full_plsr", "correlation_topk", "attention_topk_all",
    "attention_topk_exclude_qk", "attention_topk_none", "plain_dqn",
    "correlation_hint", "gate_identity", "gate_uniform", "gate_learned",
    "gate_learned_qprior", "random_search",
]
LABELS = {
    "full_plsr": "Full PLSR", "correlation_topk": "Correlation TopK",
    "attention_topk_all": "TopK: all decay",
    "attention_topk_exclude_qk": "TopK: exclude Q/K decay",
    "attention_topk_none": "TopK: no decay",
    "plain_dqn": "Plain DDQN", "correlation_hint": "Correlation hint",
    "gate_identity": "Gate + identity R", "gate_uniform": "Gate + uniform R",
    "gate_learned": "Gate + learned R", "gate_learned_qprior": "Learned R + Q prior",
    "random_search": "Random search",
}
NUMERIC_FIELDS = [
    "validation_nmse", "mean_r2", "SSC_r2", "pH_r2", "firmness_r2",
    "SSC_rmse", "pH_rmse", "firmness_rmse", "SSC_mae", "pH_mae", "firmness_mae",
    "fit_count", "total_calls", "cache_hits", "complete_unique_subsets",
    "search_seconds", "attention_pretrain_seconds", "selection_total_seconds",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float("nan")
    result = float(value)
    return result


def _stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def _mean_sd(rows: list[dict[str, Any]], field: str, digits: int = 4) -> str:
    mean, sd = _stats([_number(row, field) for row in rows])
    return f"{_fmt(mean, digits)} ± {_fmt(sd, digits)}"


def _ordered(methods: set[str] | list[str]) -> list[str]:
    return sorted(set(methods), key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 999, m))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_metrics(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], list[dict[str, str]]]:
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, int, str, int]] = set()
    for row in rows:
        key = (row["method"], int(row["episodes"]), row["role"])
        run_key = (*key, int(row["seed"]))
        if run_key in seen:
            raise ValueError(f"Duplicate method/budget/role/seed: {run_key}")
        seen.add(run_key)
        for field in ("validation_nmse", "mean_r2", "fit_count", "selection_total_seconds"):
            if not np.isfinite(_number(row, field)):
                raise ValueError(f"Missing/nonfinite required metric {field}: {run_key}")
        groups[key].append(row)
    return groups


def _summary_rows(groups: dict[tuple[str, int, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    output = []
    for (method, budget, role), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], METHOD_ORDER.index(item[0][0]) if item[0][0] in METHOD_ORDER else 999)):
        summary: dict[str, Any] = {
            "method": method, "episodes": budget, "role": role,
            "n_runs": len(rows), "n_seeds": len({int(r["seed"]) for r in rows}),
            "seeds": json.dumps(sorted({int(r["seed"]) for r in rows})),
        }
        for field in NUMERIC_FIELDS:
            mean, sd = _stats([_number(row, field) for row in rows])
            summary[field + "_mean"] = mean if np.isfinite(mean) else ""
            summary[field + "_sd"] = sd if np.isfinite(sd) else ""
        output.append(summary)
    return output


def _paired(
    groups: dict[tuple[str, int, str], list[dict[str, str]]],
    method: str, budget: int, reference: str = "random_search",
) -> list[dict[str, Any]]:
    candidate = {int(row["seed"]): row for row in groups.get((method, budget, "search_best"), [])}
    baseline = {int(row["seed"]): row for row in groups.get((reference, budget, "search_best"), [])}
    return [
        {
            "method": method, "reference": reference, "episodes": budget, "seed": seed,
            "delta_validation_nmse": _number(candidate[seed], "validation_nmse") - _number(baseline[seed], "validation_nmse"),
            "delta_mean_r2": _number(candidate[seed], "mean_r2") - _number(baseline[seed], "mean_r2"),
        }
        for seed in sorted(set(candidate) & set(baseline))
    ]


def _attention_plot(out: Path, attention: list[dict[str, str]], selected: str | None) -> None:
    import matplotlib.pyplot as plt
    modes = sorted({row["regularization"] for row in attention}, key=lambda v: ({"all": 0, "exclude_qk": 1, "none": 2}.get(v, 99), v))
    by_mode = {mode: [r for r in attention if r["regularization"] == mode] for mode in modes}
    x = np.arange(len(modes))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, field, title in [
        (axes[0, 0], "best_early_stop_mse", "Internal early-stop MSE (mode selection only)"),
        (axes[1, 1], "topk_validation_nmse", "Direct TopK reward-validation NMSE (diagnostic only)"),
    ]:
        means, sds = zip(*[_stats([_number(r, field) for r in by_mode[m]]) for m in modes])
        ax.bar(x, means, yerr=sds, capsize=4, color=["#2563eb", "#0d9488", "#d97706"][:len(modes)])
        for i, mode in enumerate(modes):
            vals = [_number(r, field) for r in by_mode[mode]]
            ax.scatter(np.full(len(vals), i), vals, color="black", s=15, zorder=3)
        ax.set_xticks(x, modes)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Standardized MSE")
        ax.grid(axis="y", alpha=.2)
    ax = axes[0, 1]
    for offset, field, color in [(-.10, "q_norm", "#7c3aed"), (.10, "k_norm", "#dc2626")]:
        means = [_stats([_number(r, field) for r in by_mode[m]])[0] for m in modes]
        ax.plot(x + offset, np.maximum(means, 1e-12), marker="o", label=field, color=color)
        for i, mode in enumerate(modes):
            values = [_number(r, field) for r in by_mode[mode]]
            ax.scatter(np.full(len(values), i + offset), np.maximum(values, 1e-12), color=color, alpha=.5, s=15)
    ax.set_yscale("log")
    ax.set_xticks(x, modes)
    ax.set_ylabel("Frobenius norm (log scale)")
    ax.set_title("Learned Q/K projection scale", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    ax = axes[1, 0]
    means = [_stats([_number(r, "R_max_abs_from_uniform") for r in by_mode[m]])[0] for m in modes]
    ax.plot(x, np.maximum(means, 1e-12), "o-", color="#0891b2")
    for i, mode in enumerate(modes):
        vals = [_number(r, "R_max_abs_from_uniform") for r in by_mode[mode]]
        ax.scatter(np.full(len(vals), i), np.maximum(vals, 1e-12), color="#0891b2", alpha=.5, s=15)
    ax.set_yscale("log")
    ax.set_xticks(x, modes)
    ax.set_ylabel("max |R - uniform| (values <= 1e-12 plotted at floor)")
    ax.set_title("Distinguish learned relations from uniform mixing", fontsize=10)
    ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Attention diagnostic: selected regularization = {selected or 'not recorded'}; dots are individual seeds", fontsize=11)
    fig.savefig(out / "attention_diagnostics.png", dpi=170)
    plt.close(fig)


def _search_plot(out: Path, traces: list[dict[str, str]], budgets: list[int]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, max(len(budgets), 1), figsize=(7 * max(len(budgets), 1), 5), squeeze=False, layout="constrained")
    all_methods = _ordered({r["method"] for r in traces if int(r["episodes"]) > 0})
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(all_methods)}
    for ax, budget in zip(axes[0], budgets):
        run_points: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        for row in traces:
            if int(row["episodes"]) != budget or str(row.get("complete_subset", "")).lower() not in {"1", "true", "yes"}:
                continue
            fits, score = _number(row, "fit_count"), _number(row, "validation_nmse")
            if np.isfinite(fits) and np.isfinite(score):
                run_points[(row["method"], int(row["seed"]))].append((fits, score))
        if not run_points:
            ax.text(.5, .5, "No complete-subset search trace", ha="center", transform=ax.transAxes)
            continue
        first = max(min(p[0] for p in points) for points in run_points.values())
        last = max(max(p[0] for p in points) for points in run_points.values())
        grid = np.arange(int(first), int(last) + 1, dtype=float)
        for method in all_methods:
            curves = []
            for (run_method, _seed), points in sorted(run_points.items()):
                if run_method != method:
                    continue
                array = np.asarray(sorted(points), dtype=float)
                # At a duplicate fit count, include every candidate seen at that
                # cost and retain the best. Carry the incumbent after a run ends.
                running_best = np.minimum.accumulate(array[:, 1])
                indices = np.searchsorted(array[:, 0], grid, side="right") - 1
                curves.append(running_best[np.maximum(indices, 0)])
            if curves:
                values = np.stack(curves)
                ax.step(grid, values.mean(0), where="post", label=LABELS.get(method, method), color=colors[method], linewidth=1.5)
                ax.fill_between(grid, values.min(0), values.max(0), step="post", alpha=.06, color=colors[method])
        ax.set_title(f"{budget} independently trained episodes")
        ax.set_xlabel("Actual PLSR fits (includes prefix fits)")
        ax.set_ylabel("Best complete-subset reward-validation NMSE")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Mean incumbent across seeds; shading = observed seed range; includes final greedy candidate", fontsize=10)
    fig.savefig(out / "search_comparison.png", dpi=170)
    plt.close(fig)


def _budget_plot(out: Path, groups: dict[tuple[str, int, str], list[dict[str, str]]], budgets: list[int]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, max(len(budgets), 2), figsize=(14, 9), squeeze=False, layout="constrained")
    methods = _ordered({m for (m, b, r) in groups if b > 0 and r == "search_best"})
    for ax, budget in zip(axes[0], budgets):
        available = [m for m in methods if (m, budget, "search_best") in groups]
        means, sds = zip(*[_stats([_number(r, "validation_nmse") for r in groups[(m, budget, "search_best")]]) for m in available])
        x = np.arange(len(available))
        ax.bar(x, means, yerr=sds, capsize=3, color=["#64748b" if m == "random_search" else "#2563eb" for m in available])
        greedy_x, greedy_y = [], []
        for i, method in enumerate(available):
            greedy = groups.get((method, budget, "final_policy"), [])
            if greedy:
                greedy_x.append(i)
                greedy_y.append(_stats([_number(r, "validation_nmse") for r in greedy])[0])
        if greedy_x:
            ax.scatter(greedy_x, greedy_y, marker="D", color="#dc2626", s=28, label="Final policy (mean)", zorder=3)
            ax.legend(fontsize=8)
        ax.set_xticks(x, [LABELS.get(m, m) for m in available], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Reward-validation NMSE (lower is better)")
        ax.set_title(f"{budget} episodes: search best, sample SD across seeds")
        ax.grid(axis="y", alpha=.2)
    ax = axes[1, 0]
    if len(budgets) >= 2:
        low, high = min(budgets), max(budgets)
        differences, paired_methods = [], []
        for method in methods:
            a = {int(r["seed"]): _number(r, "validation_nmse") for r in groups.get((method, low, "search_best"), [])}
            b = {int(r["seed"]): _number(r, "validation_nmse") for r in groups.get((method, high, "search_best"), [])}
            paired = [b[s] - a[s] for s in sorted(set(a) & set(b))]
            if paired:
                differences.append(_stats(paired))
                paired_methods.append(method)
        x = np.arange(len(paired_methods))
        ax.bar(x, [d[0] for d in differences], yerr=[d[1] for d in differences], capsize=3, color="#0d9488")
        ax.set_xticks(x, [LABELS.get(m, m) for m in paired_methods], rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_title(f"Paired seed change: {high} minus {low} episodes")
        ax.set_ylabel("Delta search-best NMSE (negative = lower)")
        ax.grid(axis="y", alpha=.2)
    else:
        ax.text(.5, .5, "Two budgets required", ha="center", transform=ax.transAxes)
    ax = axes[1, 1]
    policy_methods = [m for m in methods if any((m, b, "final_policy") in groups for b in budgets)]
    x = np.arange(len(policy_methods))
    width = .7 / max(len(budgets), 1)
    for index, budget in enumerate(budgets):
        gaps = []
        for method in policy_methods:
            best = {int(r["seed"]): _number(r, "validation_nmse") for r in groups.get((method, budget, "search_best"), [])}
            policy = {int(r["seed"]): _number(r, "validation_nmse") for r in groups.get((method, budget, "final_policy"), [])}
            gap = [policy[s] - best[s] for s in sorted(set(best) & set(policy))]
            gaps.append(_stats(gap)[0])
        ax.bar(x + (index - (len(budgets) - 1) / 2) * width, gaps, width=width, label=f"{budget} episodes")
    ax.set_xticks(x, [LABELS.get(m, m) for m in policy_methods], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Final-policy minus search-best NMSE")
    ax.set_title("Search quality and final-policy quality are different")
    ax.grid(axis="y", alpha=.2)
    if policy_methods:
        ax.legend(fontsize=8)
    fig.savefig(out / "budget_comparison.png", dpi=170)
    plt.close(fig)


def build_report(out_dir: Path) -> None:
    """Build development-only tables and plots from the completed run outputs."""
    out = Path(out_dir)
    attention = _read_csv(out / "attention_training_summary.csv")
    metrics = _read_csv(out / "selection_metrics.csv")
    traces = _read_csv(out / "search_trace.csv")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    selection_path = out / "attention_mode_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    if not attention or not metrics:
        raise ValueError("Cannot report an empty attention or selection diagnostic.")
    groups = _group_metrics(metrics)
    summaries = _summary_rows(groups)
    _write_csv(out / "summary_by_method_budget.csv", summaries)
    budgets = sorted({int(row["episodes"]) for row in metrics if int(row["episodes"]) > 0})
    methods = _ordered({row["method"] for row in metrics if row["role"] == "search_best"})
    seeds = sorted({int(row["seed"]) for row in metrics})
    selected = selection.get("selected_regularization")
    attention_modes = sorted({row["regularization"] for row in attention}, key=lambda v: ({"all": 0, "exclude_qk": 1, "none": 2}.get(v, 99), v))
    lines = [
        "# 注意力与RL开发诊断结果", "",
        "本报告只汇总拟合数据内部早停成绩及奖励验证选择成绩。没有读取测试数据、测试预测或旧测试结果；这些分数不能作为独立泛化性能证据。", "",
        f"算法种子：{', '.join(map(str, seeds))}；独立回合预算：{', '.join(map(str, budgets))}。表中±为种子间样本标准差（ddof=1），不是置信区间。", "",
        "48与200回合为各自从头训练的配置，采用各自完整探索日程；200回合配置的前48回合不能作为独立48回合配置使用。", "",
        "## 注意力正则化诊断与模式选择", "",
        f"选择规则：{selection.get('selection_rule', protocol.get('attention_mode_selection_rule', '见protocol.json'))}", "",
        f"搜索前选定模式：`{selected}`。TopK奖励验证成绩仅用于事后诊断，不参与这个模式选择。", "",
        "|正则模式|种子数|内部早停MSE|Q范数均值|K范数均值|max\|R−均匀\|均值|门控归一化熵均值|TopK奖励验证NMSE|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in attention_modes:
        rows = [row for row in attention if row["regularization"] == mode]
        means = {f: _stats([_number(r, f) for r in rows])[0] for f in ["q_norm", "k_norm", "R_max_abs_from_uniform", "gate_normalized_entropy_mean"]}
        lines.append(
            f"|{mode}|{len(rows)}|{_mean_sd(rows, 'best_early_stop_mse')}|{means['q_norm']:.3e}|{means['k_norm']:.3e}|"
            f"{means['R_max_abs_from_uniform']:.3e}|{means['gate_normalized_entropy_mean']:.6f}|{_mean_sd(rows, 'topk_validation_nmse')}|"
        )
    lines += ["", "门控归一化熵接近1表示权重接近均匀；关系偏离均匀或Q/K范数增大本身不证明关系对选谱有用。内部早停预测MSE选择的是预训练模式，并不保证该模式对下游PLSR选列最优。", "",
              "![注意力诊断](attention_diagnostics.png)", "", "## 静态参考方法", "",
              "|方法|奖励验证NMSE|平均R²|SSC R²|pH R²|硬度R²|fit均值|独立运行秒数均值|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    static_methods = _ordered({m for m, b, role in groups if b == 0 and role == "static"})
    for method in static_methods:
        rows = groups[(method, 0, "static")]
        lines.append(f"|{method}|{_mean_sd(rows,'validation_nmse')}|{_mean_sd(rows,'mean_r2')}|{_mean_sd(rows,'SSC_r2')}|{_mean_sd(rows,'pH_r2')}|{_mean_sd(rows,'firmness_r2')}|{_fmt(_stats([_number(r,'fit_count') for r in rows])[0],1)}|{_fmt(_stats([_number(r,'selection_total_seconds') for r in rows])[0],2)}|")
    paired_rows = []
    for budget in budgets:
        lines += ["", f"## {budget}回合：搜索最好完整子集", "",
                  "|方法|种子数|奖励验证NMSE|平均R²|fit均值|独立运行秒数均值|", "|---|---:|---:|---:|---:|---:|"]
        for method in methods:
            rows = groups.get((method, budget, "search_best"), [])
            if rows:
                lines.append(f"|{method}|{len(rows)}|{_mean_sd(rows,'validation_nmse')}|{_mean_sd(rows,'mean_r2')}|{_fmt(_stats([_number(r,'fit_count') for r in rows])[0],1)}|{_fmt(_stats([_number(r,'selection_total_seconds') for r in rows])[0],2)}|")
        lines += ["", "与同种子、同预算随机搜索配对。ΔNMSE=方法−随机搜索，负值较好；ΔR²=方法−随机搜索，正值较好。NMSE是本轮选择目标，两者的排序可能不同。", "",
                  "|方法|各seed ΔNMSE|ΔNMSE均值|NMSE胜出数|各seed ΔR²|ΔR²均值|R²胜出数|", "|---|---|---:|---:|---|---:|---:|"]
        for method in methods:
            if method == "random_search":
                continue
            pairs = _paired(groups, method, budget)
            paired_rows.extend(pairs)
            if pairs:
                nmse = [r["delta_validation_nmse"] for r in pairs]
                r2 = [r["delta_mean_r2"] for r in pairs]
                nmse_text = "; ".join(f"{r['seed']}: {r['delta_validation_nmse']:+.5f}" for r in pairs)
                r2_text = "; ".join(f"{r['seed']}: {r['delta_mean_r2']:+.5f}" for r in pairs)
                lines.append(f"|{method}|{nmse_text}|{np.mean(nmse):+.5f}|{sum(v<0 for v in nmse)}/{len(nmse)}|{r2_text}|{np.mean(r2):+.5f}|{sum(v>0 for v in r2)}/{len(r2)}|")
    _write_csv(out / "paired_vs_random.csv", paired_rows)
    lines += ["", "## 最终贪心策略与搜索最好子集", "",
              "|回合|方法|最终策略NMSE|搜索最好NMSE|逐seed最终−最好均值|", "|---:|---|---:|---:|---:|"]
    for budget in budgets:
        for method in methods:
            rows = groups.get((method, budget, "final_policy"), [])
            best_rows = groups.get((method, budget, "search_best"), [])
            if not rows or not best_rows:
                continue
            best_by_seed = {int(r["seed"]): _number(r,"validation_nmse") for r in best_rows}
            gaps = [_number(r,"validation_nmse") - best_by_seed[int(r["seed"])] for r in rows if int(r["seed"]) in best_by_seed]
            lines.append(f"|{budget}|{method}|{_mean_sd(rows,'validation_nmse')}|{_mean_sd(best_rows,'validation_nmse')}|{np.mean(gaps):+.5f}|")
    lines += ["", "![搜索曲线](search_comparison.png)", "",
              "曲线仅比较完整子集；DQN前缀评价仍计入横轴fit成本。每种子采用已观测最好子集的阶梯曲线，再求均值，阴影为观测种子范围；运行结束后保留最后最好值，绘图从所有运行均已有完整子集的共同起点开始。最终贪心候选也参与搜索最好子集比较。", "",
              "![预算与策略差距](budget_comparison.png)", "", "## 依据当前开发结果的后续判断", ""]
    for budget in budgets:
        random_rows = groups.get(("random_search", budget, "search_best"), [])
        if not random_rows:
            lines.append(f"- {budget}回合缺少随机搜索记录，暂不能完成该预算的公平比较。")
            continue
        random_mean = _stats([_number(r,"validation_nmse") for r in random_rows])[0]
        candidates = [(m, _stats([_number(r,"validation_nmse") for r in groups.get((m,budget,"search_best"),[])])[0]) for m in methods if m != "random_search" and groups.get((m,budget,"search_best"))]
        if candidates:
            method, value = min(candidates, key=lambda pair: pair[1])
            direction = "低于" if value < random_mean else "未低于"
            lines.append(f"- {budget}回合下，预定DQN版本中平均NMSE最低的是`{method}`（{value:.5f}），{direction}随机搜索（{random_mean:.5f}）；这是开发选择成绩，不能据此声称独立泛化优势。")
        for candidate, reference, question in [
            ("gate_learned", "gate_identity", "学习关系相对单位阵"),
            ("gate_learned", "gate_uniform", "学习关系相对均匀混合"),
            ("gate_learned_qprior", "gate_learned", "增加Q先验"),
        ]:
            pairs = _paired(groups, candidate, budget, reference)
            if pairs:
                vals = [p["delta_validation_nmse"] for p in pairs]
                lines.append(f"- {budget}回合，{question}的配对ΔNMSE均值为{np.mean(vals):+.5f}，{sum(v<0 for v in vals)}/{len(vals)}种子更低。三个种子不足以仅凭均值或胜出数认定稳定收益。")
    lines += ["", "若更长预算只改善搜索最好值、最终策略仍明显落后，应先诊断策略学习与提示接入；若关系相对单位阵/均匀混合没有一致收益，应简化相应模块。不要依据本轮开发结果继续追加预算直到某版本获胜。正式主评价需要另行锁定流程与预算，重新在每个外层训练折完成整个选择流程。", "",
              "## 成本与证据边界", "",
              "- 随机搜索按每个种子、预算下预定DQN集合的最大实际底层PLSR拟合次数对齐；这不等于完整子集数量或墙钟时间一致。",
              "- 独立运行秒数包括方法需要承担的注意力预训练成本。共享训练只实际执行一次，但不能将共享复用视作方法免费；本表不通过累加各方法独立成本估算整次运行墙钟。",
              "- 本轮固定PLSR配置用于控制评价器；全谱不是调优后的性能上限。RMSE/MAE保持原表尺度。",
              "- 三个算法种子共用同一拟合/奖励验证划分，不能当作三个独立数据划分。这里不计算显著性p值。",
              "- 模式选择只依据拟合部分内部早停MSE；奖励验证TopK、RL成绩与最终测试成绩不参与该选择。",
              "- 完整逐目标指标及全部角色见selection_metrics.csv；汇总见summary_by_method_budget.csv；配对原始差值见paired_vs_random.csv。", ""]
    notes = protocol.get("notes", [])
    if notes:
        lines += ["## 本次协议备注", ""]
        lines += [f"- {note}" for note in notes] if isinstance(notes, list) else [str(notes)]
        lines.append("")
    import matplotlib
    matplotlib.use("Agg")
    _attention_plot(out, attention, selected)
    _search_plot(out, traces, budgets)
    _budget_plot(out, groups, budgets)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
