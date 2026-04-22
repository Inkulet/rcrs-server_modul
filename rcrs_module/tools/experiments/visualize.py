"""Сохранение таблиц и диаграмм по набору прогонов.

matplotlib импортируется «лениво» — если библиотеки нет, таблицы
создаются всё равно, только графики пропускаются.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from .configs import RunSpec
from .metrics import RunMetrics


logger = logging.getLogger("experiments.visualize")


# ---------------------------------------------------------------------------
# CSV / Markdown
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_name", "role", "axis", "axis_value",
    "w_c_fire", "w_d_fire", "w_e_fire", "w_n_fire",
    "w_c_amb", "w_d_amb", "w_e_amb", "w_n_amb",
    "w_c_pol", "w_d_pol", "w_e_pol", "w_n_pol",
    "radio_enabled",
    "score", "final_tick",
    "freeze_pct", "tick_p95_ms", "tick_mean_ms",
    "utility_ms_mean", "target_switches", "stuck_detected",
    "ak_say_total", "agents_count", "wall_clock_s", "error",
]


def _expand_weights(w, prefix: str) -> dict:
    if w is None:
        return {f"w_c_{prefix}": "", f"w_d_{prefix}": "", f"w_e_{prefix}": "", f"w_n_{prefix}": ""}
    return {
        f"w_c_{prefix}": f"{w[0]:.4f}",
        f"w_d_{prefix}": f"{w[1]:.4f}",
        f"w_e_{prefix}": f"{w[2]:.4f}",
        f"w_n_{prefix}": f"{w[3]:.4f}",
    }


def _row(spec: RunSpec, metrics: RunMetrics) -> dict:
    row: dict = {
        "run_name": spec.name,
        "role": spec.ablation_role,
        "axis": spec.ablation_axis,
        "axis_value": spec.ablation_value,
        "radio_enabled": int(spec.radio_enabled),
        "score": metrics.kernel.score if metrics.kernel.score is not None else "",
        "final_tick": metrics.kernel.final_tick if metrics.kernel.final_tick is not None else "",
        "freeze_pct": round(metrics.agents.freeze_pct, 3),
        "tick_p95_ms": round(metrics.agents.tick_p95_ms, 2),
        "tick_mean_ms": round(metrics.agents.tick_mean_ms, 2),
        "utility_ms_mean": round(metrics.agents.utility_ms_mean, 2),
        "target_switches": metrics.agents.target_switches_total,
        "stuck_detected": metrics.agents.stuck_detected_total,
        "ak_say_total": metrics.agents.ak_say_total,
        "agents_count": metrics.agents.agents_count,
        "wall_clock_s": round(metrics.wall_clock_s, 1),
        "error": metrics.error or "",
    }
    row.update(_expand_weights(spec.weights_fire, "fire"))
    row.update(_expand_weights(spec.weights_ambulance, "amb"))
    row.update(_expand_weights(spec.weights_police, "pol"))
    return row


def save_csv(results: list[tuple[RunSpec, RunMetrics]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for spec, metrics in results:
            w.writerow(_row(spec, metrics))
    logger.info("CSV сохранён: %s", path)


def save_markdown(results: list[tuple[RunSpec, RunMetrics]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Результаты ablation-исследования RCRS")
    lines.append("")
    lines.append(f"Всего прогонов: **{len(results)}**")
    lines.append("")
    scored = [(s, m) for s, m in results if m.kernel.score is not None]
    if scored:
        best = max(scored, key=lambda p: p[1].kernel.score)
        worst = min(scored, key=lambda p: p[1].kernel.score)
        lines.append(f"- Лучший score: **{best[1].kernel.score:.4f}** ({best[0].name})")
        lines.append(f"- Худший score: **{worst[1].kernel.score:.4f}** ({worst[0].name})")
        lines.append("")

    lines.append("| run | role | axis | value | score | final_tick | freeze% | tick_p95 ms | util_mean ms | switches | error |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for spec, metrics in results:
        score = f"{metrics.kernel.score:.4f}" if metrics.kernel.score is not None else "—"
        lines.append(
            f"| {spec.name} | {spec.ablation_role} | {spec.ablation_axis or '—'} | "
            f"{spec.ablation_value:.2f} | {score} | {metrics.kernel.final_tick or '—'} | "
            f"{metrics.agents.freeze_pct:.2f} | {metrics.agents.tick_p95_ms:.1f} | "
            f"{metrics.agents.utility_ms_mean:.1f} | {metrics.agents.target_switches_total} | "
            f"{metrics.error or '—'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Markdown сохранён: %s", path)


# ---------------------------------------------------------------------------
# Графики (matplotlib)
# ---------------------------------------------------------------------------

def _try_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib не установлен — графики пропускаются")
        return None


def plot_weight_sweep(results: list[tuple[RunSpec, RunMetrics]], out_dir: Path) -> None:
    plt = _try_import_plt()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Группируем по (role, axis).
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for spec, metrics in results:
        if not spec.ablation_axis or spec.ablation_axis == "baseline":
            continue
        if metrics.kernel.score is None:
            continue
        key = (spec.ablation_role, spec.ablation_axis)
        groups.setdefault(key, []).append((spec.ablation_value, metrics.kernel.score))

    for (role, axis), pairs in groups.items():
        if len(pairs) < 2:
            continue
        pairs.sort()
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title(f"Score vs {axis}  ({role})")
        ax.set_xlabel(axis)
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep_{role}_{axis}.png", dpi=140)
        plt.close(fig)
    logger.info("sweep-графики сохранены: %s", out_dir)


def plot_heatmap_wc_wd(results: list[tuple[RunSpec, RunMetrics]], out_dir: Path) -> None:
    plt = _try_import_plt()
    if plt is None:
        return
    heatmap_runs = [(s, m) for s, m in results if s.ablation_role == "heatmap" and m.kernel.score is not None]
    if len(heatmap_runs) < 4:
        return

    # Достаём w_c и w_d из весов fire (на heatmap все роли одинаковы).
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for spec, metrics in heatmap_runs:
        if not spec.weights_fire:
            continue
        xs.append(spec.weights_fire[0])
        ys.append(spec.weights_fire[1])
        zs.append(metrics.kernel.score)

    if not xs:
        return

    x_levels = sorted(set(xs))
    y_levels = sorted(set(ys))
    grid = [[None] * len(x_levels) for _ in y_levels]
    for x, y, z in zip(xs, ys, zs):
        grid[y_levels.index(y)][x_levels.index(x)] = z

    fig, ax = plt.subplots(figsize=(7, 5))
    import numpy as np
    arr = np.array(
        [[v if v is not None else np.nan for v in row] for row in grid],
        dtype=float,
    )
    im = ax.imshow(arr, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x_levels)))
    ax.set_yticks(range(len(y_levels)))
    ax.set_xticklabels([f"{v:.2f}" for v in x_levels])
    ax.set_yticklabels([f"{v:.2f}" for v in y_levels])
    ax.set_xlabel("w_c (срочность)")
    ax.set_ylabel("w_d (расстояние)")
    ax.set_title("Heatmap: Score vs (w_c, w_d)  при w_e=w_n=(1−w_c−w_d)/2")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v is None:
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_wc_wd.png", dpi=140)
    plt.close(fig)
    logger.info("heatmap сохранён: %s/heatmap_wc_wd.png", out_dir)


def plot_score_bars(results: list[tuple[RunSpec, RunMetrics]], out_dir: Path, top_n: int = 15) -> None:
    plt = _try_import_plt()
    if plt is None:
        return
    scored = [(s, m) for s, m in results if m.kernel.score is not None]
    if not scored:
        return
    scored.sort(key=lambda p: p[1].kernel.score, reverse=True)
    scored = scored[:top_n]
    names = [s.name for s, _ in scored]
    vals = [m.kernel.score for _, m in scored]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(names))))
    ax.barh(range(len(names)), vals, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title(f"Топ-{len(names)} конфигураций по score")
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "top_configs.png", dpi=140)
    plt.close(fig)


def plot_perf_vs_score(results: list[tuple[RunSpec, RunMetrics]], out_dir: Path) -> None:
    plt = _try_import_plt()
    if plt is None:
        return
    data = [(s, m) for s, m in results if m.kernel.score is not None and m.agents.tick_p95_ms > 0]
    if len(data) < 3:
        return
    xs = [m.agents.tick_p95_ms for _, m in data]
    ys = [m.kernel.score for _, m in data]
    labels = [s.name for s, _ in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.7, color="darkorange")
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), fontsize=7, alpha=0.6)
    ax.set_xlabel("tick_p95 (мс)")
    ax.set_ylabel("Score")
    ax.set_title("Производительность агента vs качество (score)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perf_vs_score.png", dpi=140)
    plt.close(fig)


def render_all(results: list[tuple[RunSpec, RunMetrics]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(results, out_dir / "results.csv")
    save_markdown(results, out_dir / "results.md")
    plot_weight_sweep(results, out_dir)
    plot_heatmap_wc_wd(results, out_dir)
    plot_score_bars(results, out_dir)
    plot_perf_vs_score(results, out_dir)
