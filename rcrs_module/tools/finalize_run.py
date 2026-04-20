"""Пост-прогонный агрегатор.

Читаю `rcrs_module/agent.cfg`, собираю метрики из `summary_*.json` и
`tick_*.csv` в `log_dir`, добавляю одну строку в `log_dir/runs.md`.
Поля Map / ADF Score / Наш Score / Δ % оставляю как «?» — я их
дописываю вручную; остальное заполняется автоматически.

Работает автономным скриптом: никаких зависимостей от кода агента,
не трогает алгоритмы и не пишет ничего, кроме сводной таблицы.
"""

from __future__ import annotations

import configparser
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "agent.cfg"


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _gini(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    tot = sum(s)
    if tot == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(s))
    return (2 * cum) / (n * tot) - (n + 1) / n


def _resolve_log_dir(cfg: configparser.ConfigParser) -> Path:
    raw = cfg.get("logging", "log_dir", fallback=str(ROOT.parent / "log"))
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _aggregate_summaries(log_dir: Path) -> tuple[float, float, float]:
    """Возвращаю (freeze_pct, t_total_p95_ms, t_util_mean_ms)."""
    freeze_total = 0
    ticks_total = 0
    tick_p95s: list[float] = []
    util_means: list[float] = []
    for f in sorted(glob.glob(str(log_dir / "summary_*.json"))):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        freeze_total += int(j.get("freeze_count", 0))
        ticks_total += int(j.get("total_ticks", 0))
        p95 = j.get("tick_ms", {}).get("p95")
        if p95 is not None:
            tick_p95s.append(float(p95))
        util_mean = j.get("utility_ms", {}).get("mean")
        if util_mean is not None:
            util_means.append(float(util_mean))

    freeze_pct = (100.0 * freeze_total / ticks_total) if ticks_total else 0.0
    t_total_p95 = statistics.fmean(tick_p95s) if tick_p95s else 0.0
    t_util_mean = statistics.fmean(util_means) if util_means else 0.0
    return freeze_pct, t_total_p95, t_util_mean


def _aggregate_density(log_dir: Path) -> tuple[str, str]:
    """Кросс-агентный анализ из tick_*.csv: peak_density и Gini по типам."""
    # density[agent_type][tick][target_id] = count
    density: dict[str, dict[int, dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)),
    )
    for f in glob.glob(str(log_dir / "tick_*.csv")):
        try:
            fh = open(f, newline="", encoding="utf-8")
        except OSError:
            continue
        with fh:
            r = csv.DictReader(fh)
            for row in r:
                try:
                    tid = int(row.get("target_id") or -1)
                    tk = int(row["tick_id"])
                except (TypeError, ValueError, KeyError):
                    continue
                if tid == -1:
                    continue
                at = row.get("agent_type", "") or "?"
                density[at][tk][tid] += 1

    peaks: list[int] = []
    ginis: list[float] = []
    for at, ticks in density.items():
        peaks_type: list[int] = []
        ginis_type: list[float] = []
        for counts in ticks.values():
            vals = list(counts.values())
            if not vals:
                continue
            peaks_type.append(max(vals))
            ginis_type.append(_gini(vals))
        if peaks_type:
            peaks.append(max(peaks_type))
        if ginis_type:
            ginis.append(statistics.fmean(ginis_type))

    if not peaks:
        peak_str = "?"
    elif len(set(peaks)) == 1:
        peak_str = str(peaks[0])
    else:
        peak_str = f"{min(peaks)}–{max(peaks)}"

    if not ginis:
        gini_str = "?"
    elif len(ginis) == 1:
        gini_str = f"{ginis[0]:.2f}"
    else:
        gini_str = f"{min(ginis):.2f}–{max(ginis):.2f}"
    return peak_str, gini_str


HEADER = (
    "| # | Map | ADF Score | Наш Score | Δ % | freeze % | "
    "t_total p95 (мс) | t_util mean (мс) | peak_density | Gini |"
)
SEPARATOR = (
    "|---|-----|-----------|-----------|-----|----------|"
    "------------------|------------------|--------------|------|"
)


def _next_run_id(table_path: Path) -> tuple[int, list[str]]:
    if not table_path.exists():
        return 1, [HEADER, SEPARATOR]
    lines = table_path.read_text(encoding="utf-8").splitlines()
    data_rows = [
        ln for ln in lines
        if ln.startswith("| ")
        and not ln.startswith("| #")
        and not ln.startswith("|---")
    ]
    return len(data_rows) + 1, lines


def main() -> int:
    cfg = configparser.ConfigParser()
    cfg.read(CFG_PATH)
    if not cfg.getboolean("run_table", "enabled", fallback=False):
        print("[finalize_run] отключено в agent.cfg ([run_table] enabled=false), выхожу.")
        return 0

    log_dir = _resolve_log_dir(cfg)
    if not log_dir.exists():
        print(f"[finalize_run] каталог логов не найден: {log_dir}")
        return 1

    table_name = cfg.get("run_table", "file", fallback="runs.md")
    table_path = log_dir / table_name

    freeze_pct, t_total_p95, t_util_mean = _aggregate_summaries(log_dir)
    peak_str, gini_str = _aggregate_density(log_dir)

    run_id, existing_lines = _next_run_id(table_path)
    row = (
        f"| {run_id} | ? | ? | ? | ? | "
        f"{freeze_pct:.3f} | {t_total_p95:.1f} | {t_util_mean:.2f} | "
        f"{peak_str} | {gini_str} |"
    )
    out = "\n".join(existing_lines + [row]) + "\n"
    try:
        table_path.write_text(out, encoding="utf-8")
    except OSError as exc:
        print(f"[finalize_run] не смог записать таблицу: {exc}")
        return 1

    print(
        f"[finalize_run] прогон #{run_id} → {table_path}\n"
        f"  freeze%={freeze_pct:.3f}, t_total_p95≈{t_total_p95:.1f} мс, "
        f"t_util_mean≈{t_util_mean:.2f} мс, peak_density={peak_str}, "
        f"gini={gini_str}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
