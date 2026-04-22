"""Сбор результатов одного прогона: score из логов ядра и метрики агентов."""

from __future__ import annotations

import glob
import json
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Типы
# ---------------------------------------------------------------------------

@dataclass
class KernelOutcome:
    score: Optional[float] = None
    score_initial: Optional[float] = None
    civilians_alive: Optional[int] = None
    civilians_total: Optional[int] = None
    buildings_intact: Optional[int] = None
    buildings_total: Optional[int] = None
    final_tick: Optional[int] = None


@dataclass
class AgentsOutcome:
    freeze_pct: float = 0.0
    tick_p95_ms: float = 0.0
    tick_mean_ms: float = 0.0
    utility_ms_mean: float = 0.0
    target_switches_total: int = 0
    stuck_detected_total: int = 0
    ak_say_total: int = 0
    total_ticks: int = 0
    agents_count: int = 0


@dataclass
class RunMetrics:
    kernel: KernelOutcome
    agents: AgentsOutcome
    wall_clock_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {}
        d.update({f"kernel.{k}": v for k, v in asdict(self.kernel).items()})
        d.update({f"agents.{k}": v for k, v in asdict(self.agents).items()})
        d["wall_clock_s"] = round(self.wall_clock_s, 2)
        d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Парсинг kernel-лога
# ---------------------------------------------------------------------------

# Порядок важен: сначала ищем «финальный», затем «обычный» "Score: X".
# Строгое двоеточие после слова Score отсекает мусор "Score calculation took : Nms".
SCORE_PATTERNS = [
    re.compile(r"[Ff]inal\s+score\s*[:=]\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"[Ss]core\s*function\s*result\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"RSL\d+ScoreFunction[^-+\d\n]*?([-+]?\d+(?:\.\d+)?)"),
    re.compile(r"\bScore\s*:\s*([-+]?\d+(?:\.\d+)?)"),
]


def _last_float(pattern: re.Pattern[str], text: str) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_kernel_logs(kernel_log_dir: Path) -> KernelOutcome:
    """Извлекает score и статистику из логов ядра RCRS (kernel.log / *.log).

    Формат вывода в RCRS менялся между версиями; пробуем несколько
    regex-ов. Если score не найден — возвращается KernelOutcome с None.
    """
    outcome = KernelOutcome()
    if not kernel_log_dir.exists():
        return outcome

    texts: list[str] = []
    for log_file in sorted(kernel_log_dir.glob("*.log")):
        try:
            texts.append(log_file.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    combined = "\n".join(texts)
    if not combined:
        return outcome

    for pat in SCORE_PATTERNS:
        val = _last_float(pat, combined)
        if val is not None:
            outcome.score = val
            break

    # Финальный такт.
    tick_match = re.findall(r"[Tt]imestep\s*(\d+)", combined)
    if tick_match:
        try:
            outcome.final_tick = int(tick_match[-1])
        except ValueError:
            pass

    # Civilians / buildings сводки (опционально).
    civ = re.search(r"civilians\s+alive[^\d]*(\d+)\s*/\s*(\d+)", combined, re.IGNORECASE)
    if civ:
        outcome.civilians_alive = int(civ.group(1))
        outcome.civilians_total = int(civ.group(2))

    bld = re.search(r"buildings\s+intact[^\d]*(\d+)\s*/\s*(\d+)", combined, re.IGNORECASE)
    if bld:
        outcome.buildings_intact = int(bld.group(1))
        outcome.buildings_total = int(bld.group(2))

    return outcome


# ---------------------------------------------------------------------------
# Парсинг summary_*.json от наших агентов
# ---------------------------------------------------------------------------

def collect_agents_outcome(agents_log_dir: Path) -> AgentsOutcome:
    outcome = AgentsOutcome()
    if not agents_log_dir.exists():
        return outcome

    summaries = sorted(glob.glob(str(agents_log_dir / "summary_*.json")))
    tick_p95s: list[float] = []
    tick_means: list[float] = []
    util_means: list[float] = []
    freeze_total = 0
    ticks_total = 0
    target_switches = 0
    stuck_total = 0
    ak_say_total = 0

    for f in summaries:
        try:
            j = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        outcome.agents_count += 1
        ticks = int(j.get("total_ticks", 0))
        ticks_total += ticks
        freeze_total += int(j.get("freeze_count", 0))
        tick_ms = j.get("tick_ms", {}) or {}
        if tick_ms.get("p95") is not None:
            tick_p95s.append(float(tick_ms["p95"]))
        if tick_ms.get("mean") is not None:
            tick_means.append(float(tick_ms["mean"]))
        util_ms = j.get("utility_ms", {}) or {}
        if util_ms.get("mean") is not None:
            util_means.append(float(util_ms["mean"]))
        counters = j.get("counters", {}) or {}
        target_switches += int(counters.get("target_switched", 0))
        stuck_total += int(counters.get("stuck_detected", 0))
        ak_say_total += int(counters.get("ak_say", 0))

    outcome.total_ticks = ticks_total
    outcome.freeze_pct = (100.0 * freeze_total / ticks_total) if ticks_total else 0.0
    outcome.tick_p95_ms = statistics.fmean(tick_p95s) if tick_p95s else 0.0
    outcome.tick_mean_ms = statistics.fmean(tick_means) if tick_means else 0.0
    outcome.utility_ms_mean = statistics.fmean(util_means) if util_means else 0.0
    outcome.target_switches_total = target_switches
    outcome.stuck_detected_total = stuck_total
    outcome.ak_say_total = ak_say_total
    return outcome
