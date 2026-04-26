from __future__ import annotations

import csv
import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def _fmt(xs: list[float]) -> tuple[float, float, float, float, float]:
    return (
        statistics.fmean(xs),
        _percentile(xs, 50),
        _percentile(xs, 95),
        _percentile(xs, 99),
        max(xs),
    )


# Фазы, которые я замеряю каждый такт. Порядок влияет только на
# оформление отчёта. Все фазы атомарны (start→stop без вложенности),
# кроме `tick` — он охватывает всё.
_PHASES: tuple[str, ...] = (
    "perception",     # разбор KASense + world_model.apply_perception
    "dijkstra",       # fill_path_distances (O(V+E·logV) на всю карту)
    "pre_filter",     # dispatcher.filter_tasks (heuristic-отсев)
    "utility",        # compute_urgency/effort/distance/social + aggregator
    "selection",      # TargetSelector (blacklist + hysteresis)
    "dispatch",       # dispatch_action (отправка команды + геометрия)
    "explore",        # fallback random_walk / exploration_path
    "say",            # AKSay broadcast
    "tick",           # полный такт (sanity check, сумма ≈ tick)
)

# Счётчики событий — я агрегирую по типам команд и по событиям
# решений (hysteresis, blacklist, stuck).
_COUNTERS: tuple[str, ...] = (
    "ak_move", "ak_rescue", "ak_extinguish", "ak_clear_area",
    "ak_load", "ak_unload", "ak_rest", "ak_say",
    "target_selected", "target_switched", "target_blacklisted",
    "stuck_detected", "over_budget", "unreachable_target",
    "anti_stuck_forced_move", "early_unload_dead",
    "fire_wait_for_police",
)


class MetricsCollector:
    """Цельная система замеров: фазы, счётчики, gauge-ы → time.md.

    Использование:
        m = MetricsCollector(enabled=True)
        m.start_tick()
        with m.phase("perception"):
            ...
        m.inc("ak_move")
        m.gauge("n_tasks", len(tasks))
        m.stop_tick(tick_number=42)
        ...
        m.write_report(path, agent_id=1, agent_type="FIRE_BRIGADE")

    При `enabled=False` все методы становятся no-op, но класс остаётся
    валидным — не надо ветвить каждую точку замера в вызывающем коде.
    """

    def __init__(
        self,
        enabled: bool = True,
        budget_ms: float = 100.0,
        tick_csv_path: Optional[Path] = None,
        events_csv_path: Optional[Path] = None,
        summary_json_path: Optional[Path] = None,
        budget_total_ms: float = 1000.0,
    ) -> None:
        self.enabled = enabled
        self.budget_ms = budget_ms
        self.budget_total_ms = budget_total_ms

        # Времена фаз текущего такта (мс). Ключи совпадают с _PHASES.
        self._current_tick_phases: dict[str, float] = {}
        self._current_tick_start: float = 0.0

        # Полная история: список записей на такт.
        # record = {
        #   "tick": int, "n_tasks": int, "cache": int, "visible": int,
        #   "allies": int, "phases": {phase → ms},
        # }
        self._records: list[dict] = []

        # Счётчики и gauge-ы текущего такта.
        self._current_counters: dict[str, int] = {c: 0 for c in _COUNTERS}
        self._counters_total: dict[str, int] = {c: 0 for c in _COUNTERS}
        self._current_gauges: dict[str, int] = {}

        # Расширенные per-tick поля (воронка задач, таргет, разложение utility,
        # ресурсы, флаги событий). Заполняются извне через set_* методы,
        # сбрасываются в start_tick, пишутся одной строкой в stop_tick.
        self._tick_fields: dict[str, Any] = {}

        # CSV/JSON writers. Ленивая инициализация — не создаю файл, пока не
        # захлопнется первый stop_tick, чтобы пустых файлов не оставалось.
        self._tick_csv_path = tick_csv_path
        self._events_csv_path = events_csv_path
        self._summary_json_path = summary_json_path
        self._tick_csv_file: Optional[object] = None
        self._tick_csv_writer: Optional[csv.DictWriter] = None
        self._events_csv_file: Optional[object] = None
        self._events_csv_writer: Optional[csv.DictWriter] = None
        self._freeze_count: int = 0
        self._comm_bytes_coord: int = 0
        self._comm_bytes_sensor: int = 0
        self._session_t0 = time.perf_counter()

    # --- фазы -----------------------------------------------------------

    def start_tick(self) -> None:
        if not self.enabled:
            return
        self._current_tick_phases = {}
        self._current_counters = {c: 0 for c in _COUNTERS}
        self._current_gauges = {}
        self._tick_fields = {}
        self._current_tick_start = time.perf_counter()

    def stop_tick(self, tick_number: int) -> None:
        if not self.enabled:
            return
        total_ms = (time.perf_counter() - self._current_tick_start) * 1000.0
        self._current_tick_phases.setdefault("tick", total_ms)

        if self._current_tick_phases.get("utility", 0.0) > self.budget_ms:
            self._current_counters["over_budget"] += 1

        for k, v in self._current_counters.items():
            self._counters_total[k] += v

        rec: dict = {
            "tick": tick_number,
            "phases": dict(self._current_tick_phases),
            "counters": dict(self._current_counters),
            "gauges": dict(self._current_gauges),
        }
        self._records.append(rec)

        # Freeze = превышение 1000 мс по полному такту. Считаю всегда, даже
        # если events-CSV отключён, — нужно для summary.json / freeze_rate.
        if total_ms > self.budget_total_ms:
            self._freeze_count += 1
            self.record_event(
                "freeze",
                tick=tick_number,
                agent_id=self._tick_fields.get("agent_id", -1),
                t_total_ms=round(total_ms, 3),
            )

        # Запись per-tick CSV (уровень 1 детализированных замеров).
        if self._tick_csv_path is not None:
            try:
                self._write_tick_row(tick_number, total_ms)
            except (OSError, ValueError) as exc:
                logger.warning("Metrics: запись tick-CSV не выполнена — %s", exc)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Если фаза вложенно вызывается дважды — суммируем.
            self._current_tick_phases[name] = (
                self._current_tick_phases.get(name, 0.0) + elapsed_ms
            )

    # --- счётчики и gauge-ы ----------------------------------------------

    def inc(self, name: str, delta: int = 1) -> None:
        if not self.enabled:
            return
        # Неизвестные счётчики не теряем: добавляем в словарь динамически.
        # Итоговые тоже обновим в stop_tick.
        if name not in self._current_counters:
            self._current_counters[name] = 0
            self._counters_total.setdefault(name, 0)
        self._current_counters[name] += delta

    def gauge(self, name: str, value: int) -> None:
        if not self.enabled:
            return
        self._current_gauges[name] = value

    # --- аксессоры для внешнего кода -------------------------------------

    @property
    def tick_count(self) -> int:
        return len(self._records)

    def last_tick_ms(self) -> float:
        """Длительность последнего такта (мс), 0 если запись ещё не закрыта."""
        if not self._records:
            return 0.0
        return float(self._records[-1]["phases"].get("tick", 0.0))

    # --- расширенные per-tick поля (уровень 1 детализированных метрик) ---

    def set_tick_field(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        self._tick_fields[name] = value

    def set_tick_fields(self, **fields: Any) -> None:
        if not self.enabled:
            return
        self._tick_fields.update(fields)

    # --- события (уровень 2) --------------------------------------------

    def record_event(self, kind: str, **fields: Any) -> None:
        if not self.enabled or self._events_csv_path is None:
            # Freeze-события я считаю и без CSV — через _freeze_count.
            return
        try:
            if self._events_csv_writer is None:
                header = ["event_type", "tick", "agent_id"] + [
                    k for k in fields.keys() if k not in ("tick", "agent_id")
                ]
                self._events_csv_file = open(
                    self._events_csv_path, "a", newline="", encoding="utf-8",
                )
                self._events_csv_writer = csv.DictWriter(
                    self._events_csv_file, fieldnames=header, extrasaction="ignore",
                )
                if self._events_csv_path.stat().st_size == 0:
                    self._events_csv_writer.writeheader()
            row: dict[str, Any] = {"event_type": kind}
            row.update(fields)
            self._events_csv_writer.writerow(row)
            assert self._events_csv_file is not None
            self._events_csv_file.flush()  # type: ignore[attr-defined]
        except (OSError, ValueError) as exc:
            logger.warning("Metrics: запись события не выполнена [kind=%s]: %s", kind, exc)

    def record_communication(self, tick: int, agent_id: int, payload_bytes: int,
                             kind: str = "coord", channel: str = "say") -> None:
        if not self.enabled:
            return
        if kind == "coord":
            self._comm_bytes_coord += int(payload_bytes)
        else:
            self._comm_bytes_sensor += int(payload_bytes)
        self.record_event(
            "communication",
            tick=tick, agent_id=agent_id,
            bytes=int(payload_bytes), type=kind, channel=channel,
        )

    # --- запись строки per-tick CSV -------------------------------------

    _TICK_CSV_FIELDS: tuple[str, ...] = (
        "tick_id", "agent_id", "agent_type", "agent_status",
        "pos_entity_id", "pos_x", "pos_y",
        "t_perception_us", "t_dijkstra_us", "t_pre_filter_us",
        "t_utility_us", "t_selection_us", "t_dispatch_us", "t_total_us",
        "n_visible", "n_type_relevant", "n_pre_filter_pass",
        "n_ttl_filtered", "n_fieryness_filtered", "n_already_rescued_filtered",
        "target_id", "target_id_prev", "target_changed",
        "target_utility", "target_distance",
        "U_total", "f_urgency", "f_dist", "f_effort", "f_social",
        "w_c", "w_d", "w_e", "w_n",
        "water_level", "is_transporting", "hp",
        "is_stuck", "is_idle", "action_dispatched",
    )

    def _write_tick_row(self, tick_number: int, total_ms: float) -> None:
        assert self._tick_csv_path is not None
        if self._tick_csv_writer is None:
            self._tick_csv_file = open(
                self._tick_csv_path, "a", newline="", encoding="utf-8",
            )
            self._tick_csv_writer = csv.DictWriter(
                self._tick_csv_file, fieldnames=list(self._TICK_CSV_FIELDS),
                extrasaction="ignore",
            )
            if self._tick_csv_path.stat().st_size == 0:
                self._tick_csv_writer.writeheader()

        phases = self._current_tick_phases
        ms_to_us = lambda v: int(round(v * 1000.0))
        row: dict[str, Any] = {f: "" for f in self._TICK_CSV_FIELDS}
        row.update(self._tick_fields)
        row["tick_id"] = tick_number
        row["t_perception_us"] = ms_to_us(phases.get("perception", 0.0))
        row["t_dijkstra_us"]   = ms_to_us(phases.get("dijkstra", 0.0))
        row["t_pre_filter_us"] = ms_to_us(phases.get("pre_filter", 0.0))
        row["t_utility_us"]    = ms_to_us(phases.get("utility", 0.0))
        row["t_selection_us"]  = ms_to_us(phases.get("selection", 0.0))
        row["t_dispatch_us"]   = ms_to_us(phases.get("dispatch", 0.0))
        row["t_total_us"]      = ms_to_us(total_ms)
        self._tick_csv_writer.writerow(row)
        assert self._tick_csv_file is not None
        self._tick_csv_file.flush()  # type: ignore[attr-defined]

    # --- финальный summary (уровень 4) ----------------------------------

    def write_summary_json(self, agent_id: int, agent_type_name: str) -> None:
        if not self.enabled or self._summary_json_path is None:
            return
        n = len(self._records)
        if n == 0:
            return

        total_series = [float(r["phases"].get("tick", 0.0)) for r in self._records]
        util_series = [float(r["phases"].get("utility", 0.0)) for r in self._records]
        over_budget = sum(1 for v in util_series if v > self.budget_ms)

        summary: dict[str, Any] = {
            "agent_id": agent_id,
            "agent_type": agent_type_name,
            "pid": os.getpid(),
            "total_ticks": n,
            "first_tick": int(self._records[0]["tick"]),
            "last_tick": int(self._records[-1]["tick"]),
            "session_wall_seconds": round(time.perf_counter() - self._session_t0, 3),
            "tick_ms": {
                "mean": round(statistics.fmean(total_series), 3),
                "p50": round(_percentile(total_series, 50), 3),
                "p95": round(_percentile(total_series, 95), 3),
                "p99": round(_percentile(total_series, 99), 3),
                "max": round(max(total_series), 3),
            },
            "utility_ms": {
                "mean": round(statistics.fmean(util_series), 3),
                "p95": round(_percentile(util_series, 95), 3),
                "over_budget": over_budget,
                "budget_ms": self.budget_ms,
            },
            "freeze_count": self._freeze_count,
            "freeze_rate": round(self._freeze_count / n, 6),
            "counters": dict(self._counters_total),
            "coord_bytes_total": self._comm_bytes_coord,
            "sensor_bytes_total": self._comm_bytes_sensor,
            "ablation_config": {
                "w_c": self._tick_fields.get("w_c"),
                "w_d": self._tick_fields.get("w_d"),
                "w_e": self._tick_fields.get("w_e"),
                "w_n": self._tick_fields.get("w_n"),
            },
        }
        try:
            self._summary_json_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Metrics: запись summary.json не выполнена — %s", exc)

    def close(self) -> None:
        for f in (self._tick_csv_file, self._events_csv_file):
            if f is not None:
                try:
                    f.close()  # type: ignore[attr-defined]
                except OSError:
                    pass

    # --- отчёт ------------------------------------------------------------

    def write_report(
        self,
        path: Path,
        agent_id: int,
        agent_type_name: str,
    ) -> None:
        if not self.enabled or not self._records:
            return

        records = self._records
        n = len(records)

        # Сбор серий по фазам.
        phase_series: dict[str, list[float]] = {p: [] for p in _PHASES}
        for rec in records:
            for p in _PHASES:
                phase_series[p].append(float(rec["phases"].get(p, 0.0)))

        util_series = phase_series["utility"]
        total_series = phase_series["tick"]
        over = sum(1 for v in util_series if v > self.budget_ms)

        # Серии gauge-ов (n_tasks и т.п.) — по всем встретившимся ключам.
        gauge_keys: set[str] = set()
        for rec in records:
            gauge_keys.update(rec["gauges"].keys())
        gauge_series: dict[str, list[int]] = {g: [] for g in sorted(gauge_keys)}
        for rec in records:
            for g in gauge_series:
                gauge_series[g].append(int(rec["gauges"].get(g, 0)))

        lines: list[str] = []
        lines.append(
            f"# Метрики агента — agent_id={agent_id} ({agent_type_name})"
        )
        lines.append("")
        lines.append(f"- Сэмплов (тактов): **{n}**")
        lines.append(f"- Бюджет матмодели: **{self.budget_ms:.0f} мс** (фаза `utility`)")
        lines.append(
            f"- Превышений бюджета: **{over}/{n} "
            f"({100.0 * over / n:.2f}%)**"
        )
        first_tick = records[0]["tick"]
        last_tick = records[-1]["tick"]
        lines.append(f"- Диапазон тиков: **{first_tick}..{last_tick}**")
        lines.append("")

        # --- Сводка по фазам (мс) ---
        lines.append("## Время по фазам (мс)")
        lines.append("")
        lines.append("| Фаза | mean | p50 | p95 | p99 | max | Σ |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in _PHASES:
            xs = phase_series[p]
            if not xs:
                continue
            mean, p50, p95, p99, pmax = _fmt(xs)
            total = sum(xs)
            lines.append(
                f"| {p} | {mean:.2f} | {p50:.2f} | {p95:.2f} | "
                f"{p99:.2f} | {pmax:.2f} | {total:.1f} |"
            )
        lines.append("")

        # --- Gauge-ы (n_tasks, cache_size и пр.) ---
        if gauge_series:
            lines.append("## Показатели состояния (по тактам)")
            lines.append("")
            lines.append("| Метрика | mean | p50 | p95 | max |")
            lines.append("|---|---|---|---|---|")
            for g, xs_int in gauge_series.items():
                xs = [float(x) for x in xs_int]
                if not xs:
                    continue
                mean, p50, p95, _p99, pmax = _fmt(xs)
                lines.append(
                    f"| {g} | {mean:.1f} | {p50:.0f} | {p95:.0f} | {pmax:.0f} |"
                )
            lines.append("")

        # --- Счётчики действий ---
        lines.append("## Счётчики событий (всего за сессию)")
        lines.append("")
        lines.append("| Событие | Всего |")
        lines.append("|---|---|")
        for k in sorted(self._counters_total.keys()):
            v = self._counters_total[k]
            if v > 0:
                lines.append(f"| {k} | {v} |")
        lines.append("")

        # --- Оценка мат-модели (utility) ---
        if util_series:
            mean_u, p50_u, p95_u, p99_u, max_u = _fmt(util_series)
            pct_over = 100.0 * over / n if n > 0 else 0.0
            verdict = (
                "укладываемся" if pct_over < 1.0
                else "редкие превышения" if pct_over < 5.0
                else "систематически выше"
            )
            lines.append(f"## Матмодель (фаза `utility`)")
            lines.append("")
            lines.append(
                f"- mean={mean_u:.2f} ms, p50={p50_u:.2f}, p95={p95_u:.2f}, "
                f"p99={p99_u:.2f}, max={max_u:.2f}"
            )
            lines.append(f"- Вердикт по бюджету {self.budget_ms:.0f} мс: **{verdict}**")
            lines.append("")

        # --- Последние 20 тактов ---
        last = records[-20:]
        lines.append("## Последние 20 тактов")
        lines.append("")
        header = "| tick | " + " | ".join(_PHASES) + " | actions |"
        sep = "|---" * (len(_PHASES) + 2) + "|"
        lines.append(header)
        lines.append(sep)
        for rec in last:
            phases = rec["phases"]
            actions_count = sum(
                rec["counters"].get(k, 0) for k in (
                    "ak_move", "ak_rescue", "ak_extinguish",
                    "ak_clear_area", "ak_load", "ak_unload", "ak_rest",
                )
            )
            row = [str(rec["tick"])]
            for p in _PHASES:
                row.append(f"{float(phases.get(p, 0.0)):.1f}")
            row.append(str(actions_count))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("Metrics: запись отчёта по пути %s не выполнена — %s", path, exc)


# Synонимы команд → имена счётчиков, чтобы call-site мог писать
# `metrics.inc_command("send_move")` без запоминания констант.
_COMMAND_TO_COUNTER: dict[str, str] = {
    "send_move": "ak_move",
    "send_rescue": "ak_rescue",
    "send_extinguish": "ak_extinguish",
    "send_clear_area": "ak_clear_area",
    "send_load": "ak_load",
    "send_unload": "ak_unload",
    "send_rest": "ak_rest",
    "send_say": "ak_say",
}


def command_to_counter(name: str) -> Optional[str]:
    return _COMMAND_TO_COUNTER.get(name)


__all__ = [
    "MetricsCollector",
    "command_to_counter",
]
