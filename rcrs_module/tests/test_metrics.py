from __future__ import annotations

import tempfile
from pathlib import Path

from metrics import MetricsCollector


def test_disabled_collector_is_noop() -> None:
    m = MetricsCollector(enabled=False, budget_ms=100.0)
    m.start_tick()
    with m.phase("perception"):
        pass
    m.inc("ak_move")
    m.gauge("n_tasks", 42)
    m.stop_tick(tick_number=1)
    assert m.tick_count == 0  # ничего не сохраняем
    assert m.last_tick_ms() == 0.0


def test_phase_accumulates_ms() -> None:
    m = MetricsCollector(enabled=True, budget_ms=100.0)
    m.start_tick()
    with m.phase("perception"):
        sum(range(50_000))
    m.stop_tick(tick_number=1)
    assert m.tick_count == 1
    assert m.last_tick_ms() > 0.0


def test_counter_increments_and_survives_across_ticks() -> None:
    m = MetricsCollector(enabled=True, budget_ms=100.0)
    for tick in range(3):
        m.start_tick()
        m.inc("ak_move")
        m.inc("ak_move")
        m.stop_tick(tick_number=tick)
    # Внутренний total-счётчик должен суммироваться.
    assert m._counters_total["ak_move"] == 6


def test_over_budget_counted() -> None:
    m = MetricsCollector(enabled=True, budget_ms=0.001)  # 1 мкс — всегда превышено
    m.start_tick()
    with m.phase("utility"):
        sum(range(10_000))
    m.stop_tick(tick_number=1)
    # over_budget проверяется в stop_tick по utility.
    assert m._counters_total["over_budget"] == 1


def test_write_report_creates_file_with_sections() -> None:
    m = MetricsCollector(enabled=True, budget_ms=100.0)
    for tick in range(5):
        m.start_tick()
        with m.phase("perception"):
            sum(range(1000))
        with m.phase("utility"):
            sum(range(500))
        m.inc("ak_move")
        m.gauge("n_tasks", 10 + tick)
        m.stop_tick(tick_number=tick)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.md"
        m.write_report(p, agent_id=1, agent_type_name="FIRE_BRIGADE")
        text = p.read_text(encoding="utf-8")
        assert "agent_id=1" in text
        assert "FIRE_BRIGADE" in text
        # Заголовки секций должны присутствовать.
        assert "Время по фазам" in text
        assert "Показатели состояния" in text
        assert "Счётчики событий" in text
        assert "Матмодель" in text
        # Счётчик ak_move должен попасть в таблицу (значение 5).
        assert "ak_move" in text


def test_empty_report_not_written() -> None:
    m = MetricsCollector(enabled=True, budget_ms=100.0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.md"
        m.write_report(p, agent_id=1, agent_type_name="FIRE_BRIGADE")
        # Нет тактов — нет файла.
        assert not p.exists()


def test_unknown_counter_is_tracked_dynamically() -> None:
    m = MetricsCollector(enabled=True, budget_ms=100.0)
    m.start_tick()
    m.inc("custom_event")
    m.stop_tick(tick_number=1)
    assert m._counters_total["custom_event"] == 1
