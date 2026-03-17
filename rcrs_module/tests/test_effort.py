from __future__ import annotations

"""В этом модуле я проверяю корректность фактора трудоёмкости f_effort по диплому."""

# Ключевые инварианты: нормировка к [0,1], граничные значения 0 и MAX дают 0 и 1,
# каждый тип агента использует свою формулу из вектора obs_i(t).

import pytest

from decision.utility.effort import (
    MAX_BURIEDNESS,
    MAX_REPAIR_COST,
    MAX_TOTAL_AREA,
    compute_effort,
    effort_for_ambulance,
    effort_for_fire,
    effort_for_police,
)
from world.entities import AgentType

from conftest import make_agent, make_blockade, make_building, make_civilian


# ===========================================================================
# Тесты effort_for_ambulance
# ===========================================================================


class TestEffortAmbulance:
    """Я проверяю формулу Buriedness / MaxBuriedness для медиков."""

    def test_zero_buriedness_returns_zero(self) -> None:
        """Я проверяю: нет завала → трудоёмкость 0."""
        entity = make_civilian(buriedness=0)
        assert effort_for_ambulance(entity) == 0.0

    def test_max_buriedness_returns_one(self) -> None:
        """Я проверяю: максимальный завал → f_effort = 1.0."""
        entity = make_civilian(buriedness=int(MAX_BURIEDNESS))
        assert effort_for_ambulance(entity) == 1.0

    def test_above_max_clamped_to_one(self) -> None:
        """Я проверяю: buriedness > MAX зажимается к 1.0, не выходит за [0,1]."""
        entity = make_civilian(buriedness=int(MAX_BURIEDNESS) * 2)
        assert effort_for_ambulance(entity) == 1.0

    def test_half_buriedness_returns_half(self) -> None:
        """Я проверяю линейность формулы: 50 % завала → 0.5."""
        entity = make_civilian(buriedness=int(MAX_BURIEDNESS // 2))
        result = effort_for_ambulance(entity)
        assert abs(result - 0.5) < 1e-9

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку на широком диапазоне входных значений."""
        for b in [0, 1, 50, 100, 200]:
            entity = make_civilian(buriedness=b)
            r = effort_for_ambulance(entity)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: buriedness={b}, result={r}"


# ===========================================================================
# Тесты effort_for_fire
# ===========================================================================


class TestEffortFire:
    """Я проверяю формулу TotalArea / MaxTotalArea для пожарных."""

    def test_zero_area_returns_zero(self) -> None:
        """Я проверяю: здание без площади → трудоёмкость 0."""
        entity = make_building(ground_area=0, floors=3)
        assert effort_for_fire(entity) == 0.0

    def test_zero_floors_returns_zero(self) -> None:
        """Я проверяю: нулевой этаж → TotalArea = 0 → трудоёмкость 0."""
        entity = make_building(ground_area=100, floors=0)
        assert effort_for_fire(entity) == 0.0

    def test_max_total_area_returns_one(self) -> None:
        """Я проверяю: total_area == MAX → f_effort = 1.0."""
        # MAX_TOTAL_AREA = 100000, ищем ground_area * floors = 100000
        entity = make_building(ground_area=int(MAX_TOTAL_AREA // 10), floors=10)
        assert effort_for_fire(entity) == 1.0

    def test_above_max_clamped_to_one(self) -> None:
        """Я проверяю: total_area > MAX зажимается к 1.0."""
        entity = make_building(ground_area=int(MAX_TOTAL_AREA), floors=2)
        assert effort_for_fire(entity) == 1.0

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку: f_effort_fire всегда ∈ [0, 1]."""
        for area, floors in [(0, 1), (100, 3), (10000, 5), (50000, 2)]:
            entity = make_building(ground_area=area, floors=floors)
            r = effort_for_fire(entity)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: area={area}, floors={floors}, result={r}"


# ===========================================================================
# Тесты effort_for_police
# ===========================================================================


class TestEffortPolice:
    """Я проверяю формулу RepairCost / MaxRepairCost для полицейских."""

    def test_zero_repair_cost_returns_zero(self) -> None:
        """Я проверяю: бесплатный ремонт → трудоёмкость 0."""
        entity = make_blockade(repair_cost=0)
        assert effort_for_police(entity) == 0.0

    def test_max_repair_cost_returns_one(self) -> None:
        """Я проверяю: максимальная стоимость → f_effort = 1.0."""
        entity = make_blockade(repair_cost=int(MAX_REPAIR_COST))
        assert effort_for_police(entity) == 1.0

    def test_above_max_clamped_to_one(self) -> None:
        """Я проверяю: repair_cost > MAX зажимается к 1.0."""
        entity = make_blockade(repair_cost=int(MAX_REPAIR_COST) * 3)
        assert effort_for_police(entity) == 1.0

    def test_proportional_value(self) -> None:
        """Я проверяю линейность: repair_cost = 25000 (25 % от MAX) → 0.25."""
        entity = make_blockade(repair_cost=25000)
        result = effort_for_police(entity)
        assert abs(result - 0.25) < 1e-9

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку на широком диапазоне входных значений."""
        for cost in [0, 1000, 50000, 100000, 999999]:
            entity = make_blockade(repair_cost=cost)
            r = effort_for_police(entity)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: repair_cost={cost}, result={r}"


# ===========================================================================
# Тесты compute_effort — маршрутизация по типу агента
# ===========================================================================


class TestComputeEffortDispatch:
    """Я проверяю корректность маршрутизации compute_effort по типу агента."""

    def test_ambulance_uses_buriedness(self) -> None:
        """Я проверяю: AMBULANCE_TEAM → effort_for_ambulance."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(buriedness=50)
        result = compute_effort(agent, entity=entity)
        assert result == effort_for_ambulance(entity)

    def test_fire_uses_area(self) -> None:
        """Я проверяю: FIRE_BRIGADE → effort_for_fire."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        entity = make_building(ground_area=100, floors=3)
        result = compute_effort(agent, entity=entity)
        assert result == effort_for_fire(entity)

    def test_police_uses_repair_cost(self) -> None:
        """Я проверяю: POLICE_FORCE → effort_for_police."""
        agent = make_agent(agent_type=AgentType.POLICE_FORCE)
        entity = make_blockade(repair_cost=50000)
        result = compute_effort(agent, entity=entity)
        assert result == effort_for_police(entity)

    def test_missing_entity_returns_zero(self) -> None:
        """Я проверяю защиту: нет сущности → безопасный 0."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        assert compute_effort(agent, entity=None) == 0.0
