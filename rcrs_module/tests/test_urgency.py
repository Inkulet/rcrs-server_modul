from __future__ import annotations

"""В этом модуле я проверяю корректность фактора срочности f_urgency по диплому."""

# Я тестирую каждую ветку формулы для трёх типов агентов и граничные условия.
# Ключевые инварианты: результат всегда в [0, 1], ветки TTL отрабатывают верно.

import pytest

from decision.utility.urgency import (
    EPSILON,
    STABLE_URGENCY,
    T_MAX,
    T_MAX_TTL,
    compute_urgency,
    urgency_for_ambulance,
    urgency_for_fire,
    urgency_for_police,
)
from world.entities import AgentType

from conftest import make_agent, make_building, make_civilian


# ===========================================================================
# Тесты urgency_for_ambulance
# ===========================================================================


class TestUrgencyAmbulance:
    """Я проверяю формулу TTL для медицинских агентов."""

    def test_ttl_greater_than_time_budget_returns_normalized(self) -> None:
        """Я проверяю: если TTL > t_travel + t_work, результат = 1 - TTL/T_MAX_TTL.

        Новая формула нормирует результат в [0, 1]:
        - Малый TTL (умирает быстро) → urgency близко к 1.
        - Большой TTL → urgency близко к 0.
        """
        # hp=10000, damage=100 → TTL=100; t_travel=10, t_work=5 → budget=15 < 100 → успеваем
        entity = make_civilian(hp=10000, damage=100, buriedness=5)
        result = urgency_for_ambulance(entity, t_travel=10.0, t_work=5.0)
        expected = 1.0 - 100.0 / T_MAX_TTL  # 1 - 100/10000 = 0.99
        assert abs(result - expected) < 1e-9

    def test_ttl_equals_time_budget_returns_zero(self) -> None:
        """Я проверяю: если TTL == t_travel + t_work, агент не успевает — результат 0."""
        entity = make_civilian(hp=10000, damage=100)  # TTL=100
        result = urgency_for_ambulance(entity, t_travel=60.0, t_work=40.0)  # budget=100
        assert result == 0.0

    def test_ttl_less_than_time_budget_returns_zero(self) -> None:
        """Я проверяю: если TTL < бюджету времени, жертву не спасти — результат 0."""
        entity = make_civilian(hp=1000, damage=100)  # TTL=10
        result = urgency_for_ambulance(entity, t_travel=20.0, t_work=5.0)  # budget=25 > 10
        assert result == 0.0

    def test_stable_civilian_damage_zero_returns_stable_value(self) -> None:
        """Я проверяю: damage=0 → гражданский стабилен, TTL=∞ → возвращаю STABLE_URGENCY."""
        entity = make_civilian(hp=10000, damage=0)
        result = urgency_for_ambulance(entity, t_travel=10.0, t_work=5.0)
        assert result == STABLE_URGENCY

    def test_result_always_in_unit_interval(self) -> None:
        """Я проверяю нормировку: при любых входных данных f_urgency ∈ [0, 1]."""
        for hp, damage, tt, tw in [
            (1, 1000, 0.0, 0.0),   # очень высокий damage → TTL мал
            (10000, 1, 0.0, 0.0),  # TTL огромен → 1/TTL близко к 0
            (0, 1, 0.0, 0.0),      # hp=0 → TTL=0 → не успеть
        ]:
            entity = make_civilian(hp=hp, damage=damage)
            r = urgency_for_ambulance(entity, t_travel=tt, t_work=tw)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: hp={hp}, damage={damage}, result={r}"

    def test_missing_hp_returns_zero(self) -> None:
        """Я проверяю защиту от None: если hp отсутствует, возвращаю 0."""
        from world.entities import ComputedMetrics, EntityType, RawSensorData, VisibleEntity

        entity = VisibleEntity(
            id=99,
            type=EntityType.CIVILIAN,
            raw_sensor_data=RawSensorData(damage=50),
            computed_metrics=ComputedMetrics(path_distance=10.0, estimated_death_time=100, total_area=0),
            utility_score=0.0,
        )
        assert urgency_for_ambulance(entity, t_travel=5.0, t_work=5.0) == 0.0

    def test_negative_time_returns_zero(self) -> None:
        """Я проверяю: отрицательное t_travel — невалидный ввод, возвращаю 0."""
        entity = make_civilian(hp=10000, damage=100)
        assert urgency_for_ambulance(entity, t_travel=-1.0, t_work=5.0) == 0.0


# ===========================================================================
# Тесты urgency_for_fire
# ===========================================================================


class TestUrgencyFire:
    """Я проверяю формулу температура/T_max для пожарных агентов."""

    @pytest.mark.parametrize("fieryness", [1, 2, 3])
    def test_active_fire_returns_temperature_ratio(self, fieryness: int) -> None:
        """Я проверяю: для активного пожара (fieryness ∈ {1,2,3}) результат = temp/T_MAX."""
        entity = make_building(temperature=500.0, fieryness=fieryness)
        result = urgency_for_fire(entity, t_max=T_MAX)
        assert abs(result - 500.0 / T_MAX) < 1e-9

    @pytest.mark.parametrize("fieryness", [4, 5, 6, 7, 8])
    def test_inactive_fire_returns_zero(self, fieryness: int) -> None:
        """Я проверяю: потушенные/сгоревшие здания (fieryness ≥ 4) → приоритет 0."""
        entity = make_building(temperature=999.0, fieryness=fieryness)
        assert urgency_for_fire(entity, t_max=T_MAX) == 0.0

    def test_temperature_at_t_max_returns_one(self) -> None:
        """Я проверяю: максимальная температура → f_urgency = 1.0."""
        entity = make_building(temperature=T_MAX, fieryness=3)
        assert urgency_for_fire(entity, t_max=T_MAX) == 1.0

    def test_temperature_above_t_max_clamped_to_one(self) -> None:
        """Я проверяю: температура выше T_MAX зажимается к 1.0, не выходит за границу."""
        entity = make_building(temperature=T_MAX * 2, fieryness=1)
        assert urgency_for_fire(entity, t_max=T_MAX) == 1.0

    def test_zero_temperature_active_fire_returns_zero(self) -> None:
        """Я проверяю: здание горит, но температура 0 → f_urgency = 0."""
        entity = make_building(temperature=0.0, fieryness=2)
        assert urgency_for_fire(entity, t_max=T_MAX) == 0.0

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку: f_urgency_fire всегда ∈ [0, 1]."""
        for temp in [0.0, 100.0, 500.0, 1000.0, 9999.0]:
            entity = make_building(temperature=temp, fieryness=1)
            r = urgency_for_fire(entity)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: temp={temp}, result={r}"


# ===========================================================================
# Тесты urgency_for_police
# ===========================================================================


class TestUrgencyPolice:
    """Я проверяю формулу 1/(dist + ε) для агентов-полицейских."""

    def test_nonzero_distance_returns_inverse(self) -> None:
        """Я проверяю: при d > 0 результат = 1/(d+ε), зажатый к [0,1]."""
        d = 10.0
        result = urgency_for_police(min_distance_to_targets=d, epsilon=EPSILON)
        expected = min(1.0, 1.0 / (d + EPSILON))
        assert abs(result - expected) < 1e-9

    def test_zero_distance_clamped_to_one(self) -> None:
        """Я проверяю: расстояние 0 → 1/ε >> 1 → зажимается к 1.0."""
        result = urgency_for_police(min_distance_to_targets=0.0)
        assert result == 1.0

    def test_large_distance_approaches_zero(self) -> None:
        """Я проверяю: очень большое расстояние → f_urgency близко к 0."""
        result = urgency_for_police(min_distance_to_targets=1e9)
        assert result < 1e-6

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку: f_urgency_police всегда ∈ [0, 1]."""
        for d in [0.0, 1.0, 100.0, 10000.0, 1e9]:
            r = urgency_for_police(d)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: distance={d}, result={r}"


# ===========================================================================
# Тесты compute_urgency — маршрутизация по типу агента
# ===========================================================================


class TestComputeUrgencyDispatch:
    """Я проверяю, что compute_urgency правильно маршрутизирует по типу агента."""

    def test_ambulance_dispatched_correctly(self) -> None:
        """Я проверяю: AMBULANCE_TEAM → вызывает urgency_for_ambulance."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(hp=10000, damage=100)
        result = compute_urgency(agent, entity=entity, t_travel=10.0, t_work=5.0)
        assert result == urgency_for_ambulance(entity, t_travel=10.0, t_work=5.0)

    def test_fire_brigade_dispatched_correctly(self) -> None:
        """Я проверяю: FIRE_BRIGADE → вызывает urgency_for_fire."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        entity = make_building(temperature=600.0, fieryness=2)
        result = compute_urgency(agent, entity=entity)
        assert result == urgency_for_fire(entity)

    def test_police_dispatched_correctly(self) -> None:
        """Я проверяю: POLICE_FORCE → вызывает urgency_for_police."""
        agent = make_agent(agent_type=AgentType.POLICE_FORCE)
        result = compute_urgency(agent, min_distance_to_targets=50.0)
        assert result == urgency_for_police(50.0)

    def test_ambulance_missing_entity_returns_zero(self) -> None:
        """Я проверяю защиту: медику не передана сущность → безопасный 0."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        assert compute_urgency(agent, t_travel=10.0, t_work=5.0) == 0.0

    def test_police_missing_distance_returns_zero(self) -> None:
        """Я проверяю защиту: полиции не передано расстояние → безопасный 0."""
        agent = make_agent(agent_type=AgentType.POLICE_FORCE)
        assert compute_urgency(agent) == 0.0
