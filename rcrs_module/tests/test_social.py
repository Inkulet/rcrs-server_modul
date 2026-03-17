from __future__ import annotations

"""В этом модуле я проверяю корректность социального фактора f_social по диплому."""

# Ключевые инварианты диплома:
# - f_social = N_i(r) / |A_same_type| ∈ [0, 1]
# - Расстояние — евклидово ||pos_k - loc_i||, а не граф-дистанция
# - Исключается только текущий агент (current_agent_id); союзники другого типа игнорируются

import pytest

from decision.utility.social import DEFAULT_RADIUS, social_factor
from world.entities import AgentType, Position

from conftest import make_agent, make_world_with_graph


def make_pos(entity_id: int, x: int, y: int) -> Position:
    """Я создаю Position с конкретными координатами для евклидовых вычислений."""
    return Position(entity_id=entity_id, x=x, y=y)


# ===========================================================================
# Тесты social_factor
# ===========================================================================


class TestSocialFactor:
    """Я проверяю формулу N_i(r)/|A_same_type| с евклидовой метрикой."""

    def test_no_agents_returns_zero(self) -> None:
        """Я проверяю: нет союзников в WorldModel → f_social = 0.0."""
        wm = make_world_with_graph()
        # agents пуст — только self (current_agent_id=1), но он не добавлен
        result = social_factor(wm, make_pos(2, 500, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1)
        assert result == 0.0

    def test_only_self_in_agents_returns_zero(self) -> None:
        """Я проверяю: только сам агент зарегистрирован → после исключения self → 0.0."""
        wm = make_world_with_graph()
        wm.set_agent(make_agent(agent_id=1, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0))
        result = social_factor(wm, make_pos(2, 0, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1)
        assert result == 0.0

    def test_other_type_agents_excluded(self) -> None:
        """Я проверяю: агенты другого типа не учитываются в социальном факторе."""
        wm = make_world_with_graph()
        # Пожарный рядом с целью — но тип другой
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.FIRE_BRIGADE, x=0, y=0))
        target = make_pos(2, 10, 0)  # почти рядом
        result = social_factor(wm, target, AgentType.AMBULANCE_TEAM, current_agent_id=1)
        assert result == 0.0

    def test_all_same_type_within_radius_returns_one(self) -> None:
        """Я проверяю: все союзники в радиусе → f_social = 1.0."""
        wm = make_world_with_graph()
        target = make_pos(1, 0, 0)
        # Три союзника вплотную к цели (расстояние 0)
        for aid in [2, 3, 4]:
            wm.set_agent(make_agent(agent_id=aid, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0, entity_id=1))
        result = social_factor(wm, target, AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=DEFAULT_RADIUS)
        assert result == 1.0

    def test_some_allies_within_radius(self) -> None:
        """Я проверяю: часть союзников в радиусе → f_social = count/total."""
        wm = make_world_with_graph()
        target = make_pos(99, 0, 0)
        # Союзник 2 в радиусе (расстояние 100), союзник 3 вне (расстояние 5000)
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.FIRE_BRIGADE, x=100, y=0, entity_id=1))
        wm.set_agent(make_agent(agent_id=3, agent_type=AgentType.FIRE_BRIGADE, x=5000, y=0, entity_id=1))
        result = social_factor(
            wm, target, AgentType.FIRE_BRIGADE, current_agent_id=1, radius=1000.0
        )
        # 1 из 2 в радиусе → 0.5
        assert abs(result - 0.5) < 1e-9

    def test_euclidean_distance_not_graph(self) -> None:
        """Я проверяю: используется евклидово расстояние, а не граф-дистанция.

        Цель и союзник физически близко (euclid < radius), но граф-маршрута между ними нет.
        Если бы код использовал Дейкстра, вернул бы 0.0. С евклидовой метрикой — 1.0.
        """
        wm = make_world_with_graph()
        target = make_pos(99, 0, 0)  # узел не в графе — граф-путь невозможен
        # Союзник физически в 100 ед. от цели
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=100, y=0, entity_id=1))
        result = social_factor(
            wm, target, AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=500.0
        )
        # Физически близко → евклидово 100 < 500 → должно считаться
        assert result == 1.0

    def test_zero_radius_returns_zero(self) -> None:
        """Я проверяю: нулевой радиус — невалидный ввод → защита, возвращаю 0.0."""
        wm = make_world_with_graph()
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0, entity_id=1))
        result = social_factor(wm, make_pos(1, 0, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=0.0)
        assert result == 0.0

    def test_negative_radius_returns_zero(self) -> None:
        """Я проверяю: отрицательный радиус → защита, возвращаю 0.0."""
        wm = make_world_with_graph()
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0, entity_id=1))
        result = social_factor(wm, make_pos(1, 0, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=-100.0)
        assert result == 0.0

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку: f_social всегда ∈ [0, 1] при любом числе агентов."""
        wm = make_world_with_graph()
        target = make_pos(1, 0, 0)
        for n_allies in range(1, 6):
            # Каждый раз создаю свежую модель с n союзниками рядом
            wm2 = make_world_with_graph()
            for aid in range(2, 2 + n_allies):
                wm2.set_agent(make_agent(agent_id=aid, agent_type=AgentType.POLICE_FORCE, x=0, y=0, entity_id=1))
            r = social_factor(wm2, target, AgentType.POLICE_FORCE, current_agent_id=1)
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: n_allies={n_allies}, result={r}"
