from __future__ import annotations

"""В этом модуле я проверяю корректность социального фактора f_social по диплому."""

# Ключевые инварианты диплома (формула 15):
# - f_social = N_i(r) — целочисленный счётчик однотипных союзников в радиусе r от цели
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
    """Я проверяю формулу N_i(r) с евклидовой метрикой и spatial-grid."""

    def test_no_agents_returns_zero(self) -> None:
        """Я проверяю: нет союзников в WorldModel → N_i = 0."""
        wm = make_world_with_graph()
        # agents пуст — только self (current_agent_id=1), но он не добавлен
        result = social_factor(wm, make_pos(2, 500, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1)
        assert result == 0.0

    def test_only_self_in_agents_returns_zero(self) -> None:
        """Я проверяю: только сам агент зарегистрирован → после исключения self → 0."""
        wm = make_world_with_graph()
        wm.set_agent(make_agent(agent_id=1, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0))
        result = social_factor(wm, make_pos(2, 0, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1)
        assert result == 0.0

    def test_other_type_agents_excluded(self) -> None:
        """Я проверяю: агенты другого типа не учитываются в социальном факторе."""
        wm = make_world_with_graph()
        # Пожарный рядом с целью — но тип другой
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.FIRE_BRIGADE, x=100, y=0))
        target = make_pos(2, 110, 0)  # почти рядом
        result = social_factor(wm, target, AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=1000.0)
        assert result == 0.0

    def test_three_allies_within_radius_returns_three(self) -> None:
        """Я проверяю: три союзника в радиусе → N_i = 3 (счётчик, не доля)."""
        wm = make_world_with_graph()
        target = make_pos(99, 0, 0)
        # Три союзника вблизи цели (расстояние ≤ 50 < 1000)
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=10,  y=0))
        wm.set_agent(make_agent(agent_id=3, agent_type=AgentType.AMBULANCE_TEAM, x=20,  y=0))
        wm.set_agent(make_agent(agent_id=4, agent_type=AgentType.AMBULANCE_TEAM, x=-30, y=0))
        result = social_factor(wm, target, AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=1000.0)
        assert result == 3.0

    def test_some_allies_within_radius(self) -> None:
        """Я проверяю: часть союзников в радиусе → N_i = их число."""
        wm = make_world_with_graph()
        target = make_pos(99, 0, 0)
        # Союзник 2 в радиусе (расстояние 100), союзник 3 вне (расстояние 5000)
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.FIRE_BRIGADE, x=100,  y=0))
        wm.set_agent(make_agent(agent_id=3, agent_type=AgentType.FIRE_BRIGADE, x=5000, y=0))
        result = social_factor(
            wm, target, AgentType.FIRE_BRIGADE, current_agent_id=1, radius=1000.0
        )
        # Только агент 2 попадает в радиус → N_i = 1
        assert result == 1.0

    def test_euclidean_distance_not_graph(self) -> None:
        """Я проверяю: используется евклидово расстояние, а не граф-дистанция.

        Цель и союзник физически близко (euclid < radius), но граф-маршрута между ними нет.
        Если бы код использовал Дейкстра, вернул бы 0. С евклидовой метрикой — 1.
        """
        wm = make_world_with_graph()
        target = make_pos(99, 0, 0)  # узел не в графе — граф-путь невозможен
        # Союзник физически в 100 ед. от цели
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=100, y=0))
        result = social_factor(
            wm, target, AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=500.0
        )
        # Физически близко → евклидово 100 < 500 → должно считаться
        assert result == 1.0

    def test_zero_radius_returns_zero(self) -> None:
        """Я проверяю: нулевой радиус — невалидный ввод → защита, возвращаю 0.0."""
        wm = make_world_with_graph()
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0))
        result = social_factor(wm, make_pos(1, 0, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=0.0)
        assert result == 0.0

    def test_negative_radius_returns_zero(self) -> None:
        """Я проверяю: отрицательный радиус → защита, возвращаю 0.0."""
        wm = make_world_with_graph()
        wm.set_agent(make_agent(agent_id=2, agent_type=AgentType.AMBULANCE_TEAM, x=0, y=0))
        result = social_factor(wm, make_pos(1, 0, 0), AgentType.AMBULANCE_TEAM, current_agent_id=1, radius=-100.0)
        assert result == 0.0

    def test_counter_grows_linearly_with_allies(self) -> None:
        """Я проверяю, что f_social — именно счётчик: при N однотипных союзниках
        результат строго равен N (формула 15)."""
        for n_allies in range(1, 6):
            wm = make_world_with_graph()
            for aid in range(2, 2 + n_allies):
                # Координаты разнесены, чтобы spatial-grid не схлопнул в одну ячейку.
                wm.set_agent(make_agent(
                    agent_id=aid, agent_type=AgentType.POLICE_FORCE,
                    x=10 * aid, y=10 * aid,
                ))
            target = make_pos(1, 0, 0)
            result = social_factor(wm, target, AgentType.POLICE_FORCE, current_agent_id=1, radius=DEFAULT_RADIUS)
            assert result == float(n_allies), f"Ожидал N_i={n_allies}, получил {result}"
