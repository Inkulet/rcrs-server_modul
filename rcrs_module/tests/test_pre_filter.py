from __future__ import annotations

"""В этом модуле я проверяю правила предварительной фильтрации задач (UC-2)."""

# Я тестирую каждое правило отсева диплома отдельно, чтобы при сбое сразу знать,
# какой именно критерий нарушен. Тесты независимы — каждый проверяет одно правило.

import pytest

from decision.filters.pre_filter import NeedRefugeException, PreFilterDispatcher
from world.entities import AgentType

from conftest import make_agent, make_blockade, make_building, make_civilian


def make_dispatcher(work_rate: float = 1.0, average_speed: float = 1.0) -> PreFilterDispatcher:
    """Я создаю диспетчер с заданными скоростями работ и движения.

    average_speed=1.0 мм/такт — тестовое значение, при котором path_distance (мм)
    напрямую переводится в такты (t_travel = path_distance / 1.0 = path_distance).
    Это упрощает проверку дедлайна: time_to_action = path_distance + buriedness/work_rate.
    """
    return PreFilterDispatcher(work_rate=work_rate, average_speed=average_speed)


# ===========================================================================
# Тесты правил фильтрации для гражданских
# ===========================================================================


class TestCivilianFiltering:
    """Я проверяю правила отсева гражданских из диплома (раздел 4, шаг Pre-Filtering)."""

    def test_dead_civilian_filtered(self) -> None:
        """Я проверяю: hp=0 → гражданский мёртв → отсеивается."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(hp=0, damage=50, buriedness=5)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert result == []

    def test_healthy_civilian_no_damage_no_buriedness_filtered(self) -> None:
        """Я проверяю: damage=0 и buriedness=0 → помощь не нужна → отсеивается."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(hp=10000, damage=0, buriedness=0)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert result == []

    def test_civilian_with_damage_kept(self) -> None:
        """Я проверяю: damage > 0 → нужна помощь → проходит фильтр."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(hp=10000, damage=50, buriedness=0, estimated_death_time=9999)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert entity in result

    def test_civilian_with_buriedness_kept(self) -> None:
        """Я проверяю: buriedness > 0 → завален → проходит фильтр."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(
            hp=10000, damage=0, buriedness=5, path_distance=10.0, estimated_death_time=9999
        )
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert entity in result


# ===========================================================================
# Тесты правил фильтрации для зданий
# ===========================================================================


class TestBuildingFiltering:
    """Я проверяю правила отсева зданий по fieryness из диплома."""

    @pytest.mark.parametrize("fieryness", [4, 5, 6, 7, 8])
    def test_burned_building_filtered(self, fieryness: int) -> None:
        """Я проверяю: fieryness ∈ {4..8} → здание потушено/сгорело → отсеивается."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        entity = make_building(fieryness=fieryness)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert result == []

    @pytest.mark.parametrize("fieryness", [1, 2, 3])
    def test_active_fire_kept(self, fieryness: int) -> None:
        """Я проверяю: fieryness ∈ {1,2,3} → активный пожар → проходит фильтр."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        entity = make_building(fieryness=fieryness)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert entity in result


# ===========================================================================
# Тесты правила дедлайна
# ===========================================================================


class TestDeadlineFiltering:
    """Я проверяю условие TTL <= t_travel + t_work из диплома."""

    def test_deadline_exceeded_filtered(self) -> None:
        """Я проверяю: estimated_death_time <= path_distance + buriedness/rate → отсеивается."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        # path_distance=100, buriedness=50, rate=1.0 → time_to_action=150
        # estimated_death_time=100 ≤ 150 → не успеть
        entity = make_civilian(
            hp=10000,
            damage=50,
            buriedness=50,
            path_distance=100.0,
            estimated_death_time=100,
        )
        result = make_dispatcher(work_rate=1.0).filter_tasks(agent, [entity])
        assert result == []

    def test_deadline_not_exceeded_kept(self) -> None:
        """Я проверяю: estimated_death_time > time_to_action → успеваем → задача остаётся."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        # path_distance=10, buriedness=5, rate=1.0 → time_to_action=15
        # estimated_death_time=9999 >> 15 → успеваем
        entity = make_civilian(
            hp=10000,
            damage=50,
            buriedness=5,
            path_distance=10.0,
            estimated_death_time=9999,
        )
        result = make_dispatcher(work_rate=1.0).filter_tasks(agent, [entity])
        assert entity in result


# ===========================================================================
# Тесты специальных состояний агента
# ===========================================================================


class TestAgentStateFiltering:
    """Я проверяю правила, зависящие от состояния агента, а не задачи."""

    def test_transporting_agent_returns_empty(self) -> None:
        """Я проверяю: is_transporting=True → пропустить всё → пустой список."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM, transporting=True)
        entities = [
            make_civilian(hp=10000, damage=50, buriedness=5, estimated_death_time=9999),
            make_building(fieryness=2),
        ]
        result = make_dispatcher().filter_tasks(agent, entities)
        assert result == []

    def test_fire_brigade_no_water_raises_need_refuge(self) -> None:
        """Я проверяю: FIRE_BRIGADE с water=0 → исключение NeedRefugeException."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE, water=0)
        entity = make_building(fieryness=2)
        with pytest.raises(NeedRefugeException):
            make_dispatcher().filter_tasks(agent, [entity])

    def test_fire_brigade_with_water_no_exception(self) -> None:
        """Я проверяю: FIRE_BRIGADE с водой → нет исключения, работает нормально."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE, water=1000)
        entity = make_building(fieryness=2)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert entity in result

    def test_multiple_entities_mixed_filtering(self) -> None:
        """Я проверяю: из смешанного списка отсеиваются только нерелевантные задачи."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        dead = make_civilian(entity_id=1, hp=0)
        alive = make_civilian(entity_id=2, hp=10000, damage=50, estimated_death_time=9999)
        healthy = make_civilian(entity_id=3, hp=8000, damage=0, buriedness=0)

        result = make_dispatcher().filter_tasks(agent, [dead, alive, healthy])
        assert alive in result
        assert dead not in result
        assert healthy not in result

    def test_invalid_work_rate_raises_value_error(self) -> None:
        """Я проверяю: work_rate ≤ 0 недопустим → конструктор кидает ValueError."""
        with pytest.raises(ValueError):
            PreFilterDispatcher(work_rate=0.0)
        with pytest.raises(ValueError):
            PreFilterDispatcher(work_rate=-1.0)

    def test_invalid_average_speed_raises_value_error(self) -> None:
        """Я проверяю: average_speed ≤ 0 недопустим → конструктор кидает ValueError."""
        with pytest.raises(ValueError):
            PreFilterDispatcher(average_speed=0.0)
        with pytest.raises(ValueError):
            PreFilterDispatcher(average_speed=-5.0)


# ===========================================================================
# Тесты фильтрации для полиции и завалов
# ===========================================================================


class TestPoliceBlockadeFiltering:
    """Я проверяю: полиция работает только с завалами (BLOCKADE)."""

    def test_police_keeps_blockades(self) -> None:
        """Я проверяю: POLICE_FORCE получает задачу типа BLOCKADE."""
        agent = make_agent(agent_type=AgentType.POLICE_FORCE)
        blockade = make_blockade(entity_id=50, repair_cost=500)
        result = make_dispatcher().filter_tasks(agent, [blockade])
        assert blockade in result

    def test_police_filters_civilians(self) -> None:
        """Я проверяю: POLICE_FORCE не получает задачи типа CIVILIAN."""
        agent = make_agent(agent_type=AgentType.POLICE_FORCE)
        civ = make_civilian(hp=10000, damage=50, buriedness=5, estimated_death_time=9999)
        result = make_dispatcher().filter_tasks(agent, [civ])
        assert result == []

    def test_police_filters_buildings(self) -> None:
        """Я проверяю: POLICE_FORCE не получает задачи типа BUILDING."""
        agent = make_agent(agent_type=AgentType.POLICE_FORCE)
        bld = make_building(fieryness=2)
        result = make_dispatcher().filter_tasks(agent, [bld])
        assert result == []

    def test_ambulance_filters_blockades(self) -> None:
        """Я проверяю: AMBULANCE_TEAM не получает задачи типа BLOCKADE."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        blockade = make_blockade(entity_id=50)
        result = make_dispatcher().filter_tasks(agent, [blockade])
        assert result == []

    def test_fire_brigade_filters_civilians(self) -> None:
        """Я проверяю: FIRE_BRIGADE не получает задачи типа CIVILIAN."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE, water=5000)
        civ = make_civilian(hp=10000, damage=50, buriedness=5, estimated_death_time=9999)
        result = make_dispatcher().filter_tasks(agent, [civ])
        assert result == []


# ===========================================================================
# Тесты граничных случаев fieryness для зданий
# ===========================================================================


class TestBuildingFierynessEdgeCases:
    """Я проверяю граничные случаи fieryness: None и 0."""

    def test_fieryness_none_filtered(self) -> None:
        """Я проверяю: fieryness=None → данные устарели → здание отсеивается."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE, water=5000)
        entity = make_building(fieryness=None)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert result == []

    def test_fieryness_zero_filtered(self) -> None:
        """Я проверяю: fieryness=0 → здание не горит → отсеивается."""
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE, water=5000)
        entity = make_building(fieryness=0)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert result == []

    def test_civilian_hp_none_filtered(self) -> None:
        """Я проверяю: hp=None → устаревшие данные → гражданский отсеивается."""
        agent = make_agent(agent_type=AgentType.AMBULANCE_TEAM)
        entity = make_civilian(hp=None, damage=50, buriedness=5)
        result = make_dispatcher().filter_tasks(agent, [entity])
        assert result == []
