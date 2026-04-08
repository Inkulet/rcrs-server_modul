from __future__ import annotations

"""В этом модуле я тестирую WorldModel: слияние кэша, граф, удаление сущностей."""

import networkx as nx
import pytest

from conftest import make_agent, make_building, make_civilian, make_blockade
from world.cache import WorldModel
from world.entities import (
    AgentState,
    AgentType,
    ComputedMetrics,
    EntityType,
    MapEdge,
    MapNode,
    PerceptionPacket,
    Position,
    RawSensorData,
    Resources,
    VisibleEntity,
)


# ---------------------------------------------------------------------------
# build_graph_from_map
# ---------------------------------------------------------------------------

class TestBuildGraphFromMap:

    def test_builds_nodes_and_edges(self) -> None:
        wm = WorldModel()
        nodes = [MapNode(entity_id=1, x=0, y=0), MapNode(entity_id=2, x=100, y=0)]
        edges = [MapEdge(source_id=1, target_id=2, weight=100.0)]
        wm.build_graph_from_map(nodes, edges)

        assert wm.road_graph.number_of_nodes() == 2
        assert wm.road_graph.number_of_edges() == 1
        assert wm.road_graph[1][2]["weight"] == 100.0

    def test_node_attributes_stored(self) -> None:
        wm = WorldModel()
        wm.build_graph_from_map([MapNode(entity_id=5, x=42, y=99)], [])
        assert wm.road_graph.nodes[5]["x"] == 42
        assert wm.road_graph.nodes[5]["y"] == 99

    def test_empty_graph(self) -> None:
        wm = WorldModel()
        wm.build_graph_from_map([], [])
        assert wm.road_graph.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# add_road_node / add_road_edge
# ---------------------------------------------------------------------------

class TestGraphHelpers:

    def test_add_road_node(self) -> None:
        wm = WorldModel()
        wm.add_road_node(10, x=5, y=6)
        assert wm.road_graph.has_node(10)
        assert wm.road_graph.nodes[10]["x"] == 5

    def test_add_road_edge(self) -> None:
        wm = WorldModel()
        wm.add_road_node(1)
        wm.add_road_node(2)
        wm.add_road_edge(1, 2, weight=50.0)
        assert wm.road_graph.has_edge(1, 2)
        assert wm.road_graph[1][2]["weight"] == 50.0


# ---------------------------------------------------------------------------
# set_agent / get_agent / get_task
# ---------------------------------------------------------------------------

class TestAgentAndTaskAccess:

    def test_set_and_get_agent(self) -> None:
        wm = WorldModel()
        agent = make_agent(agent_id=7)
        wm.set_agent(agent)
        assert wm.get_agent(7) is agent

    def test_get_agent_missing_returns_none(self) -> None:
        wm = WorldModel()
        assert wm.get_agent(999) is None

    def test_get_task_missing_returns_none(self) -> None:
        wm = WorldModel()
        assert wm.get_task(999) is None

    def test_get_task_after_add(self) -> None:
        wm = WorldModel()
        civ = make_civilian(entity_id=42)
        wm.tasks[42] = civ
        assert wm.get_task(42) is civ


# ---------------------------------------------------------------------------
# update_agents
# ---------------------------------------------------------------------------

class TestUpdateAgents:

    def test_adds_new_agents(self) -> None:
        wm = WorldModel()
        a1 = make_agent(agent_id=1)
        a2 = make_agent(agent_id=2)
        wm.update_agents([a1, a2])
        assert len(wm.agents) == 2

    def test_overwrites_existing_agent(self) -> None:
        wm = WorldModel()
        a1 = make_agent(agent_id=1, x=0, y=0)
        wm.update_agents([a1])
        a1_updated = make_agent(agent_id=1, x=999, y=999)
        wm.update_agents([a1_updated])
        assert wm.agents[1].position.x == 999


# ---------------------------------------------------------------------------
# update_perception — кэш-слияние
# ---------------------------------------------------------------------------

class TestUpdatePerception:

    def _make_entity(
        self,
        entity_id: int = 10,
        hp: int | None = 5000,
        damage: int | None = 50,
        buriedness: int | None = 5,
        fieryness: int | None = None,
        temperature: float | None = None,
    ) -> VisibleEntity:
        return VisibleEntity(
            id=entity_id,
            type=EntityType.CIVILIAN,
            raw_sensor_data=RawSensorData(
                hp=hp, damage=damage, buriedness=buriedness,
                fieryness=fieryness, temperature=temperature,
            ),
            computed_metrics=ComputedMetrics(
                path_distance=0.0,
                estimated_death_time=100,
                total_area=0,
            ),
            utility_score=0.0,
        )

    def test_new_entity_added_to_cache(self) -> None:
        wm = WorldModel()
        entity = self._make_entity(entity_id=10)
        wm.update_perception([entity])
        assert 10 in wm.tasks

    def test_merge_preserves_old_when_new_is_none(self) -> None:
        """Я проверяю: при повторном наблюдении с None-полями кэш сохраняет старые значения."""
        wm = WorldModel()
        # Первое наблюдение: все поля заполнены.
        wm.update_perception([self._make_entity(hp=8000, damage=100, buriedness=10)])
        # Второе наблюдение: hp и damage — None (вышли из видимости), buriedness обновлён.
        wm.update_perception([self._make_entity(hp=None, damage=None, buriedness=5)])
        merged = wm.tasks[10]
        assert merged.raw_sensor_data.hp == 8000        # сохранено из кэша
        assert merged.raw_sensor_data.damage == 100     # сохранено из кэша
        assert merged.raw_sensor_data.buriedness == 5   # обновлено

    def test_merge_overwrites_with_fresh_data(self) -> None:
        wm = WorldModel()
        wm.update_perception([self._make_entity(hp=8000, damage=100)])
        wm.update_perception([self._make_entity(hp=6000, damage=150)])
        merged = wm.tasks[10]
        assert merged.raw_sensor_data.hp == 6000
        assert merged.raw_sensor_data.damage == 150

    def test_estimated_death_time_recalculated_after_merge(self) -> None:
        """Я проверяю: estimated_death_time пересчитывается из слитых данных."""
        wm = WorldModel()
        wm.update_perception([self._make_entity(hp=1000, damage=10)])
        # TTL = 1000/10 = 100
        assert wm.tasks[10].computed_metrics.estimated_death_time == 100

        # Второе наблюдение: hp=None, damage=None (частичное обновление).
        # Без пересчёта estimated_death_time было бы 99999 (из-за None hp/damage в raw).
        wm.update_perception([self._make_entity(hp=None, damage=None)])
        # Merged: hp=1000, damage=10 → TTL = 100
        assert wm.tasks[10].computed_metrics.estimated_death_time == 100


# ---------------------------------------------------------------------------
# apply_perception — полный цикл
# ---------------------------------------------------------------------------

class TestApplyPerception:

    def _packet(
        self,
        tick: int = 0,
        visible: list[VisibleEntity] | None = None,
        allies: list[AgentState] | None = None,
        nodes: list[MapNode] | None = None,
        edges: list[MapEdge] | None = None,
        refuge_ids: list[int] | None = None,
        deleted_ids: list[int] | None = None,
    ) -> PerceptionPacket:
        return PerceptionPacket(
            tick=tick,
            own_state=make_agent(agent_id=1),
            visible_entities=visible or [],
            ally_states=allies or [],
            map_nodes=nodes or [],
            map_edges=edges or [],
            refuge_ids=refuge_ids or [],
            deleted_entity_ids=deleted_ids or [],
        )

    def test_builds_graph_on_tick_zero(self) -> None:
        wm = WorldModel()
        packet = self._packet(
            nodes=[MapNode(entity_id=1, x=0, y=0), MapNode(entity_id=2, x=100, y=0)],
            edges=[MapEdge(source_id=1, target_id=2, weight=100.0)],
        )
        wm.apply_perception(packet)
        assert wm.road_graph.number_of_nodes() == 2

    def test_stores_refuge_ids(self) -> None:
        wm = WorldModel()
        wm.apply_perception(self._packet(refuge_ids=[501, 502]))
        assert wm.refuge_ids == [501, 502]

    def test_updates_allies(self) -> None:
        wm = WorldModel()
        ally = make_agent(agent_id=99, agent_type=AgentType.FIRE_BRIGADE)
        wm.apply_perception(self._packet(allies=[ally]))
        assert 99 in wm.agents

    def test_adds_visible_entities_to_tasks(self) -> None:
        wm = WorldModel()
        civ = make_civilian(entity_id=42)
        wm.apply_perception(self._packet(visible=[civ]))
        assert 42 in wm.tasks

    def test_deleted_entity_ids_removes_from_tasks(self) -> None:
        """Я проверяю: deleted_entity_ids из ChangeSet удаляет сущности из кэша."""
        wm = WorldModel()
        blockade = make_blockade(entity_id=77)
        wm.tasks[77] = blockade
        wm.apply_perception(self._packet(deleted_ids=[77]))
        assert 77 not in wm.tasks

    def test_deleted_nonexistent_entity_is_safe(self) -> None:
        wm = WorldModel()
        # Удаление несуществующей сущности не бросает исключение.
        wm.apply_perception(self._packet(deleted_ids=[999]))
        assert 999 not in wm.tasks
