from __future__ import annotations

"""В этом модуле я проверяю корректность навигационных функций (UC-6)."""

import networkx as nx
import pytest

from action.navigation import (
    MAX_MAP_DISTANCE,
    compute_path,
    compute_path_distance,
    fill_path_distances,
    nearest_refuge_path,
)
from world.entities import (
    ComputedMetrics,
    EntityType,
    RawSensorData,
    VisibleEntity,
)


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_graph() -> nx.Graph:
    """Я создаю граф из четырёх узлов по прямой линии: 1 - 2 - 3 - 4."""
    g = nx.Graph()
    g.add_node(1, x=0,    y=0)
    g.add_node(2, x=1000, y=0)
    g.add_node(3, x=2000, y=0)
    g.add_node(4, x=3000, y=0)
    g.add_edge(1, 2, weight=1000.0)
    g.add_edge(2, 3, weight=1000.0)
    g.add_edge(3, 4, weight=1000.0)
    return g


@pytest.fixture()
def disconnected_graph() -> nx.Graph:
    """Я создаю граф с двумя несвязными компонентами: 1-2 и 3-4."""
    g = nx.Graph()
    g.add_node(1, x=0,    y=0)
    g.add_node(2, x=1000, y=0)
    g.add_node(3, x=5000, y=5000)
    g.add_node(4, x=6000, y=5000)
    g.add_edge(1, 2, weight=1000.0)
    g.add_edge(3, 4, weight=1000.0)
    return g


def _make_entity(entity_id: int, path_distance: float = 0.0) -> VisibleEntity:
    """Я создаю минимальную VisibleEntity для теста навигации."""
    return VisibleEntity(
        id=entity_id,
        type=EntityType.BUILDING,
        raw_sensor_data=RawSensorData(temperature=300.0, fieryness=1, floors=1, ground_area=100),
        computed_metrics=ComputedMetrics(
            path_distance=path_distance,
            estimated_death_time=999,
            total_area=100,
        ),
        utility_score=0.0,
    )


# ---------------------------------------------------------------------------
# compute_path
# ---------------------------------------------------------------------------


class TestComputePath:
    def test_same_node_returns_single_element(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что путь из узла в себя — список из одного элемента."""
        path = compute_path(linear_graph, 1, 1)
        assert path == [1]

    def test_direct_neighbor(self, linear_graph: nx.Graph) -> None:
        """Я проверяю прямой путь между соседними узлами."""
        path = compute_path(linear_graph, 1, 2)
        assert path == [1, 2]

    def test_multi_hop_path(self, linear_graph: nx.Graph) -> None:
        """Я проверяю путь через несколько промежуточных узлов."""
        path = compute_path(linear_graph, 1, 4)
        assert path == [1, 2, 3, 4]

    def test_unreachable_node_returns_empty(self, disconnected_graph: nx.Graph) -> None:
        """Я проверяю, что недостижимый узел возвращает пустой список."""
        path = compute_path(disconnected_graph, 1, 3)
        assert path == []

    def test_missing_node_returns_empty(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что отсутствующий узел возвращает пустой список."""
        path = compute_path(linear_graph, 1, 999)
        assert path == []


# ---------------------------------------------------------------------------
# compute_path_distance
# ---------------------------------------------------------------------------


class TestComputePathDistance:
    def test_same_node_is_zero(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что расстояние от узла до себя равно нулю."""
        assert compute_path_distance(linear_graph, 1, 1) == 0.0

    def test_single_edge_distance(self, linear_graph: nx.Graph) -> None:
        """Я проверяю точность расстояния по одному ребру."""
        dist = compute_path_distance(linear_graph, 1, 2)
        assert dist == pytest.approx(1000.0)

    def test_multi_hop_distance(self, linear_graph: nx.Graph) -> None:
        """Я проверяю суммарное расстояние через несколько рёбер."""
        dist = compute_path_distance(linear_graph, 1, 4)
        assert dist == pytest.approx(3000.0)

    def test_unreachable_returns_max(self, disconnected_graph: nx.Graph) -> None:
        """Я проверяю, что недостижимый узел возвращает MAX_MAP_DISTANCE."""
        dist = compute_path_distance(disconnected_graph, 1, 3)
        assert dist == MAX_MAP_DISTANCE

    def test_missing_node_returns_max(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что отсутствующий узел возвращает MAX_MAP_DISTANCE."""
        dist = compute_path_distance(linear_graph, 1, 999)
        assert dist == MAX_MAP_DISTANCE


# ---------------------------------------------------------------------------
# fill_path_distances
# ---------------------------------------------------------------------------


class TestFillPathDistances:
    def test_updates_path_distance(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что fill_path_distances корректно обновляет дистанцию."""
        entities = [_make_entity(4, path_distance=0.0)]
        updated = fill_path_distances(linear_graph, agent_node_id=1, entities=entities)
        assert len(updated) == 1
        assert updated[0].computed_metrics.path_distance == pytest.approx(3000.0)

    def test_mutates_in_place(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что fill_path_distances обновляет path_distance in-place."""
        original = _make_entity(4, path_distance=0.0)
        fill_path_distances(linear_graph, agent_node_id=1, entities=[original])
        assert original.computed_metrics.path_distance == pytest.approx(3000.0)

    def test_multiple_entities(self, linear_graph: nx.Graph) -> None:
        """Я проверяю обработку нескольких сущностей за один вызов."""
        entities = [_make_entity(2), _make_entity(3), _make_entity(4)]
        updated = fill_path_distances(linear_graph, agent_node_id=1, entities=entities)
        distances = [e.computed_metrics.path_distance for e in updated]
        assert distances == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_empty_entities_list(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что пустой список сущностей возвращается без изменений."""
        result = fill_path_distances(linear_graph, agent_node_id=1, entities=[])
        assert result == []

    def test_unreachable_entity_gets_max_distance(self, disconnected_graph: nx.Graph) -> None:
        """Я проверяю, что недостижимая сущность получает MAX_MAP_DISTANCE."""
        entities = [_make_entity(3)]
        updated = fill_path_distances(disconnected_graph, agent_node_id=1, entities=entities)
        assert updated[0].computed_metrics.path_distance == MAX_MAP_DISTANCE


# ---------------------------------------------------------------------------
# nearest_refuge_path
# ---------------------------------------------------------------------------


class TestNearestRefugePath:
    def test_finds_nearest_refuge(self, linear_graph: nx.Graph) -> None:
        """Я проверяю выбор ближайшего убежища из нескольких."""
        # Убежища на узлах 2 (1000 мм) и 4 (3000 мм) от узла 1.
        path = nearest_refuge_path(linear_graph, from_id=1, refuge_ids=[4, 2])
        # Я ожидаю путь к ближайшему убежищу (узел 2).
        assert path[-1] == 2

    def test_single_refuge(self, linear_graph: nx.Graph) -> None:
        """Я проверяю путь к единственному убежищу."""
        path = nearest_refuge_path(linear_graph, from_id=1, refuge_ids=[3])
        assert path == [1, 2, 3]

    def test_empty_refuges_returns_empty(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что пустой список убежищ возвращает пустой путь."""
        path = nearest_refuge_path(linear_graph, from_id=1, refuge_ids=[])
        assert path == []

    def test_unreachable_refuges_returns_empty(self, disconnected_graph: nx.Graph) -> None:
        """Я проверяю, что недостижимые убежища возвращают пустой путь."""
        path = nearest_refuge_path(disconnected_graph, from_id=1, refuge_ids=[3, 4])
        assert path == []

    def test_at_refuge_already(self, linear_graph: nx.Graph) -> None:
        """Я проверяю, что агент уже в убежище возвращает путь из одного узла."""
        path = nearest_refuge_path(linear_graph, from_id=2, refuge_ids=[2])
        assert path == [2]
