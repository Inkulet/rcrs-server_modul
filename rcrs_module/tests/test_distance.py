from __future__ import annotations

"""В этом модуле я проверяю корректность геометрического фактора f_dist по диплому."""

# Ключевые инварианты: нормировка к [0,1], корректный Дейкстра на графе G=(V,E),
# безопасное поведение при отсутствии пути или неизвестных вершинах.

import networkx as nx
import pytest

from decision.utility.distance import MAX_MAP_DISTANCE, distance_factor
from world.entities import Position

from conftest import make_world_with_graph


def make_pos(entity_id: int, x: int = 0, y: int = 0) -> Position:
    """Я создаю Position с минимальными данными для тестов расстояния."""
    return Position(entity_id=entity_id, x=x, y=y)


# ===========================================================================
# Тесты distance_factor
# ===========================================================================


class TestDistanceFactor:
    """Я проверяю формулу f_dist = d_ij / MaxMapDistance на графе G."""

    def test_same_node_returns_zero(self) -> None:
        """Я проверяю: агент и цель в одном узле → d=0 → f_dist = 0.0."""
        wm = make_world_with_graph()
        result = distance_factor(wm.road_graph, make_pos(1), make_pos(1))
        assert result == 0.0

    def test_adjacent_nodes_returns_correct_ratio(self) -> None:
        """Я проверяю: d=500 при MaxMapDistance=100000 → f_dist=0.005."""
        wm = make_world_with_graph()
        result = distance_factor(wm.road_graph, make_pos(1), make_pos(2))
        expected = 500.0 / MAX_MAP_DISTANCE
        assert abs(result - expected) < 1e-9

    def test_two_hops_sums_edge_weights(self) -> None:
        """Я проверяю: путь через промежуточный узел суммирует веса рёбер."""
        wm = make_world_with_graph()
        # 1 → 2 → 3: вес 500 + 500 = 1000
        result = distance_factor(wm.road_graph, make_pos(1), make_pos(3))
        expected = 1000.0 / MAX_MAP_DISTANCE
        assert abs(result - expected) < 1e-9

    def test_no_path_returns_max_penalty(self) -> None:
        """Я проверяю: нет пути → возвращаю штраф 1.0 (максимальная дистанция)."""
        wm = make_world_with_graph()
        # Узел 99 не связан с графом
        wm.road_graph.add_node(99)
        result = distance_factor(wm.road_graph, make_pos(1), make_pos(99))
        assert result == 1.0

    def test_unknown_source_node_returns_max_penalty(self) -> None:
        """Я проверяю: неизвестный исходный узел → штраф 1.0."""
        wm = make_world_with_graph()
        result = distance_factor(wm.road_graph, make_pos(999), make_pos(1))
        assert result == 1.0

    def test_unknown_target_node_returns_max_penalty(self) -> None:
        """Я проверяю: неизвестный целевой узел → штраф 1.0."""
        wm = make_world_with_graph()
        result = distance_factor(wm.road_graph, make_pos(1), make_pos(999))
        assert result == 1.0

    def test_empty_graph_returns_max_penalty(self) -> None:
        """Я проверяю: пустой граф → штраф 1.0, без исключений."""
        empty = nx.Graph()
        result = distance_factor(empty, make_pos(1), make_pos(2))
        assert result == 1.0

    def test_result_in_unit_interval(self) -> None:
        """Я проверяю нормировку: f_dist всегда ∈ [0, 1]."""
        wm = make_world_with_graph()
        for src_id, tgt_id in [(1, 1), (1, 2), (1, 3), (2, 3)]:
            r = distance_factor(wm.road_graph, make_pos(src_id), make_pos(tgt_id))
            assert 0.0 <= r <= 1.0, f"Выход за [0,1]: src={src_id}, tgt={tgt_id}, result={r}"

    def test_max_map_distance_zero_returns_penalty(self) -> None:
        """Я проверяю: MaxMapDistance=0 → защита от деления на ноль, возвращаю 1.0."""
        wm = make_world_with_graph()
        result = distance_factor(wm.road_graph, make_pos(1), make_pos(2), max_map_distance=0.0)
        assert result == 1.0

    def test_large_distance_clamped_to_one(self) -> None:
        """Я проверяю: путь > MaxMapDistance зажимается к 1.0."""
        g = nx.Graph()
        g.add_edge(1, 2, weight=MAX_MAP_DISTANCE * 2)
        result = distance_factor(g, make_pos(1), make_pos(2))
        assert result == 1.0
