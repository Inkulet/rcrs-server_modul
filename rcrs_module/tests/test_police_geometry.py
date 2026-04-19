from __future__ import annotations

import math

from action.police_geometry import (
    intersects_area_edge,
    intersects_blockade,
    nearest_apex,
    point_inside_apexes,
    scale_clear_vector,
)


# Квадратный завал 1000×1000 с центром (5000, 5000).
SQUARE_BLOCKADE = [4500, 4500, 5500, 4500, 5500, 5500, 4500, 5500]


def test_nearest_apex_finds_closest_vertex() -> None:
    agent = (0.0, 0.0)
    result = nearest_apex(agent, SQUARE_BLOCKADE)
    assert result is not None
    x, y, d = result
    assert (x, y) == (4500, 4500)
    assert math.isclose(d, math.hypot(4500, 4500), rel_tol=1e-6)


def test_nearest_apex_none_on_empty() -> None:
    assert nearest_apex((0.0, 0.0), None) is None
    assert nearest_apex((0.0, 0.0), []) is None


def test_scale_clear_vector_gives_unit_times_distance() -> None:
    clear_distance = 1000.0
    cx, cy = scale_clear_vector((0.0, 0.0), (3.0, 4.0), clear_distance)
    # 3-4-5 треугольник: единичный вектор (0.6, 0.8) × 1000.
    assert cx == 600
    assert cy == 800


def test_scale_clear_vector_handles_zero_vector() -> None:
    cx, cy = scale_clear_vector((100.0, 200.0), (100.0, 200.0), 500.0)
    assert (cx, cy) == (100, 200)


def test_intersects_blockade_true_when_line_crosses_square() -> None:
    assert intersects_blockade((0, 5000), (10000, 5000), SQUARE_BLOCKADE) is True


def test_intersects_blockade_false_when_line_outside() -> None:
    assert intersects_blockade((0, 0), (1000, 0), SQUARE_BLOCKADE) is False


def test_intersects_blockade_false_on_empty_apexes() -> None:
    assert intersects_blockade((0, 0), (100, 100), None) is False
    assert intersects_blockade((0, 0), (100, 100), [1, 2]) is False


def test_intersects_area_edge_detects_crossing() -> None:
    # Квадратный road 2000×2000 вокруг (0,0).
    road = [-1000, -1000, 1000, -1000, 1000, 1000, -1000, 1000]
    # Линия от внутри к снаружи пересекает границу.
    assert intersects_area_edge((0, 0), (5000, 0), road) is True


def test_intersects_area_edge_no_cross_inside() -> None:
    road = [-1000, -1000, 1000, -1000, 1000, 1000, -1000, 1000]
    # Линия полностью внутри road — границу не пересекает.
    assert intersects_area_edge((0, 0), (500, 500), road) is False


def test_point_inside_apexes() -> None:
    assert point_inside_apexes((5000, 5000), SQUARE_BLOCKADE) is True
    assert point_inside_apexes((0, 0), SQUARE_BLOCKADE) is False
