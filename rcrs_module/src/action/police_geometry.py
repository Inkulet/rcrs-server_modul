from __future__ import annotations

import math
from typing import Optional, Sequence

from shapely import LineString, Point, Polygon


def nearest_apex(
    agent_xy: tuple[float, float],
    apexes: Sequence[int] | None,
) -> Optional[tuple[int, int, float]]:
    if apexes is None or len(apexes) < 2:
        return None

    ax, ay = agent_xy
    best_x: int = 0
    best_y: int = 0
    best_d: float = math.inf
    for i in range(0, len(apexes) - 1, 2):
        px = int(apexes[i])
        py = int(apexes[i + 1])
        d = math.hypot(px - ax, py - ay)
        if d < best_d:
            best_d = d
            best_x = px
            best_y = py

    if best_d is math.inf:
        return None
    return best_x, best_y, best_d


def scale_clear_vector(
    agent_xy: tuple[float, float],
    target_xy: tuple[float, float],
    clear_distance: float,
) -> tuple[int, int]:
    ax, ay = agent_xy
    tx, ty = target_xy
    dx = tx - ax
    dy = ty - ay
    length = math.hypot(dx, dy)
    if length <= 0:
        return int(ax), int(ay)
    k = clear_distance / length
    return int(ax + dx * k), int(ay + dy * k)


BACK_OFFSET: float = 510.0


def scale_back_vector(
    agent_xy: tuple[float, float],
    target_xy: tuple[float, float],
    back_distance: float = BACK_OFFSET,
) -> tuple[int, int]:
    ax, ay = agent_xy
    tx, ty = target_xy
    dx = tx - ax
    dy = ty - ay
    length = math.hypot(dx, dy)
    if length <= 0:
        return int(ax), int(ay)
    k = back_distance / length
    return int(ax - dx * k), int(ay - dy * k)


def segment_crosses_edges(
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    apexes: Sequence[int] | None,
) -> bool:
    if apexes is None or len(apexes) < 4:
        return False
    try:
        segment = LineString([line_start, line_end])
    except (ValueError, TypeError):
        return False
    try:
        for i in range(0, len(apexes) - 3, 2):
            edge = LineString(
                [
                    (int(apexes[i]), int(apexes[i + 1])),
                    (int(apexes[i + 2]), int(apexes[i + 3])),
                ]
            )
            if segment.intersects(edge):
                return True
    except (ValueError, TypeError):
        return False
    return False


def _apexes_to_polygon(apexes: Sequence[int] | None) -> Optional[Polygon]:
    if apexes is None or len(apexes) < 6:
        return None
    try:
        pts = [(int(apexes[i]), int(apexes[i + 1])) for i in range(0, len(apexes) - 1, 2)]
    except (TypeError, ValueError):
        return None
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid or poly.is_empty:
            return None
        return poly
    except (ValueError, TypeError):
        return None


def intersects_blockade(
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    blockade_apexes: Sequence[int] | None,
) -> bool:
    poly = _apexes_to_polygon(blockade_apexes)
    if poly is None:
        return False
    try:
        line = LineString([line_start, line_end])
        return bool(line.intersects(poly))
    except (ValueError, TypeError):
        return False


def intersects_area_edge(
    agent_xy: tuple[float, float],
    target_xy: tuple[float, float],
    area_apexes: Sequence[int] | None,
) -> bool:
    poly = _apexes_to_polygon(area_apexes)
    if poly is None:
        return False
    try:
        line = LineString([agent_xy, target_xy])
        return bool(line.crosses(poly.boundary) or line.intersects(poly.boundary))
    except (ValueError, TypeError):
        return False


def point_inside_apexes(
    xy: tuple[float, float],
    apexes: Sequence[int] | None,
) -> bool:
    poly = _apexes_to_polygon(apexes)
    if poly is None:
        return False
    try:
        return bool(poly.contains(Point(xy)))
    except (ValueError, TypeError):
        return False


__all__ = [
    "nearest_apex",
    "scale_clear_vector",
    "scale_back_vector",
    "segment_crosses_edges",
    "intersects_blockade",
    "intersects_area_edge",
    "point_inside_apexes",
    "BACK_OFFSET",
]
