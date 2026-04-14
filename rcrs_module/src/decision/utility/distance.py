from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from world.entities import Position


logger = logging.getLogger(__name__)

MAX_MAP_DISTANCE: float = 100_000_000.0


def distance_factor(
    graph: nx.Graph,
    agent_position: Position,
    target_position: Position,
    max_map_distance: float = MAX_MAP_DISTANCE,
) -> float:
    if max_map_distance <= 0:
        logger.warning("Я получил неположительную MaxMapDistance, возвращаю штраф 1.0")
        return 1.0

    try:
        distance: Optional[float] = float(
            nx.shortest_path_length(
                graph,
                agent_position.entity_id,
                target_position.entity_id,
                weight="weight",
            )
        )
    except nx.NetworkXNoPath:
        logger.warning(
            "Я не нашёл путь между %s и %s",
            agent_position.entity_id, target_position.entity_id,
        )
        distance = None
    except nx.NodeNotFound as exc:
        logger.warning("Я получил неизвестный узел: %s", exc)
        distance = None

    if distance is None:
        return 1.0

    try:
        value = distance / max_map_distance
        return max(0.0, min(1.0, value))
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчёте геометрического фактора")
        return 1.0


def distance_factor_precomputed(
    path_distance: float,
    max_map_distance: float = MAX_MAP_DISTANCE,
) -> float:
    if max_map_distance <= 0:
        logger.warning("Я получил неположительную MaxMapDistance, возвращаю штраф 1.0")
        return 1.0
    try:
        value = path_distance / max_map_distance
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при нормировке предвычисленной дистанции")
        return 1.0

__all__ = ["MAX_MAP_DISTANCE", "distance_factor", "distance_factor_precomputed"]

