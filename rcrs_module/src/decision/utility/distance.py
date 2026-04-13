from __future__ import annotations


import logging
from typing import Optional

import networkx as nx

from world.entities import Position


logger = logging.getLogger(__name__)

MAX_MAP_DISTANCE: float = 100_000_000.0


def _safe_shortest_path_length(
    graph: nx.Graph,
    source_id: int,
    target_id: int,
) -> Optional[float]:

    try:
        return float(nx.shortest_path_length(graph, source_id, target_id, weight="weight"))
    except nx.NetworkXNoPath:
        logger.warning(
            "Я не нашел путь в графе между %s и %s",
            source_id,
            target_id,
        )
        return None
    except nx.NodeNotFound as exc:
        logger.warning("Я получил неизвестный узел при поиске пути: %s", exc)
        return None


def distance_factor(
    graph: nx.Graph,
    agent_position: Position,
    target_position: Position,
    max_map_distance: float = MAX_MAP_DISTANCE,
) -> float:

    if max_map_distance <= 0:
        logger.warning("Я получил неположительную MaxMapDistance, поэтому возвращаю штраф 1.0")
        return 1.0

    distance = _safe_shortest_path_length(
        graph,
        agent_position.entity_id,
        target_position.entity_id,
    )

    if distance is None:
        return 1.0

    try:
        value = distance / max_map_distance
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете геометрического фактора")
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

__all__ = ["MAX_MAP_DISTANCE", "distance_factor_precomputed"]
