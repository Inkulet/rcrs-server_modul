from __future__ import annotations

import logging

import networkx as nx

from decision.utility.distance import MAX_MAP_DISTANCE
from world.entities import MapNode, VisibleEntity


logger = logging.getLogger(__name__)


def compute_path(
    graph: nx.Graph,
    from_id: int,
    to_id: int,
) -> list[int]:
    if from_id == to_id:
        return [from_id]

    try:
        path: list[int] = nx.astar_path(graph, from_id, to_id, weight="weight")
        logger.debug("Я нашёл путь от %d до %d: %d шагов", from_id, to_id, len(path))
        return path
    except nx.NetworkXNoPath:
        logger.warning("Я не нашёл пути от %d до %d — граф несвязный", from_id, to_id)
        return []
    except nx.NodeNotFound as exc:
        logger.warning("Я не нашёл узел в графе: %s", exc)
        return []


def compute_path_distance(
    graph: nx.Graph,
    from_id: int,
    to_id: int,
) -> float:
    if from_id == to_id:
        return 0.0

    try:
        distance: float = nx.dijkstra_path_length(graph, from_id, to_id, weight="weight")
        return distance
    except nx.NetworkXNoPath:
        logger.warning(
            "Я не нашёл пути от %d до %d, возвращаю MAX_MAP_DISTANCE=%.0f",
            from_id, to_id, MAX_MAP_DISTANCE,
        )
        return MAX_MAP_DISTANCE
    except nx.NodeNotFound as exc:
        logger.warning("Я не нашёл узел при расчёте дистанции: %s", exc)
        return MAX_MAP_DISTANCE


def fill_path_distances(
    graph: nx.Graph,
    agent_node_id: int,
    entities: list[VisibleEntity],
) -> list[VisibleEntity]:
    try:
        dist_map: dict[int, float] = nx.single_source_dijkstra_path_length(
            graph, agent_node_id, weight="weight"
        )
    except nx.NodeNotFound:
        logger.warning(
            "Я не нашёл позицию агента %d в графе, все дистанции = MAX", agent_node_id
        )
        dist_map = {}

    for entity in entities:
        nav_id = entity.id
        if nav_id not in dist_map:
            pos = entity.raw_sensor_data.position_on_edge
            if pos is not None and pos in dist_map:
                nav_id = pos

        distance = dist_map.get(nav_id, MAX_MAP_DISTANCE)
        entity.computed_metrics.path_distance = distance
        logger.debug(
            "Я обновил path_distance для entity_id=%d (nav_id=%d): %.1f мм",
            entity.id, nav_id, distance,
        )
    return entities


def nearest_refuge_path(
    graph: nx.Graph,
    from_id: int,
    refuge_ids: list[int],
) -> list[int]:
    if not refuge_ids:
        logger.warning("Я не знаю ни одного убежища — невозможно построить маршрут")
        return []

    best_path: list[int] = []
    best_distance: float = MAX_MAP_DISTANCE

    for refuge_id in refuge_ids:
        path = compute_path(graph, from_id, refuge_id)
        if not path:
            continue
        dist = sum(
            graph[u][v].get("weight", 1.0)
            for u, v in zip(path, path[1:])
        )
        if dist < best_distance:
            best_distance = dist
            best_path = path

    if best_path:
        logger.info(
            "Я выбрал ближайшее убежище refuge_id=%d: дистанция=%.1f мм",
            best_path[-1], best_distance,
        )
    else:
        logger.error("Я не смог найти путь ни к одному убежищу из node=%d", from_id)

    return best_path


__all__ = [
    "compute_path",
    "compute_path_distance",
    "fill_path_distances",
    "nearest_refuge_path",
]
