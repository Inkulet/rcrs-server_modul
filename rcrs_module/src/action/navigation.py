from __future__ import annotations

import logging
import random

import networkx as nx

from decision.utility.distance import MAX_MAP_DISTANCE
from world.entities import MapNode, VisibleEntity


logger = logging.getLogger(__name__)

# Я определяю долю самых удалённых узлов, из которых выбираю цель исследования.
# 0.3 означает «верхние 30% узлов по расстоянию» — даёт агенту широкий разброс
# между районами карты, но отсекает слишком локальные шаги.
EXPLORATION_FAR_QUANTILE: float = 0.3

# Я ставлю минимальное расстояние для цели исследования, чтобы агент реально
# перешёл из своего района, а не выбрал соседний узел.
EXPLORATION_MIN_DISTANCE: float = 30_000.0


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


import math


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
            else:
                if entity.entity_x is not None and entity.entity_y is not None:
                    best_node = None
                    best_dist = float('inf')
                    for node_id, data in graph.nodes(data=True):
                        nx_val = data.get("x", 0)
                        ny_val = data.get("y", 0)
                        d = math.hypot(entity.entity_x - nx_val, entity.entity_y - ny_val)
                        if d < best_dist:
                            best_dist = d
                            best_node = node_id

                    if best_node is not None and best_node in dist_map:
                        nav_id = best_node

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


def pick_exploration_target(
    graph: nx.Graph,
    start_node: int,
    visited: set[int] | None = None,
    min_distance: float = EXPLORATION_MIN_DISTANCE,
    far_quantile: float = EXPLORATION_FAR_QUANTILE,
    rng: random.Random | None = None,
) -> int | None:
    # Я выбираю дальнюю цель для исследования: предпочитаю непосещённые узлы из
    # верхнего квантиля по расстоянию, чтобы агенты расходились по разным районам.
    if not graph.has_node(start_node):
        logger.warning("Я не нашёл start_node=%d в графе для exploration target", start_node)
        return None

    try:
        dist_map: dict[int, float] = nx.single_source_dijkstra_path_length(
            graph, start_node, weight="weight"
        )
    except nx.NodeNotFound:
        logger.warning("Я не нашёл start_node=%d при расчёте дистанций для exploration", start_node)
        return None

    # Я исключаю сам start_node из кандидатов.
    reachable = [(node, d) for node, d in dist_map.items() if node != start_node]
    if not reachable:
        logger.debug("Я не нашёл достижимых узлов из start_node=%d", start_node)
        return None

    # Я фильтрую по минимальной дистанции — иначе агент может выбрать соседний узел
    # и не покинуть текущий район.
    far_candidates = [item for item in reachable if item[1] >= min_distance]
    if not far_candidates:
        # Я отступаю к верхнему квантилю, если нет узлов за min_distance
        # (например, мелкая карта или агент уже в углу).
        reachable.sort(key=lambda item: item[1], reverse=True)
        cutoff = max(1, int(len(reachable) * far_quantile))
        far_candidates = reachable[:cutoff]

    # Я отдаю приоритет непосещённым узлам, чтобы агент исследовал новые территории.
    unvisited: list[tuple[int, float]] = []
    if visited is not None:
        unvisited = [item for item in far_candidates if item[0] not in visited]

    pool = unvisited if unvisited else far_candidates
    if not pool:
        return None

    rnd = rng if rng is not None else random
    # Я выбираю случайным образом из пула, чтобы разные агенты расходились в разные районы.
    chosen, chosen_dist = rnd.choice(pool)
    logger.debug(
        "Я выбрал exploration target=%d (dist=%.0f, pool=%d, unvisited=%d)",
        chosen, chosen_dist, len(pool), len(unvisited),
    )
    return chosen


def plan_exploration_path(
    graph: nx.Graph,
    start_node: int,
    target_node: int,
    max_steps: int = 50,
) -> list[int]:
    # Я строю путь к цели исследования и ограничиваю длину, чтобы не отправлять
    # серверу огромные маршруты (сервер всё равно будет корректировать каждый такт).
    path = compute_path(graph, start_node, target_node)
    if not path:
        return []
    if len(path) > max_steps:
        return path[:max_steps]
    return path


__all__ = [
    "EXPLORATION_FAR_QUANTILE",
    "EXPLORATION_MIN_DISTANCE",
    "compute_path",
    "compute_path_distance",
    "fill_path_distances",
    "nearest_refuge_path",
    "pick_exploration_target",
    "plan_exploration_path",
]
