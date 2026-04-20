from __future__ import annotations

import logging
import math
import random
from collections import OrderedDict

import networkx as nx

from decision.utility.distance import MAX_MAP_DISTANCE
from world.entities import VisibleEntity


logger = logging.getLogger(__name__)

EXPLORATION_FAR_QUANTILE: float = 0.3

EXPLORATION_MIN_DISTANCE: float = 30_000.0

_PATH_CACHE_MAX_SIZE: int = 4096
_PATH_CACHE: OrderedDict[tuple[object, int, int, int], tuple[int, ...]] = OrderedDict()


def find_nearest_node(graph: nx.Graph, x: int, y: int) -> int | None:
    best_node: int | None = None
    best_dist: float = float("inf")

    for node_id, data in graph.nodes(data=True):
        nx_val: int = data.get("x", 0)
        ny_val: int = data.get("y", 0)
        dist = math.hypot(x - nx_val, y - ny_val)
        if dist < best_dist:
            best_dist = dist
            best_node = node_id

    return best_node


def compute_path(
    graph: nx.Graph,
    from_id: int,
    to_id: int,
) -> list[int]:
    if from_id == to_id:
        return [from_id]

    cache_token = graph.graph.setdefault("_path_cache_token", object())
    revision = int(graph.graph.get("revision", 0))
    cache_key = (cache_token, revision, from_id, to_id)
    cached = _PATH_CACHE.get(cache_key)
    if cached is not None:
        _PATH_CACHE.move_to_end(cache_key)
        return list(cached)

    reverse_key = (cache_token, revision, to_id, from_id)
    reverse_cached = _PATH_CACHE.get(reverse_key)
    if reverse_cached is not None:
        _PATH_CACHE.move_to_end(reverse_key)
        return list(reversed(reverse_cached))

    def heuristic(node_id: int, goal_id: int) -> float:
        node_attrs = graph.nodes.get(node_id, {})
        goal_attrs = graph.nodes.get(goal_id, {})
        nx_val = node_attrs.get("x")
        ny_val = node_attrs.get("y")
        gx_val = goal_attrs.get("x")
        gy_val = goal_attrs.get("y")
        if nx_val is None or ny_val is None or gx_val is None or gy_val is None:
            return 0.0
        return math.hypot(float(nx_val) - float(gx_val), float(ny_val) - float(gy_val))

    try:
        path: list[int] = nx.astar_path(
            graph, from_id, to_id, heuristic=heuristic, weight="weight",
        )
        _remember_path(cache_key, path)
        logger.debug("Я нашёл путь от %d до %d: %d шагов", from_id, to_id, len(path))
        return path
    except nx.NetworkXNoPath:
        _remember_path(cache_key, [])
        logger.warning("Я не нашёл пути от %d до %d — граф несвязный", from_id, to_id)
        return []
    except nx.NodeNotFound as exc:
        logger.warning("Я не нашёл узел в графе: %s", exc)
        return []


def _remember_path(
    cache_key: tuple[object, int, int, int],
    path: list[int],
) -> None:
    _PATH_CACHE[cache_key] = tuple(path)
    _PATH_CACHE.move_to_end(cache_key)
    while len(_PATH_CACHE) > _PATH_CACHE_MAX_SIZE:
        _PATH_CACHE.popitem(last=False)


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
    blockades_by_node: dict[int, set[int]] | None = None,
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
            elif entity.entity_x is not None and entity.entity_y is not None:
                nearest = find_nearest_node(graph, entity.entity_x, entity.entity_y)
                if nearest is not None and nearest in dist_map:
                    nav_id = nearest

        # Fallback: reverse lookup через blockades_by_node.
        # Если завал не удалось привязать к графу через position_on_edge
        # или entity_x/entity_y, ищем его в индексе «узел → завалы».
        if nav_id not in dist_map and blockades_by_node:
            for node_id, blk_set in blockades_by_node.items():
                if entity.id in blk_set and node_id in dist_map:
                    nav_id = node_id
                    logger.debug(
                        "Я привязал завал entity_id=%d к узлу %d через blockades_by_node",
                        entity.id, node_id,
                    )
                    break

        distance = dist_map.get(nav_id, MAX_MAP_DISTANCE)

        if distance >= MAX_MAP_DISTANCE:
            logger.warning(
                "Я не смог привязать entity_id=%d к графу: "
                "pos_on_edge=%s, entity_xy=(%s,%s), nav_id=%d → distance=MAX",
                entity.id,
                entity.raw_sensor_data.position_on_edge,
                entity.entity_x, entity.entity_y,
                nav_id,
            )

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


def choose_refuge_with_exit(
    graph: nx.Graph,
    from_id: int,
    refuge_ids: list[int],
    next_target_node: int | None,
) -> list[int]:
    if not refuge_ids:
        return []
    if next_target_node is None or not graph.has_node(next_target_node):
        return nearest_refuge_path(graph, from_id, refuge_ids)

    scored: list[tuple[float, int, list[int]]] = []
    for refuge_id in refuge_ids:
        path = compute_path(graph, from_id, refuge_id)
        if not path:
            continue
        dist = sum(
            graph[u][v].get("weight", 1.0) for u, v in zip(path, path[1:])
        )
        scored.append((dist, refuge_id, path))

    scored.sort(key=lambda s: s[0])

    for dist, refuge_id, path in scored:
        exit_path = compute_path(graph, refuge_id, next_target_node)
        if exit_path:
            logger.info(
                "Я выбрал refuge_id=%d с проверкой exit→node=%d, "
                "дистанция=%.1f мм", refuge_id, next_target_node, dist,
            )
            return path

    if scored:
        logger.info(
            "Я не нашёл refuge с exit→node=%d, fallback на ближайший",
            next_target_node,
        )
        return scored[0][2]
    return []


def pick_search_target(
    graph: nx.Graph,
    start_node: int,
    visited: set[int] | None = None,
    partition_index: int | None = None,
    partition_count: int | None = None,
    excluded: set[int] | None = None,
) -> int | None:
    if not graph.has_node(start_node):
        logger.warning("Я не нашёл start_node=%d в графе для search target", start_node)
        return None

    try:
        dist_map: dict[int, float] = nx.single_source_dijkstra_path_length(
            graph, start_node, weight="weight"
        )
    except nx.NodeNotFound:
        logger.warning(
            "Я не нашёл start_node=%d при расчёте дистанций для search target",
            start_node,
        )
        return None

    candidates: list[int] = [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("area_type") == "BUILDING" and node_id in dist_map
    ]
    if not candidates:
        logger.debug("Я не нашёл building-узлов для целевого поиска")
        return None

    if visited is not None:
        unvisited = [node_id for node_id in candidates if node_id not in visited]
        if unvisited:
            candidates = unvisited

    if excluded:
        filtered = [node_id for node_id in candidates if node_id not in excluded]
        if filtered:
            candidates = filtered

    if (
        partition_index is not None
        and partition_count is not None
        and partition_count > 1
        and 0 <= partition_index < partition_count
    ):
        partitioned = [
            node_id
            for idx, node_id in enumerate(sorted(candidates))
            if idx % partition_count == partition_index
        ]
        if partitioned:
            candidates = partitioned
        else:
            logger.debug(
                "Я не нашёл building-целей в своём секторе поиска "
                "(sector=%d/%d), переключаюсь на глобальный fallback",
                partition_index, partition_count,
            )

    best_target: int | None = None
    best_distance: float = float("inf")
    for node_id in candidates:
        distance = dist_map[node_id]
        if distance < best_distance:
            best_distance = distance
            best_target = node_id

    if best_target is not None:
        logger.debug(
            "Я выбрал search target=%d (dist=%.0f, candidates=%d)",
            best_target, best_distance, len(candidates),
        )
    return best_target


def random_walk(
    graph: nx.Graph,
    start_node: int,
    length: int = 50,
    visited: set[int] | None = None,
) -> list[int]:
    if not graph.has_node(start_node):
        logger.warning("Я не нашёл start_node=%d в графе для random walk", start_node)
        return [start_node]

    if visited is not None:
        visited.add(start_node)

    path: list[int] = [start_node]
    current = start_node

    for _ in range(length):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break

        if visited is not None:
            unvisited = [n for n in neighbors if n not in visited]
            next_node = random.choice(unvisited) if unvisited else random.choice(neighbors)
        else:
            next_node = random.choice(neighbors)

        path.append(next_node)
        current = next_node
        if visited is not None:
            visited.add(next_node)

    logger.debug(
        "Я построил random walk: start=%d, длина=%d узлов, visited=%d",
        start_node, len(path), len(visited) if visited is not None else 0,
    )
    return path


def pick_exploration_target(
    graph: nx.Graph,
    start_node: int,
    visited: set[int] | None = None,
    min_distance: float = EXPLORATION_MIN_DISTANCE,
    far_quantile: float = EXPLORATION_FAR_QUANTILE,
    rng: random.Random | None = None,
) -> int | None:
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

    reachable = [(node, d) for node, d in dist_map.items() if node != start_node]
    if not reachable:
        logger.debug("Я не нашёл достижимых узлов из start_node=%d", start_node)
        return None

    far_candidates = [item for item in reachable if item[1] >= min_distance]
    if not far_candidates:
        reachable.sort(key=lambda item: item[1], reverse=True)
        cutoff = max(1, int(len(reachable) * far_quantile))
        far_candidates = reachable[:cutoff]

    unvisited: list[tuple[int, float]] = []
    if visited is not None:
        unvisited = [item for item in far_candidates if item[0] not in visited]

    pool = unvisited if unvisited else far_candidates
    if not pool:
        return None

    rnd = rng if rng is not None else random
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
    path = compute_path(graph, start_node, target_node)
    if not path:
        return []
    return path[:max_steps]


__all__ = [
    "EXPLORATION_FAR_QUANTILE",
    "EXPLORATION_MIN_DISTANCE",
    "find_nearest_node",
    "compute_path",
    "compute_path_distance",
    "fill_path_distances",
    "nearest_refuge_path",
    "choose_refuge_with_exit",
    "pick_search_target",
    "random_walk",
    "pick_exploration_target",
    "plan_exploration_path",
]
