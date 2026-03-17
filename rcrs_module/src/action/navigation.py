from __future__ import annotations

"""В этом модуле я реализую навигацию агента по дорожному графу (UC-6).

Я вычисляю кратчайшие пути алгоритмом A* через NetworkX и обновляю
поле path_distance у всех видимых сущностей до расчёта полезности.
Это позволяет f_dist опираться на реальное расстояние по дорогам,
а не на эвклидово приближение.
"""

import logging
from typing import List, Optional

import networkx as nx

from world.entities import MapNode, VisibleEntity


logger = logging.getLogger(__name__)

# Я задаю предельное расстояние для случаев, когда путь не найден.
# Значение должно быть больше любого реального расстояния на карте RCRS.
MAX_MAP_DISTANCE: float = 100_000_000.0


def compute_path(
    graph: nx.Graph,
    from_id: int,
    to_id: int,
) -> List[int]:
    """Здесь я вычисляю кратчайший путь в графе между двумя узлами.

    Я использую встроенный алгоритм A* NetworkX с весом «weight» на рёбрах.
    Если путь не найден (узел отсутствует или граф несвязный), возвращаю
    пустой список — вызывающий код должен трактовать это как «не достижимо».
    """
    if from_id == to_id:
        return [from_id]

    try:
        path: List[int] = nx.astar_path(graph, from_id, to_id, weight="weight")
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
    """Здесь я вычисляю дистанцию кратчайшего пути между двумя узлами.

    Я использую Dijkstra из NetworkX — он работает на C-бэкенде и
    обеспечивает O(M log N) сложность вместо полного A* с эвристикой.
    При недостижимости возвращаю MAX_MAP_DISTANCE как безопасный дефолт:
    это гарантирует, что f_dist = 1.0 и агент не выберет такую цель.
    """
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
    entities: List[VisibleEntity],
) -> List[VisibleEntity]:
    """Здесь я обновляю path_distance у каждой сущности перед расчётом полезности.

    Я запускаю ОДИН вызов Dijkstra из позиции агента, получая словарь
    {node_id: distance} для ВСЕХ узлов графа за O(N log N + E).
    Затем я обновляю path_distance у каждой сущности за O(1) по словарю —
    итого O(N log N + E + M) вместо O(M·N log N) при N вызовах Dijkstra.

    Для сущностей вне графа (завалы, гражданские) я использую position_on_edge
    (ID дороги/здания, где находится сущность) как навигационный узел.
    Возвращаю новый список: исходные объекты неизменны (Pydantic immutable).
    """
    # Я получаю все дистанции из позиции агента одним вызовом Dijkstra.
    try:
        dist_map: dict[int, float] = nx.single_source_dijkstra_path_length(
            graph, agent_node_id, weight="weight"
        )
    except nx.NodeNotFound:
        logger.warning(
            "Я не нашёл позицию агента %d в графе, все дистанции = MAX", agent_node_id
        )
        dist_map = {}

    updated: List[VisibleEntity] = []
    for entity in entities:
        # Я определяю навигационный узел: для сущностей не в графе
        # (гражданские, завалы) использую position_on_edge — дорогу/здание, где они находятся.
        nav_id = entity.id
        if nav_id not in dist_map:
            pos = entity.raw_sensor_data.position_on_edge
            if pos is not None and pos in dist_map:
                nav_id = pos

        distance = dist_map.get(nav_id, MAX_MAP_DISTANCE)
        updated_metrics = entity.computed_metrics.model_copy(
            update={"path_distance": distance}
        )
        updated_entity = entity.model_copy(
            update={"computed_metrics": updated_metrics}
        )
        updated.append(updated_entity)
        logger.debug(
            "Я обновил path_distance для entity_id=%d (nav_id=%d): %.1f мм",
            entity.id, nav_id, distance,
        )
    return updated


def nearest_refuge_path(
    graph: nx.Graph,
    from_id: int,
    refuge_ids: List[int],
) -> List[int]:
    """Здесь я нахожу ближайшее убежище и возвращаю путь к нему.

    Я вычисляю путь для каждого убежища ровно один раз и определяю длину
    суммированием весов рёбер — без повторного запуска Dijkstra для победителя.
    Это вызывается только при NeedRefugeException — не в основном цикле.
    Если убежищ нет или ни одно не достижимо, возвращаю пустой список.
    """
    if not refuge_ids:
        logger.warning("Я не знаю ни одного убежища — невозможно построить маршрут")
        return []

    best_path: List[int] = []
    best_distance: float = MAX_MAP_DISTANCE

    for refuge_id in refuge_ids:
        path = compute_path(graph, from_id, refuge_id)
        if not path:
            continue
        # Я вычисляю длину найденного пути суммой весов рёбер — избегаю повторного Dijkstra.
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
    "MAX_MAP_DISTANCE",
    "compute_path",
    "compute_path_distance",
    "fill_path_distances",
    "nearest_refuge_path",
]
