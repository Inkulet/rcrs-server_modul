from __future__ import annotations

"""В этом модуле я вычисляю геометрический фактор расстояния для функции полезности."""

import logging
from typing import Optional

import networkx as nx

from world.entities import Position


logger = logging.getLogger(__name__)

# Я задаю максимальную дистанцию по дорожному графу в мм.
# Значение совпадает с MAX_MAP_DISTANCE в action/navigation.py:
# типичная карта RCRS — несколько км, 100 000 км — безопасный потолок нормировки.
MAX_MAP_DISTANCE: float = 100_000_000.0


def _safe_shortest_path_length(
    graph: nx.Graph,
    source_id: int,
    target_id: int,
) -> Optional[float]:
    """Здесь я безопасно считаю длину кратчайшего пути с учетом весов ребер."""

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
    """Здесь я вычисляю f_dist как отношение длины пути к максимальной дистанции карты."""

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
    """Здесь я нормирую уже вычисленную дистанцию в фактор f_dist ∈ [0, 1].

    Я использую эту функцию в aggregator.py вместо повторного запуска Dijkstra:
    fill_path_distances() уже заполнил entity.computed_metrics.path_distance
    перед вызовом агрегатора, поэтому повторный запрос к графу избыточен (O(2·M·logN)).
    """
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
