from __future__ import annotations

"""В этом модуле я вычисляю социальный фактор для предотвращения интерференции агентов."""

import logging
from typing import Optional

import networkx as nx

from world.cache import WorldModel
from world.entities import AgentType, Position


logger = logging.getLogger(__name__)

DEFAULT_RADIUS: float = 2000.0


def _safe_shortest_path_length(
    graph: nx.Graph,
    source_id: int,
    target_id: int,
) -> Optional[float]:
    """Здесь я безопасно считаю длину пути для оценки близости агентов."""

    try:
        return float(nx.shortest_path_length(graph, source_id, target_id, weight="weight"))
    except nx.NetworkXNoPath:
        logger.debug(
            "Я не нашел путь в графе между %s и %s при расчете социального фактора",
            source_id,
            target_id,
        )
        return None
    except nx.NodeNotFound as exc:
        logger.debug("Я получил неизвестный узел при расчете социального фактора: %s", exc)
        return None


def social_factor(
    world_model: WorldModel,
    target_position: Position,
    agent_type: AgentType,
    current_agent_id: int,
    radius: float = DEFAULT_RADIUS,
) -> float:
    """Здесь я считаю количество однотипных агентов в радиусе r вокруг цели."""

    if radius <= 0:
        logger.warning("Я получил неположительный радиус для социального фактора")
        return 0.0

    count = 0
    for agent in world_model.agents.values():
        if agent.id == current_agent_id:
            continue
        if agent.type != agent_type:
            continue

        distance = _safe_shortest_path_length(
            world_model.road_graph,
            agent.position.entity_id,
            target_position.entity_id,
        )
        if distance is None:
            continue

        if distance < radius:
            count += 1

    return float(count)


__all__ = ["DEFAULT_RADIUS", "social_factor"]
