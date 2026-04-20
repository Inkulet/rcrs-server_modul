from __future__ import annotations

import logging
import math

from world.cache import WorldModel
from world.entities import AgentType, Position


logger = logging.getLogger(__name__)

DEFAULT_RADIUS: float = 30_000.0


def _euclidean_distance(pos_a: Position, pos_b: Position) -> float:
    return math.hypot(pos_a.x - pos_b.x, pos_a.y - pos_b.y)


def social_factor(
    world_model: WorldModel,
    target_position: Position,
    agent_type: AgentType,
    current_agent_id: int,
    radius: float = DEFAULT_RADIUS,
) -> float:
    if radius <= 0:
        logger.warning(
            "Social factor: неположительный радиус [radius=%.2f] — возвращается 0.0",
            radius,
        )
        return 0.0

    same_type_agents = [
        agent
        for agent_id, agent in world_model.agents.items()
        if agent_id != current_agent_id and agent.type == agent_type
    ]

    total = len(same_type_agents)
    if total == 0:
        logger.debug(
            "Social factor: однотипные агенты не найдены [target_id=%s] — возвращается 0.0",
            target_position.entity_id,
        )
        return 0.0

    count = sum(
        1
        for agent in same_type_agents
        if _euclidean_distance(agent.position, target_position) < radius
    )

    try:
        result = float(count) / float(total)
    except ZeroDivisionError:
        logger.warning(
            "ZeroDivisionError в social_factor [target_id=%s] — возвращается 0.0",
            target_position.entity_id,
        )
        return 0.0

    logger.debug(
        "Social factor вычислен [target_id=%s, f_social=%.4f, agents_in_radius=%d/%d, radius=%.0f]",
        target_position.entity_id,
        result,
        count,
        total,
        radius,
    )
    return result


__all__ = ["DEFAULT_RADIUS", "social_factor"]
