from __future__ import annotations

import logging
import math
from typing import Optional

from world.cache import WorldModel
from world.entities import AgentType, Position


logger = logging.getLogger(__name__)

DEFAULT_RADIUS: float = 30_000.0


def _euclidean_distance(pos_a: Position, pos_b: Position) -> float:
    return math.hypot(pos_a.x - pos_b.x, pos_a.y - pos_b.y)


class SocialFactorCache:
    # Прекэш позиций однотипных союзников через грид с шагом radius.
    # Стоимость: O(A) на построение, O(k) на запрос (k ≤ 9 соседних ячеек),
    # где A — количество однотипных агентов. В сумме по M задачам —
    # O(A + M·k) вместо O(M·A) у наивной реализации social_factor.

    __slots__ = ("_cells", "_radius", "_total", "_radius_sq")

    def __init__(
        self,
        world_model: WorldModel,
        agent_type: AgentType,
        current_agent_id: int,
        radius: float,
    ) -> None:
        self._cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._radius: float = radius
        self._radius_sq: float = radius * radius
        self._total: int = 0

        if radius <= 0:
            return

        for agent_id, agent in world_model.agents.items():
            if agent_id == current_agent_id or agent.type != agent_type:
                continue
            ax, ay = agent.position.x, agent.position.y
            cell = (int(ax // radius), int(ay // radius))
            self._cells.setdefault(cell, []).append((ax, ay))
            self._total += 1

    @property
    def total(self) -> int:
        return self._total

    def count_in_radius(self, x: int, y: int) -> int:
        if self._total == 0 or self._radius <= 0:
            return 0
        cx = int(x // self._radius)
        cy = int(y // self._radius)
        hit = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = self._cells.get((cx + dx, cy + dy))
                if not bucket:
                    continue
                for ax, ay in bucket:
                    if (ax - x) * (ax - x) + (ay - y) * (ay - y) < self._radius_sq:
                        hit += 1
        return hit

    def factor_for(self, target_position: Position) -> float:
        if self._total == 0:
            return 0.0
        count = self.count_in_radius(target_position.x, target_position.y)
        try:
            return float(count) / float(self._total)
        except ZeroDivisionError:
            return 0.0


def social_factor(
    world_model: WorldModel,
    target_position: Position,
    agent_type: AgentType,
    current_agent_id: int,
    radius: float = DEFAULT_RADIUS,
    cache: Optional[SocialFactorCache] = None,
) -> float:
    if radius <= 0:
        logger.warning(
            "Social factor: неположительный радиус [radius=%.2f] — возвращается 0.0",
            radius,
        )
        return 0.0

    if cache is not None:
        result = cache.factor_for(target_position)
        logger.debug(
            "Social factor (cached) [target_id=%s, f_social=%.4f, total_agents=%d, radius=%.0f]",
            target_position.entity_id,
            result,
            cache.total,
            radius,
        )
        return result

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


__all__ = ["DEFAULT_RADIUS", "social_factor", "SocialFactorCache"]
