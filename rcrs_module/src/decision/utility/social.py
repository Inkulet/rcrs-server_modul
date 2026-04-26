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
    """Я возвращаю N_i(r) — счётчик однотипных союзников в радиусе r от
    цели. Это ровно формула (15) из диплома:

        N_i(r) = Σ_{k∈A, k≠j} I(||pos_k − loc_i|| < r) · I(type_k = type_j)

    Возвращаю float только для совместимости с агрегатором (он умножает
    на w_n и складывает с другими f_*). Семантика — целочисленная: 0, 1, 2, …

    Быстрый путь: WorldModel держит spatial-grid союзников по типу,
    собранный на каждом такте. Запрос проходит за O(1) ожидаемое время.
    Если индекс ещё не построен (например, в юнит-тесте, где
    apply_perception не вызывался), WorldModel пересобирает его лениво.
    """
    if radius <= 0:
        logger.warning(
            "Social factor: неположительный радиус [radius=%.2f] — возвращается 0.0",
            radius,
        )
        return 0.0

    count = world_model.count_allies_in_radius(
        target_x=int(target_position.x),
        target_y=int(target_position.y),
        agent_type=agent_type,
        exclude_agent_id=current_agent_id,
        radius=radius,
    )

    logger.debug(
        "Social factor вычислен [target_id=%s, N_i=%d, radius=%.0f]",
        target_position.entity_id,
        count,
        radius,
    )
    return float(count)


__all__ = ["DEFAULT_RADIUS", "social_factor"]
