from __future__ import annotations


import logging
from typing import Optional

from world.cache import WorldModel
from world.entities import AgentState, Position, VisibleEntity

from decision.utility.distance import MAX_MAP_DISTANCE, distance_factor_precomputed
from decision.utility.effort import compute_effort
from decision.utility.social import DEFAULT_RADIUS, social_factor
from decision.utility.urgency import compute_urgency


logger = logging.getLogger(__name__)


class UtilityAggregator:
    def __init__(self, w_c: float, w_d: float, w_e: float, w_n: float) -> None:
        if any(w < 0 for w in (w_c, w_d, w_e, w_n)):
            raise ValueError(
                f"Я ожидаю неотрицательные веса: w_c={w_c}, w_d={w_d}, w_e={w_e}, w_n={w_n}"
            )
        weight_sum = w_c + w_d + w_e + w_n

        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Я ожидаю веса с суммой 1.0, получено {weight_sum:.6f}"
            )

        self.w_c = w_c
        self.w_d = w_d
        self.w_e = w_e
        self.w_n = w_n

    def calculate_utility(
        self,
        agent_state: AgentState,
        entity: VisibleEntity,
        world_model: WorldModel,
        target_position: Position,
        t_travel: Optional[float] = None,
        t_work: Optional[float] = None,
        task_distance: Optional[float] = None,
        social_radius: float = DEFAULT_RADIUS,
        max_map_distance: float = MAX_MAP_DISTANCE,
    ) -> float:

        try:
            f_urgency = compute_urgency(
                agent_state,
                entity=entity,
                t_travel=t_travel,
                t_work=t_work,
                task_distance=task_distance,
            )

            f_effort = compute_effort(agent_state, entity=entity)
            f_dist = distance_factor_precomputed(
                entity.computed_metrics.path_distance,
                max_map_distance=max_map_distance,
            )

            f_social = social_factor(
                world_model,
                target_position,
                agent_state.type,
                current_agent_id=agent_state.id,
                radius=social_radius,
            )

            utility = (
                self.w_c * f_urgency
                - self.w_d * f_dist
                + self.w_e * f_effort
                - self.w_n * f_social
            )
            return utility
        except ZeroDivisionError:
            logger.warning("Я поймал деление на ноль при вычислении полезности")
            return 0.0


__all__ = ["UtilityAggregator"]
