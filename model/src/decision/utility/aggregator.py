from __future__ import annotations

"""В этом модуле я объединяю факторы срочности, расстояния, трудоемкости и социального влияния."""

import logging
from typing import Optional

from world.cache import WorldModel
from world.entities import AgentState, Position, VisibleEntity

from decision.utility.distance import MAX_MAP_DISTANCE, distance_factor
from decision.utility.effort import compute_effort
from decision.utility.social import DEFAULT_RADIUS, social_factor
from decision.utility.urgency import compute_urgency


logger = logging.getLogger(__name__)


class UtilityAggregator:
    """В этом классе я реализую аддитивную модель полезности с весами факторов."""

    def __init__(self, w_c: float, w_d: float, w_e: float, w_n: float) -> None:
        """Здесь я фиксирую веса факторов, чтобы настраивать вклад каждого из них."""

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
        min_distance_to_targets: Optional[float] = None,
        social_radius: float = DEFAULT_RADIUS,
        max_map_distance: float = MAX_MAP_DISTANCE,
    ) -> float:
        """Здесь я объединяю факторы в итоговую полезность U_ij по аддитивной формуле."""

        try:
            f_urgency = compute_urgency(
                agent_state,
                entity=entity,
                t_travel=t_travel,
                t_work=t_work,
                min_distance_to_targets=min_distance_to_targets,
            )
            f_effort = compute_effort(agent_state, entity=entity)
            f_dist = distance_factor(
                world_model.road_graph,
                agent_state.position,
                target_position,
                max_map_distance=max_map_distance,
            )
            f_social = social_factor(
                world_model,
                target_position,
                agent_state.type,
                radius=social_radius,
            )

            utility = (
                self.w_c * f_urgency
                - self.w_d * f_dist
                + self.w_e * f_effort
                + self.w_n * f_social
            )
            return utility
        except ZeroDivisionError:
            logger.warning("Я поймал деление на ноль при вычислении полезности")
            return 0.0


__all__ = ["UtilityAggregator"]
