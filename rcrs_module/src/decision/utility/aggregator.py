from __future__ import annotations

"""В этом модуле я объединяю факторы срочности, расстояния, трудоемкости и социального влияния."""

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
    """В этом классе я реализую аддитивную модель полезности с весами факторов."""

    def __init__(self, w_c: float, w_d: float, w_e: float, w_n: float) -> None:
        """Здесь я фиксирую веса факторов, чтобы настраивать вклад каждого из них.

        Я валидирую веса: все должны быть неотрицательными, а сумма — равна 1.0.
        Отрицательный w_d перевернул бы знак дистанции и сделал бы f_dist «бонусом».
        """
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
        """Здесь я объединяю факторы в итоговую полезность U_ij по аддитивной формуле.

        f_dist вычисляется из entity.computed_metrics.path_distance (уже заполнено
        вызовом fill_path_distances() до вызова агрегатора) — без повторного Dijkstra.
        target_position используется только для f_social (евклидово расстояние).
        """

        try:
            f_urgency = compute_urgency(
                agent_state,
                entity=entity,
                t_travel=t_travel,
                t_work=t_work,
                task_distance=task_distance,
            )
            f_effort = compute_effort(agent_state, entity=entity)
            # Я использую уже вычисленную дистанцию из fill_path_distances() (UC-6),
            # избегая повторного запуска Dijkstra — это было бы O(2·M·logN).
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

            # Я вычисляю итоговую полезность по формуле диплома:
            # U_ij = w_c·f_urgency − w_d·f_dist + w_e·f_effort − w_n·f_social
            # f_social вычитается: чем больше союзников рядом с целью, тем ниже
            # полезность — это реализует механизм антироения (Критерий 4 диплома).
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
