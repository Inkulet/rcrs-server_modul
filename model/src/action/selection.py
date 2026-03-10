from __future__ import annotations

"""В этом модуле я реализую выбор целевой задачи с гистерезисом."""

import logging
from typing import Dict, Optional


logger = logging.getLogger(__name__)


class TargetSelector:
    """В этом классе я реализую механизм гистерезиса для выбора цели."""

    def __init__(self, c_switch: float = 0.1) -> None:
        """Здесь я задаю порог переключения C_switch для устойчивого выбора."""

        self.c_switch = c_switch

    def select_best_target(
        self,
        current_target_id: Optional[int],
        utilities_dict: Dict[int, float],
    ) -> Optional[int]:
        """Здесь я выбираю цель по максимуму полезности с учетом гистерезиса."""

        if not utilities_dict:
            logger.info("Я не получил доступных задач и возвращаю режим ожидания")
            return None

        best_target_id = max(utilities_dict, key=utilities_dict.get)
        best_utility = utilities_dict[best_target_id]

        if current_target_id is None or current_target_id not in utilities_dict:
            logger.info("Я выбираю новую цель без гистерезиса: target_id=%s", best_target_id)
            return best_target_id

        current_utility = utilities_dict[current_target_id]
        if best_target_id != current_target_id and best_utility > current_utility + self.c_switch:
            logger.info(
                "Я переключаюсь на новую цель по гистерезису: %s -> %s",
                current_target_id,
                best_target_id,
            )
            return best_target_id

        logger.info(
            "Я сохраняю текущую цель из-за гистерезиса: target_id=%s",
            current_target_id,
        )
        return current_target_id


__all__ = ["TargetSelector"]
