from __future__ import annotations

import logging

from config import STUCK_BLACKLIST_TICKS, STUCK_TICKS, UNREACHABLE_BLACKLIST_TICKS


logger = logging.getLogger(__name__)


class TargetSelector:
    def __init__(self, c_switch: float = 0.1) -> None:
        self.c_switch = c_switch

        self._blacklisted_until: dict[int, int] = {}

        # Я отслеживаю застревание не по «последний узел == текущий»,
        # а через множество уже посещённых узлов для текущей цели.
        # Это ловит осцилляции (агент прыгает A↔B между двумя узлами
        # вокруг завала) — узел формально «меняется», но реального
        # прогресса нет, потому что мы уже были в обоих.
        self._stuck_target: int | None = None
        self._seen_nodes_for_target: set[int] = set()
        self._stuck_counter: int = 0

    def select_best_target(
        self,
        current_target_id: int | None,
        utilities_dict: dict[int, float],
        current_tick: int = 0,
    ) -> int | None:
        self._expire_blacklist(current_tick)

        available = {
            tid: u for tid, u in utilities_dict.items()
            if not self._is_blacklisted(tid, current_tick)
        }

        if not available:
            logger.info("Я не получил доступных задач (все %d в ЧС или список пуст)", len(utilities_dict))
            return None

        best_target_id = max(available, key=lambda k: available[k])
        best_utility = available[best_target_id]

        if current_target_id is None or current_target_id not in available:
            logger.info("Я выбираю новую цель без гистерезиса: target_id=%s", best_target_id)
            return best_target_id

        current_utility = available[current_target_id]
        if best_target_id != current_target_id and best_utility > current_utility + self.c_switch:
            logger.info(
                "Я переключаюсь на новую цель по гистерезису: %s → %s",
                current_target_id, best_target_id,
            )
            return best_target_id

        logger.info(
            "Я сохраняю текущую цель из-за гистерезиса: target_id=%s",
            current_target_id,
        )
        return current_target_id

    def blacklist(self, target_id: int, current_tick: int, duration: int) -> None:
        until = current_tick + duration
        self._blacklisted_until[target_id] = until
        logger.info(
            "Я занёс target_id=%d в чёрный список до такта %d (на %d тактов)",
            target_id, until, duration,
        )

    def blacklist_unreachable(self, target_id: int, current_tick: int) -> None:
        self.blacklist(target_id, current_tick, UNREACHABLE_BLACKLIST_TICKS)

    def blacklist_stuck(self, target_id: int, current_tick: int) -> None:
        self.blacklist(target_id, current_tick, STUCK_BLACKLIST_TICKS)

    def _is_blacklisted(self, target_id: int, current_tick: int) -> bool:
        until = self._blacklisted_until.get(target_id)
        return until is not None and until > current_tick

    def _expire_blacklist(self, current_tick: int) -> None:
        expired = [tid for tid, until in self._blacklisted_until.items() if until <= current_tick]
        for tid in expired:
            del self._blacklisted_until[tid]
            logger.info("Я снял блокировку с target_id=%d на такте %d", tid, current_tick)

    def report_progress(
        self,
        current_target_id: int | None,
        current_node: int,
        current_tick: int,
    ) -> bool:
        if current_target_id != self._stuck_target:
            self._stuck_target = current_target_id
            self._seen_nodes_for_target = {current_node}
            self._stuck_counter = 0
            return False

        if current_target_id is None:
            return False

        # Я считаю прогрессом только посещение НОВОГО узла: осцилляция
        # между двумя соседними узлами (типичный случай на узких улицах
        # Kobe) больше не сбрасывает счётчик в 0.
        if current_node not in self._seen_nodes_for_target:
            self._seen_nodes_for_target.add(current_node)
            self._stuck_counter = 0
            return False

        self._stuck_counter += 1
        if self._stuck_counter >= STUCK_TICKS:
            logger.warning(
                "Я обнаружил застой у target_id=%d: %d тактов без нового узла "
                "(последний=%d, посещено=%d) — блокирую цель",
                current_target_id, self._stuck_counter, current_node,
                len(self._seen_nodes_for_target),
            )
            self.blacklist_stuck(current_target_id, current_tick)
            self._stuck_target = None
            self._seen_nodes_for_target = set()
            self._stuck_counter = 0
            return True

        return False

    def reset_stuck(self) -> None:
        self._stuck_target = None
        self._seen_nodes_for_target = set()
        self._stuck_counter = 0


__all__ = ["TargetSelector"]
