from __future__ import annotations

"""В этом модуле я реализую предварительную фильтрацию задач (UC-2)."""

import logging
from typing import Iterable, List

from world.entities import AgentState, AgentType, EntityType, VisibleEntity


logger = logging.getLogger(__name__)


class NeedRefugeException(RuntimeError):
    """В этом исключении я сигнализирую, что пожарному нужно направиться в убежище."""


class PreFilterDispatcher:
    """В этом классе я реализую предварительную фильтрацию задач до расчетов полезности."""

    def __init__(self, work_rate: float = 1.0) -> None:
        """Здесь я задаю скорость работ для оценки временных ограничений."""

        if work_rate <= 0:
            raise ValueError("Я ожидаю положительный work_rate для расчета дедлайнов")
        self.work_rate = work_rate

    def filter_tasks(self, agent_state: AgentState, tasks: Iterable[VisibleEntity]) -> List[VisibleEntity]:
        """В этом методе я применяю правила UC-2 и возвращаю релевантные задачи."""

        try:
            if agent_state.resources.is_transporting:
                logger.debug(
                    "Я пропускаю фильтрацию, потому что агент занят: agent_id=%s",
                    agent_state.id,
                )
                return []

            if agent_state.type == AgentType.FIRE_BRIGADE and agent_state.resources.water_quantity == 0:
                logger.debug(
                    "Я направляю пожарного в убежище из-за отсутствия воды: agent_id=%s",
                    agent_state.id,
                )
                raise NeedRefugeException("Я обнаружил, что у пожарного нет воды")

            filtered: List[VisibleEntity] = []
            for entity in tasks:
                try:
                    if self._is_relevant(entity):
                        filtered.append(entity)
                except ValueError as exc:
                    logger.error(
                        "Я получил некорректные данные для entity_id=%s: %s",
                        entity.id,
                        exc,
                    )
            return filtered
        except ValueError as exc:
            logger.error("Я получил некорректные входные данные фильтра: %s", exc)
            return []

    def _is_relevant(self, entity: VisibleEntity) -> bool:
        """Здесь я применяю правила отсева для одной сущности."""

        if entity.type == EntityType.CIVILIAN:
            hp = entity.raw_sensor_data.hp
            damage = entity.raw_sensor_data.damage
            buriedness = entity.raw_sensor_data.buriedness

            if hp == 0:
                logger.debug("Я исключаю погибшего гражданского: entity_id=%s", entity.id)
                return False

            if damage == 0 and buriedness == 0:
                logger.debug(
                    "Я исключаю здорового гражданского без завала: entity_id=%s",
                    entity.id,
                )
                return False

        if entity.type == EntityType.BUILDING:
            fieryness = entity.raw_sensor_data.fieryness
            if fieryness in {4, 5, 6, 7, 8}:
                logger.debug(
                    "Я исключаю сгоревшее здание по fieryness=%s: entity_id=%s",
                    fieryness,
                    entity.id,
                )
                return False

        if entity.raw_sensor_data.buriedness is not None:
            if self.work_rate <= 0:
                raise ValueError("Я получил неположительную скорость работ")

            time_to_action = entity.computed_metrics.path_distance + (
                entity.raw_sensor_data.buriedness / self.work_rate
            )
            if entity.computed_metrics.estimated_death_time <= time_to_action:
                logger.debug(
                    "Я исключаю задачу из-за дедлайна: entity_id=%s",
                    entity.id,
                )
                return False

        return True


__all__ = ["NeedRefugeException", "PreFilterDispatcher"]
