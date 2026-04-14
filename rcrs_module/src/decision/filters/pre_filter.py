from __future__ import annotations

import logging
from typing import Iterable, List

from decision.utility.distance import MAX_MAP_DISTANCE
from world.entities import AgentState, AgentType, EntityType, VisibleEntity


logger = logging.getLogger(__name__)

ESTIMATED_TRIP_TO_REFUGE: float = 20.0

UNREACHABLE_DISTANCE_THRESHOLD: float = MAX_MAP_DISTANCE * 0.9


class NeedRefugeException(RuntimeError):
    """пожарному нужно направиться в убежище."""


class PreFilterDispatcher:
    def __init__(self, work_rate: float = 1.0, average_speed: float = 70_000.0) -> None:
        if work_rate <= 0:
            raise ValueError("Я ожидаю положительный work_rate для расчета дедлайнов")
        if average_speed <= 0:
            raise ValueError("Я ожидаю положительный average_speed для конвертации дистанции")
        self.work_rate = work_rate
        self.average_speed = average_speed

    def _allowed_entity_types(self, agent_type: AgentType) -> frozenset[EntityType]:
        if agent_type == AgentType.FIRE_BRIGADE:
            return frozenset({EntityType.BUILDING})
        if agent_type == AgentType.AMBULANCE_TEAM:
            return frozenset({EntityType.CIVILIAN, EntityType.HUMAN})
        return frozenset({EntityType.BLOCKADE})

    def filter_tasks(self, agent_state: AgentState, tasks: Iterable[VisibleEntity]) -> List[VisibleEntity]:

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

            allowed_types = self._allowed_entity_types(agent_state.type)
            tasks_list = list(tasks)
            tasks = [e for e in tasks_list if e.type in allowed_types]
            logger.debug(
                "Я отфильтровал сущности по типам %s: было=%d, осталось=%d, agent_id=%s",
                ",".join(t.value for t in sorted(allowed_types, key=lambda t: t.value)),
                len(tasks_list),
                len(tasks),
                agent_state.id,
            )

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
        if entity.type == EntityType.CIVILIAN:
            hp = entity.raw_sensor_data.hp
            damage = entity.raw_sensor_data.damage
            buriedness = entity.raw_sensor_data.buriedness

            if hp is not None and hp == 0:
                logger.debug("Я исключаю погибшего гражданского: entity_id=%s", entity.id)
                return False

            if damage == 0 and buriedness == 0:
                logger.debug(
                    "Я исключаю здорового гражданского без завала: entity_id=%s",
                    entity.id,
                )
                return False

            if damage is None and buriedness is None:
                logger.debug(
                    "Я исключаю гражданского с неизвестным состоянием: entity_id=%s",
                    entity.id,
                )
                return False

        if entity.type == EntityType.HUMAN:
            hp = entity.raw_sensor_data.hp
            buriedness = entity.raw_sensor_data.buriedness

            if hp is not None and hp == 0:
                logger.debug("Я исключаю погибшего агента: entity_id=%s", entity.id)
                return False

            if buriedness is None or buriedness == 0:
                logger.debug(
                    "Я исключаю незаваленного агента: entity_id=%s, buriedness=%s",
                    entity.id,
                    buriedness,
                )
                return False

        if entity.type == EntityType.BUILDING:
            fieryness = entity.raw_sensor_data.fieryness
            if fieryness is not None and fieryness not in {1, 2, 3, 4, 5, 6}:
                logger.debug(
                    "Я исключаю здание: fieryness=%s не в {1..6}: entity_id=%s",
                    fieryness,
                    entity.id,
                )
                return False

        if entity.type == EntityType.BLOCKADE:
            repair_cost = entity.raw_sensor_data.repair_cost
            if repair_cost is None or repair_cost == 0:
                logger.debug(
                    "Я исключаю завал: repair_cost=%s: entity_id=%s",
                    repair_cost, entity.id,
                )
                return False

        if entity.computed_metrics.path_distance >= UNREACHABLE_DISTANCE_THRESHOLD:
            logger.info(
                "Я исключаю недостижимую цель entity_id=%s: path_distance=%.0f ≥ %.0f (нет прохода)",
                entity.id,
                entity.computed_metrics.path_distance,
                UNREACHABLE_DISTANCE_THRESHOLD,
            )
            return False

        if entity.raw_sensor_data.buriedness is not None:
            if self.work_rate <= 0:
                raise ValueError("Я получил неположительную скорость работ")

            try:
                t_travel = entity.computed_metrics.path_distance / self.average_speed
            except ZeroDivisionError:
                t_travel = 0.0

            time_to_action = t_travel + (
                entity.raw_sensor_data.buriedness / self.work_rate
            )
            if entity.computed_metrics.estimated_death_time < (
                time_to_action + ESTIMATED_TRIP_TO_REFUGE
            ):
                logger.debug(
                    "Я исключаю обреченного (не довезет живым): entity_id=%s, "
                    "estimated_death=%d, time_to_action=%.1f, trip_to_refuge=%.1f",
                    entity.id,
                    entity.computed_metrics.estimated_death_time,
                    time_to_action,
                    ESTIMATED_TRIP_TO_REFUGE,
                )
                return False

        return True


__all__ = ["NeedRefugeException", "PreFilterDispatcher"]
