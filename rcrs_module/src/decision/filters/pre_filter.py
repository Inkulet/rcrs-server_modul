from __future__ import annotations

import logging
from typing import Iterable, List, Optional, TYPE_CHECKING

from decision.utility.distance import MAX_MAP_DISTANCE
from world.entities import AgentState, AgentType, EntityType, VisibleEntity

if TYPE_CHECKING:
    from world.cache import WorldModel


logger = logging.getLogger(__name__)

ESTIMATED_TRIP_TO_REFUGE: float = 10.0

UNREACHABLE_DISTANCE_THRESHOLD: float = MAX_MAP_DISTANCE * 0.9


class NeedRefugeException(RuntimeError):
    """Сигнал о необходимости направить пожарного в убежище."""


class PreFilterDispatcher:
    def __init__(self, work_rate: float = 1.0, average_speed: float = 70_000.0) -> None:
        if work_rate <= 0:
            raise ValueError("PreFilterDispatcher: требуется положительный work_rate для расчёта дедлайнов")
        if average_speed <= 0:
            raise ValueError("PreFilterDispatcher: требуется положительный average_speed для конвертации дистанции в время")
        self.work_rate = work_rate
        self.average_speed = average_speed

    def _allowed_entity_types(self, agent_type: AgentType) -> frozenset[EntityType]:
        if agent_type == AgentType.FIRE_BRIGADE:
            return frozenset({EntityType.BUILDING, EntityType.CIVILIAN, EntityType.HUMAN})
        if agent_type == AgentType.AMBULANCE_TEAM:
            return frozenset({EntityType.CIVILIAN, EntityType.HUMAN})
        return frozenset({EntityType.BLOCKADE})

    def filter_tasks(
        self,
        agent_state: AgentState,
        tasks: Iterable[VisibleEntity],
        world_model: Optional["WorldModel"] = None,
    ) -> List[VisibleEntity]:

        try:
            if agent_state.resources.is_transporting:
                logger.debug(
                    "Pre-filter: агент транспортирует гражданского, фильтрация пропущена [agent_id=%s]",
                    agent_state.id,
                )
                return []

            allowed_types = self._allowed_entity_types(agent_state.type)
            tasks_list = list(tasks)
            tasks = [e for e in tasks_list if e.type in allowed_types]

            if agent_state.type == AgentType.FIRE_BRIGADE and agent_state.resources.water_quantity == 0:
                tasks = [e for e in tasks if e.type != EntityType.BUILDING]
                if not tasks:
                    logger.debug(
                        "Pre-filter: FIRE_BRIGADE без воды и без задач на спасение — требуется убежище [agent_id=%s]",
                        agent_state.id,
                    )
                    raise NeedRefugeException("FIRE_BRIGADE: отсутствует вода и нет задач на спасение — требуется возврат в убежище")
            logger.debug(
                "Pre-filter: отсев по типам завершён [agent_id=%s, allowed_types=%s, before=%d, after=%d]",
                agent_state.id,
                ",".join(t.value for t in sorted(allowed_types, key=lambda t: t.value)),
                len(tasks_list),
                len(tasks),
            )

            has_live_fire_brigades = self._has_live_fire_brigades(world_model)
            refuge_set = self._refuge_set(world_model)

            filtered: List[VisibleEntity] = []
            for entity in tasks:
                try:
                    if self._is_relevant(
                        entity,
                        agent_type=agent_state.type,
                        has_live_fire_brigades=has_live_fire_brigades,
                        refuge_set=refuge_set,
                    ):
                        filtered.append(entity)
                except ValueError as exc:
                    logger.error(
                        "Pre-filter: некорректные данные сущности [entity_id=%s]: %s",
                        entity.id,
                        exc,
                    )

            if not filtered and tasks:
                logger.info(
                    "Pre-filter: все кандидаты отсеяны [agent_id=%s, candidates_in=%d, sample=%s]",
                    agent_state.id,
                    len(tasks),
                    ", ".join(
                        f"{e.id}(f={e.raw_sensor_data.fieryness},hp={e.raw_sensor_data.hp},"
                        f"d={e.raw_sensor_data.damage},b={e.raw_sensor_data.buriedness},"
                        f"pd={e.computed_metrics.path_distance:.0f})"
                        for e in list(tasks)[:5]
                    ),
                )

            return filtered
        except ValueError as exc:
            logger.error("Pre-filter: некорректные входные данные — %s", exc)
            return []

    @staticmethod
    def _has_live_fire_brigades(world_model: Optional["WorldModel"]) -> bool:
        if world_model is None:
            return True
        for ally in world_model.agents.values():
            if ally.type != AgentType.FIRE_BRIGADE:
                continue
            if ally.hp is None or ally.hp > 0:
                return True
        return False

    @staticmethod
    def _refuge_set(world_model: Optional["WorldModel"]) -> frozenset[int]:
        if world_model is None:
            return frozenset()
        return frozenset(world_model.refuge_ids)

    def _is_relevant(
        self,
        entity: VisibleEntity,
        agent_type: AgentType,
        has_live_fire_brigades: bool,
        refuge_set: frozenset[int],
    ) -> bool:
        hp = entity.raw_sensor_data.hp
        damage = entity.raw_sensor_data.damage
        buriedness = entity.raw_sensor_data.buriedness
        pos_edge = entity.raw_sensor_data.position_on_edge

        if entity.type == EntityType.CIVILIAN:
            if hp is not None and hp == 0:
                logger.debug("Pre-filter: гражданский погиб (hp=0), исключён [entity_id=%s]", entity.id)
                return False

            if damage == 0 and buriedness == 0:
                logger.debug(
                    "Pre-filter: гражданский здоров и не завален, исключён [entity_id=%s]",
                    entity.id,
                )
                return False

            if pos_edge is not None and pos_edge in refuge_set:
                logger.debug(
                    "Pre-filter: гражданский уже в убежище, исключён [entity_id=%s, pos_edge=%s]",
                    entity.id, pos_edge,
                )
                return False

            if agent_type == AgentType.FIRE_BRIGADE:
                if damage is None or damage <= 0:
                    return False
                if buriedness is None or buriedness <= 0:
                    return False

        if entity.type == EntityType.HUMAN:
            if hp is not None and hp == 0:
                logger.debug("Pre-filter: агент-спасатель погиб (hp=0), исключён [entity_id=%s]", entity.id)
                return False

            if buriedness is None or buriedness == 0:
                logger.debug(
                    "Pre-filter: агент-спасатель не завален, исключён [entity_id=%s, buriedness=%s]",
                    entity.id,
                    buriedness,
                )
                return False

            if pos_edge is not None and pos_edge in refuge_set:
                return False

            if agent_type == AgentType.FIRE_BRIGADE:
                if damage is None or damage <= 0:
                    return False

        if entity.type == EntityType.BUILDING:
            fieryness = entity.raw_sensor_data.fieryness
            if fieryness is not None and fieryness not in {1, 2, 3}:
                logger.debug(
                    "Pre-filter: здание не горит, исключено [entity_id=%s, fieryness=%s]",
                    entity.id, fieryness,
                )
                return False

        if entity.type == EntityType.BLOCKADE:
            repair_cost = entity.raw_sensor_data.repair_cost
            if repair_cost == 0:
                logger.debug(
                    "Pre-filter: завал уже расчищен (repair_cost=0), исключён [entity_id=%s]",
                    entity.id,
                )
                return False

        if entity.computed_metrics.path_distance >= UNREACHABLE_DISTANCE_THRESHOLD:
            logger.info(
                "Pre-filter: цель недостижима по графу, исключена [entity_id=%s, path_distance=%.0f, threshold=%.0f]",
                entity.id,
                entity.computed_metrics.path_distance,
                UNREACHABLE_DISTANCE_THRESHOLD,
            )
            return False

        return True


__all__ = ["NeedRefugeException", "PreFilterDispatcher"]
