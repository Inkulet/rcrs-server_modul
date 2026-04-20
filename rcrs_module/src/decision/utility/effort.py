from __future__ import annotations


import logging
from typing import Optional

from world.entities import AgentState, AgentType, EntityType, VisibleEntity

from decision.utility._utils import _clamp_to_unit


logger = logging.getLogger(__name__)

MAX_BURIEDNESS: float = 100.0
MAX_TOTAL_AREA: float = 100000.0
MAX_REPAIR_COST: float = 100000.0


def effort_for_ambulance(entity: VisibleEntity, max_buriedness: float = MAX_BURIEDNESS) -> float:

    try:
        buriedness = entity.raw_sensor_data.buriedness
        if buriedness is None:
            logger.debug("Фактор effort (ambulance) пропущен: отсутствует buriedness [entity_id=%s]", entity.id)
            return 0.0
        return _clamp_to_unit(buriedness / max_buriedness)
    except ZeroDivisionError:
        logger.warning("ZeroDivisionError в effort (ambulance) [entity_id=%s] — возвращается 0.0", entity.id)
        return 0.0


def effort_for_fire(entity: VisibleEntity, max_total_area: float = MAX_TOTAL_AREA) -> float:

    try:
        ground_area = entity.raw_sensor_data.ground_area
        floors = entity.raw_sensor_data.floors
        if ground_area is None or floors is None:
            logger.debug("Фактор effort (fire) пропущен: отсутствует ground_area/floors [entity_id=%s]", entity.id)
            return 0.0
        total_area = ground_area * floors
        return _clamp_to_unit(total_area / max_total_area)
    except ZeroDivisionError:
        logger.warning("ZeroDivisionError в effort (fire) [entity_id=%s] — возвращается 0.0", entity.id)
        return 0.0


def effort_for_police(entity: VisibleEntity, max_repair_cost: float = MAX_REPAIR_COST) -> float:

    try:
        repair_cost = entity.raw_sensor_data.repair_cost
        if repair_cost is None:
            logger.debug("Фактор effort (police) пропущен: отсутствует repair_cost [entity_id=%s]", entity.id)
            return 0.0
        return _clamp_to_unit(repair_cost / max_repair_cost)
    except ZeroDivisionError:
        logger.warning("ZeroDivisionError в effort (police) [entity_id=%s] — возвращается 0.0", entity.id)
        return 0.0


def compute_effort(
    agent_state: AgentState,
    entity: Optional[VisibleEntity] = None,
    max_buriedness: float = MAX_BURIEDNESS,
    max_total_area: float = MAX_TOTAL_AREA,
    max_repair_cost: float = MAX_REPAIR_COST,
) -> float:

    try:
        if entity is None:
            logger.warning("Сущность для расчёта effort не передана [agent_id=%s]", agent_state.id)
            return 0.0

        if agent_state.type == AgentType.AMBULANCE_TEAM:
            return effort_for_ambulance(entity, max_buriedness)

        if agent_state.type == AgentType.FIRE_BRIGADE:
            if entity.type in (EntityType.CIVILIAN, EntityType.HUMAN):
                return effort_for_ambulance(entity, max_buriedness)
            return effort_for_fire(entity, max_total_area)

        if agent_state.type == AgentType.POLICE_FORCE:
            return effort_for_police(entity, max_repair_cost)

        return 0.0
    except ZeroDivisionError:
        logger.warning("ZeroDivisionError в compute_effort [agent_id=%s] — возвращается 0.0", agent_state.id)
        return 0.0


__all__ = [
    "MAX_BURIEDNESS",
    "MAX_TOTAL_AREA",
    "MAX_REPAIR_COST",
    "effort_for_ambulance",
    "effort_for_fire",
    "effort_for_police",
    "compute_effort",
]
