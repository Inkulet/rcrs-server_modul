from __future__ import annotations

"""В этом модуле я рассчитываю фактор трудоемкости для функции полезности."""

import logging
from typing import Optional

from world.entities import AgentState, AgentType, VisibleEntity


logger = logging.getLogger(__name__)

MAX_BURIEDNESS: float = 100.0
MAX_TOTAL_AREA: float = 100000.0
MAX_REPAIR_COST: float = 100000.0


def _clamp_to_unit(value: float) -> float:
    """Здесь я ограничиваю значение диапазоном [0, 1], чтобы сохранить нормировку."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def effort_for_ambulance(entity: VisibleEntity, max_buriedness: float = MAX_BURIEDNESS) -> float:
    """Здесь я рассчитываю трудоемкость для медиков по глубине завала."""

    try:
        buriedness = entity.raw_sensor_data.buriedness
        if buriedness is None:
            logger.warning("Я не могу вычислить трудоемкость без buriedness для entity_id=%s", entity.id)
            return 0.0
        return _clamp_to_unit(buriedness / max_buriedness)
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете трудоемкости для entity_id=%s", entity.id)
        return 0.0


def effort_for_fire(entity: VisibleEntity, max_total_area: float = MAX_TOTAL_AREA) -> float:
    """Здесь я рассчитываю трудоемкость для пожарных по площади здания."""

    try:
        ground_area = entity.raw_sensor_data.ground_area
        floors = entity.raw_sensor_data.floors
        if ground_area is None or floors is None:
            logger.warning("Я не могу вычислить площадь без ground_area и floors для entity_id=%s", entity.id)
            return 0.0
        total_area = ground_area * floors
        return _clamp_to_unit(total_area / max_total_area)
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете трудоемкости для entity_id=%s", entity.id)
        return 0.0


def effort_for_police(entity: VisibleEntity, max_repair_cost: float = MAX_REPAIR_COST) -> float:
    """Здесь я рассчитываю трудоемкость для полиции по стоимости расчистки."""

    try:
        repair_cost = entity.raw_sensor_data.repair_cost
        if repair_cost is None:
            logger.warning("Я не могу вычислить трудоемкость без repair_cost для entity_id=%s", entity.id)
            return 0.0
        return _clamp_to_unit(repair_cost / max_repair_cost)
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете трудоемкости для entity_id=%s", entity.id)
        return 0.0


def compute_effort(
    agent_state: AgentState,
    entity: Optional[VisibleEntity] = None,
    max_buriedness: float = MAX_BURIEDNESS,
    max_total_area: float = MAX_TOTAL_AREA,
    max_repair_cost: float = MAX_REPAIR_COST,
) -> float:
    """Здесь я объединяю вычисление трудоемкости и маршрутизирую по типу агента."""

    try:
        if entity is None:
            logger.warning("Я не получил сущность для расчета трудоемкости: agent_id=%s", agent_state.id)
            return 0.0

        if agent_state.type == AgentType.AMBULANCE_TEAM:
            return effort_for_ambulance(entity, max_buriedness)

        if agent_state.type == AgentType.FIRE_BRIGADE:
            return effort_for_fire(entity, max_total_area)

        if agent_state.type == AgentType.POLICE_FORCE:
            return effort_for_police(entity, max_repair_cost)

        return 0.0
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при общем расчете трудоемкости: agent_id=%s", agent_state.id)
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
