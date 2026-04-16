from __future__ import annotations

import logging
from typing import Optional

from world.entities import AgentState, AgentType, VisibleEntity

from decision.utility._utils import _clamp_to_unit


logger = logging.getLogger(__name__)

T_MAX: float = 1500.0
EPSILON: float = 1e-6
STABLE_URGENCY: float = 0.01

T_MAX_TTL: float = 1_000.0


def urgency_for_ambulance(
    entity: VisibleEntity,
    t_travel: float,
    t_work: float,
    stable_value: float = STABLE_URGENCY,
    t_max_ttl: float = T_MAX_TTL,
) -> float:
    try:
        hp = entity.raw_sensor_data.hp
        damage = entity.raw_sensor_data.damage
        if hp is None or damage is None:
            logger.warning("Я не могу вычислить TTL без hp и damage для entity_id=%s", entity.id)
            return 0.0

        if t_travel < 0 or t_work < 0:
            logger.warning("Я получил отрицательное время пути или работы для entity_id=%s", entity.id)
            return 0.0

        if damage <= 0:
            return _clamp_to_unit(stable_value)

        ttl = hp / damage

        if ttl <= t_travel + t_work:
            return 0.0

        return _clamp_to_unit(1.0 - ttl / t_max_ttl)
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете срочности для entity_id=%s", entity.id)
        return 0.0


def urgency_for_fire(entity: VisibleEntity, t_max: float = T_MAX) -> float:
    try:
        temperature = entity.raw_sensor_data.temperature
        fieryness = entity.raw_sensor_data.fieryness
        if temperature is None or fieryness is None:
            logger.warning("Я не могу вычислить срочность без temperature и fieryness для entity_id=%s", entity.id)
            return 0.0
        if fieryness not in {1, 2, 3}:
            return 0.0
        return _clamp_to_unit(temperature / t_max)
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете срочности для entity_id=%s", entity.id)
        return 0.0


def urgency_for_police(task_distance: float, epsilon: float = EPSILON) -> float:
    try:
        return _clamp_to_unit(1.0 / (task_distance + epsilon))

    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете срочности для полиции")
        return 0.0


def compute_urgency(
    agent_state: AgentState,
    entity: Optional[VisibleEntity] = None,
    t_travel: Optional[float] = None,
    t_work: Optional[float] = None,
    task_distance: Optional[float] = None,
    t_max: float = T_MAX,
    epsilon: float = EPSILON,
    stable_value: float = STABLE_URGENCY,
) -> float:

    try:
        if agent_state.type == AgentType.AMBULANCE_TEAM:
            if entity is None or t_travel is None or t_work is None:
                logger.warning("Я не получил достаточно данных для срочности медика: agent_id=%s", agent_state.id)
                return 0.0
            return urgency_for_ambulance(entity, t_travel, t_work, stable_value)

        if agent_state.type == AgentType.FIRE_BRIGADE:
            if entity is None:
                logger.warning("Я не получил сущность для срочности пожарного: agent_id=%s", agent_state.id)
                return 0.0
            return urgency_for_fire(entity, t_max)

        if agent_state.type == AgentType.POLICE_FORCE:
            if task_distance is None:
                logger.warning("Я не получил расстояние для срочности полиции: agent_id=%s", agent_state.id)
                return 0.0
            return urgency_for_police(task_distance, epsilon)

        return 0.0
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при общем расчете срочности: agent_id=%s", agent_state.id)
        return 0.0


__all__ = [
    "T_MAX",
    "T_MAX_TTL",
    "EPSILON",
    "STABLE_URGENCY",
    "urgency_for_ambulance",
    "urgency_for_fire",
    "urgency_for_police",
    "compute_urgency",
]
