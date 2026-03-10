from __future__ import annotations

"""В этом модуле я рассчитываю фактор срочности для функции полезности."""

import logging
from math import inf
from typing import Optional

from world.entities import AgentState, AgentType, VisibleEntity


logger = logging.getLogger(__name__)

T_MAX: float = 1000.0
EPSILON: float = 1e-6
STABLE_URGENCY: float = 0.01


def _clamp_to_unit(value: float) -> float:
    """Здесь я ограничиваю значение диапазоном [0, 1], чтобы сохранить нормировку."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def urgency_for_ambulance(
    entity: VisibleEntity,
    t_travel: float,
    t_work: float,
    stable_value: float = STABLE_URGENCY,
) -> float:
    """Здесь я рассчитываю срочность для медиков по TTL и времени прибытия."""

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
        if ttl > t_travel + t_work:
            return _clamp_to_unit(1.0 / ttl)
        return 0.0
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете срочности для entity_id=%s", entity.id)
        return 0.0


def urgency_for_fire(entity: VisibleEntity, t_max: float = T_MAX) -> float:
    """Здесь я рассчитываю срочность для пожарных по температуре и стадии горения."""

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


def urgency_for_police(min_distance_to_targets: float, epsilon: float = EPSILON) -> float:
    """Здесь я рассчитываю срочность для полиции по расстоянию до целей."""

    try:
        return _clamp_to_unit(1.0 / (min_distance_to_targets + epsilon))
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при расчете срочности для полиции")
        return 0.0


def compute_urgency(
    agent_state: AgentState,
    entity: Optional[VisibleEntity] = None,
    t_travel: Optional[float] = None,
    t_work: Optional[float] = None,
    min_distance_to_targets: Optional[float] = None,
    t_max: float = T_MAX,
    epsilon: float = EPSILON,
    stable_value: float = STABLE_URGENCY,
) -> float:
    """Здесь я объединяю вычисление срочности и маршрутизирую по типу агента."""

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
            if min_distance_to_targets is None:
                logger.warning("Я не получил расстояние для срочности полиции: agent_id=%s", agent_state.id)
                return 0.0
            return urgency_for_police(min_distance_to_targets, epsilon)

        return 0.0
    except ZeroDivisionError:
        logger.warning("Я поймал деление на ноль при общем расчете срочности: agent_id=%s", agent_state.id)
        return 0.0


__all__ = [
    "T_MAX",
    "EPSILON",
    "STABLE_URGENCY",
    "urgency_for_ambulance",
    "urgency_for_fire",
    "urgency_for_police",
    "compute_urgency",
]
