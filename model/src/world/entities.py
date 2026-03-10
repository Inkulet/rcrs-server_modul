from __future__ import annotations

"""В этом модуле я описываю базовые структуры данных состояния агентов и сущностей."""

import logging
from enum import Enum
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, ValidationError


logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """В этом перечислении я фиксирую типы агентов симуляции."""

    FIRE_BRIGADE = "FIRE_BRIGADE"
    AMBULANCE_TEAM = "AMBULANCE_TEAM"
    POLICE_FORCE = "POLICE_FORCE"


class EntityType(str, Enum):
    """В этом перечислении я фиксирую типы наблюдаемых сущностей."""

    BUILDING = "BUILDING"
    CIVILIAN = "CIVILIAN"
    BLOCKADE = "BLOCKADE"


class BaseEntityModel(BaseModel):
    """В этом базовом классе я задаю строгую типизацию и единую обработку ошибок."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    @classmethod
    def parse(cls, data: object) -> Optional[Self]:
        """Здесь я валидирую входные данные и явно логирую ошибки парсинга."""

        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            logger.error("Я получил некорректные данные для %s: %s", cls.__name__, exc)
            return None


class Position(BaseEntityModel):
    """В этом классе я описываю положение сущности на карте."""

    entity_id: StrictInt = Field(..., ge=0)
    x: StrictInt
    y: StrictInt


class Resources(BaseEntityModel):
    """В этом классе я описываю ключевые ресурсы агента."""

    water_quantity: StrictInt = Field(..., ge=0)
    is_transporting: StrictBool


class AgentState(BaseEntityModel):
    """В этом классе я описываю состояние агента в момент времени."""

    id: StrictInt = Field(..., ge=0)
    type: AgentType
    position: Position
    resources: Resources


class RawSensorData(BaseEntityModel):
    """В этом классе я храню сырые сенсорные параметры, которые могут быть неизвестны."""

    hp: Optional[StrictInt] = Field(default=None, ge=0)
    damage: Optional[StrictInt] = Field(default=None, ge=0)
    buriedness: Optional[StrictInt] = Field(default=None, ge=0)
    temperature: Optional[StrictFloat] = Field(default=None)
    fieryness: Optional[StrictInt] = Field(default=None, ge=1, le=8)
    floors: Optional[StrictInt] = Field(default=None, ge=0)
    ground_area: Optional[StrictInt] = Field(default=None, ge=0)
    repair_cost: Optional[StrictInt] = Field(default=None, ge=0)
    position_on_edge: Optional[StrictInt] = Field(default=None, ge=0)


class ComputedMetrics(BaseEntityModel):
    """В этом классе я описываю вычисленные метрики для принятия решений."""

    path_distance: StrictFloat = Field(..., ge=0)
    estimated_death_time: StrictInt
    total_area: StrictInt = Field(..., ge=0)


class VisibleEntity(BaseEntityModel):
    """В этом классе я описываю наблюдаемую сущность в мире."""

    id: StrictInt = Field(..., ge=0)
    type: EntityType
    raw_sensor_data: RawSensorData
    computed_metrics: ComputedMetrics
    utility_score: StrictFloat


def parse_agent_state(data: object) -> Optional[AgentState]:
    """Здесь я даю удобную обертку для безопасного парсинга состояния агента."""

    return AgentState.parse(data)


def parse_visible_entity(data: object) -> Optional[VisibleEntity]:
    """Здесь я даю удобную обертку для безопасного парсинга наблюдаемой сущности."""

    return VisibleEntity.parse(data)


__all__ = [
    "AgentType",
    "EntityType",
    "Position",
    "Resources",
    "AgentState",
    "RawSensorData",
    "ComputedMetrics",
    "VisibleEntity",
    "parse_agent_state",
    "parse_visible_entity",
]
