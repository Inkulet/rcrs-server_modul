from __future__ import annotations


import logging
from enum import Enum
from typing import List, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, ValidationError


logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    FIRE_BRIGADE = "FIRE_BRIGADE"
    AMBULANCE_TEAM = "AMBULANCE_TEAM"
    POLICE_FORCE = "POLICE_FORCE"
    FIRE_STATION = "FIRE_STATION"
    AMBULANCE_CENTRE = "AMBULANCE_CENTRE"
    POLICE_OFFICE = "POLICE_OFFICE"


class EntityType(str, Enum):
    BUILDING = "BUILDING"
    CIVILIAN = "CIVILIAN"
    HUMAN = "HUMAN"
    BLOCKADE = "BLOCKADE"


class BaseEntityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    @classmethod
    def parse(cls, data: object) -> Optional[Self]:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            logger.error("Я получил некорректные данные для %s: %s", cls.__name__, exc)
            return None


class Position(BaseEntityModel):
    entity_id: StrictInt = Field(..., ge=0)
    x: StrictInt
    y: StrictInt


class Resources(BaseEntityModel):
    water_quantity: StrictInt = Field(..., ge=0)
    is_transporting: StrictBool


class AgentState(BaseEntityModel):
    id: StrictInt = Field(..., ge=0)
    type: AgentType
    position: Position
    resources: Resources
    hp: Optional[StrictInt] = Field(default=None, ge=0)
    damage: Optional[StrictInt] = Field(default=None, ge=0)
    buriedness: Optional[StrictInt] = Field(default=None, ge=0)


class RawSensorData(BaseEntityModel):
    hp: Optional[StrictInt] = Field(default=None, ge=0)
    damage: Optional[StrictInt] = Field(default=None, ge=0)
    buriedness: Optional[StrictInt] = Field(default=None, ge=0)
    temperature: Optional[StrictFloat] = Field(default=None)
    fieryness: Optional[StrictInt] = Field(default=None, ge=0, le=8)
    floors: Optional[StrictInt] = Field(default=None, ge=0)
    ground_area: Optional[StrictInt] = Field(default=None, ge=0)
    repair_cost: Optional[StrictInt] = Field(default=None, ge=0)
    position_on_edge: Optional[StrictInt] = Field(default=None, ge=0)


class ComputedMetrics(BaseEntityModel):
    path_distance: StrictFloat = Field(..., ge=0)
    estimated_death_time: StrictInt
    total_area: StrictInt = Field(..., ge=0)


class VisibleEntity(BaseEntityModel):
    id: StrictInt = Field(..., ge=0)
    type: EntityType
    raw_sensor_data: RawSensorData
    computed_metrics: ComputedMetrics
    utility_score: StrictFloat
    entity_x: Optional[StrictInt] = Field(default=None)
    entity_y: Optional[StrictInt] = Field(default=None)


class MapNode(BaseEntityModel):
    entity_id: StrictInt = Field(..., ge=0)
    x: StrictInt
    y: StrictInt


class MapEdge(BaseEntityModel):
    source_id: StrictInt = Field(..., ge=0)
    target_id: StrictInt = Field(..., ge=0)
    weight: StrictFloat = Field(..., gt=0)


class PerceptionPacket(BaseEntityModel):
    tick: StrictInt = Field(..., ge=0)
    own_state: AgentState
    visible_entities: List[VisibleEntity] = Field(default_factory=list)
    ally_states: List[AgentState] = Field(default_factory=list)

    map_nodes: List[MapNode] = Field(default_factory=list)
    map_edges: List[MapEdge] = Field(default_factory=list)
    refuge_ids: List[StrictInt] = Field(default_factory=list)

    deleted_entity_ids: List[StrictInt] = Field(default_factory=list)

    heard_target_ids: set[int] = Field(default_factory=set)

    # Обратный индекс blockade_id → road_id, извлечённый из PROP_BLOCKADES
    # дорожных сущностей. Используется как резервный источник position_on_edge
    # в случае, если PROP_POSITION для завала не пришёл.
    blockade_to_road: dict[int, int] = Field(default_factory=dict)


# Я умножаю наивную оценку hp/damage на этот коэффициент, чтобы учесть
# нелинейный рост damage (рядом с пожаром, кровотечение нарастает).
# 0.7 — консервативный «запас» на то, что гражданский умрёт раньше,
# чем предсказывает линейная экстраполяция.
DEATH_TIME_SAFETY_FACTOR: float = 0.7


def estimate_death_time(raw: RawSensorData) -> int:
    hp = raw.hp
    damage = raw.damage
    if hp is None or damage is None or damage <= 0:
        return 99999
    try:
        return int((hp / damage) * DEATH_TIME_SAFETY_FACTOR)
    except ZeroDivisionError:
        return 99999


def compute_total_area(raw: RawSensorData) -> int:
    ga = raw.ground_area
    fl = raw.floors
    if ga is None or fl is None:
        return 0
    return ga * fl


def parse_agent_state(data: object) -> Optional[AgentState]:
    return AgentState.parse(data)


def parse_visible_entity(data: object) -> Optional[VisibleEntity]:
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
    "MapNode",
    "MapEdge",
    "PerceptionPacket",
    "StrictInt",
    "parse_agent_state",
    "parse_visible_entity",
    "estimate_death_time",
    "compute_total_area",
]
