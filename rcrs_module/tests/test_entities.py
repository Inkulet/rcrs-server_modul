from __future__ import annotations

"""В этом модуле я тестирую модели данных и вспомогательные функции из world/entities.py."""

import pytest

from world.entities import (
    AgentType,
    ComputedMetrics,
    DEATH_TIME_SAFETY_FACTOR,
    EntityType,
    Position,
    RawSensorData,
    Resources,
    VisibleEntity,
    estimate_death_time,
    compute_total_area,
)


# ---------------------------------------------------------------------------
# estimate_death_time
# ---------------------------------------------------------------------------

class TestEstimateDeathTime:

    def test_normal_case(self) -> None:
        raw = RawSensorData(hp=1000, damage=10)
        # Я применяю safety factor для пессимистичной оценки.
        assert estimate_death_time(raw) == int(100 * DEATH_TIME_SAFETY_FACTOR)

    def test_hp_none_returns_large(self) -> None:
        raw = RawSensorData(hp=None, damage=10)
        assert estimate_death_time(raw) == 99999

    def test_damage_none_returns_large(self) -> None:
        raw = RawSensorData(hp=1000, damage=None)
        assert estimate_death_time(raw) == 99999

    def test_damage_zero_returns_large(self) -> None:
        raw = RawSensorData(hp=1000, damage=0)
        assert estimate_death_time(raw) == 99999

    def test_both_none_returns_large(self) -> None:
        raw = RawSensorData()
        assert estimate_death_time(raw) == 99999

    def test_small_hp_high_damage(self) -> None:
        raw = RawSensorData(hp=10, damage=100)
        assert estimate_death_time(raw) == 0  # int(10/100) = 0

    def test_large_values(self) -> None:
        raw = RawSensorData(hp=10000, damage=1)
        assert estimate_death_time(raw) == int(10000 * DEATH_TIME_SAFETY_FACTOR)


# ---------------------------------------------------------------------------
# compute_total_area
# ---------------------------------------------------------------------------

class TestComputeTotalArea:

    def test_normal_case(self) -> None:
        raw = RawSensorData(ground_area=200, floors=3)
        assert compute_total_area(raw) == 600

    def test_ground_area_none(self) -> None:
        raw = RawSensorData(ground_area=None, floors=3)
        assert compute_total_area(raw) == 0

    def test_floors_none(self) -> None:
        raw = RawSensorData(ground_area=200, floors=None)
        assert compute_total_area(raw) == 0

    def test_both_none(self) -> None:
        raw = RawSensorData()
        assert compute_total_area(raw) == 0

    def test_zero_floors(self) -> None:
        raw = RawSensorData(ground_area=200, floors=0)
        assert compute_total_area(raw) == 0

    def test_zero_area(self) -> None:
        raw = RawSensorData(ground_area=0, floors=5)
        assert compute_total_area(raw) == 0


# ---------------------------------------------------------------------------
# RawSensorData — валидация Pydantic
# ---------------------------------------------------------------------------

class TestRawSensorData:

    def test_all_none_defaults(self) -> None:
        raw = RawSensorData()
        assert raw.hp is None
        assert raw.damage is None
        assert raw.fieryness is None

    def test_fieryness_zero_is_valid(self) -> None:
        raw = RawSensorData(fieryness=0)
        assert raw.fieryness == 0

    def test_fieryness_range_1_to_8(self) -> None:
        for f in range(0, 9):
            raw = RawSensorData(fieryness=f)
            assert raw.fieryness == f


# ---------------------------------------------------------------------------
# ComputedMetrics
# ---------------------------------------------------------------------------

class TestComputedMetrics:

    def test_default_fields(self) -> None:
        cm = ComputedMetrics(path_distance=0.0, estimated_death_time=99999, total_area=0)
        assert cm.path_distance == 0.0


# ---------------------------------------------------------------------------
# Position / Resources / AgentType / EntityType
# ---------------------------------------------------------------------------

class TestEnumsAndModels:

    def test_agent_types(self) -> None:
        assert AgentType.FIRE_BRIGADE.value == "FIRE_BRIGADE"
        assert AgentType.AMBULANCE_TEAM.value == "AMBULANCE_TEAM"
        assert AgentType.POLICE_FORCE.value == "POLICE_FORCE"

    def test_entity_types(self) -> None:
        assert EntityType.CIVILIAN.value == "CIVILIAN"
        assert EntityType.HUMAN.value == "HUMAN"
        assert EntityType.BUILDING.value == "BUILDING"
        assert EntityType.BLOCKADE.value == "BLOCKADE"

    def test_position_model(self) -> None:
        pos = Position(entity_id=5, x=100, y=200)
        assert pos.entity_id == 5

    def test_resources_defaults(self) -> None:
        res = Resources(water_quantity=0, is_transporting=False)
        assert res.water_quantity == 0
        assert res.is_transporting is False
