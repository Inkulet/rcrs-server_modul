from __future__ import annotations

"""В этом модуле я определяю общие pytest-фикстуры для всего тестового набора."""

# Я добавляю src/ в sys.path один раз здесь, чтобы все тесты могли импортировать
# модули без установки пакета — это стандартная практика для in-tree тестов.

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import networkx as nx

from world.cache import WorldModel
from world.entities import (
    AgentState,
    AgentType,
    ComputedMetrics,
    EntityType,
    MapEdge,
    MapNode,
    Position,
    RawSensorData,
    Resources,
    VisibleEntity,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики — я делаю их функциями, а не фикстурами,
# чтобы тесты могли задавать конкретные значения через аргументы.
# ---------------------------------------------------------------------------


def make_agent(
    agent_id: int = 1,
    agent_type: AgentType = AgentType.AMBULANCE_TEAM,
    x: int = 0,
    y: int = 0,
    entity_id: int = 1,
    water: int = 5000,
    transporting: bool = False,
) -> AgentState:
    """Я создаю AgentState с заданными параметрами для тестов."""
    return AgentState(
        id=agent_id,
        type=agent_type,
        position=Position(entity_id=entity_id, x=x, y=y),
        resources=Resources(water_quantity=water, is_transporting=transporting),
    )


def make_civilian(
    entity_id: int = 10,
    hp: int = 10000,
    damage: int = 50,
    buriedness: int = 10,
    path_distance: float = 100.0,
    estimated_death_time: int = 500,
) -> VisibleEntity:
    """Я создаю VisibleEntity-гражданского с заданными параметрами."""
    return VisibleEntity(
        id=entity_id,
        type=EntityType.CIVILIAN,
        raw_sensor_data=RawSensorData(hp=hp, damage=damage, buriedness=buriedness),
        computed_metrics=ComputedMetrics(
            path_distance=path_distance,
            estimated_death_time=estimated_death_time,
            total_area=0,
        ),
        utility_score=0.0,
    )


def make_building(
    entity_id: int = 20,
    temperature: float = 500.0,
    fieryness: int = 2,
    floors: int = 3,
    ground_area: int = 200,
    path_distance: float = 200.0,
) -> VisibleEntity:
    """Я создаю VisibleEntity-здание с заданными параметрами."""
    return VisibleEntity(
        id=entity_id,
        type=EntityType.BUILDING,
        raw_sensor_data=RawSensorData(
            temperature=temperature,
            fieryness=fieryness,
            floors=floors,
            ground_area=ground_area,
        ),
        computed_metrics=ComputedMetrics(
            path_distance=path_distance,
            estimated_death_time=999,
            total_area=floors * ground_area,
        ),
        utility_score=0.0,
    )


def make_blockade(
    entity_id: int = 30,
    repair_cost: int = 1000,
    path_distance: float = 50.0,
) -> VisibleEntity:
    """Я создаю VisibleEntity-завал с заданными параметрами."""
    return VisibleEntity(
        id=entity_id,
        type=EntityType.BLOCKADE,
        raw_sensor_data=RawSensorData(repair_cost=repair_cost),
        computed_metrics=ComputedMetrics(
            path_distance=path_distance,
            estimated_death_time=9999,
            total_area=0,
        ),
        utility_score=0.0,
    )


def make_world_with_graph() -> WorldModel:
    """Я создаю WorldModel с простым линейным графом для тестов дистанции."""
    wm = WorldModel()
    # Граф: 1 --500-- 2 --500-- 3
    nodes = [
        MapNode(entity_id=1, x=0, y=0),
        MapNode(entity_id=2, x=500, y=0),
        MapNode(entity_id=3, x=1000, y=0),
    ]
    edges = [
        MapEdge(source_id=1, target_id=2, weight=500.0),
        MapEdge(source_id=2, target_id=3, weight=500.0),
    ]
    wm.build_graph_from_map(nodes, edges)
    return wm
