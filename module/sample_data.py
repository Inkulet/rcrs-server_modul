from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from module.agents import AmbulanceTeamAgent, FireBrigadeAgent, PoliceForceAgent
from module.calculator import UtilityCalculator
from module.config import ModelConfig, ModelConstants, UtilityWeights
from module.data_models import (
    AgentOperationalState,
    AgentState,
    AgentType,
    ComputedMetrics,
    EntityType,
    Position,
    RawSensorData,
    Resources,
    VisibleEntity,
)
from module.simulation import SimulationEngine
from module.strategy import UtilityBasedTargetSelectionStrategy


BASELINE_PROFILE_NAME = "baseline_diploma_2_2"
DEFAULT_LIVE_STATE_PATH = Path(__file__).resolve().parents[0] / "data" / "live_state.json"


def build_demo_engine(
    config: Optional[ModelConfig] = None,
    visible_entities: Optional[List[VisibleEntity]] = None,
    refuges: Optional[List[Position]] = None,
) -> SimulationEngine:
    """Сборка движка вынесена в фабрику, чтобы UI мог перезапускать симуляцию с новым профилем."""
    if visible_entities is None or refuges is None:
        scenario_entities, scenario_refuges = build_demo_scenario()
    else:
        scenario_entities = copy.deepcopy(visible_entities)
        scenario_refuges = copy.deepcopy(refuges)

    active_config = config if config is not None else build_baseline_config(scenario_entities, scenario_refuges)

    calculator = UtilityCalculator(active_config)
    strategy = UtilityBasedTargetSelectionStrategy(calculator)

    fire_agent = FireBrigadeAgent(
        state=AgentState(
            id=101,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=1010, x=2500, y=2300),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=3000, is_transporting=False),
        ),
        strategy=strategy,
    )

    ambulance_agent = AmbulanceTeamAgent(
        state=AgentState(
            id=201,
            type=AgentType.AMBULANCE_TEAM,
            position=Position(entity_id=2010, x=1200, y=1000),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=0, is_transporting=False),
        ),
        strategy=strategy,
    )

    police_agent = PoliceForceAgent(
        state=AgentState(
            id=301,
            type=AgentType.POLICE_FORCE,
            position=Position(entity_id=3010, x=1800, y=1200),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=0, is_transporting=False),
        ),
        strategy=strategy,
    )

    return SimulationEngine(
        agents={
            fire_agent.state.id: fire_agent,
            ambulance_agent.state.id: ambulance_agent,
            police_agent.state.id: police_agent,
        },
        visible_entities=scenario_entities,
        refuges=scenario_refuges,
    )


def build_demo_scenario() -> Tuple[List[VisibleEntity], List[Position]]:
    """Сценарий формируется отдельной функцией, чтобы baseline и custom профили работали на одинаковых данных."""
    return _build_visible_entities(), _build_refuges()


def build_baseline_config(
    visible_entities: Optional[List[VisibleEntity]] = None,
    refuges: Optional[List[Position]] = None,
) -> ModelConfig:
    """Базовый профиль соответствует формуле из 2.2 и служит точкой возврата в UI."""
    if visible_entities is None or refuges is None:
        scenario_entities, scenario_refuges = build_demo_scenario()
    else:
        scenario_entities = visible_entities
        scenario_refuges = refuges

    return _build_config(scenario_entities, scenario_refuges)


def _build_config(visible_entities: List[VisibleEntity], refuges: List[Position]) -> ModelConfig:
    """Числа нормализации берём из текущей карты, чтобы не зашивать произвольные константы."""
    max_map_distance = _calculate_map_diagonal(visible_entities, refuges)
    max_buriedness = max(
        1,
        max(
            [
                entity.raw_sensor_data.buriedness or 0
                for entity in visible_entities
                if entity.type == EntityType.CIVILIAN
            ],
            default=1,
        ),
    )

    max_total_area = max(
        1,
        max(
            [
                (entity.raw_sensor_data.ground_area or 0) * (entity.raw_sensor_data.floors or 0)
                for entity in visible_entities
                if entity.type == EntityType.BUILDING
            ],
            default=1,
        ),
    )

    max_repair_cost = max(
        1,
        max(
            [
                entity.raw_sensor_data.repair_cost or 0
                for entity in visible_entities
                if entity.type == EntityType.BLOCKADE
            ],
            default=1,
        ),
    )

    temperature_max = max(
        1.0,
        max(
            [
                float(entity.raw_sensor_data.temperature or 0.0)
                for entity in visible_entities
                if entity.type == EntityType.BUILDING
            ],
            default=1.0,
        ),
    )

    # Веса вынесены в конфиг и могут редактироваться пользователем без изменения математической формулы U_ij.
    weights_by_agent: Dict[AgentType, UtilityWeights] = {
        AgentType.FIRE_BRIGADE: UtilityWeights(w_c=1.0, w_d=1.0, w_e=1.0, w_n=1.0),
        AgentType.AMBULANCE_TEAM: UtilityWeights(w_c=1.0, w_d=1.0, w_e=1.0, w_n=1.0),
        AgentType.POLICE_FORCE: UtilityWeights(w_c=1.0, w_d=1.0, w_e=1.0, w_n=1.0),
    }

    constants = ModelConstants(
        max_map_distance=max_map_distance,
        max_buriedness=float(max_buriedness),
        max_total_area=float(max_total_area),
        max_repair_cost=float(max_repair_cost),
        temperature_max=float(temperature_max),
        social_radius=max(1.0, max_map_distance * 0.25),
        ambulance_clear_rate=12.0,
        travel_speed=300.0,
        epsilon=1e-6,
        c_switch=0.03,
    )

    config = ModelConfig(weights_by_agent=weights_by_agent, constants=constants)
    config.validate()
    return config


def _build_refuges() -> List[Position]:
    """Refuge точки требуются для формулы приоритета PoliceForce и ветки GoToRefuge у пожарных."""
    return [
        Position(entity_id=9001, x=800, y=700),
        Position(entity_id=9002, x=5200, y=600),
    ]


def _build_visible_entities() -> List[VisibleEntity]:
    """Набор целей покрывает все обязательные фильтры из раздела 4 (живые, мертвые, сгоревшие, невозможные)."""
    return [
        VisibleEntity(
            id=5001,
            type=EntityType.BUILDING,
            position=Position(entity_id=5001, x=4300, y=3700),
            raw_sensor_data=RawSensorData(
                temperature=680.0,
                fieryness=2,
                floors=3,
                ground_area=900,
            ),
            computed_metrics=ComputedMetrics(),
        ),
        VisibleEntity(
            id=5002,
            type=EntityType.BUILDING,
            position=Position(entity_id=5002, x=4800, y=4200),
            raw_sensor_data=RawSensorData(
                temperature=100.0,
                fieryness=8,
                floors=2,
                ground_area=700,
            ),
            computed_metrics=ComputedMetrics(),
        ),
        VisibleEntity(
            id=6001,
            type=EntityType.CIVILIAN,
            position=Position(entity_id=6001, x=1900, y=1300),
            raw_sensor_data=RawSensorData(hp=8000, damage=25, buriedness=35),
            computed_metrics=ComputedMetrics(),
        ),
        VisibleEntity(
            id=6002,
            type=EntityType.CIVILIAN,
            position=Position(entity_id=6002, x=1400, y=900),
            raw_sensor_data=RawSensorData(hp=4500, damage=0, buriedness=0),
            computed_metrics=ComputedMetrics(),
        ),
        VisibleEntity(
            id=6003,
            type=EntityType.CIVILIAN,
            position=Position(entity_id=6003, x=4700, y=3800),
            raw_sensor_data=RawSensorData(hp=1200, damage=100, buriedness=40),
            computed_metrics=ComputedMetrics(),
        ),
        VisibleEntity(
            id=7001,
            type=EntityType.BLOCKADE,
            position=Position(entity_id=7101, x=2100, y=1500),
            raw_sensor_data=RawSensorData(repair_cost=260, position_on_edge=7101),
            computed_metrics=ComputedMetrics(),
        ),
        VisibleEntity(
            id=7002,
            type=EntityType.BLOCKADE,
            position=Position(entity_id=7102, x=4200, y=3500),
            raw_sensor_data=RawSensorData(repair_cost=120, position_on_edge=7102),
            computed_metrics=ComputedMetrics(),
        ),
    ]


def _calculate_map_diagonal(visible_entities: List[VisibleEntity], refuges: List[Position]) -> float:
    """MaxMapDistance берём как диагональ bbox карты, как предписано примечанием к f_dist."""
    points = [entity.position for entity in visible_entities] + refuges
    safe_points = [point for point in points if point.x is not None and point.y is not None]
    if len(safe_points) < 2:
        return 1.0

    min_x = min(point.x for point in safe_points if point.x is not None)
    max_x = max(point.x for point in safe_points if point.x is not None)
    min_y = min(point.y for point in safe_points if point.y is not None)
    max_y = max(point.y for point in safe_points if point.y is not None)

    dx = float(max_x - min_x)
    dy = float(max_y - min_y)
    diagonal = math.sqrt(dx * dx + dy * dy)
    return max(1.0, diagonal)


def load_live_state_snapshot(snapshot_path: Optional[Path] = None) -> Optional[Dict[str, object]]:
    """Live Mode читает снапшот из файла, который обновляется сетевыми агентами каждый тик."""
    target_path = (snapshot_path or DEFAULT_LIVE_STATE_PATH).expanduser().resolve()
    if not target_path.exists():
        return None

    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload
    except (OSError, json.JSONDecodeError):
        return None
