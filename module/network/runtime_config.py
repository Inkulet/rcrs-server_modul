from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from module.config import ModelConfig, ModelConstants, UtilityWeights
from module.config_profile import model_config_from_dict
from module.data_models import AgentType
from module.network.protocol import URN
from module.network.world_model import WorldModel


def build_runtime_baseline_config(world_model: WorldModel) -> ModelConfig:
    """Конфигурация runtime вычисляет нормировочные константы из текущей карты, не меняя формулы матмодели."""
    max_map_distance = _compute_map_diagonal(world_model)
    max_buriedness = _max_non_zero(_collect_property_values(world_model, int(URN.Entity.CIVILIAN), int(URN.Property.BURIEDNESS)))
    max_repair_cost = _max_non_zero(_collect_property_values(world_model, int(URN.Entity.BLOCKADE), int(URN.Property.REPAIR_COST)))
    temperature_max = _max_non_zero(
        _collect_property_values(world_model, int(URN.Entity.BUILDING), int(URN.Property.TEMPERATURE)),
        minimum=1.0,
    )

    total_areas: List[float] = []
    for entity in world_model.entities.values():
        if entity.urn not in {int(URN.Entity.BUILDING), int(URN.Entity.REFUGE), int(URN.Entity.GAS_STATION)}:
            continue
        ground_area = entity.properties.get(int(URN.Property.BUILDING_AREA_GROUND))
        floors = entity.properties.get(int(URN.Property.FLOORS))
        try:
            if ground_area is None or floors is None:
                continue
            total_areas.append(float(ground_area) * float(floors))
        except (TypeError, ValueError):
            continue
    max_total_area = _max_non_zero(total_areas)

    weights_by_agent: Dict[AgentType, UtilityWeights] = {
        AgentType.FIRE_BRIGADE: UtilityWeights(w_c=1.0, w_d=1.0, w_e=1.0, w_n=1.0),
        AgentType.AMBULANCE_TEAM: UtilityWeights(w_c=1.0, w_d=1.0, w_e=1.0, w_n=1.0),
        AgentType.POLICE_FORCE: UtilityWeights(w_c=1.0, w_d=1.0, w_e=1.0, w_n=1.0),
    }

    constants = ModelConstants(
        max_map_distance=max_map_distance,
        max_buriedness=max_buriedness,
        max_total_area=max_total_area,
        max_repair_cost=max_repair_cost,
        temperature_max=temperature_max,
        social_radius=max(1.0, max_map_distance * 0.25),
        ambulance_clear_rate=12.0,
        travel_speed=1.0,
        epsilon=1e-6,
        c_switch=0.03,
    )

    config = ModelConfig(weights_by_agent=weights_by_agent, constants=constants)
    config.validate()
    return config


def load_profile_or_baseline(profile_path: Optional[Path], baseline: ModelConfig) -> ModelConfig:
    """Пользовательский профиль применяется только при валидном JSON, иначе используется baseline без остановки цикла."""
    if profile_path is None:
        return baseline

    if not profile_path.exists():
        return baseline

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        return model_config_from_dict(payload)
    except Exception:  # noqa: BLE001
        return baseline


def _collect_property_values(world_model: WorldModel, entity_urn: int, property_urn: int) -> List[float]:
    """Сбор значений выделен в helper, чтобы повторно использовать логику максимумов для разных свойств."""
    values: List[float] = []
    for entity in world_model.entities.values():
        if entity.urn != int(entity_urn):
            continue
        value = entity.properties.get(int(property_urn))
        try:
            if value is None:
                continue
            float_value = float(value)
            if float_value >= 0:
                values.append(float_value)
        except (TypeError, ValueError):
            continue
    return values


def _max_non_zero(values: List[float], minimum: float = 1.0) -> float:
    """Нижняя граница >0 обязательна, иначе нормализация в формуле полезности приведёт к делению на ноль."""
    if not values:
        return minimum
    return max(minimum, max(values))


def _compute_map_diagonal(world_model: WorldModel) -> float:
    """MaxMapDistance задаётся диагональю карты согласно примечанию в разделе 2.2 документации."""
    points: List[Tuple[int, int]] = []

    for entity in world_model.entities.values():
        if not world_model.is_area(entity.urn):
            continue
        x_value = entity.properties.get(int(URN.Property.X))
        y_value = entity.properties.get(int(URN.Property.Y))
        try:
            if x_value is None or y_value is None:
                continue
            points.append((int(x_value), int(y_value)))
        except (TypeError, ValueError):
            continue

    if len(points) < 2:
        return 1.0

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    dx = float(max_x - min_x)
    dy = float(max_y - min_y)
    diagonal = math.sqrt(dx * dx + dy * dy)
    return max(1.0, diagonal)
