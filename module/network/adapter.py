from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from module.data_models import (
    AgentObservation,
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
from module.network.protocol import URN
from module.network.world_model import ServerEntity, WorldModel


@dataclass
class AdaptedWorldState:
    """Результат адаптации объединяет observation, список коллег и refuge-точки для полного расчёта U_ij."""

    observation: AgentObservation
    all_agents: List[AgentState]
    refuges: List[Position]
    warnings: List[str]


class WorldModelAdapter:
    """Adapter Pattern: переводит server-worldmodel в структуру данных, ожидаемую calculator.py без его изменений."""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def adapt(self, agent_id: int, agent_type: AgentType, visible_entity_ids: List[int]) -> AdaptedWorldState:
        """Основной метод адаптации выполняет валидацию и защищает цикл агента от битых сенсорных данных."""
        warnings: List[str] = []

        agent_state = self._build_agent_state(agent_id=agent_id, agent_type=agent_type, warnings=warnings)
        visible_entities = self._build_visible_entities(
            observer_state=agent_state,
            visible_entity_ids=visible_entity_ids,
            warnings=warnings,
        )

        observation = AgentObservation(agent_state=agent_state, visible_entities=visible_entities)
        all_agents = self._build_all_agents(warnings=warnings)
        refuges = self._build_refuge_positions(warnings=warnings)

        return AdaptedWorldState(
            observation=observation,
            all_agents=all_agents,
            refuges=refuges,
            warnings=warnings,
        )

    def _build_agent_state(self, agent_id: int, agent_type: AgentType, warnings: List[str]) -> AgentState:
        """Состояние агента берётся из world-model и нормализуется под JSON-схему раздела 3."""
        area_id = self.world_model.resolve_area_position(agent_id)
        x, y = self.world_model.get_area_coordinates(area_id) if area_id is not None else (None, None)

        water_quantity = self._safe_non_negative_int(
            self.world_model.get_property(agent_id, int(URN.Property.WATER_QUANTITY)),
            default=0,
            warnings=warnings,
            field_name=f"agent[{agent_id}].water_quantity",
        )

        is_transporting = self._is_transporting(agent_id)

        return AgentState(
            id=int(agent_id),
            type=agent_type,
            position=Position(entity_id=area_id, x=x, y=y),
            state=(
                AgentOperationalState.TRANSPORTING
                if agent_type == AgentType.AMBULANCE_TEAM and is_transporting
                else AgentOperationalState.IDLE
            ),
            resources=Resources(water_quantity=water_quantity, is_transporting=is_transporting),
        )

    def _build_all_agents(self, warnings: List[str]) -> List[AgentState]:
        """Список агентов нужен для корректного f_social, иначе показатель координации будет искажён."""
        all_agents: List[AgentState] = []
        urn_to_type: Dict[int, AgentType] = {
            int(URN.Entity.FIRE_BRIGADE): AgentType.FIRE_BRIGADE,
            int(URN.Entity.AMBULANCE_TEAM): AgentType.AMBULANCE_TEAM,
            int(URN.Entity.POLICE_FORCE): AgentType.POLICE_FORCE,
        }

        for entity in self.world_model.entities.values():
            if entity.urn not in urn_to_type:
                continue

            agent_id = entity.entity_id
            area_id = self.world_model.resolve_area_position(agent_id)
            x, y = self.world_model.get_area_coordinates(area_id) if area_id is not None else (None, None)

            water_quantity = self._safe_non_negative_int(
                self.world_model.get_property(agent_id, int(URN.Property.WATER_QUANTITY)),
                default=0,
                warnings=warnings,
                field_name=f"agent[{agent_id}].water_quantity",
            )

            all_agents.append(
                AgentState(
                    id=agent_id,
                    type=urn_to_type[entity.urn],
                    position=Position(entity_id=area_id, x=x, y=y),
                    state=(
                        AgentOperationalState.TRANSPORTING
                        if urn_to_type[entity.urn] == AgentType.AMBULANCE_TEAM and self._is_transporting(agent_id)
                        else AgentOperationalState.IDLE
                    ),
                    resources=Resources(water_quantity=water_quantity, is_transporting=self._is_transporting(agent_id)),
                )
            )

        return all_agents

    def _build_refuge_positions(self, warnings: List[str]) -> List[Position]:
        """Refuge-позиции обязательны для f_urgency^police и команды GoToRefuge у пожарных."""
        refuges: List[Position] = []
        for entity in self.world_model.entities.values():
            if entity.urn != int(URN.Entity.REFUGE):
                continue
            x, y = self.world_model.get_area_coordinates(entity.entity_id)
            if x is None or y is None:
                warnings.append(f"Refuge {entity.entity_id} не содержит валидных координат X/Y")
            refuges.append(Position(entity_id=entity.entity_id, x=x, y=y))
        return refuges

    def _build_visible_entities(
        self,
        observer_state: AgentState,
        visible_entity_ids: List[int],
        warnings: List[str],
    ) -> List[VisibleEntity]:
        """Visible-список формируется только из текущего KA_SENSE, чтобы UI отражал реальное восприятие агента."""
        unique_ids = sorted({int(entity_id) for entity_id in visible_entity_ids})
        result: List[VisibleEntity] = []

        for entity_id in unique_ids:
            entity = self.world_model.get_entity(entity_id)
            if entity is None:
                warnings.append(f"Visible entity {entity_id} отсутствует в world-model")
                continue

            mapped_type = self._map_entity_type(entity.urn)
            if mapped_type is None:
                continue

            visible_entity = self._convert_entity(observer_state, entity, mapped_type, warnings)
            if visible_entity is not None:
                result.append(visible_entity)

        return result

    def _convert_entity(
        self,
        observer_state: AgentState,
        server_entity: ServerEntity,
        mapped_type: EntityType,
        warnings: List[str],
    ) -> Optional[VisibleEntity]:
        """Конвертация одной сущности включает строгий маппинг полей и расчёт базовых computed_metrics."""
        try:
            position = self._extract_entity_position(server_entity, warnings)
            raw_data = self._extract_raw_sensor_data(server_entity, mapped_type, warnings)
            computed_metrics = self._build_computed_metrics(observer_state, server_entity, mapped_type, raw_data)

            return VisibleEntity(
                id=server_entity.entity_id,
                type=mapped_type,
                position=position,
                raw_sensor_data=raw_data,
                computed_metrics=computed_metrics,
                utility_score=None,
            )
        except Exception as error:  # noqa: BLE001
            warnings.append(f"Не удалось адаптировать entity={server_entity.entity_id}: {error}")
            return None

    def _map_entity_type(self, entity_urn: int) -> Optional[EntityType]:
        """Маппинг URN -> EntityType фиксирует договор между сервером и calculator.py."""
        if entity_urn in {int(URN.Entity.BUILDING), int(URN.Entity.REFUGE), int(URN.Entity.GAS_STATION)}:
            return EntityType.BUILDING
        if entity_urn == int(URN.Entity.CIVILIAN):
            return EntityType.CIVILIAN
        if entity_urn == int(URN.Entity.BLOCKADE):
            return EntityType.BLOCKADE
        return None

    def _extract_entity_position(self, entity: ServerEntity, warnings: List[str]) -> Position:
        """Позиция нормализуется к area+координатам, чтобы f_dist/f_social работали на единой геометрии."""
        if entity.urn in {int(URN.Entity.BUILDING), int(URN.Entity.REFUGE), int(URN.Entity.GAS_STATION)}:
            x, y = self.world_model.get_area_coordinates(entity.entity_id)
            return Position(entity_id=entity.entity_id, x=x, y=y)

        if entity.urn == int(URN.Entity.CIVILIAN):
            area_id = self.world_model.resolve_area_position(entity.entity_id)
            x, y = self.world_model.get_area_coordinates(area_id) if area_id is not None else (None, None)
            if area_id is None:
                warnings.append(f"Civilian {entity.entity_id} без валидного POSITION")
            return Position(entity_id=area_id, x=x, y=y)

        if entity.urn == int(URN.Entity.BLOCKADE):
            area_ref = entity.properties.get(int(URN.Property.POSITION))
            area_id = self._safe_non_negative_int(area_ref, default=None, warnings=warnings, field_name=f"blockade[{entity.entity_id}].position")
            x, y = self.world_model.get_area_coordinates(area_id) if area_id is not None else (None, None)
            return Position(entity_id=area_id, x=x, y=y)

        return Position(entity_id=None, x=None, y=None)

    def _extract_raw_sensor_data(self, entity: ServerEntity, mapped_type: EntityType, warnings: List[str]) -> RawSensorData:
        """Raw sensor поля приводятся к ожидаемым именам схемы раздела 3 и валидируются по диапазонам."""
        if mapped_type == EntityType.BUILDING:
            fieryness = self._safe_int_in_range(
                entity.properties.get(int(URN.Property.FIERYNESS)),
                low=1,
                high=8,
                default=None,
                warnings=warnings,
                field_name=f"building[{entity.entity_id}].fieryness",
            )
            return RawSensorData(
                temperature=self._safe_float_non_negative(
                    entity.properties.get(int(URN.Property.TEMPERATURE)),
                    default=None,
                    warnings=warnings,
                    field_name=f"building[{entity.entity_id}].temperature",
                ),
                fieryness=fieryness,
                floors=self._safe_non_negative_int(
                    entity.properties.get(int(URN.Property.FLOORS)),
                    default=None,
                    warnings=warnings,
                    field_name=f"building[{entity.entity_id}].floors",
                ),
                ground_area=self._safe_non_negative_int(
                    entity.properties.get(int(URN.Property.BUILDING_AREA_GROUND)),
                    default=None,
                    warnings=warnings,
                    field_name=f"building[{entity.entity_id}].ground_area",
                ),
            )

        if mapped_type == EntityType.CIVILIAN:
            return RawSensorData(
                hp=self._safe_non_negative_int(
                    entity.properties.get(int(URN.Property.HP)),
                    default=None,
                    warnings=warnings,
                    field_name=f"civilian[{entity.entity_id}].hp",
                ),
                damage=self._safe_non_negative_int(
                    entity.properties.get(int(URN.Property.DAMAGE)),
                    default=None,
                    warnings=warnings,
                    field_name=f"civilian[{entity.entity_id}].damage",
                ),
                buriedness=self._safe_non_negative_int(
                    entity.properties.get(int(URN.Property.BURIEDNESS)),
                    default=None,
                    warnings=warnings,
                    field_name=f"civilian[{entity.entity_id}].buriedness",
                ),
            )

        if mapped_type == EntityType.BLOCKADE:
            position_on_edge = self._safe_non_negative_int(
                entity.properties.get(int(URN.Property.POSITION)),
                default=None,
                warnings=warnings,
                field_name=f"blockade[{entity.entity_id}].position_on_edge",
            )
            return RawSensorData(
                repair_cost=self._safe_non_negative_int(
                    entity.properties.get(int(URN.Property.REPAIR_COST)),
                    default=None,
                    warnings=warnings,
                    field_name=f"blockade[{entity.entity_id}].repair_cost",
                ),
                position_on_edge=position_on_edge,
            )

        return RawSensorData()

    def _build_computed_metrics(
        self,
        observer_state: AgentState,
        entity: ServerEntity,
        mapped_type: EntityType,
        raw_data: RawSensorData,
    ) -> ComputedMetrics:
        """Computed metrics считаются до calculator.py, чтобы она получала уже нормализованный вход по схеме 3."""
        observer_area = observer_state.position.entity_id
        target_area = self._resolve_target_area(entity, mapped_type)

        path_distance = self.world_model.shortest_distance(observer_area, target_area)

        estimated_death_time: Optional[float] = None
        if mapped_type == EntityType.CIVILIAN:
            hp = raw_data.hp
            damage = raw_data.damage
            if hp is not None and damage is not None:
                if damage > 0:
                    estimated_death_time = float(hp) / float(damage)
                elif damage == 0:
                    estimated_death_time = math.inf

        total_area: Optional[float] = None
        if mapped_type == EntityType.BUILDING:
            if raw_data.ground_area is not None and raw_data.floors is not None:
                total_area = float(raw_data.ground_area * raw_data.floors)

        return ComputedMetrics(
            path_distance=path_distance,
            estimated_death_time=estimated_death_time,
            total_area=total_area,
        )

    def _resolve_target_area(self, entity: ServerEntity, mapped_type: EntityType) -> Optional[int]:
        """Target area-id нужен pathfinder'у, так как команды move/clear/extinguish выполняются в graph-координатах."""
        if mapped_type == EntityType.BUILDING:
            return entity.entity_id
        if mapped_type == EntityType.CIVILIAN:
            return self.world_model.resolve_area_position(entity.entity_id)
        if mapped_type == EntityType.BLOCKADE:
            area_ref = entity.properties.get(int(URN.Property.POSITION))
            try:
                return int(area_ref) if area_ref is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def _is_transporting(self, ambulance_id: int) -> bool:
        """Флаг transport вычисляется по POSITION гражданских, когда их position указывает на ID AmbulanceTeam."""
        for entity in self.world_model.entities.values():
            if entity.urn != int(URN.Entity.CIVILIAN):
                continue
            position_ref = entity.properties.get(int(URN.Property.POSITION))
            try:
                if position_ref is not None and int(position_ref) == int(ambulance_id):
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _safe_non_negative_int(
        self,
        value: object,
        default: Optional[int],
        warnings: List[str],
        field_name: str,
    ) -> Optional[int]:
        """Нецелевые/отрицательные значения помечаются warning, чтобы не искажать Utility-факторы."""
        if value is None:
            return default
        try:
            int_value = int(value)
            if int_value < 0:
                warnings.append(f"{field_name}: отрицательное значение {int_value}, заменено на {default}")
                return default
            return int_value
        except (TypeError, ValueError):
            warnings.append(f"{field_name}: нецелочисленное значение {value}, заменено на {default}")
            return default

    def _safe_float_non_negative(
        self,
        value: object,
        default: Optional[float],
        warnings: List[str],
        field_name: str,
    ) -> Optional[float]:
        """Float-валидация защищает f_urgency^fire от нечисловых/отрицательных температур."""
        if value is None:
            return default
        try:
            float_value = float(value)
            if float_value < 0:
                warnings.append(f"{field_name}: отрицательное значение {float_value}, заменено на {default}")
                return default
            return float_value
        except (TypeError, ValueError):
            warnings.append(f"{field_name}: нечисловое значение {value}, заменено на {default}")
            return default

    def _safe_int_in_range(
        self,
        value: object,
        low: int,
        high: int,
        default: Optional[int],
        warnings: List[str],
        field_name: str,
    ) -> Optional[int]:
        """Диапазонная проверка нужна для полей с конечным доменом (например Fieryness 1..8)."""
        safe_value = self._safe_non_negative_int(value=value, default=default, warnings=warnings, field_name=field_name)
        if safe_value is None:
            return default
        if safe_value < low or safe_value > high:
            warnings.append(f"{field_name}: значение {safe_value} вне диапазона [{low}, {high}], заменено на {default}")
            return default
        return safe_value
