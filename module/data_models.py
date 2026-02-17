from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(str, Enum):
    """Типы агентов фиксированы документацией (раздел 3, JSON schema)."""

    FIRE_BRIGADE = "FIRE_BRIGADE"
    AMBULANCE_TEAM = "AMBULANCE_TEAM"
    POLICE_FORCE = "POLICE_FORCE"


class EntityType(str, Enum):
    """Типы задач из схемы данных 2.2; REFUGE добавлен как сервисная цель для шага Water=0."""

    BUILDING = "BUILDING"
    CIVILIAN = "CIVILIAN"
    BLOCKADE = "BLOCKADE"
    REFUGE = "REFUGE"


class AgentOperationalState(str, Enum):
    """Статусы включают значения из фильтрации алгоритма выбора цели (раздел 4)."""

    IDLE = "IDLE"
    MOVING = "MOVING"
    WORKING = "WORKING"
    LOADING = "LOADING"
    UNLOADING = "UNLOADING"
    TRANSPORTING = "TRANSPORTING"
    GOING_TO_REFUGE = "GOING_TO_REFUGE"


@dataclass
class Position:
    """Координаты храним явно, чтобы корректно считать расстояния в f_dist и f_social."""

    entity_id: Optional[int]
    x: Optional[int]
    y: Optional[int]


@dataclass
class Resources:
    """Ресурсы агента соответствуют схеме, где критичен water_quantity для пожарного фильтра."""

    water_quantity: int = 0
    is_transporting: bool = False


@dataclass
class AgentState:
    """Состояние агента повторяет JSON-модель из раздела 3 и хранит текущую цель для гистерезиса."""

    id: int
    type: AgentType
    position: Position
    state: AgentOperationalState
    resources: Resources
    current_target_id: Optional[int] = None


@dataclass
class RawSensorData:
    """Поля сенсоров optional, потому что в RCRS видимость неполная и данные могут отсутствовать."""

    hp: Optional[int] = None
    damage: Optional[int] = None
    buriedness: Optional[int] = None
    temperature: Optional[float] = None
    fieryness: Optional[int] = None
    floors: Optional[int] = None
    ground_area: Optional[int] = None
    repair_cost: Optional[int] = None
    position_on_edge: Optional[int] = None


@dataclass
class ComputedMetrics:
    """Производные метрики выделены отдельно по схеме 2.2 для прозрачности расчёта U_ij."""

    path_distance: Optional[float] = None
    estimated_death_time: Optional[float] = None
    total_area: Optional[float] = None


@dataclass
class VisibleEntity:
    """Цель агента в едином формате позволяет строить матрицу полезности в UI без специальных веток."""

    id: int
    type: EntityType
    position: Position
    raw_sensor_data: RawSensorData = field(default_factory=RawSensorData)
    computed_metrics: ComputedMetrics = field(default_factory=ComputedMetrics)
    utility_score: Optional[float] = None


@dataclass
class AgentObservation:
    """Корневой контейнер один-в-один повторяет раздел 3: agent_state + visible_entities."""

    agent_state: AgentState
    visible_entities: List[VisibleEntity]

    def to_schema_dict(self) -> Dict[str, Any]:
        """Сериализация нужна для отладки соответствия JSON-схеме раздела 3."""
        return asdict(self)


@dataclass
class DecisionLogEntry:
    """Лог фиксирует причину решения, чтобы UI показывал объяснимость выбора цели."""

    tick: int
    agent_id: int
    action: str
    target_id: Optional[int]
    reason: str


@dataclass
class UtilityBreakdown:
    """Расшифровка факторов для UI-матрицы полезности по требованиям задачи."""

    entity_id: int
    entity_type: EntityType
    f_urgency: float
    f_dist: float
    f_effort: float
    f_social: float
    utility_score: float
    passed_prefilter: bool
    prefilter_reason: str


def parse_agent_state(payload: Dict[str, Any]) -> AgentState:
    """Парсер отделён, чтобы аккуратно обработать неполные сообщения от сенсоров через try/except."""
    try:
        position_data = payload.get("position", {}) or {}
        resources_data = payload.get("resources", {}) or {}
        return AgentState(
            id=int(payload["id"]),
            type=AgentType(payload["type"]),
            position=Position(
                entity_id=_safe_int(position_data.get("entity_id")),
                x=_safe_int(position_data.get("x")),
                y=_safe_int(position_data.get("y")),
            ),
            state=AgentOperationalState(payload.get("state", AgentOperationalState.IDLE.value)),
            resources=Resources(
                water_quantity=int(resources_data.get("water_quantity", 0) or 0),
                is_transporting=bool(resources_data.get("is_transporting", False)),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Некорректный формат agent_state: {error}") from error


def parse_visible_entity(payload: Dict[str, Any]) -> VisibleEntity:
    """Защищаемся от Null/отсутствующих полей, потому что это штатный режим неполной наблюдаемости."""
    try:
        sensor_data = payload.get("raw_sensor_data", {}) or {}
        metrics_data = payload.get("computed_metrics", {}) or {}
        position_data = payload.get("position", {}) or {}
        return VisibleEntity(
            id=int(payload["id"]),
            type=EntityType(payload["type"]),
            position=Position(
                entity_id=_safe_int(position_data.get("entity_id")),
                x=_safe_int(position_data.get("x")),
                y=_safe_int(position_data.get("y")),
            ),
            raw_sensor_data=RawSensorData(
                hp=_safe_int(sensor_data.get("hp")),
                damage=_safe_int(sensor_data.get("damage")),
                buriedness=_safe_int(sensor_data.get("buriedness")),
                temperature=_safe_float(sensor_data.get("temperature")),
                fieryness=_safe_int(sensor_data.get("fieryness")),
                floors=_safe_int(sensor_data.get("floors")),
                ground_area=_safe_int(sensor_data.get("ground_area")),
                repair_cost=_safe_int(sensor_data.get("repair_cost")),
                position_on_edge=_safe_int(sensor_data.get("position_on_edge")),
            ),
            computed_metrics=ComputedMetrics(
                path_distance=_safe_float(metrics_data.get("path_distance")),
                estimated_death_time=_safe_float(metrics_data.get("estimated_death_time")),
                total_area=_safe_float(metrics_data.get("total_area")),
            ),
            utility_score=_safe_float(payload.get("utility_score")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Некорректный формат visible_entity: {error}") from error


def _safe_int(value: Any) -> Optional[int]:
    """Возвращаем None вместо исключения, чтобы агент продолжал работу при частично битых данных."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    """Аналог safe-int для дробных полей сенсоров и производных метрик."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
