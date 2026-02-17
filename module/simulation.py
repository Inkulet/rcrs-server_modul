from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from module.agents import BaseAgent
from module.data_models import AgentOperationalState, AgentState, AgentType, EntityType, Position, VisibleEntity
from module.strategy import DecisionResult


class SimulationEngine:
    """Движок управляет тиками, чтобы агенты и UI работали параллельно и читали согласованное состояние."""

    def __init__(self, agents: Dict[int, BaseAgent], visible_entities: List[VisibleEntity], refuges: List[Position]):
        self._agents = agents
        self._visible_entities = visible_entities
        self._refuges = refuges
        self._tick = 0
        self._last_decisions: Dict[int, DecisionResult] = {}
        self._lock = threading.RLock()

    @property
    def tick(self) -> int:
        """Текущий тик нужен для журнала решений и синхронизации UI."""
        with self._lock:
            return self._tick

    def get_agent_ids(self) -> List[int]:
        """UI использует этот список для выбора агента, не зная внутренней структуры движка."""
        with self._lock:
            return sorted(self._agents.keys())

    def get_agent_state(self, agent_id: int) -> AgentState:
        """Возвращаем копию, чтобы UI не модифицировал состояние симуляции напрямую."""
        with self._lock:
            return copy.deepcopy(self._agents[agent_id].state)

    def get_agent_decision_log(self, agent_id: int, limit: int = 30) -> List[Dict[str, object]]:
        """Лог ограничивается по длине, чтобы интерфейс оставался отзывчивым при длительном запуске."""
        with self._lock:
            history = self._agents[agent_id].decision_log[-limit:]
            return [asdict(entry) for entry in history]

    def get_agent_utility_matrix(self, agent_id: int) -> List[Dict[str, object]]:
        """Матрица полезности показывается в UI как основной артефакт объяснимого выбора цели."""
        with self._lock:
            rows = []
            for breakdown in self._agents[agent_id].last_utility_matrix:
                rows.append(
                    {
                        "entity_id": breakdown.entity_id,
                        "entity_type": breakdown.entity_type.value,
                        "f_urgency": round(breakdown.f_urgency, 6),
                        "f_dist": round(breakdown.f_dist, 6),
                        "f_effort": round(breakdown.f_effort, 6),
                        "f_social": round(breakdown.f_social, 6),
                        "U_ij": round(breakdown.utility_score, 6) if math.isfinite(breakdown.utility_score) else "-inf",
                        "passed_prefilter": breakdown.passed_prefilter,
                        "prefilter_reason": breakdown.prefilter_reason,
                    }
                )
            return rows

    def get_visible_entities_snapshot(self) -> List[Dict[str, object]]:
        """Снимок visible-целей нужен UI для проверки pre-filter правил и демонстрации сенсорных полей 2.2."""
        with self._lock:
            result: List[Dict[str, object]] = []
            for entity in self._visible_entities:
                serialized = asdict(entity)
                serialized["type"] = entity.type.value
                result.append(serialized)
            return result

    def step(self) -> None:
        """Один тик: обновляем метрики восприятия, выбираем цели и применяем эффекты действий."""
        with self._lock:
            self._tick += 1
            all_states = [copy.deepcopy(agent.state) for agent in self._agents.values()]
            decisions_by_agent: Dict[int, DecisionResult] = {}

            for agent_id in sorted(self._agents.keys()):
                agent = self._agents[agent_id]
                perception = copy.deepcopy(self._visible_entities)
                self._recalculate_computed_metrics(agent.state, perception)
                decision = agent.step(
                    tick=self._tick,
                    visible_entities=perception,
                    all_agents=all_states,
                    refuges=self._refuges,
                )
                decisions_by_agent[agent_id] = decision

            self._last_decisions = decisions_by_agent
            for agent_id in sorted(self._agents.keys()):
                self._apply_decision(self._agents[agent_id], decisions_by_agent[agent_id])

            self._advance_world_state()

    def run_background(self, stop_event: threading.Event, interval_seconds: float = 1.0) -> None:
        """Фоновый цикл позволяет UI рендериться независимо от шагов агентов (параллельный режим)."""
        while not stop_event.is_set():
            self.step()
            time.sleep(interval_seconds)

    def _recalculate_computed_metrics(self, agent_state: AgentState, visible_entities: List[VisibleEntity]) -> None:
        """Производные метрики пересчитываются каждый тик, чтобы U_ij опирался на актуальное восприятие."""
        for entity in visible_entities:
            entity.computed_metrics.path_distance = self._distance(agent_state.position, entity.position)
            if entity.type == EntityType.CIVILIAN:
                hp = entity.raw_sensor_data.hp
                damage = entity.raw_sensor_data.damage
                entity.computed_metrics.estimated_death_time = self._ttl(hp, damage)
            if entity.type == EntityType.BUILDING:
                ground_area = 0 if entity.raw_sensor_data.ground_area is None else entity.raw_sensor_data.ground_area
                floors = 0 if entity.raw_sensor_data.floors is None else entity.raw_sensor_data.floors
                entity.computed_metrics.total_area = float(ground_area * floors)

    def _apply_decision(self, agent: BaseAgent, decision: DecisionResult) -> None:
        """Изменяем состояние агента и мира детерминированно, чтобы воспроизводимость не зависела от random()."""
        if decision.action == "GO_TO_REFUGE":
            agent.state.state = AgentOperationalState.GOING_TO_REFUGE
            # В демонстрационном модуле пополнение воды делается мгновенно, чтобы показать ветку Water=0 из раздела 4.
            if agent.state.type == AgentType.FIRE_BRIGADE:
                agent.state.resources.water_quantity = max(agent.state.resources.water_quantity, 5000)
            agent.state.state = AgentOperationalState.IDLE
            return

        if decision.action == "EXECUTE_TASK":
            agent.state.state = AgentOperationalState.WORKING
            target = self._find_entity_by_id(decision.selected_target_id)
            if target is not None:
                self._apply_task_effect(agent, target)
            agent.state.state = AgentOperationalState.IDLE
            return

        if decision.action in {"NO_TARGET", "SKIP_BUSY", "ERROR"}:
            if decision.action != "SKIP_BUSY":
                agent.state.state = AgentOperationalState.IDLE

    def _apply_task_effect(self, agent: BaseAgent, target: VisibleEntity) -> None:
        """Эффекты задач упрощены, но сохраняют причинно-следственную связь между выбором цели и состоянием мира."""
        if agent.state.type == AgentType.FIRE_BRIGADE and target.type == EntityType.BUILDING:
            if target.raw_sensor_data.temperature is not None:
                target.raw_sensor_data.temperature = max(0.0, target.raw_sensor_data.temperature - 120.0)
            if target.raw_sensor_data.temperature == 0 and target.raw_sensor_data.fieryness in {1, 2, 3}:
                target.raw_sensor_data.fieryness = 4
            agent.state.resources.water_quantity = max(0, agent.state.resources.water_quantity - 250)

        if agent.state.type == AgentType.AMBULANCE_TEAM and target.type == EntityType.CIVILIAN:
            if target.raw_sensor_data.buriedness is not None:
                target.raw_sensor_data.buriedness = max(0, target.raw_sensor_data.buriedness - 12)

        if agent.state.type == AgentType.POLICE_FORCE and target.type == EntityType.BLOCKADE:
            if target.raw_sensor_data.repair_cost is not None:
                target.raw_sensor_data.repair_cost = max(0, target.raw_sensor_data.repair_cost - 80)
                if target.raw_sensor_data.repair_cost == 0:
                    self._visible_entities = [entity for entity in self._visible_entities if entity.id != target.id]

    def _advance_world_state(self) -> None:
        """Мир эволюционирует детерминированно: у пострадавших падает HP, активные пожары нагреваются."""
        for entity in self._visible_entities:
            if entity.type == EntityType.CIVILIAN:
                hp = entity.raw_sensor_data.hp
                damage = entity.raw_sensor_data.damage
                if hp is not None and damage is not None and hp > 0 and damage > 0:
                    entity.raw_sensor_data.hp = max(0, hp - damage)
            if entity.type == EntityType.BUILDING:
                fieryness = entity.raw_sensor_data.fieryness
                temperature = entity.raw_sensor_data.temperature
                if fieryness in {1, 2, 3} and temperature is not None:
                    entity.raw_sensor_data.temperature = min(1200.0, temperature + 20.0)

    def _find_entity_by_id(self, entity_id: Optional[int]) -> Optional[VisibleEntity]:
        """Линейный поиск достаточен, потому что список видимых целей мал и пересобирается каждый тик."""
        if entity_id is None:
            return None
        for entity in self._visible_entities:
            if entity.id == entity_id:
                return entity
        return None

    @staticmethod
    def _ttl(hp: Optional[int], damage: Optional[int]) -> float:
        """Локальный TTL нужен для метрики estimated_death_time и повторяет формулу из раздела 2.2."""
        try:
            if hp is None:
                return math.inf
            if damage is None:
                return math.inf
            if damage > 0:
                return hp / damage
            if damage == 0:
                return math.inf
            return math.inf
        except (TypeError, ZeroDivisionError):
            return math.inf

    @staticmethod
    def _distance(source: Position, target: Position) -> float:
        """Защита от пустых координат предотвращает падения при неполных сенсорных пакетах."""
        try:
            if source.x is None or source.y is None or target.x is None or target.y is None:
                return float("inf")
            dx = float(source.x) - float(target.x)
            dy = float(source.y) - float(target.y)
            return math.sqrt(dx * dx + dy * dy)
        except (TypeError, ValueError):
            return float("inf")
