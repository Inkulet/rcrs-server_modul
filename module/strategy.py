from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from module.calculator import UtilityCalculator
from module.data_models import AgentOperationalState, AgentState, AgentType, Position, UtilityBreakdown, VisibleEntity


@dataclass
class DecisionResult:
    """Результат выбора цели отделён от агента, чтобы UI и движок использовали единый формат данных."""

    action: str
    selected_target_id: Optional[int]
    reason: str
    utility_matrix: List[UtilityBreakdown]


class TargetSelectionStrategy(ABC):
    """Strategy-паттерн позволяет менять политику выбора цели без изменения классов агентов."""

    @abstractmethod
    def select_target(
        self,
        agent: AgentState,
        visible_entities: List[VisibleEntity],
        all_agents: List[AgentState],
        refuges: List[Position],
    ) -> DecisionResult:
        """Возвращает выбранную цель и матрицу полезности для объяснимого принятия решений."""


class UtilityBasedTargetSelectionStrategy(TargetSelectionStrategy):
    """Базовая стратегия реализует алгоритм Sensing->PreFilter->Calculation->Selection->Action (раздел 4)."""

    BUSY_STATES = {
        AgentOperationalState.LOADING,
        AgentOperationalState.UNLOADING,
        AgentOperationalState.TRANSPORTING,
    }

    def __init__(self, calculator: UtilityCalculator):
        self.calculator = calculator

    def select_target(
        self,
        agent: AgentState,
        visible_entities: List[VisibleEntity],
        all_agents: List[AgentState],
        refuges: List[Position],
    ) -> DecisionResult:
        """Шаг Selection учитывает гистерезис C_switch, чтобы уменьшить дергание целей между тиками."""
        try:
            if agent.state in self.BUSY_STATES:
                return DecisionResult(
                    action="SKIP_BUSY",
                    selected_target_id=agent.current_target_id,
                    reason="Статус агента в {Loading, Unloading, Transporting}: шаг выбора пропущен",
                    utility_matrix=[],
                )

            if agent.type == AgentType.FIRE_BRIGADE and agent.resources.water_quantity <= 0:
                refuge_target = self._choose_nearest_refuge(agent.position, refuges)
                return DecisionResult(
                    action="GO_TO_REFUGE",
                    selected_target_id=refuge_target.entity_id if refuge_target else None,
                    reason="Water = 0: согласно разделу 4 выбрана команда GoToRefuge",
                    utility_matrix=[],
                )

            relevant_entities = [
                entity
                for entity in visible_entities
                if self.calculator.is_entity_relevant_for_agent(agent.type, entity.type)
            ]

            utility_matrix = [
                self.calculator.calculate_utility_breakdown(
                    agent=agent,
                    entity=entity,
                    all_agents=all_agents,
                    all_visible_entities=visible_entities,
                    refuges=refuges,
                )
                for entity in relevant_entities
            ]

            passed_candidates = [candidate for candidate in utility_matrix if candidate.passed_prefilter]
            if not passed_candidates:
                return DecisionResult(
                    action="NO_TARGET",
                    selected_target_id=None,
                    reason="После Pre-Filtering не осталось допустимых задач",
                    utility_matrix=utility_matrix,
                )

            best_candidate = sorted(
                passed_candidates,
                key=lambda item: (-item.utility_score, item.entity_id),
            )[0]

            selected_candidate = best_candidate
            hysteresis_reason = "Выбрана цель с максимальным U_ij"
            if agent.current_target_id is not None:
                current_candidate = next(
                    (candidate for candidate in passed_candidates if candidate.entity_id == agent.current_target_id),
                    None,
                )
                if current_candidate is not None:
                    gain = best_candidate.utility_score - current_candidate.utility_score
                    if gain < self.calculator.config.constants.c_switch:
                        selected_candidate = current_candidate
                        hysteresis_reason = (
                            f"Сработал гистерезис C_switch={self.calculator.config.constants.c_switch:.4f}, "
                            f"прирост U_ij={gain:.4f}"
                        )

            reason = (
                f"{hysteresis_reason}; target={selected_candidate.entity_id}, "
                f"U={selected_candidate.utility_score:.4f}, "
                f"f_urgency={selected_candidate.f_urgency:.4f}, "
                f"f_dist={selected_candidate.f_dist:.4f}, "
                f"f_effort={selected_candidate.f_effort:.4f}, "
                f"f_social={selected_candidate.f_social:.4f}"
            )
            return DecisionResult(
                action="EXECUTE_TASK",
                selected_target_id=selected_candidate.entity_id,
                reason=reason,
                utility_matrix=utility_matrix,
            )
        except Exception as error:  # noqa: BLE001
            return DecisionResult(
                action="ERROR",
                selected_target_id=None,
                reason=f"Ошибка выбора цели: {error}",
                utility_matrix=[],
            )

    @staticmethod
    def _choose_nearest_refuge(agent_position: Position, refuges: List[Position]) -> Optional[Position]:
        """Для GoToRefuge выбираем ближайшее убежище детерминированно по расстоянию и ID."""
        if not refuges:
            return None

        safe_refuges = [refuge for refuge in refuges if refuge.x is not None and refuge.y is not None]
        if not safe_refuges:
            return None

        def distance_to_refuge(refuge: Position) -> float:
            if agent_position.x is None or agent_position.y is None:
                return float("inf")
            dx = float(agent_position.x) - float(refuge.x)
            dy = float(agent_position.y) - float(refuge.y)
            return dx * dx + dy * dy

        return sorted(
            safe_refuges,
            key=lambda refuge: (distance_to_refuge(refuge), refuge.entity_id if refuge.entity_id is not None else 10**9),
        )[0]
