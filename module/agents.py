from __future__ import annotations

from abc import ABC
from typing import List

from module.data_models import AgentState, AgentType, DecisionLogEntry, Position, UtilityBreakdown, VisibleEntity
from module.strategy import DecisionResult, TargetSelectionStrategy


class BaseAgent(ABC):
    """Абстрактный агент инкапсулирует общий цикл принятия решения и журналирование причин выбора."""

    expected_type: AgentType

    def __init__(self, state: AgentState, strategy: TargetSelectionStrategy):
        if state.type != self.expected_type:
            raise ValueError(
                f"Неверный тип состояния для {self.__class__.__name__}: "
                f"ожидался {self.expected_type.value}, получен {state.type.value}"
            )
        self.state = state
        self.strategy = strategy
        self.decision_log: List[DecisionLogEntry] = []
        self.last_utility_matrix: List[UtilityBreakdown] = []

    def step(
        self,
        tick: int,
        visible_entities: List[VisibleEntity],
        all_agents: List[AgentState],
        refuges: List[Position],
    ) -> DecisionResult:
        """Шаг агента повторяет алгоритм раздела 4 и сохраняет объяснимый след решения для UI."""
        decision = self.strategy.select_target(
            agent=self.state,
            visible_entities=visible_entities,
            all_agents=all_agents,
            refuges=refuges,
        )

        if decision.action in {"EXECUTE_TASK", "GO_TO_REFUGE"}:
            self.state.current_target_id = decision.selected_target_id
        elif decision.action in {"NO_TARGET", "ERROR"}:
            self.state.current_target_id = None

        self.last_utility_matrix = decision.utility_matrix
        self.decision_log.append(
            DecisionLogEntry(
                tick=tick,
                agent_id=self.state.id,
                action=decision.action,
                target_id=decision.selected_target_id,
                reason=decision.reason,
            )
        )
        return decision


class FireBrigadeAgent(BaseAgent):
    """Пожарный агент использует профиль задач BUILDING и формулы fire из раздела 2.2."""

    expected_type = AgentType.FIRE_BRIGADE


class AmbulanceTeamAgent(BaseAgent):
    """Медицинский агент использует TTL-модель и фильтрацию спасаемости из разделов 2.2 и 4."""

    expected_type = AgentType.AMBULANCE_TEAM


class PoliceForceAgent(BaseAgent):
    """Полицейский агент работает с BLOCKADE и формулой доступа к Targets U Refuges."""

    expected_type = AgentType.POLICE_FORCE
