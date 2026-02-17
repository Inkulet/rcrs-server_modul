from __future__ import annotations

import math
from typing import List, Optional, Tuple

from module.config import ModelConfig
from module.data_models import AgentState, AgentType, EntityType, Position, UtilityBreakdown, VisibleEntity


class UtilityCalculator:
    """Калькулятор реализует формулы из раздела 2.2 без изменения структуры весов и знаков."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.config.validate()

    def calculate_ttl(self, hp: Optional[int], damage: Optional[int]) -> float:
        """TTL считается строго по формуле 2.2 с отдельной обработкой Damage=0 -> inf."""
        try:
            if hp is None or hp <= 0:
                return 0.0
            if damage is None:
                return math.inf
            if damage > 0:
                return float(hp) / float(damage)
            if damage == 0:
                return math.inf
            return math.inf
        except (TypeError, ValueError, ZeroDivisionError):
            return math.inf

    def calculate_travel_time(self, distance: Optional[float]) -> float:
        """T_travel вводится явно, чтобы проверка TTL <= T_travel+T_work выполнялась по разделу 4."""
        try:
            if distance is None:
                return math.inf
            return max(0.0, distance) / self.config.constants.travel_speed
        except (TypeError, ValueError, ZeroDivisionError):
            return math.inf

    def calculate_work_time(self, buriedness: Optional[int]) -> float:
        """T_work = Buriedness / Rate по 2.2; деление защищено валидацией и try/except."""
        try:
            if buriedness is None:
                return 0.0
            return max(0.0, float(buriedness)) / self.config.constants.ambulance_clear_rate
        except (TypeError, ValueError, ZeroDivisionError):
            return math.inf

    def is_entity_relevant_for_agent(self, agent_type: AgentType, entity_type: EntityType) -> bool:
        """Функциональные ограничения 1-в-1: каждый тип агента работает только со своим типом задач."""
        if agent_type == AgentType.AMBULANCE_TEAM:
            return entity_type == EntityType.CIVILIAN
        if agent_type == AgentType.FIRE_BRIGADE:
            return entity_type == EntityType.BUILDING
        if agent_type == AgentType.POLICE_FORCE:
            return entity_type == EntityType.BLOCKADE
        return False

    def prefilter_candidate(
        self,
        agent: AgentState,
        entity: VisibleEntity,
    ) -> Tuple[bool, str]:
        """Pre-filter реализует правила раздела 4 до вычисления U_ij, чтобы исключить заведомо недопустимые цели."""
        if not self.is_entity_relevant_for_agent(agent.type, entity.type):
            return False, "Тип цели не соответствует специализации агента"

        try:
            raw = entity.raw_sensor_data
            metrics = entity.computed_metrics

            if agent.type == AgentType.AMBULANCE_TEAM:
                hp = raw.hp
                damage = raw.damage
                buriedness = raw.buriedness
                if hp is None or damage is None or buriedness is None:
                    return False, "Недостаточно данных сенсоров для Ambulance (HP/Damage/Buriedness)"
                if hp is not None and hp <= 0:
                    return False, "HP_i = 0: пострадавший мертв"
                if damage == 0 and (buriedness is None or buriedness == 0):
                    return False, "Damage_i = 0 и Buriedness_i = 0: задача неактуальна"

                ttl = self.calculate_ttl(hp, damage)
                travel_time = self.calculate_travel_time(metrics.path_distance)
                work_time = self.calculate_work_time(buriedness)
                if not math.isinf(ttl) and ttl <= (travel_time + work_time):
                    return False, "TTL_i <= T_travel + T_work: невозможно спасти"

            if agent.type == AgentType.FIRE_BRIGADE:
                fieryness = raw.fieryness
                if fieryness is None:
                    return False, "Отсутствует Fieryness для фильтрации здания"
                if fieryness in {4, 5, 6, 7, 8}:
                    return False, "Fieryness_i в {4..8}: исключено по разделу 4"

            if agent.type == AgentType.POLICE_FORCE:
                if raw.repair_cost is None:
                    return False, "Отсутствует RepairCost для расчета трудоемкости"
                if raw.repair_cost <= 0:
                    return False, "RepairCost <= 0: завал уже расчищен"

            return True, "OK"
        except Exception as error:  # noqa: BLE001
            return False, f"Ошибка pre-filter: {error}"

    def calculate_utility_breakdown(
        self,
        agent: AgentState,
        entity: VisibleEntity,
        all_agents: List[AgentState],
        all_visible_entities: List[VisibleEntity],
        refuges: List[Position],
    ) -> UtilityBreakdown:
        """Рассчитываем все факторы и итоговый U_ij для таблицы объяснимости в UI."""
        passed_prefilter, prefilter_reason = self.prefilter_candidate(agent, entity)

        if not passed_prefilter:
            return UtilityBreakdown(
                entity_id=entity.id,
                entity_type=entity.type,
                f_urgency=0.0,
                f_dist=0.0,
                f_effort=0.0,
                f_social=0.0,
                utility_score=float("-inf"),
                passed_prefilter=False,
                prefilter_reason=prefilter_reason,
            )

        f_urgency = self._calculate_urgency(agent, entity, all_visible_entities, refuges)
        f_dist = self._calculate_distance_factor(entity)
        f_effort = self._calculate_effort(agent, entity)
        f_social = self._calculate_social_factor(agent, entity, all_agents)

        weights = self.config.weights_by_agent[agent.type]
        utility_score = weights.w_c * f_urgency - (
            weights.w_d * f_dist + weights.w_e * f_effort + weights.w_n * f_social
        )

        return UtilityBreakdown(
            entity_id=entity.id,
            entity_type=entity.type,
            f_urgency=f_urgency,
            f_dist=f_dist,
            f_effort=f_effort,
            f_social=f_social,
            utility_score=utility_score,
            passed_prefilter=True,
            prefilter_reason="OK",
        )

    def _calculate_urgency(
        self,
        agent: AgentState,
        entity: VisibleEntity,
        all_visible_entities: List[VisibleEntity],
        refuges: List[Position],
    ) -> float:
        """Единая точка расчета f_urgency нужна для соблюдения разных формул для каждого типа агента."""
        try:
            if agent.type == AgentType.AMBULANCE_TEAM:
                return self._calculate_ambulance_urgency(entity)
            if agent.type == AgentType.FIRE_BRIGADE:
                return self._calculate_fire_urgency(entity)
            if agent.type == AgentType.POLICE_FORCE:
                return self._calculate_police_urgency(entity, all_visible_entities, refuges)
            return 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _calculate_ambulance_urgency(self, entity: VisibleEntity) -> float:
        """Формула f_urgency^amb реализована строго по кусочной функции из 2.2."""
        raw = entity.raw_sensor_data
        metrics = entity.computed_metrics

        ttl = self.calculate_ttl(raw.hp, raw.damage)
        travel_time = self.calculate_travel_time(metrics.path_distance)
        work_time = self.calculate_work_time(raw.buriedness)

        if math.isinf(ttl):
            return 0.01
        if ttl > (travel_time + work_time):
            return self._clip_01(self._safe_div(1.0, ttl, 0.0))
        return 0.0

    def _calculate_fire_urgency(self, entity: VisibleEntity) -> float:
        """Для пожарного учитывается только ранняя стадия (Fieryness in {1,2,3}) по 2.2."""
        raw = entity.raw_sensor_data
        fieryness = raw.fieryness
        temperature = raw.temperature

        if fieryness not in {1, 2, 3}:
            return 0.0
        if temperature is None:
            return 0.0

        normalized_temperature = self._safe_div(temperature, self.config.constants.temperature_max, 0.0)
        return self._clip_01(normalized_temperature)

    def _calculate_police_urgency(
        self,
        blockade_entity: VisibleEntity,
        all_visible_entities: List[VisibleEntity],
        refuges: List[Position],
    ) -> float:
        """Приоритет полиции: завал ближе к активным целям/убежищам имеет больший f_urgency (раздел 2.2)."""
        blockade_position = blockade_entity.position
        if blockade_position.x is None or blockade_position.y is None:
            return 0.0

        objective_positions: List[Position] = []
        for entity in all_visible_entities:
            # Исправление self-reference: блокада-кандидат не может быть объектом, к которому меряем доступ.
            if entity.id == blockade_entity.id:
                continue
            if entity.type == EntityType.BUILDING and entity.raw_sensor_data.fieryness in {1, 2, 3}:
                objective_positions.append(entity.position)
            if entity.type == EntityType.CIVILIAN:
                hp = entity.raw_sensor_data.hp
                damage = entity.raw_sensor_data.damage
                buriedness = entity.raw_sensor_data.buriedness
                is_active_casualty = (hp is None or hp > 0) and not (damage == 0 and (buriedness is None or buriedness == 0))
                if is_active_casualty:
                    objective_positions.append(entity.position)

        objective_positions.extend(refuges)

        distances = [
            self._euclidean_distance(blockade_position, objective_position)
            for objective_position in objective_positions
            if objective_position.x is not None and objective_position.y is not None
        ]
        if not distances:
            return 0.0

        min_distance = min(distances)
        value = self._safe_div(1.0, min_distance + self.config.constants.epsilon, 0.0)
        return self._clip_01(value)

    def _calculate_distance_factor(self, entity: VisibleEntity) -> float:
        """f_dist = d_ij / MaxMapDistance по 2.2 с fallback, если путь еще не построен."""
        distance = entity.computed_metrics.path_distance
        if distance is None:
            return 1.0
        normalized = self._safe_div(distance, self.config.constants.max_map_distance, 1.0)
        return self._clip_01(normalized)

    def _calculate_effort(self, agent: AgentState, entity: VisibleEntity) -> float:
        """f_effort зависит от типа агента и строго следует формулам раздела 2.2."""
        raw = entity.raw_sensor_data
        metrics = entity.computed_metrics

        if agent.type == AgentType.AMBULANCE_TEAM:
            buriedness = 0.0 if raw.buriedness is None else float(raw.buriedness)
            return self._clip_01(self._safe_div(buriedness, self.config.constants.max_buriedness, 1.0))

        if agent.type == AgentType.FIRE_BRIGADE:
            total_area = metrics.total_area
            if total_area is None:
                ground_area = 0.0 if raw.ground_area is None else float(raw.ground_area)
                floors = 0.0 if raw.floors is None else float(raw.floors)
                total_area = ground_area * floors
            return self._clip_01(self._safe_div(total_area, self.config.constants.max_total_area, 1.0))

        if agent.type == AgentType.POLICE_FORCE:
            repair_cost = 0.0 if raw.repair_cost is None else float(raw.repair_cost)
            return self._clip_01(self._safe_div(repair_cost, self.config.constants.max_repair_cost, 1.0))

        return 0.0

    def _calculate_social_factor(self, agent: AgentState, entity: VisibleEntity, all_agents: List[AgentState]) -> float:
        """f_social реализует N_i(r) из 2.2 и нормализацию к [0,1], как требует текст раздела."""
        target_position = entity.position
        if target_position.x is None or target_position.y is None:
            return 0.0

        same_type_agents = [candidate for candidate in all_agents if candidate.type == agent.type]
        max_neighbors = max(0, len(same_type_agents) - 1)
        if max_neighbors == 0:
            return 0.0

        neighbor_count = 0
        for candidate in same_type_agents:
            if candidate.id == agent.id:
                continue
            distance = self._euclidean_distance(candidate.position, target_position)
            if distance < self.config.constants.social_radius:
                neighbor_count += 1

        return self._clip_01(self._safe_div(float(neighbor_count), float(max_neighbors), 0.0))

    def _euclidean_distance(self, source: Position, target: Position) -> float:
        """Евклидова метрика используется как устойчивая аппроксимация ||pos - loc|| в формулах 2.2."""
        try:
            if source.x is None or source.y is None or target.x is None or target.y is None:
                return float("inf")
            dx = float(source.x) - float(target.x)
            dy = float(source.y) - float(target.y)
            return math.sqrt(dx * dx + dy * dy)
        except (TypeError, ValueError):
            return float("inf")

    @staticmethod
    def _safe_div(numerator: float, denominator: float, fallback: float) -> float:
        """Централизуем защиту от деления на ноль для TTL/дистанций/нормализации факторов."""
        try:
            if denominator == 0:
                return fallback
            return numerator / denominator
        except (TypeError, ZeroDivisionError, ValueError):
            return fallback

    @staticmethod
    def _clip_01(value: float) -> float:
        """Ограничиваем факторы диапазоном [0,1], как требует раздел 2.2."""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
