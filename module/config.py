from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from module.data_models import AgentType


@dataclass(frozen=True)
class UtilityWeights:
    """Веса из формулы U_ij (раздел 2.2) задаются извне, чтобы не зашивать произвольные коэффициенты в код."""

    w_c: float
    w_d: float
    w_e: float
    w_n: float

    def validate(self) -> None:
        """Проверка нужна для раннего отказа, если конфигурация нарушает предпосылки модели 2.2."""
        if self.w_c < 0 or self.w_d < 0 or self.w_e < 0 or self.w_n < 0:
            raise ValueError("Коэффициенты полезности должны быть неотрицательными.")


@dataclass(frozen=True)
class ModelConstants:
    """Константы нормализации и временных расчётов из раздела 2.2."""

    max_map_distance: float
    max_buriedness: float
    max_total_area: float
    max_repair_cost: float
    temperature_max: float
    social_radius: float
    ambulance_clear_rate: float
    travel_speed: float = 1.0
    epsilon: float = 1e-6
    c_switch: float = 0.0

    def validate(self) -> None:
        """Ограничения защищают формулы от деления на ноль и некорректных диапазонов."""
        if self.max_map_distance <= 0:
            raise ValueError("max_map_distance должен быть > 0.")
        if self.max_buriedness <= 0:
            raise ValueError("max_buriedness должен быть > 0.")
        if self.max_total_area <= 0:
            raise ValueError("max_total_area должен быть > 0.")
        if self.max_repair_cost <= 0:
            raise ValueError("max_repair_cost должен быть > 0.")
        if self.temperature_max <= 0:
            raise ValueError("temperature_max должен быть > 0.")
        if self.social_radius <= 0:
            raise ValueError("social_radius должен быть > 0.")
        if self.ambulance_clear_rate <= 0:
            raise ValueError("ambulance_clear_rate должен быть > 0.")
        if self.travel_speed <= 0:
            raise ValueError("travel_speed должен быть > 0.")
        if self.epsilon <= 0:
            raise ValueError("epsilon должен быть > 0.")
        if self.c_switch < 0:
            raise ValueError("c_switch не может быть отрицательным.")


@dataclass(frozen=True)
class ModelConfig:
    """Объединяем веса и константы в единую конфигурацию для детерминированного расчёта U_ij."""

    weights_by_agent: Dict[AgentType, UtilityWeights]
    constants: ModelConstants

    def validate(self) -> None:
        """Явная валидация гарантирует, что каждый тип агента использует свою формулу с корректными весами."""
        for agent_type in AgentType:
            if agent_type not in self.weights_by_agent:
                raise ValueError(f"Для {agent_type.value} не задан набор весов.")
            self.weights_by_agent[agent_type].validate()
        self.constants.validate()
