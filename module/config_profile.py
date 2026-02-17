from __future__ import annotations

from typing import Any, Dict

from module.config import ModelConfig, ModelConstants, UtilityWeights
from module.data_models import AgentType


def model_config_to_dict(config: ModelConfig) -> Dict[str, Any]:
    """Сериализация профиля нужна для UI и хранения пользовательских настроек без изменения формул модели."""
    return {
        "weights_by_agent": {
            agent_type.value: {
                "w_c": weights.w_c,
                "w_d": weights.w_d,
                "w_e": weights.w_e,
                "w_n": weights.w_n,
            }
            for agent_type, weights in config.weights_by_agent.items()
        },
        "constants": {
            "max_map_distance": config.constants.max_map_distance,
            "max_buriedness": config.constants.max_buriedness,
            "max_total_area": config.constants.max_total_area,
            "max_repair_cost": config.constants.max_repair_cost,
            "temperature_max": config.constants.temperature_max,
            "social_radius": config.constants.social_radius,
            "ambulance_clear_rate": config.constants.ambulance_clear_rate,
            "travel_speed": config.constants.travel_speed,
            "epsilon": config.constants.epsilon,
            "c_switch": config.constants.c_switch,
        },
    }


def model_config_from_dict(payload: Dict[str, Any]) -> ModelConfig:
    """Десериализация валидирует профиль, чтобы пользовательские изменения не ломали матмодель."""
    try:
        raw_weights = payload["weights_by_agent"]
        raw_constants = payload["constants"]

        weights_by_agent = {
            AgentType.FIRE_BRIGADE: UtilityWeights(
                w_c=float(raw_weights[AgentType.FIRE_BRIGADE.value]["w_c"]),
                w_d=float(raw_weights[AgentType.FIRE_BRIGADE.value]["w_d"]),
                w_e=float(raw_weights[AgentType.FIRE_BRIGADE.value]["w_e"]),
                w_n=float(raw_weights[AgentType.FIRE_BRIGADE.value]["w_n"]),
            ),
            AgentType.AMBULANCE_TEAM: UtilityWeights(
                w_c=float(raw_weights[AgentType.AMBULANCE_TEAM.value]["w_c"]),
                w_d=float(raw_weights[AgentType.AMBULANCE_TEAM.value]["w_d"]),
                w_e=float(raw_weights[AgentType.AMBULANCE_TEAM.value]["w_e"]),
                w_n=float(raw_weights[AgentType.AMBULANCE_TEAM.value]["w_n"]),
            ),
            AgentType.POLICE_FORCE: UtilityWeights(
                w_c=float(raw_weights[AgentType.POLICE_FORCE.value]["w_c"]),
                w_d=float(raw_weights[AgentType.POLICE_FORCE.value]["w_d"]),
                w_e=float(raw_weights[AgentType.POLICE_FORCE.value]["w_e"]),
                w_n=float(raw_weights[AgentType.POLICE_FORCE.value]["w_n"]),
            ),
        }

        constants = ModelConstants(
            max_map_distance=float(raw_constants["max_map_distance"]),
            max_buriedness=float(raw_constants["max_buriedness"]),
            max_total_area=float(raw_constants["max_total_area"]),
            max_repair_cost=float(raw_constants["max_repair_cost"]),
            temperature_max=float(raw_constants["temperature_max"]),
            social_radius=float(raw_constants["social_radius"]),
            ambulance_clear_rate=float(raw_constants["ambulance_clear_rate"]),
            travel_speed=float(raw_constants.get("travel_speed", 1.0)),
            epsilon=float(raw_constants.get("epsilon", 1e-6)),
            c_switch=float(raw_constants.get("c_switch", 0.0)),
        )

        config = ModelConfig(weights_by_agent=weights_by_agent, constants=constants)
        config.validate()
        return config
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"Некорректный формат профиля конфигурации: {error}") from error
