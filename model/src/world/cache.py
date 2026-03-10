from __future__ import annotations

"""В этом модуле я реализую in-memory модель мира для симуляции спасателей."""

import logging
from typing import Dict, Iterable, Optional

import networkx as nx

from .entities import AgentState, VisibleEntity


logger = logging.getLogger(__name__)


class WorldModel:
    """В этом классе я описываю центральный кэш мира с доступом за O(1)."""

    def __init__(self) -> None:
        """Здесь я инициализирую хранилища агентов, задач и граф дорожной сети."""

        self.agents: Dict[int, AgentState] = {}
        self.tasks: Dict[int, VisibleEntity] = {}
        self.road_graph: nx.Graph = nx.Graph()
        logger.info("Я инициализировал пустую модель мира и граф дорожной сети")

    def add_road_node(self, entity_id: int, **attrs: object) -> None:
        """Здесь я добавляю вершину в граф дорожной сети."""

        self.road_graph.add_node(entity_id, **attrs)

    def add_road_edge(self, source_id: int, target_id: int, weight: float, **attrs: object) -> None:
        """Здесь я добавляю ребро с весом для расчета дистанций."""

        self.road_graph.add_edge(source_id, target_id, weight=weight, **attrs)

    def set_agent(self, agent: AgentState) -> None:
        """Здесь я сохраняю состояние агента с доступом по его идентификатору."""

        self.agents[agent.id] = agent

    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        """Здесь я возвращаю агента по идентификатору, если он известен."""

        return self.agents.get(agent_id)

    def get_task(self, entity_id: int) -> Optional[VisibleEntity]:
        """Здесь я возвращаю задачу по идентификатору сущности, если она известна."""

        return self.tasks.get(entity_id)

    def update_perception(self, visible_entities: Iterable[VisibleEntity]) -> None:
        """В этом методе я реализую обновление модели мира на основе новых наблюдений."""

        for entity in visible_entities:
            if entity.id not in self.road_graph:
                logger.error(
                    "Я получил сенсорные данные для entity_id=%s, которого нет в графе дорожной сети",
                    entity.id,
                )

            existing = self.tasks.get(entity.id)
            if existing is None:
                self.tasks[entity.id] = entity
                logger.info("Я добавил новую сущность в кэш: entity_id=%s", entity.id)
                continue

            updated_hp = entity.raw_sensor_data.hp
            if updated_hp is None:
                updated_hp = existing.raw_sensor_data.hp

            updated_temperature = entity.raw_sensor_data.temperature
            if updated_temperature is None:
                updated_temperature = existing.raw_sensor_data.temperature

            updated_damage = entity.raw_sensor_data.damage
            if updated_damage is None:
                updated_damage = existing.raw_sensor_data.damage

            merged_raw = entity.raw_sensor_data.model_copy(
                update={
                    "hp": updated_hp,
                    "temperature": updated_temperature,
                    "damage": updated_damage,
                }
            )

            merged_entity = existing.model_copy(
                update={
                    "type": entity.type,
                    "raw_sensor_data": merged_raw,
                    "computed_metrics": entity.computed_metrics,
                    "utility_score": entity.utility_score,
                }
            )

            self.tasks[entity.id] = merged_entity
            logger.info("Я обновил сущность в кэше: entity_id=%s", entity.id)


__all__ = ["WorldModel"]
