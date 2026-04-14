from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import networkx as nx

from .entities import (
    AgentState, EntityType, MapEdge, MapNode, PerceptionPacket, VisibleEntity,
    estimate_death_time, compute_total_area,
)


logger = logging.getLogger(__name__)

# Я держу завал в кэше не более этого числа тактов без его появления в
# visible_entities. Сервер StandardPerception.java:236-247 обновляет
# road.blockades и свойства блокад, но НЕ посылает entityDeleted в перцепции
# агента — когда ClearSimulator полностью расчищает завал, Python никогда не
# получает явного сигнала удаления. Если 3 такта подряд сервер не присылает
# свойств для блокады, считаю что он её удалил (или я вышел из зоны видимости,
# тогда при возврате завал появится заново в update_perception).
_BLOCKADE_STALE_TICKS: int = 3


class WorldModel:
    def __init__(self) -> None:
        self.agents: Dict[int, AgentState] = {}
        self.tasks: Dict[int, VisibleEntity] = {}
        # Я храню номер такта, на котором каждая сущность последний раз пришла
        # в visible_entities. Это единственный надёжный сигнал "сервер удалил
        # завал" — явного entityDeleted в перцепции нет.
        self.last_seen_tick: Dict[int, int] = {}
        self.road_graph: nx.Graph = nx.Graph()

        self.refuge_ids: list[int] = []
        logger.info("Я инициализировал пустую модель мира и граф дорожной сети")

    def add_road_node(self, entity_id: int, **attrs: object) -> None:
        self.road_graph.add_node(entity_id, **attrs)

    def add_road_edge(self, source_id: int, target_id: int, weight: float, **attrs: object) -> None:
        self.road_graph.add_edge(source_id, target_id, weight=weight, **attrs)

    def set_agent(self, agent: AgentState) -> None:
        self.agents[agent.id] = agent

    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        return self.agents.get(agent_id)

    def get_nearest_node(self, x: int, y: int) -> int:
        import math

        best_node = -1
        min_dist = float("inf")

        for node_id, attrs in self.road_graph.nodes(data=True):
            nx, ny = attrs.get("x", 0), attrs.get("y", 0)
            dist = math.hypot(nx - x, ny - y)
            if dist < min_dist:
                min_dist = dist
                best_node = node_id

        return best_node

    def build_graph_from_map(
        self,
        nodes: Iterable[MapNode],
        edges: Iterable[MapEdge],
    ) -> None:
        node_count = 0
        for node in nodes:
            self.road_graph.add_node(node.entity_id, x=node.x, y=node.y)
            node_count += 1

        edge_count = 0
        for edge in edges:
            self.road_graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
            edge_count += 1

        logger.info(
            "Я построил дорожный граф: %d вершин, %d рёбер",
            node_count,
            edge_count,
        )

    def update_agents(self, ally_states: Iterable[AgentState]) -> None:
        for ally in ally_states:
            self.agents[ally.id] = ally
            logger.debug("Я обновил состояние союзника agent_id=%s", ally.id)

    def apply_perception(self, packet: PerceptionPacket) -> None:
        if packet.map_nodes or packet.map_edges:
            self.build_graph_from_map(packet.map_nodes, packet.map_edges)

        if packet.refuge_ids:
            self.refuge_ids = list(packet.refuge_ids)
            logger.info("Я сохранил %d убежищ: %s", len(self.refuge_ids), self.refuge_ids)

        self.agents.clear()
        self.update_agents(packet.ally_states)

        for eid in packet.deleted_entity_ids:
            if eid in self.tasks:
                logger.info("Я удалил сущность entity_id=%d из кэша (ядро удалило из ChangeSet)", eid)
                del self.tasks[eid]
            self.last_seen_tick.pop(eid, None)

        self.update_perception(packet.visible_entities)

        # Я фиксирую «этого увидел на такте N» для каждой пришедшей сущности.
        # Трекинг нужен только для выявления «тихих» удалений завалов ниже.
        for entity in packet.visible_entities:
            self.last_seen_tick[entity.id] = packet.tick

        # Я удаляю завалы, которых сервер не присылал _BLOCKADE_STALE_TICKS
        # тактов подряд — значит ClearSimulator их полностью расчистил
        # (model.removeEntity), но нам он об этом не сообщает явно (см. комментарий
        # у константы). Если я просто вышел из зоны видимости, завал появится
        # снова при возвращении. Для других типов (civilian/human/building)
        # чистку не делаю — их надо помнить для пересечения зон видимости.
        stale_threshold = packet.tick - _BLOCKADE_STALE_TICKS
        stale_blockades = [
            eid for eid, entity in self.tasks.items()
            if entity.type == EntityType.BLOCKADE
            and self.last_seen_tick.get(eid, packet.tick) < stale_threshold
        ]
        for eid in stale_blockades:
            logger.info(
                "Я удалил завал entity_id=%d из кэша: %d тактов без обновлений "
                "(сервер удалил после полной расчистки)",
                eid, packet.tick - self.last_seen_tick.get(eid, packet.tick),
            )
            del self.tasks[eid]
            self.last_seen_tick.pop(eid, None)

        logger.debug(
            "Я применил пакет восприятия такта %d: %d сущностей, %d союзников, "
            "выбраковано завалов=%d",
            packet.tick,
            len(packet.visible_entities),
            len(packet.ally_states),
            len(stale_blockades),
        )

    def update_perception(self, visible_entities: Iterable[VisibleEntity]) -> None:
        for entity in visible_entities:
            if entity.id not in self.road_graph:
                logger.debug(
                    "Я получил сенсорные данные для entity_id=%s, которого нет в графе дорожной сети "
                    "(ожидаемо для гражданских — они не являются узлами дорожного графа)",
                    entity.id,
                )

            existing = self.tasks.get(entity.id)
            if existing is None:
                self.tasks[entity.id] = entity
                logger.info("Я добавил новую сущность в кэш: entity_id=%s", entity.id)
                continue

            def _keep(
                new_val: int | float | None,
                old_val: int | float | None,
            ) -> int | float | None:
                return new_val if new_val is not None else old_val

            new_raw = entity.raw_sensor_data
            old_raw = existing.raw_sensor_data

            merged_raw = new_raw.model_copy(
                update={
                    "hp":              _keep(new_raw.hp,              old_raw.hp),
                    "damage":          _keep(new_raw.damage,          old_raw.damage),
                    "buriedness":      _keep(new_raw.buriedness,      old_raw.buriedness),
                    "temperature":     _keep(new_raw.temperature,     old_raw.temperature),
                    "fieryness":       _keep(new_raw.fieryness,       old_raw.fieryness),
                    "floors":          _keep(new_raw.floors,          old_raw.floors),
                    "ground_area":     _keep(new_raw.ground_area,     old_raw.ground_area),
                    "repair_cost":     _keep(new_raw.repair_cost,     old_raw.repair_cost),
                    "position_on_edge":_keep(new_raw.position_on_edge, old_raw.position_on_edge),
                }
            )

            merged_metrics = entity.computed_metrics.model_copy(
                update={
                    "estimated_death_time": estimate_death_time(merged_raw),
                    "total_area": compute_total_area(merged_raw),
                }
            )

            merged_entity = existing.model_copy(
                update={
                    "type": entity.type,
                    "raw_sensor_data": merged_raw,
                    "computed_metrics": merged_metrics,
                    "utility_score": entity.utility_score,
                    "entity_x": entity.entity_x if entity.entity_x is not None else existing.entity_x,
                    "entity_y": entity.entity_y if entity.entity_y is not None else existing.entity_y,
                }
            )

            self.tasks[entity.id] = merged_entity
            logger.info("Я обновил сущность в кэше: entity_id=%s", entity.id)


__all__ = ["WorldModel"]

