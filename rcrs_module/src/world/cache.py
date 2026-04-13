from __future__ import annotations

"""В этом модуле я реализую in-memory модель мира для симуляции спасателей."""

import logging
from typing import Dict, Iterable, Optional

import networkx as nx

from .entities import (
    AgentState, MapEdge, MapNode, PerceptionPacket, VisibleEntity,
    estimate_death_time, compute_total_area,
)


logger = logging.getLogger(__name__)


class WorldModel:
    """В этом классе я описываю центральный кэш мира с доступом за O(1)."""

    def __init__(self) -> None:
        """Здесь я инициализирую хранилища агентов, задач, убежищ и граф дорожной сети."""

        self.agents: Dict[int, AgentState] = {}
        self.tasks: Dict[int, VisibleEntity] = {}
        self.road_graph: nx.Graph = nx.Graph()
        # Я храню идентификаторы убежищ для маршрутизации пожарных при NeedRefugeException.
        self.refuge_ids: list[int] = []
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

    def get_nearest_node(self, x: int, y: int) -> int:
        """Здесь я нахожу ближайший узел графа к координатам (x, y).

        Я использую этот метод, если агент находится на дороге (ребре),
        чтобы корректно построить путь к убежищу через ближайший узел.
        """
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
        """В этом методе я строю граф G=(V,E) из топологии карты, полученной от ядра симулятора.

        Я заполняю граф один раз при старте: вершины — здания и перекрёстки,
        рёбра — дороги с весом (длиной) для алгоритма Дейкстры в f_dist.
        """
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
        """В этом методе я обновляю словарь союзников свежими данными такта.

        Я перезаписываю запись по agent.id, не накапливая устаревшие позиции,
        потому что агенты перемещаются каждый такт и старые координаты исказят f_social.
        """
        for ally in ally_states:
            self.agents[ally.id] = ally
            logger.debug("Я обновил состояние союзника agent_id=%s", ally.id)

    def apply_perception(self, packet: PerceptionPacket) -> None:
        """В этом методе я применяю полный пакет восприятия к модели мира за O(N+M).

        Порядок обновления важен: сначала граф (если пришли данные карты),
        затем союзники (чтобы f_social опирался на актуальные позиции),
        затем задачи (слияние с кэшем).
        """
        # Я заполняю граф только если ядро прислало данные карты (обычно такт 0).
        if packet.map_nodes or packet.map_edges:
            self.build_graph_from_map(packet.map_nodes, packet.map_edges)

        # Я сохраняю идентификаторы убежищ, чтобы пожарные знали куда пополнять воду.
        if packet.refuge_ids:
            self.refuge_ids = list(packet.refuge_ids)
            logger.info("Я сохранил %d убежищ: %s", len(self.refuge_ids), self.refuge_ids)

        # Я заменяю словарь союзников целиком данными текущего такта — это
        # предотвращает деградацию f_social из-за устаревших позиций агентов,
        # которые давно вышли из зоны видимости и могли переместиться куда угодно.
        # Агенты, невидимые в текущем такте, не участвуют в расчёте f_social.
        self.agents.clear()
        self.update_agents(packet.ally_states)

        # Я удаляю сущности, которые ядро пометило как удалённые (ChangeSet.deletes).
        # Это критически важно для завалов: после AKClear blockade-сущность удаляется ядром,
        # и без этого полицейский будет повторно пытаться расчистить несуществующий завал.
        for eid in packet.deleted_entity_ids:
            if eid in self.tasks:
                logger.info("Я удалил сущность entity_id=%d из кэша (ядро удалило из ChangeSet)", eid)
                del self.tasks[eid]

        # Я сливаю наблюдаемые сущности с историческим кэшем.
        self.update_perception(packet.visible_entities)

        logger.debug(
            "Я применил пакет восприятия такта %d: %d сущностей, %d союзников",
            packet.tick,
            len(packet.visible_entities),
            len(packet.ally_states),
        )

    def update_perception(self, visible_entities: Iterable[VisibleEntity]) -> None:
        """В этом методе я реализую обновление модели мира на основе новых наблюдений."""

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
                """Я сохраняю старое значение, если новое равно None (сущность вышла из зоны видимости)."""
                return new_val if new_val is not None else old_val

            new_raw = entity.raw_sensor_data
            old_raw = existing.raw_sensor_data

            # Я сливаю все поля RawSensorData: приоритет у свежих данных,
            # при None — сохраняю последнее известное значение из кэша.
            # Это особенно важно для fieryness и buriedness: если здание вышло
            # из зоны видимости, его статус горения остаётся актуальным.
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

            # Я пересчитываю estimated_death_time и total_area из слитых данных,
            # потому что entity.computed_metrics был вычислен из неслитого raw
            # (где hp/damage могли быть None), а merged_raw содержит актуальные данные.
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
                    # Я сохраняю последние известные координаты сущности:
                    # свежие данные (entity.entity_x) приоритетнее кэша.
                    "entity_x": entity.entity_x if entity.entity_x is not None else existing.entity_x,
                    "entity_y": entity.entity_y if entity.entity_y is not None else existing.entity_y,
                }
            )

            self.tasks[entity.id] = merged_entity
            logger.info("Я обновил сущность в кэше: entity_id=%s", entity.id)


__all__ = ["WorldModel"]

