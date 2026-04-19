from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, Optional

import networkx as nx

from .entities import (
    AgentState, EntityType, MapEdge, MapNode, PerceptionPacket, VisibleEntity,
    estimate_death_time, compute_total_area,
)
from config import (
    BLOCKADE_EDGE_PENALTY, BLOCKADE_PENALTY_MAX, BLOCKADE_REPAIR_COST_DIVISOR,
)
from decision.utility.distance import MAX_MAP_DISTANCE


logger = logging.getLogger(__name__)


class WorldModel:
    def __init__(self) -> None:
        self.agents: Dict[int, AgentState] = {}
        self.tasks: Dict[int, VisibleEntity] = {}
        self.last_seen_tick: Dict[int, int] = {}
        self.road_graph: nx.Graph = nx.Graph()
        self.road_graph.graph["revision"] = 0
        self.road_graph.graph["blockade_signature"] = frozenset()

        self.refuge_ids: list[int] = []

        # Я храню диагональ карты как нормировочную базу для f_dist.
        # До получения карты использую консервативный fallback из distance.py;
        # реальное значение пересчитывается при построении графа по bbox узлов.
        self.max_map_distance: float = MAX_MAP_DISTANCE

        # Индекс node_id → set of blockade entity_ids.
        # Обновляется в refresh_blockade_weights, O(1) поиск на пути.
        self.blockades_by_node: Dict[int, set[int]] = {}

        logger.info("Я инициализировал пустую модель мира и граф дорожной сети")

    def _bump_graph_revision(self) -> None:
        revision = int(self.road_graph.graph.get("revision", 0)) + 1
        self.road_graph.graph["revision"] = revision
        logger.debug("Я повысил revision дорожного графа до %d", revision)

    def add_road_node(self, entity_id: int, **attrs: object) -> None:
        self.road_graph.add_node(entity_id, **attrs)

    def add_road_edge(self, source_id: int, target_id: int, weight: float, **attrs: object) -> None:
        self.road_graph.add_edge(source_id, target_id, weight=weight, **attrs)

    def set_agent(self, agent: AgentState) -> None:
        self.agents[agent.id] = agent

    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        return self.agents.get(agent_id)

    def get_task(self, entity_id: int) -> Optional[VisibleEntity]:
        return self.tasks.get(entity_id)

    def get_nearest_node(self, x: int, y: int) -> int:
        from action.navigation import find_nearest_node
        result = find_nearest_node(self.road_graph, x, y)
        return result if result is not None else -1

    def remove_task(self, entity_id: int) -> None:
        if entity_id in self.tasks:
            del self.tasks[entity_id]
            self.last_seen_tick.pop(entity_id, None)
            logger.info("Я удалил задачу entity_id=%d из кэша через remove_task", entity_id)
        else:
            logger.debug("Я пропустил remove_task: entity_id=%d не в кэше", entity_id)

    def build_graph_from_map(
        self,
        nodes: Iterable[MapNode],
        edges: Iterable[MapEdge],
    ) -> None:
        node_count = 0
        for node in nodes:
            self.road_graph.add_node(
                node.entity_id,
                x=node.x,
                y=node.y,
                area_type=node.area_type,
                apexes=node.apexes,
            )
            node_count += 1

        edge_count = 0
        for edge in edges:
            self.road_graph.add_edge(
                edge.source_id, edge.target_id,
                weight=edge.weight, base_weight=edge.weight,
            )
            edge_count += 1

        # Я пересчитываю MaxMapDistance как диагональ bounding-box всех узлов графа.
        # Это даёт корректную нормировку f_dist независимо от размера карты
        # (Kobe, VC, Berlin — у всех разные размеры в миллиметрах).
        xs: list[int] = []
        ys: list[int] = []
        for _node_id, attrs in self.road_graph.nodes(data=True):
            x = attrs.get("x")
            y = attrs.get("y")
            if x is not None and y is not None:
                xs.append(int(x))
                ys.append(int(y))
        if xs and ys:
            diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if diagonal > 0:
                self.max_map_distance = diagonal
                logger.info(
                    "Я вычислил MaxMapDistance по bbox карты: %.0f мм (узлов=%d)",
                    self.max_map_distance, len(xs),
                )

        logger.info(
            "Я построил дорожный граф: %d вершин, %d рёбер",
            node_count,
            edge_count,
        )
        self._bump_graph_revision()

    def refresh_blockade_weights(self) -> None:
        for _u, _v, data in self.road_graph.edges(data=True):
            base = data.get("base_weight")
            if base is None:
                base = data.get("weight", 1.0)
                data["base_weight"] = base
            data["weight"] = base

        self.blockades_by_node.clear()

        penalized = 0
        active_blockades: set[tuple[int, int, int]] = set()
        for entity in self.tasks.values():
            if entity.type != EntityType.BLOCKADE:
                continue

            node_id = entity.raw_sensor_data.position_on_edge

            # Если position_on_edge не пришёл от сервера, пытаюсь
            # восстановить узел графа по координатам завала (entity_x/y).
            if node_id is None:
                if entity.entity_x is not None and entity.entity_y is not None:
                    from action.navigation import find_nearest_node
                    inferred = find_nearest_node(self.road_graph, entity.entity_x, entity.entity_y)
                    if inferred is not None:
                        node_id = inferred
                        logger.info(
                            "Я вывел position_on_edge=%d для завала entity_id=%d "
                            "из координат (%d, %d)",
                            node_id, entity.id, entity.entity_x, entity.entity_y,
                        )
                if node_id is None:
                    continue

            # Я пропускаю завалы с repair_cost=0: они уже расчищены, но
            # сервер ещё не прислал deleted_entity_ids. Без этой проверки
            # полицейские бесконечно отправляли AKClear на пустые завалы.
            repair_cost = entity.raw_sensor_data.repair_cost
            if repair_cost is not None and repair_cost <= 0:
                continue

            self.blockades_by_node.setdefault(node_id, set()).add(entity.id)
            active_blockades.add((entity.id, node_id, int(repair_cost or 0)))

            if not self.road_graph.has_node(node_id):
                continue

            repair_cost = entity.raw_sensor_data.repair_cost or 0
            penalty = BLOCKADE_EDGE_PENALTY + (repair_cost / BLOCKADE_REPAIR_COST_DIVISOR)
            penalty = min(penalty, BLOCKADE_PENALTY_MAX)

            for neighbor in self.road_graph.neighbors(node_id):
                data = self.road_graph[node_id][neighbor]
                base = data.get("base_weight", data.get("weight", 1.0))
                candidate = base * penalty
                if candidate > data.get("weight", base):
                    data["weight"] = candidate
                    penalized += 1

        if penalized:
            logger.debug("Я пересчитал штрафы обхода завалов: обновлено %d рёбер", penalized)

        signature = frozenset(active_blockades)
        prev_signature = self.road_graph.graph.get("blockade_signature", frozenset())
        if signature != prev_signature:
            self.road_graph.graph["blockade_signature"] = signature
            self._bump_graph_revision()

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

        # Я НЕ очищаю agents целиком: союзники вне зоны видимости
        # сохраняют последнее известное состояние. Без этого social_factor
        # считал 0 однотипных агентов рядом, даже если они там были.
        self.update_agents(packet.ally_states)

        for eid in packet.deleted_entity_ids:
            if eid in self.tasks:
                logger.info("Я удалил сущность entity_id=%d из кэша (ядро удалило из ChangeSet)", eid)
                del self.tasks[eid]
            self.last_seen_tick.pop(eid, None)

        self.update_perception(packet.visible_entities)

        # Резервный источник position_on_edge: PROP_BLOCKADES дорог даёт
        # обратный индекс blockade_id → road_id. Если у завала в кэше нет
        # position_on_edge — заполняю из этого индекса.
        if packet.blockade_to_road:
            for blk_id, road_id in packet.blockade_to_road.items():
                entity = self.tasks.get(blk_id)
                if entity is not None and entity.raw_sensor_data.position_on_edge is None:
                    merged_raw = entity.raw_sensor_data.model_copy(
                        update={"position_on_edge": road_id},
                    )
                    self.tasks[blk_id] = entity.model_copy(
                        update={"raw_sensor_data": merged_raw},
                    )
                    logger.info(
                        "Я восстановил position_on_edge=%d для завала entity_id=%d "
                        "из PROP_BLOCKADES дороги",
                        road_id, blk_id,
                    )

        if packet.road_blockades:
            self._remove_stale_blockades(packet.road_blockades)

        for entity in packet.visible_entities:
            self.last_seen_tick[entity.id] = packet.tick

        # Я НЕ удаляю завалы по давности наблюдения: сервер сам присылает
        # deleted_entity_ids при фактической расчистке, а stale-expiry
        # приводила к циклу «забыл завал → построил путь сквозь него →
        # уперся → добавил обратно → забыл» (Kobe, узкие улицы).
        self.refresh_blockade_weights()

        logger.debug(
            "Я применил пакет восприятия такта %d: %d сущностей, %d союзников",
            packet.tick,
            len(packet.visible_entities),
            len(packet.ally_states),
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
                apexes = entity.raw_sensor_data.apexes
                logger.info(
                    "Я добавил новую сущность в кэш: entity_id=%s, type=%s, apexes_len=%s",
                    entity.id, entity.type.value,
                    len(apexes) if apexes is not None else None,
                )
                continue

            def _keep(
                new_val: int | float | None,
                old_val: int | float | None,
            ) -> int | float | None:
                return new_val if new_val is not None else old_val

            new_raw = entity.raw_sensor_data
            old_raw = existing.raw_sensor_data

            merged_apexes = new_raw.apexes if new_raw.apexes is not None else old_raw.apexes
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
                    "apexes": merged_apexes,
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
                    # is_ally — неизменяемое свойство типа сущности (CIVILIAN vs
                    # союзный агент), отследил по URN при первом наблюдении.
                    "is_ally": entity.is_ally or existing.is_ally,
                }
            )

            self.tasks[entity.id] = merged_entity
            logger.info("Я обновил сущность в кэше: entity_id=%s", entity.id)

    def _remove_stale_blockades(self, road_blockades: dict[int, list[int]]) -> None:
        stale_ids: set[int] = set()
        for road_id, current_ids in road_blockades.items():
            current = set(current_ids)
            for entity_id, entity in self.tasks.items():
                if entity.type != EntityType.BLOCKADE:
                    continue
                if entity.raw_sensor_data.position_on_edge != road_id:
                    continue
                if entity_id in current:
                    continue
                stale_ids.add(entity_id)

        for entity_id in stale_ids:
            self.tasks.pop(entity_id, None)
            self.last_seen_tick.pop(entity_id, None)
            logger.info(
                "Я удалил stale-завал entity_id=%d: он исчез из PROP_BLOCKADES дороги",
                entity_id,
            )


__all__ = ["WorldModel"]
