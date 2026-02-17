from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from module.network.protocol import URN


@dataclass
class ServerEntity:
    """Локальная сущность мира хранит URN и свойства в сыром виде для адаптации к матмодели."""

    entity_id: int
    urn: int
    properties: Dict[int, Any] = field(default_factory=dict)


class WorldModel:
    """Локальная модель мира агрегирует initial entities и ChangeSet, чтобы видеть полное состояние между тиками."""

    def __init__(self) -> None:
        self.entities: Dict[int, ServerEntity] = {}
        self.config: Dict[str, str] = {}

    def initialize(self, entities: Iterable[Any], config: Optional[Dict[str, str]] = None) -> None:
        """Инициализация из KA_CONNECT_OK создаёт стартовую world-model агента до первого KA_SENSE."""
        self.entities.clear()
        if config is not None:
            self.config = dict(config)

        for proto_entity in entities:
            parsed_entity = self._parse_entity_proto(proto_entity)
            self.entities[parsed_entity.entity_id] = parsed_entity

    def apply_changeset(self, changeset: Any) -> None:
        """ChangeSet обновляет только изменённые свойства, поэтому состояние копится между сообщениями."""
        for entity_change in changeset.changes:
            entity_id = int(entity_change.entityID)
            entity_urn = int(entity_change.urn)

            if entity_id not in self.entities:
                self.entities[entity_id] = ServerEntity(entity_id=entity_id, urn=entity_urn, properties={})
            else:
                if entity_urn != 0:
                    self.entities[entity_id].urn = entity_urn

            for property_proto in entity_change.properties:
                property_urn = int(property_proto.urn)
                property_value = self._decode_property_value(property_proto)
                self.entities[entity_id].properties[property_urn] = property_value

        for deleted_entity_id in changeset.deletes:
            self.entities.pop(int(deleted_entity_id), None)

    def get_entity(self, entity_id: int) -> Optional[ServerEntity]:
        """Чтение сущности из словаря выделено отдельно для централизованной обработки отсутствующих ID."""
        return self.entities.get(int(entity_id))

    def get_property(self, entity_id: int, property_urn: int) -> Any:
        """Свойства читаются через метод, чтобы адаптер не работал напрямую с внутренним хранилищем."""
        entity = self.get_entity(entity_id)
        if entity is None:
            return None
        return entity.properties.get(int(property_urn))

    def is_area(self, entity_urn: int) -> bool:
        """Area-предикат нужен pathfinder'у для корректного построения графа дорог и зданий."""
        return entity_urn in {
            int(URN.Entity.ROAD),
            int(URN.Entity.BUILDING),
            int(URN.Entity.REFUGE),
            int(URN.Entity.HYDRANT),
            int(URN.Entity.GAS_STATION),
            int(URN.Entity.FIRE_STATION),
            int(URN.Entity.AMBULANCE_CENTRE),
            int(URN.Entity.POLICE_OFFICE),
        }

    def is_human(self, entity_urn: int) -> bool:
        """Human-предикат используется при резолве POSITION, когда civilian загружен в AmbulanceTeam."""
        return entity_urn in {
            int(URN.Entity.CIVILIAN),
            int(URN.Entity.FIRE_BRIGADE),
            int(URN.Entity.AMBULANCE_TEAM),
            int(URN.Entity.POLICE_FORCE),
        }

    def resolve_area_position(self, entity_id: int, max_depth: int = 5) -> Optional[int]:
        """POSITION может указывать на человека; рекурсивный резолв нужен для получения конечной area-позиции."""
        visited: Set[int] = set()
        current_id = int(entity_id)

        for _ in range(max_depth):
            if current_id in visited:
                return None
            visited.add(current_id)

            entity = self.get_entity(current_id)
            if entity is None:
                return None

            if self.is_area(entity.urn):
                return current_id

            if not self.is_human(entity.urn):
                return None

            position_ref = entity.properties.get(int(URN.Property.POSITION))
            if position_ref is None:
                return None
            current_id = int(position_ref)

        return None

    def get_area_coordinates(self, area_entity_id: int) -> Tuple[Optional[int], Optional[int]]:
        """Координаты area нужны для вычисления евклидовых факторов и визуализации в UI."""
        area_entity = self.get_entity(area_entity_id)
        if area_entity is None:
            return None, None

        x_value = area_entity.properties.get(int(URN.Property.X))
        y_value = area_entity.properties.get(int(URN.Property.Y))

        try:
            x = int(x_value) if x_value is not None else None
            y = int(y_value) if y_value is not None else None
            return x, y
        except (TypeError, ValueError):
            return None, None

    def get_neighbors(self, area_entity_id: int) -> List[int]:
        """Соседство по EDGES используется для построения shortest-path без сторонних библиотек."""
        entity = self.get_entity(area_entity_id)
        if entity is None:
            return []

        edges = entity.properties.get(int(URN.Property.EDGES))
        if not isinstance(edges, list):
            return []

        neighbors: List[int] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            neighbor_id = edge.get("neighbour")
            try:
                if neighbor_id is None:
                    continue
                neighbor_int = int(neighbor_id)
                if neighbor_int > 0:
                    neighbors.append(neighbor_int)
            except (TypeError, ValueError):
                continue

        return neighbors

    def shortest_path(self, start_area_id: Optional[int], goal_area_id: Optional[int]) -> List[int]:
        """BFS-path достаточен для онлайн-цикла агента и даёт детерминированную траекторию без random()."""
        if start_area_id is None or goal_area_id is None:
            return []

        start = int(start_area_id)
        goal = int(goal_area_id)

        if start == goal:
            return [start]

        queue: deque[int] = deque([start])
        parent: Dict[int, Optional[int]] = {start: None}

        while queue:
            current = queue.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == goal:
                    return self._reconstruct_path(parent, goal)
                queue.append(neighbor)

        return []

    def shortest_distance(self, start_area_id: Optional[int], goal_area_id: Optional[int]) -> Optional[float]:
        """Дистанция в coordinate-units согласована с диагональю карты для корректной нормализации f_dist."""
        path = self.shortest_path(start_area_id, goal_area_id)
        if not path:
            return None
        if len(path) == 1:
            return 0.0

        total_distance = 0.0
        for index in range(len(path) - 1):
            start_x, start_y = self.get_area_coordinates(path[index])
            goal_x, goal_y = self.get_area_coordinates(path[index + 1])
            if None in {start_x, start_y, goal_x, goal_y}:
                return float(max(0, len(path) - 1))

            dx = float(goal_x) - float(start_x)
            dy = float(goal_y) - float(start_y)
            total_distance += math.sqrt(dx * dx + dy * dy)

        return max(0.0, total_distance)

    def _reconstruct_path(self, parent: Dict[int, Optional[int]], goal: int) -> List[int]:
        """Восстановление пути отделено для читаемости и проверки корректности parent-дерева."""
        path: List[int] = []
        current: Optional[int] = goal
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        return path

    def _parse_entity_proto(self, proto_entity: Any) -> ServerEntity:
        """EntityProto из KA_CONNECT_OK преобразуется в компактную структуру без привязки к Java классам."""
        entity_id = int(proto_entity.entityID)
        entity_urn = int(proto_entity.urn)
        properties: Dict[int, Any] = {}

        for property_proto in proto_entity.properties:
            property_urn = int(property_proto.urn)
            properties[property_urn] = self._decode_property_value(property_proto)

        return ServerEntity(entity_id=entity_id, urn=entity_urn, properties=properties)

    def _decode_property_value(self, property_proto: Any) -> Any:
        """Декодер oneof-property гарантирует единый python-формат вне зависимости от wire-типа protobuf."""
        if not bool(property_proto.defined):
            return None

        value_case = property_proto.WhichOneof("value")
        if value_case == "intValue":
            return int(property_proto.intValue)
        if value_case == "boolValue":
            return bool(property_proto.boolValue)
        if value_case == "doubleValue":
            return float(property_proto.doubleValue)
        if value_case == "byteList":
            return bytes(property_proto.byteList)
        if value_case == "intList":
            return [int(value) for value in property_proto.intList.values]
        if value_case == "intMatrix":
            return [[int(value) for value in row.values] for row in property_proto.intMatrix.values]
        if value_case == "edgeList":
            return [
                {
                    "startX": int(edge.startX),
                    "startY": int(edge.startY),
                    "endX": int(edge.endX),
                    "endY": int(edge.endY),
                    "neighbour": int(edge.neighbour),
                }
                for edge in property_proto.edgeList.edges
            ]
        if value_case == "point2D":
            return {"x": float(property_proto.point2D.X), "y": float(property_proto.point2D.Y)}
        return None
