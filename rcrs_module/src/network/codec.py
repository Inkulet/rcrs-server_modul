from __future__ import annotations

"""В этом модуле я реализую кодек протокола RCRS Kernel: константы URN и чистые функции
сборки/разбора Protobuf-сообщений.

Я намеренно держу этот слой отделённым от TCP-транспорта (client.py):
функции здесь не касаются сокетов и легко тестируются изолированно.

Формат фрейма (StreamConnection.java):
    [4 байта INT32 big-endian: длина][N байт: MessageProto.toByteArray()]
"""

import logging
import struct
from typing import Any, List, Optional

from network.proto.RCRSProto_pb2 import (
    ChangeSetProto,
    IntListProto,
    MessageComponentProto,
    MessageProto,
)
from world.entities import (
    AgentState,
    AgentType,
    ComputedMetrics,
    EntityType,
    MapEdge,
    MapNode,
    PerceptionPacket,
    Position,
    RawSensorData,
    Resources,
    VisibleEntity,
    estimate_death_time,
    compute_total_area,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Я фиксирую все URN-константы из Java-исходников ядра симулятора в одном месте,
# чтобы любое изменение версии протокола отражалось только здесь.
# ---------------------------------------------------------------------------

# Префиксы (rescuecore2/Constants.java и standard/Constants.java)
_CTRL_MSG = 0x0100
_CTRL_CMP = 0x0200
_ENT      = 0x1100
_PROP     = 0x1200
_STD_MSG  = 0x1300
_STD_CMP  = 0x1400

# --- Управляющие сообщения (ControlMessageURN) ---
URN_AK_CONNECT       = _CTRL_MSG | 18  # 0x0112  Агент → Ядро: регистрация
URN_AK_ACKNOWLEDGE   = _CTRL_MSG | 19  # 0x0113  Агент → Ядро: подтверждение
URN_KA_CONNECT_OK    = _CTRL_MSG | 20  # 0x0114  Ядро → Агент: успешная регистрация
URN_KA_CONNECT_ERROR = _CTRL_MSG | 21  # 0x0115  Ядро → Агент: ошибка регистрации
URN_KA_SENSE         = _CTRL_MSG | 22  # 0x0116  Ядро → Агент: такт восприятия

# --- Компоненты управляющих сообщений (ControlMessageComponentURN) ---
COMP_REQUEST_ID    = _CTRL_CMP | 1   # 0x0201
COMP_AGENT_ID      = _CTRL_CMP | 2   # 0x0202
COMP_VERSION       = _CTRL_CMP | 3   # 0x0203
COMP_NAME          = _CTRL_CMP | 4   # 0x0204
COMP_ENTITY_TYPES  = _CTRL_CMP | 5   # 0x0205
COMP_ENTITIES      = _CTRL_CMP | 11  # 0x020B
COMP_AGENT_CONFIG  = _CTRL_CMP | 13  # 0x020D
COMP_TIME          = _CTRL_CMP | 14  # 0x020E
COMP_UPDATES       = _CTRL_CMP | 15  # 0x020F
COMP_HEARING       = _CTRL_CMP | 16  # 0x0210

# --- Типы сущностей (StandardEntityURN) ---
ENT_WORLD           = _ENT | 1   # 0x1101
ENT_ROAD            = _ENT | 2   # 0x1102
ENT_BLOCKADE        = _ENT | 3   # 0x1103
ENT_BUILDING        = _ENT | 4   # 0x1104
ENT_REFUGE          = _ENT | 5   # 0x1105
ENT_HYDRANT         = _ENT | 6   # 0x1106
ENT_GAS_STATION     = _ENT | 7   # 0x1107
ENT_FIRE_STATION    = _ENT | 8   # 0x1108
ENT_AMBULANCE_CENTRE= _ENT | 9   # 0x1109
ENT_POLICE_OFFICE   = _ENT | 10  # 0x110A
ENT_CIVILIAN        = _ENT | 11  # 0x110B
ENT_FIRE_BRIGADE    = _ENT | 12  # 0x110C
ENT_AMBULANCE_TEAM  = _ENT | 13  # 0x110D
ENT_POLICE_FORCE    = _ENT | 14  # 0x110E

# --- Свойства сущностей (StandardPropertyURN) ---
PROP_X                  = _PROP | 6    # 0x1206
PROP_Y                  = _PROP | 7    # 0x1207
PROP_BLOCKADES          = _PROP | 8    # 0x1208  EntityRefListProperty (список завалов на дороге)
PROP_REPAIR_COST        = _PROP | 9    # 0x1209
PROP_FLOORS             = _PROP | 10   # 0x120A
PROP_FIERYNESS          = _PROP | 13   # 0x120D
PROP_BUILDING_AREA_GROUND = _PROP | 16 # 0x1210
# Я задаю URN свойства EdgeList: ordinal=19 → 0x1213.
# В Java StandardPropertyURN: EDGES = PROPERTY_URN_PREFIX | 19.
# Ранее ошибочно использовалось ordinal=17 (это BUILDING_AREA_TOTAL).
# Каждое ребро содержит: startX, startY, endX, endY, neighbour (entity_id соседа).
PROP_EDGES              = _PROP | 19   # 0x1213  EdgeListProperty — список рёбер дорожного графа
PROP_POSITION           = _PROP | 20   # 0x1214
PROP_HP                 = _PROP | 24   # 0x1218
PROP_DAMAGE             = _PROP | 25   # 0x1219
PROP_BURIEDNESS         = _PROP | 26   # 0x121A
PROP_WATER_QUANTITY     = _PROP | 28   # 0x121C
PROP_TEMPERATURE        = _PROP | 29   # 0x121D

# --- Командные сообщения (StandardMessageURN) ---
MSG_AK_REST       = _STD_MSG | 1   # 0x1301
MSG_AK_MOVE       = _STD_MSG | 2   # 0x1302
MSG_AK_LOAD       = _STD_MSG | 3   # 0x1303
MSG_AK_UNLOAD     = _STD_MSG | 4   # 0x1304
MSG_AK_EXTINGUISH = _STD_MSG | 7   # 0x1307
MSG_AK_RESCUE     = _STD_MSG | 8   # 0x1308
MSG_AK_CLEAR      = _STD_MSG | 9   # 0x1309

# --- Компоненты командных сообщений (StandardMessageComponentURN) ---
COMP_TARGET   = _STD_CMP | 1  # 0x1401
COMP_DEST_X   = _STD_CMP | 2  # 0x1402
COMP_DEST_Y   = _STD_CMP | 3  # 0x1403
COMP_WATER    = _STD_CMP | 4  # 0x1404
COMP_PATH     = _STD_CMP | 5  # 0x1405

# Я сопоставляю URN сущности с типом агента для заполнения AgentState.type.
_ENTITY_URN_TO_AGENT_TYPE: dict[int, AgentType] = {
    ENT_FIRE_BRIGADE:  AgentType.FIRE_BRIGADE,
    ENT_AMBULANCE_TEAM: AgentType.AMBULANCE_TEAM,
    ENT_POLICE_FORCE:  AgentType.POLICE_FORCE,
}

# Я сопоставляю URN сущности с EntityType для задач (здания, гражданские, завалы).
_ENTITY_URN_TO_ENTITY_TYPE: dict[int, EntityType] = {
    ENT_CIVILIAN: EntityType.CIVILIAN,
    ENT_BUILDING: EntityType.BUILDING,
    ENT_REFUGE:   EntityType.BUILDING,
    ENT_BLOCKADE: EntityType.BLOCKADE,
}

# Версия протокола AKConnect (2 = IntList для типов сущностей, не StringList).
PROTOCOL_VERSION: int = 2


# ===========================================================================
# Фреймирование: упаковка/распаковка длины сообщения
# ===========================================================================

def pack_frame(proto_bytes: bytes) -> bytes:
    """Я упаковываю Protobuf-байты в фрейм: [INT32 big-endian длина][тело].

    Соответствует StreamConnection.serializeMessageProto() в Java.
    """
    # Я использую struct.pack для big-endian INT32, так как это гарантирует
    # идентичный порядок байт с Java EncodingTools.writeInt32().
    return struct.pack(">I", len(proto_bytes)) + proto_bytes


def unpack_frame_length(header: bytes) -> int:
    """Я извлекаю длину тела сообщения из 4-байтового заголовка фрейма."""
    return struct.unpack(">I", header)[0]


# ===========================================================================
# Сборка исходящих сообщений (Агент → Ядро)
# ===========================================================================

def build_ak_connect(
    request_id: int,
    agent_name: str,
    entity_types: List[int],
) -> bytes:
    """Я собираю AKConnect — сообщение регистрации агента в ядре симулятора.

    Компоненты (ControlMessageComponentURN):
        RequestID (0x0201) → intValue
        Version   (0x0203) → intValue = 2
        Name      (0x0204) → stringValue
        RequestedEntityTypes (0x0205) → intList
    """
    msg = MessageProto()
    msg.urn = URN_AK_CONNECT

    msg.components[COMP_REQUEST_ID].intValue = request_id
    msg.components[COMP_VERSION].intValue    = PROTOCOL_VERSION
    msg.components[COMP_NAME].stringValue    = agent_name

    int_list = IntListProto()
    int_list.values.extend(entity_types)
    msg.components[COMP_ENTITY_TYPES].intList.CopyFrom(int_list)

    logger.debug(
        "Я собрал AKConnect: request_id=%d, name=%s, types=%s",
        request_id, agent_name, entity_types,
    )
    return pack_frame(msg.SerializeToString())


def build_ak_acknowledge(request_id: int, agent_id: int = 0) -> bytes:
    """Я собираю AKAcknowledge — подтверждение получения KAConnectOK.

    Ядро RCRS ожидает два компонента:
    - COMP_REQUEST_ID (0x0201) intValue — совпадает с request_id из KAConnectOK;
    - COMP_AGENT_ID   (0x0202) entityID  — идентификатор агента (agent_id из KAConnectOK).
    Без agent_id ядро бросает NullPointerException при попытке прочитать getEntityID().
    """
    msg = MessageProto()
    msg.urn = URN_AK_ACKNOWLEDGE
    msg.components[COMP_REQUEST_ID].intValue = request_id
    msg.components[COMP_AGENT_ID].entityID   = agent_id
    return pack_frame(msg.SerializeToString())


def build_ak_move(
    agent_id: int,
    time: int,
    path: List[int],
    dest_x: int = -1,
    dest_y: int = -1,
) -> bytes:
    """Я собираю AKMove — команду движения агента по маршруту path.

    path — список entity_id узлов графа дорог (Road/Building IDs).
    dest_x, dest_y = -1 означает «двигаться по маршруту без точной точки назначения».
    """
    msg = MessageProto()
    msg.urn = MSG_AK_MOVE
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time

    id_list = IntListProto()
    id_list.values.extend(path)
    msg.components[COMP_PATH].entityIDList.CopyFrom(id_list)

    # Я добавляю координаты только если они явно переданы — иначе Protobuf
    # автоматически создаёт компоненты со значением -1, и ядро пытается
    # вычислить вектор движения к точке (-1, -1) за границей карты,
    # обнуляя скорость агента.
    if dest_x != -1 and dest_y != -1:
        msg.components[COMP_DEST_X].intValue = dest_x
        msg.components[COMP_DEST_Y].intValue = dest_y

    logger.debug("Я собрал AKMove: agent=%d, time=%d, path=%s", agent_id, time, path)
    return pack_frame(msg.SerializeToString())


def build_ak_rescue(agent_id: int, time: int, target_id: int) -> bytes:
    """Я собираю AKRescue — команду спасения гражданского target_id."""
    msg = MessageProto()
    msg.urn = MSG_AK_RESCUE
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_TARGET].entityID   = target_id
    logger.debug("Я собрал AKRescue: agent=%d, time=%d, target=%d", agent_id, time, target_id)
    return pack_frame(msg.SerializeToString())


def build_ak_extinguish(
    agent_id: int,
    time: int,
    target_id: int,
    water: int,
) -> bytes:
    """Я собираю AKExtinguish — команду тушения здания target_id с расходом воды water."""
    msg = MessageProto()
    msg.urn = MSG_AK_EXTINGUISH
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_TARGET].entityID   = target_id
    msg.components[COMP_WATER].intValue    = water
    logger.debug(
        "Я собрал AKExtinguish: agent=%d, time=%d, target=%d, water=%d",
        agent_id, time, target_id, water,
    )
    return pack_frame(msg.SerializeToString())


def build_ak_clear(agent_id: int, time: int, target_id: int) -> bytes:
    """Я собираю AKClear — команду расчистки завала target_id."""
    msg = MessageProto()
    msg.urn = MSG_AK_CLEAR
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_TARGET].entityID   = target_id
    logger.debug("Я собрал AKClear: agent=%d, time=%d, target=%d", agent_id, time, target_id)
    return pack_frame(msg.SerializeToString())


def build_ak_load(agent_id: int, time: int, target_id: int) -> bytes:
    """Я собираю AKLoad — команду погрузки гражданского target_id."""
    msg = MessageProto()
    msg.urn = MSG_AK_LOAD
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_TARGET].entityID   = target_id
    return pack_frame(msg.SerializeToString())


def build_ak_unload(agent_id: int, time: int) -> bytes:
    """Я собираю AKUnload — команду выгрузки переносимого гражданского."""
    msg = MessageProto()
    msg.urn = MSG_AK_UNLOAD
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    return pack_frame(msg.SerializeToString())


def build_ak_rest(agent_id: int, time: int) -> bytes:
    """Я собираю AKRest — команду ожидания на текущей позиции."""
    msg = MessageProto()
    msg.urn = MSG_AK_REST
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    return pack_frame(msg.SerializeToString())


# ===========================================================================
# Разбор входящих сообщений (Ядро → Агент)
# ===========================================================================

def _get_int(comp: MessageComponentProto) -> int:
    """Я извлекаю целочисленное значение из компонента, независимо от oneof-поля."""
    # intValue используется для обычных INT, entityID — для ссылок на сущности.
    which = comp.WhichOneof("component")
    if which == "intValue":
        return comp.intValue
    if which == "entityID":
        return comp.entityID
    return 0


def parse_ka_connect_ok(
    proto: MessageProto,
) -> tuple[int, int, list[MapNode], list[MapEdge], list[int]]:
    """Я разбираю KAConnectOK и возвращаю (request_id, agent_id, map_nodes, map_edges, refuge_ids).

    Из EntityList я извлекаю топологию карты: Road/Building-узлы → вершины графа G=(V,E),
    рёбра Road-сущностей → рёбра графа, ENT_REFUGE → список убежищ.
    """
    import math as _math

    request_id = _get_int(proto.components[COMP_REQUEST_ID]) if COMP_REQUEST_ID in proto.components else 0
    agent_id   = _get_int(proto.components[COMP_AGENT_ID])   if COMP_AGENT_ID   in proto.components else 0

    map_nodes:  list[MapNode] = []
    map_edges:  list[MapEdge] = []
    refuge_ids: list[int]     = []

    # Я использую двухпроходной алгоритм для построения рёбер графа.
    # Проход 1: собираю координаты всех Area-узлов в словарь {entity_id: (x, y)}.
    # Проход 2: строю рёбра — через PROP_EDGES (0x1213, EdgeListProperty с геометрией).
    # В RCRS-протоколе единственный URN для рёбер дорожного графа — PROP_EDGES (_PROP | 19).
    # Каждое ребро содержит координаты и ID соседней Area-сущности.

    _AREA_URNS = frozenset((ENT_ROAD, ENT_BUILDING, ENT_REFUGE, ENT_FIRE_STATION,
                            ENT_AMBULANCE_CENTRE, ENT_POLICE_OFFICE, ENT_HYDRANT,
                            ENT_GAS_STATION))

    # --- Проход 1: узлы и координаты ---
    node_coords: dict[int, tuple[int, int]] = {}  # eid → (x, y)
    _entity_protos_area = []  # буфер для второго прохода

    if COMP_ENTITIES in proto.components:
        entity_list = proto.components[COMP_ENTITIES].entityList
        for entity_proto in entity_list.entities:
            urn = entity_proto.urn
            eid = entity_proto.entityID
            props = {p.urn: p for p in entity_proto.properties}

            if urn in _AREA_URNS:
                x = props[PROP_X].intValue if PROP_X in props else 0
                y = props[PROP_Y].intValue if PROP_Y in props else 0
                map_nodes.append(MapNode(entity_id=eid, x=x, y=y))
                node_coords[eid] = (x, y)
                _entity_protos_area.append((eid, urn, props))

            if urn == ENT_REFUGE:
                refuge_ids.append(eid)

    # --- Проход 2: рёбра ---
    _seen_edges: set[tuple[int, int]] = set()  # дедупликация (a,b) ↔ (b,a)

    for eid, urn, props in _entity_protos_area:
        # Я обрабатываю PROP_EDGES (0x1213, EdgeListProperty) — ребро с координатами и соседом.
        if PROP_EDGES in props:
            for edge_proto in props[PROP_EDGES].edgeList.edges:
                neighbour = edge_proto.neighbour
                if neighbour > 0:
                    key = (min(eid, neighbour), max(eid, neighbour))
                    if key not in _seen_edges:
                        _seen_edges.add(key)
                        # Я использую расстояние между центрами Area-сущностей,
                        # а не длину грани (edge_proto.start/end). Длина грани —
                        # это ширина прохода (дверной проём, разделительная полоса),
                        # а не расстояние перемещения между центрами объектов.
                        if neighbour in node_coords:
                            sx, sy = node_coords[eid]
                            tx, ty = node_coords[neighbour]
                            weight = max(1.0, _math.hypot(tx - sx, ty - sy))
                        else:
                            # Координаты соседа неизвестны — я вычисляю расстояние
                            # от центра текущего узла до середины грани как нижнюю
                            # оценку длины дороги (лучше, чем ширина прохода).
                            sx, sy = node_coords[eid]
                            mid_x = (edge_proto.startX + edge_proto.endX) / 2
                            mid_y = (edge_proto.startY + edge_proto.endY) / 2
                            weight = max(1.0, _math.hypot(mid_x - sx, mid_y - sy))
                        map_edges.append(MapEdge(source_id=eid, target_id=neighbour, weight=weight))


    logger.info(
        "Я разобрал KAConnectOK: agent_id=%d, узлов=%d, рёбер=%d, убежищ=%d",
        agent_id, len(map_nodes), len(map_edges), len(refuge_ids),
    )
    return request_id, agent_id, map_nodes, map_edges, refuge_ids


def _defined_int_val(props: dict[int, Any], urn: int, default: int) -> int:
    """Я безопасно извлекаю intValue из свойства, проверяя наличие и флаг defined.

    В RCRS дельта-обновления (ChangeSet) могут содержать свойство с defined=False,
    что означает «значение не изменилось» — в этом случае я возвращаю default,
    чтобы не перезаписать актуальные данные нулём или мусором.
    """
    if urn not in props:
        return default
    p = props[urn]
    if not p.defined:
        return default
    return p.intValue


def parse_ka_sense(
    proto: MessageProto,
    agent_id: int,
    agent_type: AgentType,
    prev_position: Position | None = None,
    prev_water: int = 0,
    prev_transporting: bool = False,
) -> PerceptionPacket:
    """Я разбираю KASense и собираю PerceptionPacket для текущего такта.

    Я итерирую ChangeSet: каждая EntityChangeProto → либо обновление состояния
    союзника (FIRE_BRIGADE / AMBULANCE_TEAM / POLICE_FORCE), либо наблюдаемая
    задача (BUILDING / CIVILIAN / BLOCKADE).

    Параметры prev_position, prev_water, prev_transporting — состояние агента
    с предыдущего такта. RCRS использует дельта-обновления: если свойство не
    изменилось, оно не попадает в ChangeSet. Без сохранения предыдущего состояния
    агент «забывает» позицию (сброс в entity_id=0) и флаг перевозки между тактами.
    """
    tick = _get_int(proto.components[COMP_TIME]) if COMP_TIME in proto.components else 0

    # Я инициализирую собственное состояние предыдущими значениями —
    # дельта-обновление обновит только те поля, которые изменились.
    own_position = prev_position if prev_position is not None else Position(entity_id=0, x=0, y=0)
    own_water    = prev_water
    own_transporting = prev_transporting
    own_found    = False

    visible_entities: list[VisibleEntity] = []
    ally_states:       list[AgentState]   = []

    if COMP_UPDATES in proto.components:
        change_set: ChangeSetProto = proto.components[COMP_UPDATES].changeSet

        for change in change_set.changes:
            eid  = change.entityID
            eurn = change.urn
            props = {p.urn: p for p in change.properties}

            # --- Собственное состояние агента ---
            if eid == agent_id and eurn in _ENTITY_URN_TO_AGENT_TYPE:
                # Я использую _defined_int_val с предыдущими значениями как default,
                # чтобы дельта-обновление без PROP_POSITION не сбрасывало позицию в 0.
                x      = _defined_int_val(props, PROP_X, own_position.x)
                y      = _defined_int_val(props, PROP_Y, own_position.y)
                pos_id = _defined_int_val(props, PROP_POSITION, own_position.entity_id)
                own_position = Position(entity_id=pos_id, x=x, y=y)
                own_water    = _defined_int_val(props, PROP_WATER_QUANTITY, own_water)
                own_found    = True
                continue

            # --- Состояния союзников ---
            if eurn in _ENTITY_URN_TO_AGENT_TYPE and eid != agent_id:
                ally_agent_type = _ENTITY_URN_TO_AGENT_TYPE[eurn]
                x      = _defined_int_val(props, PROP_X, 0)
                y      = _defined_int_val(props, PROP_Y, 0)
                pos_id = _defined_int_val(props, PROP_POSITION, 0)
                water  = _defined_int_val(props, PROP_WATER_QUANTITY, 0)
                ally = AgentState(
                    id=eid,
                    type=ally_agent_type,
                    position=Position(entity_id=pos_id, x=x, y=y),
                    resources=Resources(water_quantity=water, is_transporting=False),
                )
                ally_states.append(ally)
                continue

            # --- Наблюдаемые задачи (здания, гражданские, завалы) ---
            entity_type = _ENTITY_URN_TO_ENTITY_TYPE.get(eurn)
            if entity_type is None:
                continue  # Я пропускаю неизвестные типы (дороги, мировые объекты и т.д.)

            # Я определяю, перевозит ли наш агент гражданского: если гражданский
            # находится на позиции нашего агента (его PROP_POSITION = agent_id),
            # значит медик везёт этого гражданского в убежище.
            if eurn == ENT_CIVILIAN and PROP_POSITION in props and props[PROP_POSITION].defined:
                # Я читаю intValue только при defined=True: дельта-обновление с
                # defined=False означает «значение не изменилось», а intValue будет 0
                # (protobuf-дефолт), что исказит проверку транспортировки.
                civilian_pos_id = props[PROP_POSITION].intValue
                # Я проверяю: гражданский загружен в этого агента, если его PROP_POSITION == agent_id.
                # В RCRS при AKLoad позиция гражданского устанавливается равной ID медика.
                if civilian_pos_id == agent_id:
                    own_transporting = True
                    logger.debug(
                        "Я обнаружил перевозимого гражданского entity_id=%d на агенте %d",
                        eid, agent_id,
                    )

            raw = _parse_raw_sensor_data(props, entity_urn=eurn)
            # Я читаю координаты сущности с проверкой defined-флага, чтобы
            # дельта-обновление без изменения координат не перезаписало их нулём.
            entity_x: int | None = None
            entity_y: int | None = None
            if PROP_X in props and props[PROP_X].defined:
                entity_x = props[PROP_X].intValue
            if PROP_Y in props and props[PROP_Y].defined:
                entity_y = props[PROP_Y].intValue

            entity = VisibleEntity(
                id=eid,
                type=entity_type,
                raw_sensor_data=raw,
                computed_metrics=ComputedMetrics(
                    # Я оставляю path_distance=0 — она будет заполнена
                    # алгоритмом A* в слое навигации (UC-6), вне этого слоя.
                    path_distance=0.0,
                    estimated_death_time=estimate_death_time(raw),
                    total_area=compute_total_area(raw),
                ),
                utility_score=0.0,
                entity_x=entity_x,
                entity_y=entity_y,
            )
            visible_entities.append(entity)

    # Я определяю, какие союзники перевозят гражданских: если PROP_POSITION
    # гражданского совпадает с entity_id союзника — этот союзник уже загрузил
    # гражданского. Без этой проверки f_social и распределение целей не учитывают,
    # что союзный медик уже занят, и могут отправить второго медика к той же цели.
    ally_transporting_ids: set[int] = set()
    ally_id_set = {a.id for a in ally_states}
    for ve in visible_entities:
        if ve.type == EntityType.CIVILIAN and ve.raw_sensor_data.position_on_edge is not None:
            civ_pos = ve.raw_sensor_data.position_on_edge
            if civ_pos in ally_id_set:
                ally_transporting_ids.add(civ_pos)
    if ally_transporting_ids:
        ally_states = [
            ally.model_copy(update={
                "resources": ally.resources.model_copy(update={"is_transporting": True}),
            }) if ally.id in ally_transporting_ids else ally
            for ally in ally_states
        ]

    if not own_found:
        logger.warning(
            "Я не нашёл собственного состояния агента agent_id=%d в KASense такта %d",
            agent_id, tick,
        )

    own_state = AgentState(
        id=agent_id,
        type=agent_type,
        position=own_position,
        resources=Resources(water_quantity=own_water, is_transporting=own_transporting),
    )

    # Я извлекаю список удалённых ядром сущностей из ChangeSet.deletes —
    # расчищенные завалы, спасённые гражданские и т.д. должны быть удалены из кэша.
    deleted_ids: list[int] = []
    if COMP_UPDATES in proto.components:
        deleted_ids = list(proto.components[COMP_UPDATES].changeSet.deletes)

    packet = PerceptionPacket(
        tick=tick,
        own_state=own_state,
        visible_entities=visible_entities,
        ally_states=ally_states,
        map_nodes=[],
        map_edges=[],
        deleted_entity_ids=deleted_ids,
    )

    logger.debug(
        "Я разобрал KASense такта %d: сущностей=%d, союзников=%d",
        tick, len(visible_entities), len(ally_states),
    )
    return packet


# ===========================================================================
# Вспомогательные функции разбора свойств
# ===========================================================================

def _parse_raw_sensor_data(props: dict[int, Any], entity_urn: int = 0) -> RawSensorData:
    """Я извлекаю сенсорные поля из словаря PropertyProto по соответствующим URN.

    Параметр entity_urn определяет тип сущности: для гражданских и завалов
    я заполняю position_on_edge, для остальных типов — оставляю None.
    """

    def _int(urn: int) -> Optional[int]:
        """Я возвращаю int-значение свойства или None, если свойство не определено."""
        if urn not in props:
            return None
        p = props[urn]
        if not p.defined:
            return None
        return p.intValue

    def _float(urn: int) -> Optional[float]:
        """Я возвращаю float-значение свойства (temperature хранится как intValue в RCRS)."""
        if urn not in props:
            return None
        p = props[urn]
        if not p.defined:
            return None
        # Я привожу к float, т.к. Temperature в Java IntProperty, а модель ожидает float.
        return float(p.intValue)

    return RawSensorData(
        hp=_int(PROP_HP),
        damage=_int(PROP_DAMAGE),
        buriedness=_int(PROP_BURIEDNESS),
        temperature=_float(PROP_TEMPERATURE),
        fieryness=_int(PROP_FIERYNESS),
        floors=_int(PROP_FLOORS),
        ground_area=_int(PROP_BUILDING_AREA_GROUND),
        repair_cost=_int(PROP_REPAIR_COST),
        position_on_edge=_int(PROP_POSITION) if entity_urn in (ENT_CIVILIAN, ENT_BLOCKADE) else None,
    )



__all__ = [
    # URN-константы
    "URN_AK_CONNECT", "URN_AK_ACKNOWLEDGE", "URN_KA_CONNECT_OK",
    "URN_KA_CONNECT_ERROR", "URN_KA_SENSE",
    "COMP_REQUEST_ID", "COMP_AGENT_ID", "COMP_VERSION", "COMP_NAME",
    "COMP_ENTITY_TYPES", "COMP_ENTITIES", "COMP_TIME", "COMP_UPDATES",
    "ENT_CIVILIAN", "ENT_FIRE_BRIGADE", "ENT_AMBULANCE_TEAM", "ENT_POLICE_FORCE",
    "ENT_BUILDING", "ENT_ROAD", "ENT_BLOCKADE",
    "PROP_X", "PROP_Y", "PROP_HP", "PROP_DAMAGE", "PROP_BURIEDNESS",
    "PROP_TEMPERATURE", "PROP_FIERYNESS", "PROP_FLOORS",
    "PROP_BUILDING_AREA_GROUND", "PROP_REPAIR_COST", "PROP_WATER_QUANTITY",
    "PROP_EDGES", "PROP_BLOCKADES",
    "MSG_AK_MOVE", "MSG_AK_RESCUE", "MSG_AK_EXTINGUISH",
    "MSG_AK_CLEAR", "MSG_AK_LOAD", "MSG_AK_UNLOAD", "MSG_AK_REST",
    # Фреймирование
    "pack_frame", "unpack_frame_length",
    # Сборка команд
    "build_ak_connect", "build_ak_acknowledge",
    "build_ak_move", "build_ak_rescue", "build_ak_extinguish",
    "build_ak_clear", "build_ak_load", "build_ak_unload", "build_ak_rest",
    # Разбор ответов ядра
    "parse_ka_connect_ok", "parse_ka_sense",
]
