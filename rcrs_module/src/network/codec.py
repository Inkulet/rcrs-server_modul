from __future__ import annotations


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

PROP_APEXES             = _PROP | 18   # 0x1212  IntArrayProperty — плоский список [x0,y0,x1,y1,...] вершин полигона
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
MSG_AK_CLEAR_AREA = _STD_MSG | 10  # 0x130A  AKClearArea
MSG_AK_SAY        = _STD_MSG | 5   # 0x1305  AKSay — расчистка в направлении (destX, destY)

# --- Компоненты командных сообщений (StandardMessageComponentURN) ---
COMP_TARGET   = _STD_CMP | 1  # 0x1401
COMP_DEST_X   = _STD_CMP | 2  # 0x1402
COMP_DEST_Y   = _STD_CMP | 3  # 0x1403
COMP_WATER    = _STD_CMP | 4  # 0x1404
COMP_PATH     = _STD_CMP | 5  # 0x1405
COMP_MESSAGE  = _STD_CMP | 6  # 0x1406  Message — содержимое AKSay/AKSpeak (rawData)

_ENTITY_URN_TO_AGENT_TYPE: dict[int, AgentType] = {
    ENT_FIRE_BRIGADE:    AgentType.FIRE_BRIGADE,
    ENT_AMBULANCE_TEAM:  AgentType.AMBULANCE_TEAM,
    ENT_POLICE_FORCE:    AgentType.POLICE_FORCE,
    ENT_FIRE_STATION:    AgentType.FIRE_STATION,
    ENT_AMBULANCE_CENTRE: AgentType.AMBULANCE_CENTRE,
    ENT_POLICE_OFFICE:   AgentType.POLICE_OFFICE,
}

_RESCUABLE_AGENT_URNS: frozenset[int] = frozenset({
    ENT_FIRE_BRIGADE,
    ENT_AMBULANCE_TEAM,
    ENT_POLICE_FORCE,
})

_ENTITY_URN_TO_ENTITY_TYPE: dict[int, EntityType] = {
    ENT_CIVILIAN:         EntityType.CIVILIAN,
    ENT_BUILDING:         EntityType.BUILDING,
    ENT_REFUGE:           EntityType.BUILDING,
    ENT_GAS_STATION:      EntityType.BUILDING,
    ENT_FIRE_STATION:     EntityType.BUILDING,
    ENT_AMBULANCE_CENTRE: EntityType.BUILDING,
    ENT_POLICE_OFFICE:    EntityType.BUILDING,
    ENT_HYDRANT:          EntityType.BUILDING,
    ENT_BLOCKADE:         EntityType.BLOCKADE,
}

PROTOCOL_VERSION: int = 2


def pack_frame(proto_bytes: bytes) -> bytes:
    return struct.pack(">I", len(proto_bytes)) + proto_bytes


def unpack_frame_length(header: bytes) -> int:
    return struct.unpack(">I", header)[0]


def build_ak_connect(
    request_id: int,
    agent_name: str,
    entity_types: List[int],
) -> bytes:
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
    msg = MessageProto()
    msg.urn = MSG_AK_MOVE
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time

    msg.components[COMP_PATH].entityIDList.values.extend(path)

    msg.components[COMP_DEST_X].intValue = dest_x
    msg.components[COMP_DEST_Y].intValue = dest_y

    logger.debug("Я собрал AKMove: agent=%d, time=%d, path=%s", agent_id, time, path)
    return pack_frame(msg.SerializeToString())


def build_ak_rescue(agent_id: int, time: int, target_id: int) -> bytes:
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
    msg = MessageProto()
    msg.urn = MSG_AK_CLEAR
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_TARGET].entityID   = target_id
    logger.debug("Я собрал AKClear: agent=%d, time=%d, target=%d", agent_id, time, target_id)
    return pack_frame(msg.SerializeToString())


def build_ak_clear_area(agent_id: int, time: int, dest_x: int, dest_y: int) -> bytes:
    msg = MessageProto()
    msg.urn = MSG_AK_CLEAR_AREA
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_DEST_X].intValue   = dest_x
    msg.components[COMP_DEST_Y].intValue   = dest_y
    logger.debug(
        "Я собрал AKClearArea: agent=%d, time=%d, dest=(%d,%d)",
        agent_id, time, dest_x, dest_y,
    )
    return pack_frame(msg.SerializeToString())


def build_ak_load(agent_id: int, time: int, target_id: int) -> bytes:
    msg = MessageProto()
    msg.urn = MSG_AK_LOAD
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_TARGET].entityID   = target_id
    return pack_frame(msg.SerializeToString())


def build_ak_unload(agent_id: int, time: int) -> bytes:
    msg = MessageProto()
    msg.urn = MSG_AK_UNLOAD
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    return pack_frame(msg.SerializeToString())


def build_ak_rest(agent_id: int, time: int) -> bytes:
    msg = MessageProto()
    msg.urn = MSG_AK_REST
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    return pack_frame(msg.SerializeToString())


def build_ak_say(agent_id: int, time: int, data: bytes) -> bytes:
    msg = MessageProto()
    msg.urn = MSG_AK_SAY
    msg.components[COMP_AGENT_ID].entityID = agent_id
    msg.components[COMP_TIME].intValue     = time
    msg.components[COMP_MESSAGE].rawData   = data
    return pack_frame(msg.SerializeToString())


def _get_int(comp: MessageComponentProto) -> int:
    which = comp.WhichOneof("component")
    if which == "intValue":
        return comp.intValue
    if which == "entityID":
        return comp.entityID
    return 0


def parse_ka_connect_ok(
    proto: MessageProto,
) -> tuple[int, int, list[MapNode], list[MapEdge], list[int], Position, int]:
    import math as _math

    request_id = _get_int(proto.components[COMP_REQUEST_ID]) if COMP_REQUEST_ID in proto.components else 0
    agent_id   = _get_int(proto.components[COMP_AGENT_ID])   if COMP_AGENT_ID   in proto.components else 0

    map_nodes:  list[MapNode] = []
    map_edges:  list[MapEdge] = []
    refuge_ids: list[int]     = []

    initial_position = Position(entity_id=0, x=0, y=0)
    initial_water: int = 0

    _AREA_URNS = frozenset((ENT_ROAD, ENT_BUILDING, ENT_REFUGE, ENT_FIRE_STATION,
                            ENT_AMBULANCE_CENTRE, ENT_POLICE_OFFICE, ENT_HYDRANT,
                            ENT_GAS_STATION))

    node_coords: dict[int, tuple[int, int]] = {}  # eid → (x, y)
    _entity_protos_area = []  # буфер для второго прохода

    if COMP_ENTITIES in proto.components:
        entity_list = proto.components[COMP_ENTITIES].entityList
        for entity_proto in entity_list.entities:
            urn = entity_proto.urn
            eid = entity_proto.entityID
            props = {p.urn: p for p in entity_proto.properties}

            if eid == agent_id:
                pos_id = props[PROP_POSITION].intValue if PROP_POSITION in props and props[PROP_POSITION].defined else 0
                x = props[PROP_X].intValue if PROP_X in props and props[PROP_X].defined else 0
                y = props[PROP_Y].intValue if PROP_Y in props and props[PROP_Y].defined else 0
                initial_position = Position(entity_id=pos_id, x=x, y=y)
                if PROP_WATER_QUANTITY in props and props[PROP_WATER_QUANTITY].defined:
                    initial_water = props[PROP_WATER_QUANTITY].intValue
                logger.info(
                    "Я извлёк начальную позицию агента agent_id=%d: pos=%d, x=%d, y=%d, water=%d",
                    agent_id, pos_id, x, y, initial_water,
                )

            if urn in _AREA_URNS:
                x = props[PROP_X].intValue if PROP_X in props else 0
                y = props[PROP_Y].intValue if PROP_Y in props else 0
                area_type = None
                if urn == ENT_ROAD:
                    area_type = "ROAD"
                elif urn == ENT_REFUGE:
                    area_type = "REFUGE"
                elif urn == ENT_HYDRANT:
                    area_type = "HYDRANT"
                else:
                    area_type = "BUILDING"
                area_apexes: Optional[list[int]] = None
                if PROP_APEXES in props and props[PROP_APEXES].defined:
                    try:
                        apexes_values = list(props[PROP_APEXES].intList.values)
                        if apexes_values:
                            area_apexes = apexes_values
                    except (AttributeError, TypeError):
                        pass
                map_nodes.append(
                    MapNode(
                        entity_id=eid,
                        x=x,
                        y=y,
                        area_type=area_type,
                        apexes=area_apexes,
                    )
                )
                node_coords[eid] = (x, y)
                _entity_protos_area.append((eid, urn, props))

            if urn == ENT_REFUGE:
                refuge_ids.append(eid)

    _seen_edges: set[tuple[int, int]] = set()  # дедупликация (a,b) ↔ (b,a)

    for eid, urn, props in _entity_protos_area:
        if PROP_EDGES in props:
            for edge_proto in props[PROP_EDGES].edgeList.edges:
                neighbour = edge_proto.neighbour
                if neighbour > 0:
                    key = (min(eid, neighbour), max(eid, neighbour))
                    if key not in _seen_edges:
                        _seen_edges.add(key)
                        if neighbour in node_coords:
                            sx, sy = node_coords[eid]
                            tx, ty = node_coords[neighbour]
                            weight = max(1.0, _math.hypot(tx - sx, ty - sy))
                        else:
                            sx, sy = node_coords[eid]
                            mid_x = (edge_proto.startX + edge_proto.endX) / 2
                            mid_y = (edge_proto.startY + edge_proto.endY) / 2
                            weight = max(1.0, _math.hypot(mid_x - sx, mid_y - sy))
                        map_edges.append(MapEdge(source_id=eid, target_id=neighbour, weight=weight))


    logger.info(
        "Я разобрал KAConnectOK: agent_id=%d, узлов=%d, рёбер=%d, убежищ=%d, start_pos=%d, water=%d",
        agent_id, len(map_nodes), len(map_edges), len(refuge_ids), initial_position.entity_id, initial_water,
    )
    return request_id, agent_id, map_nodes, map_edges, refuge_ids, initial_position, initial_water


def _defined_int_val(props: dict[int, Any], urn: int, default: int) -> int:
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
    tick = _get_int(proto.components[COMP_TIME]) if COMP_TIME in proto.components else 0

    own_position = prev_position if prev_position is not None else Position(entity_id=0, x=0, y=0)
    own_water    = prev_water
    own_transporting = prev_transporting
    own_hp: int | None = None
    own_damage: int | None = None
    own_buriedness: int | None = None
    own_found    = False

    visible_entities: list[VisibleEntity] = []
    ally_states:       list[AgentState]   = []
    blockade_to_road: dict[int, int]      = {}
    road_blockades: dict[int, list[int]]  = {}

    if COMP_UPDATES in proto.components:
        change_set: ChangeSetProto = proto.components[COMP_UPDATES].changeSet

        for change in change_set.changes:
            eid  = change.entityID
            eurn = change.urn
            props = {p.urn: p for p in change.properties}

            if eid == agent_id and eurn in _ENTITY_URN_TO_AGENT_TYPE:
                x      = _defined_int_val(props, PROP_X, own_position.x)
                y      = _defined_int_val(props, PROP_Y, own_position.y)
                pos_id = _defined_int_val(props, PROP_POSITION, own_position.entity_id)
                own_position = Position(entity_id=pos_id, x=x, y=y)
                own_water    = _defined_int_val(props, PROP_WATER_QUANTITY, own_water)
                hp = _defined_int_val(props, PROP_HP, -1)
                damage = _defined_int_val(props, PROP_DAMAGE, -1)
                buriedness = _defined_int_val(props, PROP_BURIEDNESS, -1)
                own_hp = None if hp < 0 else hp
                own_damage = None if damage < 0 else damage
                own_buriedness = None if buriedness < 0 else buriedness
                own_found    = True
                continue

            if eurn in _ENTITY_URN_TO_AGENT_TYPE and eid != agent_id:
                ally_agent_type = _ENTITY_URN_TO_AGENT_TYPE[eurn]
                x      = _defined_int_val(props, PROP_X, 0)
                y      = _defined_int_val(props, PROP_Y, 0)
                pos_id = _defined_int_val(props, PROP_POSITION, 0)
                water  = _defined_int_val(props, PROP_WATER_QUANTITY, 0)
                hp     = _defined_int_val(props, PROP_HP, -1)
                damage = _defined_int_val(props, PROP_DAMAGE, -1)
                buriedness = _defined_int_val(props, PROP_BURIEDNESS, -1)
                ally = AgentState(
                    id=eid,
                    type=ally_agent_type,
                    position=Position(entity_id=pos_id, x=x, y=y),
                    resources=Resources(water_quantity=water, is_transporting=False),
                    hp=None if hp < 0 else hp,
                    damage=None if damage < 0 else damage,
                    buriedness=None if buriedness < 0 else buriedness,
                )
                ally_states.append(ally)

                if eurn in _RESCUABLE_AGENT_URNS:
                    raw = RawSensorData(
                        hp=None if hp < 0 else hp,
                        damage=None if damage < 0 else damage,
                        buriedness=None if buriedness < 0 else buriedness,
                        position_on_edge=pos_id,
                    )
                    visible_entities.append(VisibleEntity(
                        id=eid,
                        type=EntityType.HUMAN,
                        raw_sensor_data=raw,
                        computed_metrics=ComputedMetrics(
                            path_distance=0.0,
                            estimated_death_time=estimate_death_time(raw),
                            total_area=0,
                        ),
                        utility_score=0.0,
                        entity_x=x,
                        entity_y=y,
                        is_ally=True,
                    ))
                continue

            # Для дорог извлекаю PROP_BLOCKADES — список завалов на этой
            # дороге. Строю обратный индекс blockade_id → road_id как
            # резервный источник position_on_edge (на случай, если
            # PROP_POSITION для завала не пришёл в ChangeSet).
            if eurn == ENT_ROAD and PROP_BLOCKADES in props and props[PROP_BLOCKADES].defined:
                try:
                    blk_ids = list(props[PROP_BLOCKADES].intList.values)
                    road_blockades[eid] = blk_ids
                    for blk_id in blk_ids:
                        blockade_to_road[blk_id] = eid
                except (AttributeError, TypeError):
                    road_blockades[eid] = []

            entity_type = _ENTITY_URN_TO_ENTITY_TYPE.get(eurn)
            if entity_type is None:
                continue

            if eurn == ENT_CIVILIAN and PROP_POSITION in props and props[PROP_POSITION].defined:
                civilian_pos_id = props[PROP_POSITION].intValue
                if civilian_pos_id == agent_id:
                    own_transporting = True
                    logger.debug(
                        "Я обнаружил перевозимого гражданского entity_id=%d на агенте %d",
                        eid, agent_id,
                    )

            raw = _parse_raw_sensor_data(props, entity_urn=eurn)
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
                    path_distance=0.0,
                    estimated_death_time=estimate_death_time(raw),
                    total_area=compute_total_area(raw),
                ),
                utility_score=0.0,
                entity_x=entity_x,
                entity_y=entity_y,
            )
            visible_entities.append(entity)

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
        hp=own_hp,
        damage=own_damage,
        buriedness=own_buriedness,
    )

    deleted_ids: list[int] = []
    if COMP_UPDATES in proto.components:
        deleted_ids = list(proto.components[COMP_UPDATES].changeSet.deletes)

    heard_target_ids: set[int] = set()
    if COMP_HEARING in proto.components:
        hearing_comp = proto.components[COMP_HEARING]
        if hearing_comp.HasField("commandList"):
            for cmd in hearing_comp.commandList.commands:
                if cmd.urn == MSG_AK_SAY and COMP_MESSAGE in cmd.components:
                    raw = cmd.components[COMP_MESSAGE].rawData
                    if len(raw) >= 4:
                        import struct
                        target_id = struct.unpack(">i", raw[:4])[0]
                        if target_id > 0:
                            heard_target_ids.add(target_id)
    if heard_target_ids:
        logger.debug(
            "Я услышал %d целей от соседних агентов: %s",
            len(heard_target_ids), heard_target_ids,
        )

    packet = PerceptionPacket(
        tick=tick,
        own_state=own_state,
        visible_entities=visible_entities,
        ally_states=ally_states,
        map_nodes=[],
        map_edges=[],
        deleted_entity_ids=deleted_ids,
        heard_target_ids=heard_target_ids,
        blockade_to_road=blockade_to_road,
        road_blockades=road_blockades,
    )

    logger.debug(
        "Я разобрал KASense такта %d: сущностей=%d, союзников=%d",
        tick, len(visible_entities), len(ally_states),
    )
    return packet

def _parse_raw_sensor_data(props: dict[int, Any], entity_urn: int = 0) -> RawSensorData:
    def _int(urn: int) -> Optional[int]:
        if urn not in props:
            return None
        p = props[urn]
        if not p.defined:
            return None
        return p.intValue

    def _float(urn: int) -> Optional[float]:
        if urn not in props:
            return None
        p = props[urn]
        if not p.defined:
            return None
        return float(p.intValue)

    def _int_list(urn: int) -> Optional[list[int]]:
        if urn not in props:
            return None
        p = props[urn]
        if not p.defined:
            return None
        try:
            values = list(p.intList.values)
        except (AttributeError, TypeError):
            return None
        return values if values else None

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
        apexes=_int_list(PROP_APEXES),
    )



__all__ = [
    # URN-константы
    "URN_AK_CONNECT", "URN_AK_ACKNOWLEDGE", "URN_KA_CONNECT_OK",
    "URN_KA_CONNECT_ERROR", "URN_KA_SENSE",
    "COMP_REQUEST_ID", "COMP_AGENT_ID", "COMP_VERSION", "COMP_NAME",
    "COMP_ENTITY_TYPES", "COMP_ENTITIES", "COMP_TIME", "COMP_UPDATES",
    "ENT_CIVILIAN", "ENT_FIRE_BRIGADE", "ENT_AMBULANCE_TEAM", "ENT_POLICE_FORCE",
    "ENT_FIRE_STATION", "ENT_AMBULANCE_CENTRE", "ENT_POLICE_OFFICE",
    "ENT_BUILDING", "ENT_ROAD", "ENT_BLOCKADE",
    "PROP_X", "PROP_Y", "PROP_HP", "PROP_DAMAGE", "PROP_BURIEDNESS",
    "PROP_TEMPERATURE", "PROP_FIERYNESS", "PROP_FLOORS",
    "PROP_BUILDING_AREA_GROUND", "PROP_REPAIR_COST", "PROP_WATER_QUANTITY",
    "PROP_EDGES", "PROP_BLOCKADES", "PROP_APEXES",
    "MSG_AK_MOVE", "MSG_AK_RESCUE", "MSG_AK_EXTINGUISH",
    "MSG_AK_CLEAR", "MSG_AK_LOAD", "MSG_AK_UNLOAD", "MSG_AK_REST",
    "MSG_AK_SAY", "COMP_MESSAGE",
    # Фреймирование
    "pack_frame", "unpack_frame_length",
    # Сборка команд
    "build_ak_connect", "build_ak_acknowledge",
    "build_ak_move", "build_ak_rescue", "build_ak_extinguish",
    "build_ak_clear", "build_ak_clear_area", "build_ak_load", "build_ak_unload", "build_ak_rest",
    "build_ak_say",
    # Разбор ответов ядра
    "parse_ka_connect_ok", "parse_ka_sense",
]
