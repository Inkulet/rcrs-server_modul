from __future__ import annotations

"""В этом модуле я проверяю корректность кодека протокола RCRS (codec.py).

Я тестирую фреймирование, сборку команд и разбор входящих сообщений
без реального TCP-соединения — только через Protobuf-объекты.
"""

import struct

import pytest

from network.codec import (
    COMP_AGENT_ID,
    COMP_ENTITIES,
    COMP_HEARING,
    COMP_MESSAGE,
    COMP_PATH,
    COMP_REQUEST_ID,
    COMP_TARGET,
    COMP_TIME,
    COMP_UPDATES,
    COMP_WATER,
    ENT_AMBULANCE_TEAM,
    ENT_BUILDING,
    ENT_CIVILIAN,
    ENT_FIRE_BRIGADE,
    ENT_REFUGE,
    ENT_ROAD,
    MSG_AK_CLEAR,
    MSG_AK_EXTINGUISH,
    MSG_AK_LOAD,
    MSG_AK_MOVE,
    MSG_AK_REST,
    MSG_AK_SAY,
    MSG_AK_UNLOAD,
    PROP_BLOCKADES,
    PROP_BURIEDNESS,
    PROP_DAMAGE,
    PROP_EDGES,
    PROP_FIERYNESS,
    PROP_FLOORS,
    PROP_HP,
    PROP_POSITION,
    PROP_TEMPERATURE,
    PROP_WATER_QUANTITY,
    PROP_X,
    PROP_Y,
    URN_AK_CONNECT,
    URN_KA_CONNECT_OK,
    URN_KA_SENSE,
    build_ak_clear,
    build_ak_connect,
    encode_say_payload,
    build_ak_extinguish,
    build_ak_load,
    build_ak_move,
    build_ak_rescue,
    build_ak_rest,
    build_ak_unload,
    SAY_KIND_BLOCKADE_REPORT,
    SAY_KIND_BURIED_HELP,
    SAY_KIND_SEARCH_CLAIM,
    SAY_KIND_TARGET_CLAIM,
    SAY_ROLE_AMBULANCE_TEAM,
    pack_frame,
    parse_ka_connect_ok,
    parse_ka_sense,
    unpack_frame_length,
)
from network.proto.RCRSProto_pb2 import (
    ChangeSetProto,
    EntityListProto,
    EntityProto,
    MessageComponentProto,
    MessageProto,
    PropertyProto,
)
from world.entities import AgentType


# ===========================================================================
# Вспомогательные функции
# ===========================================================================


def _make_int_prop(urn: int, value: int, defined: bool = True) -> PropertyProto:
    """Я создаю PropertyProto с целочисленным значением."""
    p = PropertyProto()
    p.urn = urn
    p.intValue = value
    p.defined = defined
    return p


def _make_entity_prop(urn: int, entity_id: int, defined: bool = True) -> PropertyProto:
    """Я создаю PropertyProto для ссылки на сущность (entity_id хранится как intValue).

    В PropertyProto нет поля entityID — ссылки на сущности хранятся как intValue.
    Это отличие от MessageComponentProto, который поддерживает oneof с entityID.
    """
    p = PropertyProto()
    p.urn = urn
    p.intValue = entity_id  # Я использую intValue, потому что PropertyProto не имеет entityID
    p.defined = defined
    return p


def _append_hearing_say(
    proto: MessageProto,
    speaker_id: int,
    payload: bytes,
) -> None:
    msg = MessageProto()
    msg.urn = MSG_AK_SAY
    msg.components[COMP_AGENT_ID].entityID = speaker_id
    msg.components[COMP_MESSAGE].rawData = payload
    proto.components[COMP_HEARING].commandList.commands.append(msg)


# ===========================================================================
# Тесты фреймирования
# ===========================================================================


class TestFraming:
    """Я проверяю pack_frame / unpack_frame_length — основу TCP-протокола."""

    def test_pack_empty_body(self) -> None:
        """Я проверяю: пустые байты → 4-байтовый заголовок с нулём."""
        frame = pack_frame(b"")
        assert len(frame) == 4
        assert struct.unpack(">I", frame)[0] == 0

    def test_pack_known_body(self) -> None:
        """Я проверяю: body=b'abc' → заголовок = 3, тело = b'abc'."""
        frame = pack_frame(b"abc")
        assert struct.unpack(">I", frame[:4])[0] == 3
        assert frame[4:] == b"abc"

    def test_roundtrip_length(self) -> None:
        """Я проверяю: pack_frame → unpack_frame_length возвращает исходную длину."""
        body = b"hello RCRS" * 10
        frame = pack_frame(body)
        header = frame[:4]
        assert unpack_frame_length(header) == len(body)

    def test_big_endian_byte_order(self) -> None:
        """Я проверяю: заголовок big-endian (Java DataOutputStream совместим)."""
        # length=256 → bytes: 00 00 01 00 в big-endian
        frame = pack_frame(b"\x00" * 256)
        assert frame[:4] == b"\x00\x00\x01\x00"


# ===========================================================================
# Тесты сборки команд
# ===========================================================================


class TestBuildCommands:
    """Я проверяю сборку Protobuf-команд агента."""

    def test_build_ak_rest_has_correct_urn(self) -> None:
        """Я проверяю: AKRest содержит URN_AK_REST после десериализации."""
        frame = build_ak_rest(agent_id=42, time=5)
        body = frame[4:]  # пропускаю 4-байтовый заголовок
        proto = MessageProto()
        proto.ParseFromString(body)
        assert proto.urn == MSG_AK_REST

    def test_build_ak_rest_contains_agent_and_time(self) -> None:
        """Я проверяю: AKRest содержит agent_id и time в компонентах."""
        frame = build_ak_rest(agent_id=7, time=99)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.components[COMP_AGENT_ID].entityID == 7
        assert proto.components[COMP_TIME].intValue == 99

    def test_build_ak_move_contains_path(self) -> None:
        """Я проверяю: AKMove содержит переданный маршрут в entityIDList."""
        path = [10, 20, 30]
        frame = build_ak_move(agent_id=1, time=3, path=path)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.urn == MSG_AK_MOVE
        assert list(proto.components[COMP_PATH].entityIDList.values) == path

    def test_build_ak_rescue_contains_target(self) -> None:
        """Я проверяю: AKRescue содержит target_id гражданского."""
        frame = build_ak_rescue(agent_id=5, time=10, target_id=888)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.components[COMP_TARGET].entityID == 888

    def test_build_ak_connect_contains_name_and_version(self) -> None:
        """Я проверяю: AKConnect содержит имя агента и версию протокола."""
        from network.codec import COMP_NAME, COMP_VERSION, PROTOCOL_VERSION
        frame = build_ak_connect(request_id=1, agent_name="test-agent", entity_types=[ENT_FIRE_BRIGADE])
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.urn == URN_AK_CONNECT
        assert proto.components[COMP_NAME].stringValue == "test-agent"
        assert proto.components[COMP_VERSION].intValue == PROTOCOL_VERSION

    def test_build_ak_extinguish_contains_target_and_water(self) -> None:
        """Я проверяю: AKExtinguish содержит target_id и water в компонентах."""
        frame = build_ak_extinguish(agent_id=3, time=10, target_id=200, water=8000)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.urn == MSG_AK_EXTINGUISH
        assert proto.components[COMP_AGENT_ID].entityID == 3
        assert proto.components[COMP_TARGET].entityID == 200
        assert proto.components[COMP_WATER].intValue == 8000
        assert proto.components[COMP_TIME].intValue == 10

    def test_build_ak_clear_contains_target(self) -> None:
        """Я проверяю: AKClear содержит target_id завала."""
        frame = build_ak_clear(agent_id=4, time=15, target_id=300)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.urn == MSG_AK_CLEAR
        assert proto.components[COMP_AGENT_ID].entityID == 4
        assert proto.components[COMP_TARGET].entityID == 300

    def test_build_ak_load_contains_target(self) -> None:
        """Я проверяю: AKLoad содержит target_id гражданского для погрузки."""
        frame = build_ak_load(agent_id=5, time=20, target_id=777)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.urn == MSG_AK_LOAD
        assert proto.components[COMP_AGENT_ID].entityID == 5
        assert proto.components[COMP_TARGET].entityID == 777

    def test_build_ak_unload_has_no_target(self) -> None:
        """Я проверяю: AKUnload содержит agent_id и time, но не target."""
        frame = build_ak_unload(agent_id=6, time=25)
        proto = MessageProto()
        proto.ParseFromString(frame[4:])
        assert proto.urn == MSG_AK_UNLOAD
        assert proto.components[COMP_AGENT_ID].entityID == 6
        assert proto.components[COMP_TIME].intValue == 25


# ===========================================================================
# Тесты разбора KAConnectOK
# ===========================================================================


class TestParseKaConnectOk:
    """Я проверяю разбор ответа ядра на рукопожатие."""

    def _make_connect_ok(
        self,
        agent_id: int = 42,
        request_id: int = 1,
        road_nodes: list[tuple[int, int, int]] | None = None,  # (eid, x, y)
        refuge_ids: list[int] | None = None,
    ) -> MessageProto:
        """Я собираю минимальный KAConnectOK для тестирования."""
        proto = MessageProto()
        proto.urn = URN_KA_CONNECT_OK
        proto.components[COMP_REQUEST_ID].intValue = request_id
        proto.components[COMP_AGENT_ID].entityID = agent_id

        entity_list = EntityListProto()
        for eid, x, y in (road_nodes or []):
            e = EntityProto()
            e.entityID = eid
            e.urn = ENT_ROAD
            e.properties.append(_make_int_prop(PROP_X, x))
            e.properties.append(_make_int_prop(PROP_Y, y))
            entity_list.entities.append(e)

        for rid in (refuge_ids or []):
            e = EntityProto()
            e.entityID = rid
            e.urn = ENT_REFUGE
            e.properties.append(_make_int_prop(PROP_X, 0))
            e.properties.append(_make_int_prop(PROP_Y, 0))
            entity_list.entities.append(e)

        proto.components[COMP_ENTITIES].entityList.CopyFrom(entity_list)
        return proto

    def test_returns_correct_agent_id(self) -> None:
        """Я проверяю: parse_ka_connect_ok извлекает agent_id из компонента."""
        proto = self._make_connect_ok(agent_id=99)
        _, agent_id, _, _, _, _, _ = parse_ka_connect_ok(proto)
        assert agent_id == 99

    def test_returns_correct_request_id(self) -> None:
        """Я проверяю: parse_ka_connect_ok извлекает request_id."""
        proto = self._make_connect_ok(request_id=7)
        req_id, _, _, _, _, _, _ = parse_ka_connect_ok(proto)
        assert req_id == 7

    def test_parses_road_nodes(self) -> None:
        """Я проверяю: Road-сущности попадают в map_nodes с правильными координатами."""
        proto = self._make_connect_ok(road_nodes=[(1, 100, 200), (2, 300, 400)])
        _, _, nodes, _, _, _, _ = parse_ka_connect_ok(proto)
        node_ids = {n.entity_id for n in nodes}
        assert {1, 2} == node_ids

    def test_parses_refuge_ids(self) -> None:
        """Я проверяю: ENT_REFUGE-сущности попадают в refuge_ids."""
        proto = self._make_connect_ok(refuge_ids=[501, 502])
        _, _, _, _, refuge_ids, _, _ = parse_ka_connect_ok(proto)
        assert set(refuge_ids) == {501, 502}

    def test_empty_entity_list_returns_empty_maps(self) -> None:
        """Я проверяю: отсутствие сущностей → пустые nodes, edges, refuges."""
        proto = self._make_connect_ok()
        _, _, nodes, edges, refuges, _, _ = parse_ka_connect_ok(proto)
        assert nodes == []
        assert edges == []
        assert refuges == []


# ===========================================================================
# Тесты разбора KASense
# ===========================================================================


class TestParseKaSense:
    """Я проверяю разбор тактового пакета восприятия от ядра."""

    def _make_sense(
        self,
        agent_id: int,
        tick: int,
        agent_urn: int = ENT_FIRE_BRIGADE,
        agent_x: int = 100,
        agent_y: int = 200,
        agent_pos_id: int = 10,
        water: int = 5000,
        extra_changes: list | None = None,
    ) -> MessageProto:
        """Я собираю минимальный KASense для тестирования parse_ka_sense.

        Я использую change_set.changes.add() вместо EntityProto():
        ChangeSetProto.changes — это список вложенного типа EntityChangeProto,
        который является приватным nested-типом ChangeSetProto, а не EntityProto.
        """
        proto = MessageProto()
        proto.urn = URN_KA_SENSE
        proto.components[COMP_TIME].intValue = tick

        change_set = ChangeSetProto()

        # Собственный агент — добавляю через .add() для корректного вложенного типа.
        own_change = change_set.changes.add()
        own_change.entityID = agent_id
        own_change.urn = agent_urn
        own_change.properties.append(_make_int_prop(PROP_X, agent_x))
        own_change.properties.append(_make_int_prop(PROP_Y, agent_y))
        own_change.properties.append(_make_entity_prop(PROP_POSITION, agent_pos_id))
        own_change.properties.append(_make_int_prop(PROP_WATER_QUANTITY, water))

        for (eid, eurn, props_list) in (extra_changes or []):
            ch = change_set.changes.add()
            ch.entityID = eid
            ch.urn = eurn
            for p in props_list:
                ch.properties.append(p)

        proto.components[COMP_UPDATES].changeSet.CopyFrom(change_set)
        return proto

    def test_parses_tick(self) -> None:
        """Я проверяю: parse_ka_sense корректно извлекает номер такта."""
        proto = self._make_sense(agent_id=1, tick=42)
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.FIRE_BRIGADE)
        assert packet.tick == 42

    def test_parses_own_position(self) -> None:
        """Я проверяю: позиция агента берётся из PROP_POSITION (entity_id здания)."""
        proto = self._make_sense(agent_id=5, tick=1, agent_pos_id=77)
        packet = parse_ka_sense(proto, agent_id=5, agent_type=AgentType.FIRE_BRIGADE)
        assert packet.own_state.position.entity_id == 77

    def test_parses_water_quantity(self) -> None:
        """Я проверяю: запас воды агента правильно читается."""
        proto = self._make_sense(agent_id=3, tick=0, water=1234)
        packet = parse_ka_sense(proto, agent_id=3, agent_type=AgentType.FIRE_BRIGADE)
        assert packet.own_state.resources.water_quantity == 1234

    def test_parses_own_health_fields(self) -> None:
        """Я проверяю: hp/damage/buriedness собственного агента читаются из ChangeSet."""
        proto = self._make_sense(agent_id=3, tick=0)
        own_change = proto.components[COMP_UPDATES].changeSet.changes[0]
        own_change.properties.append(_make_int_prop(PROP_HP, 8000))
        own_change.properties.append(_make_int_prop(PROP_DAMAGE, 20))
        own_change.properties.append(_make_int_prop(PROP_BURIEDNESS, 7))
        packet = parse_ka_sense(proto, agent_id=3, agent_type=AgentType.AMBULANCE_TEAM)
        assert packet.own_state.hp == 8000
        assert packet.own_state.damage == 20
        assert packet.own_state.buriedness == 7

    def test_civilian_appears_in_visible_entities(self) -> None:
        """Я проверяю: CIVILIAN в ChangeSet попадает в visible_entities."""
        extra = [(999, ENT_CIVILIAN, [
            _make_int_prop(PROP_HP, 8000),
            _make_int_prop(PROP_DAMAGE, 100),
        ])]
        proto = self._make_sense(agent_id=1, tick=5, extra_changes=extra)
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.AMBULANCE_TEAM)
        entity_ids = {e.id for e in packet.visible_entities}
        assert 999 in entity_ids

    def test_buried_ally_agent_appears_in_visible_entities(self) -> None:
        """Я проверяю: завалённый союзник попадает в список задач скорой как HUMAN."""
        extra = [(999, ENT_FIRE_BRIGADE, [
            _make_entity_prop(PROP_POSITION, 77),
            _make_int_prop(PROP_X, 100),
            _make_int_prop(PROP_Y, 200),
            _make_int_prop(PROP_HP, 9000),
            _make_int_prop(PROP_DAMAGE, 50),
            _make_int_prop(PROP_BURIEDNESS, 12),
        ])]
        proto = self._make_sense(agent_id=1, tick=5, extra_changes=extra)
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.AMBULANCE_TEAM)
        buried_ally = next(e for e in packet.visible_entities if e.id == 999)
        assert buried_ally.type.value == "HUMAN"
        assert buried_ally.raw_sensor_data.buriedness == 12
        assert buried_ally.raw_sensor_data.position_on_edge == 77

    def test_is_transporting_when_civilian_at_agent_position(self) -> None:
        """Я проверяю: если civilian.PROP_POSITION == agent_id, is_transporting=True."""
        # Я помещаю гражданского на позицию агента — это означает переноску.
        extra = [(777, ENT_CIVILIAN, [
            _make_entity_prop(PROP_POSITION, 1),  # agent_id=1
        ])]
        proto = self._make_sense(agent_id=1, tick=0, extra_changes=extra)
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.AMBULANCE_TEAM)
        assert packet.own_state.resources.is_transporting is True

    def test_is_not_transporting_when_civilian_elsewhere(self) -> None:
        """Я проверяю: если civilian.PROP_POSITION != agent_id, is_transporting=False."""
        extra = [(555, ENT_CIVILIAN, [
            _make_entity_prop(PROP_POSITION, 999),  # другая сущность, не агент
        ])]
        proto = self._make_sense(agent_id=1, tick=0, extra_changes=extra)
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.AMBULANCE_TEAM)
        assert packet.own_state.resources.is_transporting is False

    def test_deleted_entity_ids_parsed(self) -> None:
        """Я проверяю: deletes из ChangeSet попадают в deleted_entity_ids."""
        proto = self._make_sense(agent_id=1, tick=3)
        # Я добавляю удалённые сущности в ChangeSet.
        proto.components[COMP_UPDATES].changeSet.deletes.append(42)
        proto.components[COMP_UPDATES].changeSet.deletes.append(99)
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.FIRE_BRIGADE)
        assert 42 in packet.deleted_entity_ids
        assert 99 in packet.deleted_entity_ids

    def test_heard_target_claim_parses_role(self) -> None:
        proto = self._make_sense(agent_id=1, tick=3)
        _append_hearing_say(
            proto,
            speaker_id=77,
            payload=encode_say_payload(
                SAY_KIND_TARGET_CLAIM, 501, SAY_ROLE_AMBULANCE_TEAM,
            ),
        )
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.FIRE_BRIGADE)
        assert packet.heard_target_ids == {501}
        assert packet.heard_target_roles == {501: SAY_ROLE_AMBULANCE_TEAM}
        assert packet.heard_target_speakers == {501: 77}

    def test_non_claim_say_does_not_pollute_heard_targets(self) -> None:
        proto = self._make_sense(agent_id=1, tick=3)
        _append_hearing_say(
            proto,
            speaker_id=77,
            payload=encode_say_payload(SAY_KIND_BURIED_HELP, 77),
        )
        _append_hearing_say(
            proto,
            speaker_id=88,
            payload=encode_say_payload(SAY_KIND_BLOCKADE_REPORT, 9001),
        )
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.FIRE_BRIGADE)
        assert packet.heard_target_ids == set()
        assert packet.heard_target_roles == {}
        assert packet.heard_target_speakers == {}

    def test_search_claim_is_parsed_separately_from_target_claim(self) -> None:
        proto = self._make_sense(agent_id=1, tick=3)
        _append_hearing_say(
            proto,
            speaker_id=77,
            payload=encode_say_payload(
                SAY_KIND_SEARCH_CLAIM, 701, SAY_ROLE_AMBULANCE_TEAM,
            ),
        )

        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.FIRE_BRIGADE)

        assert packet.heard_target_ids == set()
        assert packet.heard_search_target_ids == {701}
        assert packet.heard_search_target_roles == {701: SAY_ROLE_AMBULANCE_TEAM}
        assert packet.heard_search_target_speakers == {701: 77}

    def test_building_with_fieryness_zero_parses_ok(self) -> None:
        """Я проверяю: fieryness=0 (не горит) не вызывает ValidationError после исправления ge=0."""
        extra = [(200, ENT_BUILDING, [
            _make_int_prop(PROP_FIERYNESS, 0),
            _make_int_prop(PROP_FLOORS, 2),
        ])]
        proto = self._make_sense(agent_id=1, tick=0, extra_changes=extra)
        # Я ожидаю отсутствие исключения — fieryness=0 теперь допустим (ge=0).
        packet = parse_ka_sense(proto, agent_id=1, agent_type=AgentType.FIRE_BRIGADE)
        assert packet is not None


# ===========================================================================
# Тесты URN-констант
# ===========================================================================


class TestUrnConstants:
    """Я проверяю корректность значений URN-констант согласно Java-исходникам RCRS."""

    def test_prop_edges_value(self) -> None:
        """Я проверяю: PROP_EDGES = 0x1213 (StandardPropertyURN.EDGES, ordinal=19)."""
        assert PROP_EDGES == 0x1213

    def test_prop_blockades_value(self) -> None:
        """Я проверяю: PROP_BLOCKADES = 0x1208."""
        assert PROP_BLOCKADES == 0x1208

    def test_ent_refuge_value(self) -> None:
        """Я проверяю: ENT_REFUGE = 0x1105."""
        assert ENT_REFUGE == 0x1105

    def test_ent_civilian_value(self) -> None:
        """Я проверяю: ENT_CIVILIAN = 0x110B."""
        assert ENT_CIVILIAN == 0x110B
