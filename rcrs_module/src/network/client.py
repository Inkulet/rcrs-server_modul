from __future__ import annotations

"""В этом модуле я реализую TCP-клиент для связи с ядром RCRS Kernel.

Я разделяю ответственность так:
- Этот файл: транспортный уровень (сокет, чтение/запись фреймов, рукопожатие).
- codec.py:   протокольный уровень (сборка и разбор Protobuf-сообщений).
- client.py не содержит бизнес-логики и не знает о функции полезности.
"""

import logging
import socket
import struct
from typing import List, Optional

from network.codec import (
    ENT_AMBULANCE_TEAM,
    ENT_FIRE_BRIGADE,
    ENT_POLICE_FORCE,
    URN_KA_CONNECT_ERROR,
    URN_KA_CONNECT_OK,
    URN_KA_SENSE,
    build_ak_acknowledge,
    build_ak_clear,
    build_ak_connect,
    build_ak_extinguish,
    build_ak_load,
    build_ak_move,
    build_ak_rescue,
    build_ak_rest,
    build_ak_unload,
    parse_ka_connect_ok,
    parse_ka_sense,
    unpack_frame_length,
)
from network.proto.RCRSProto_pb2 import MessageProto
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
)


logger = logging.getLogger(__name__)

# Я сопоставляю тип агента с URN для формирования AKConnect.requestedEntityTypes.
_AGENT_TYPE_TO_ENTITY_URN: dict[AgentType, int] = {
    AgentType.FIRE_BRIGADE:   ENT_FIRE_BRIGADE,
    AgentType.AMBULANCE_TEAM: ENT_AMBULANCE_TEAM,
    AgentType.POLICE_FORCE:   ENT_POLICE_FORCE,
}



class RCRSClient:
    """В этом классе я инкапсулирую TCP-соединение с ядром RCRS Kernel.

    При mock=True я возвращаю детерминированные тестовые данные вместо
    реального протокола — это позволяет запускать тесты без запущенного ядра.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,
        mock: bool = False,
    ) -> None:
        """Здесь я инициализирую параметры соединения и режим работы клиента."""
        self.host    = host
        self.port    = port
        self.timeout = timeout
        self.mock    = mock

        self._socket: Optional[socket.socket] = None
        self._agent_id:   int       = 0
        self._agent_type: AgentType = AgentType.FIRE_BRIGADE
        # Я использую персистентный буфер для защиты от TCP-рассинхронизации:
        # если таймаут прервал чтение на середине фрейма, уже прочитанные байты
        # сохраняются и будут использованы при следующем вызове _recv_proto.
        self._recv_buf: bytearray = bytearray()
        # Я сохраняю состояние агента между тактами для корректной обработки
        # дельта-обновлений: если ChangeSet не содержит PROP_POSITION или данных
        # о перевозке, предыдущие значения сохраняются вместо сброса в 0/False.
        self._prev_position: Optional[Position]  = None
        self._prev_water: int       = 0
        self._prev_transporting: bool = False
        # Я использую счётчик тактов только в mock-режиме.
        self._mock_tick: int = 0

    # ===========================================================================
    # Управление соединением
    # ===========================================================================

    def connect(self) -> None:
        """Здесь я открываю TCP-соединение с ядром симулятора."""
        if self.mock:
            logger.info("Я работаю в mock-режиме, реальное соединение не устанавливается")
            return

        if self._socket is not None:
            return

        try:
            self._recv_buf.clear()
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            logger.info("Я установил TCP-соединение с RCRS Kernel на %s:%d", self.host, self.port)
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            self._socket = None
            logger.error("Я не смог подключиться к RCRS Kernel: %s", exc)
            raise

    def disconnect(self) -> None:
        """Здесь я корректно закрываю TCP-соединение."""
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None
            self._recv_buf.clear()
            logger.info("Я закрыл соединение с RCRS Kernel")

    # ===========================================================================
    # Рукопожатие (выполняется один раз после connect())
    # ===========================================================================

    def handshake(self, agent_name: str, agent_type: AgentType) -> int:
        """Здесь я провожу рукопожатие AKConnect → KAConnectOK и возвращаю agent_id.

        Последовательность:
        1. Отправляю AKConnect (имя агента, тип сущности).
        2. Получаю KAConnectOK → извлекаю agent_id и топологию карты.
        3. Отправляю AKAcknowledge.

        Возвращаю agent_id, присвоенный ядром.
        """
        if self.mock:
            self._agent_id           = 1
            self._agent_type         = agent_type
            self._initial_map_nodes  = []
            self._initial_map_edges  = []
            self._initial_refuge_ids = []
            logger.info("Я провёл mock-рукопожатие: agent_id=%d", self._agent_id)
            return self._agent_id

        entity_urn = _AGENT_TYPE_TO_ENTITY_URN[agent_type]
        request_id = 1

        # Шаг 1: отправляю AKConnect
        connect_frame = build_ak_connect(request_id, agent_name, [entity_urn])
        self._send_raw(connect_frame)
        logger.info("Я отправил AKConnect: name=%s, type=%s", agent_name, agent_type)

        # Шаг 2: получаю ответ ядра
        proto = self._recv_proto()
        if proto.urn == URN_KA_CONNECT_ERROR:
            # Я извлекаю строковую причину из компонента COMP_REASON (0x020A).
            reason_comp = proto.components.get(0x020A)
            reason = reason_comp.stringValue if reason_comp is not None else "unknown"
            logger.error("Я получил KAConnectError от ядра: %s", reason)
            raise ConnectionError(f"Ядро отказало в соединении: {reason}")

        if proto.urn != URN_KA_CONNECT_OK:
            logger.error("Я ожидал KAConnectOK (0x%04X), получил URN=0x%04X", URN_KA_CONNECT_OK, proto.urn)
            raise ConnectionError(f"Неожиданный URN при рукопожатии: 0x{proto.urn:04X}")

        req_id, agent_id, map_nodes, map_edges, refuge_ids = parse_ka_connect_ok(proto)

        self._agent_id   = agent_id
        self._agent_type = agent_type
        # Я сохраняю данные карты и убежищ для передачи в первом PerceptionPacket.
        self._initial_map_nodes  = map_nodes
        self._initial_map_edges  = map_edges
        self._initial_refuge_ids = refuge_ids

        # Шаг 3: отправляю AKAcknowledge (с agent_id — ядро делает getEntityID() на нём)
        ack_frame = build_ak_acknowledge(req_id, agent_id)
        self._send_raw(ack_frame)

        # Я устанавливаю длинный таймаут вместо блокирующего режима после рукопожатия.
        # До рукопожатия короткий таймаут нужен для детектирования «ядро не запущено».
        # После рукопожатия ядро может отправить первый KASense через 30+ секунд
        # (ожидание всех агентов), поэтому таймаут должен быть большим.
        # Я использую 120 с вместо None, чтобы SIGTERM мог прервать recv() через TimeoutError.
        if self._socket is not None:
            self._socket.settimeout(120.0)

        logger.info(
            "Я завершил рукопожатие: agent_id=%d, карта=%d узлов, %d рёбер",
            agent_id, len(map_nodes), len(map_edges),
        )
        return agent_id

    # ===========================================================================
    # Восприятие (вызывается каждый такт)
    # ===========================================================================

    def receive_sense(self) -> PerceptionPacket:
        """Здесь я получаю KASense от ядра и возвращаю PerceptionPacket.

        В mock-режиме я возвращаю детерминированные тестовые данные.
        """
        if self.mock:
            return self._make_mock_packet()

        proto = self._recv_proto()
        if proto.urn != URN_KA_SENSE:
            logger.error(
                "Я ожидал KASense (0x%04X), получил URN=0x%04X",
                URN_KA_SENSE, proto.urn,
            )
            raise ConnectionError(f"Неожиданный URN вместо KASense: 0x{proto.urn:04X}")

        packet = parse_ka_sense(
            proto,
            self._agent_id,
            self._agent_type,
            prev_position=self._prev_position,
            prev_water=self._prev_water,
            prev_transporting=self._prev_transporting,
        )

        # Я сохраняю состояние агента для следующего такта — дельта-обновления
        # ядра RCRS могут не содержать неизменённые свойства.
        self._prev_position     = packet.own_state.position
        self._prev_water        = packet.own_state.resources.water_quantity
        self._prev_transporting = packet.own_state.resources.is_transporting

        # Я прикрепляю данные карты и убежищ только к первому такту (приходят в KAConnectOK).
        # Я проверяю наличие ЛЮБЫХ инициализационных данных (узлы, рёбра или убежища),
        # а не только узлов — в редком кейсе «убежища есть, а узлов нет» данные
        # убежищ не должны теряться.
        _has_initial = (
            hasattr(self, "_initial_map_nodes")
            and (self._initial_map_nodes or self._initial_map_edges or self._initial_refuge_ids)
        )
        if _has_initial:
            packet = packet.model_copy(update={
                "map_nodes":   self._initial_map_nodes,
                "map_edges":   self._initial_map_edges,
                "refuge_ids":  self._initial_refuge_ids,
            })
            self._initial_map_nodes  = []
            self._initial_map_edges  = []
            self._initial_refuge_ids = []

        return packet

    # ===========================================================================
    # Команды (вызываются после выбора цели)
    # ===========================================================================

    def send_move(self, time: int, path: List[int], dest_x: int = -1, dest_y: int = -1) -> None:
        """Я отправляю AKMove — движение агента по маршруту path."""
        self._send_command(build_ak_move(self._agent_id, time, path, dest_x, dest_y))
        logger.debug("Я отправил AKMove: time=%d, path=%s", time, path)

    def send_rescue(self, time: int, target_id: int) -> None:
        """Я отправляю AKRescue — команду спасения гражданского."""
        self._send_command(build_ak_rescue(self._agent_id, time, target_id))
        logger.info("Я отправил AKRescue: time=%d, target=%d", time, target_id)

    def send_extinguish(self, time: int, target_id: int, water: int) -> None:
        """Я отправляю AKExtinguish — команду тушения здания."""
        self._send_command(build_ak_extinguish(self._agent_id, time, target_id, water))
        logger.info("Я отправил AKExtinguish: time=%d, target=%d, water=%d", time, target_id, water)

    def send_clear(self, time: int, target_id: int) -> None:
        """Я отправляю AKClear — команду расчистки завала."""
        self._send_command(build_ak_clear(self._agent_id, time, target_id))
        logger.info("Я отправил AKClear: time=%d, target=%d", time, target_id)

    def send_load(self, time: int, target_id: int) -> None:
        """Я отправляю AKLoad — команду погрузки гражданского."""
        self._send_command(build_ak_load(self._agent_id, time, target_id))
        logger.info("Я отправил AKLoad: time=%d, target=%d", time, target_id)

    def send_unload(self, time: int) -> None:
        """Я отправляю AKUnload — команду выгрузки гражданского."""
        self._send_command(build_ak_unload(self._agent_id, time))
        logger.info("Я отправил AKUnload: time=%d", time)

    def send_rest(self, time: int) -> None:
        """Я отправляю AKRest — команду ожидания (агент бездействует такт)."""
        self._send_command(build_ak_rest(self._agent_id, time))
        logger.debug("Я отправил AKRest: time=%d", time)

    # ===========================================================================
    # Транспортный уровень — приватные методы
    # ===========================================================================

    def _send_raw(self, frame: bytes) -> None:
        """Я отправляю уже сформированный фрейм [INT32 len][proto bytes] в сокет."""
        if self._socket is None:
            logger.error("Я пытаюсь отправить данные без активного соединения")
            raise ConnectionError("Нет активного соединения с ядром симулятора")
        try:
            self._socket.sendall(frame)
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as exc:
            logger.error("Я потерял соединение при отправке: %s", exc)
            raise

    def _send_command(self, frame: bytes) -> None:
        """Я оборачиваю _send_raw для команд с предупреждением в mock-режиме."""
        if self.mock:
            logger.debug("Я в mock-режиме, команда не отправляется в ядро")
            return
        self._send_raw(frame)

    def _recv_proto(self) -> MessageProto:
        """Я принимаю один фрейм из сокета и десериализую его в MessageProto.

        Формат: [4 байта INT32 big-endian: длина][N байт: protobuf].
        Соответствует StreamConnection.deserializeMessageProto() в Java.

        Я использую персистентный буфер self._recv_buf для защиты от TCP-
        рассинхронизации: если socket.timeout прерывает чтение на середине
        фрейма (заголовок прочитан, тело — частично), уже полученные байты
        сохраняются в буфере. При следующем вызове чтение продолжается
        с того же места, а не с нуля — это предотвращает фатальный desync,
        при котором остаток предыдущего сообщения был бы интерпретирован
        как заголовок нового.
        """
        if self._socket is None:
            raise ConnectionError("Нет активного соединения с ядром симулятора")

        try:
            # Шаг 1: я накапливаю 4-байтовый заголовок длины в персистентный буфер.
            # Если буфер уже содержит байты от предыдущего прерванного вызова,
            # чтение продолжается с того же места.
            while len(self._recv_buf) < 4:
                chunk = self._socket.recv(4 - len(self._recv_buf))
                if not chunk:
                    self._recv_buf.clear()
                    raise ConnectionError("Соединение закрыто ядром при чтении заголовка фрейма")
                self._recv_buf.extend(chunk)

            size = unpack_frame_length(bytes(self._recv_buf[:4]))
            if size <= 0 or size > 10_000_000:
                self._recv_buf.clear()
                raise ConnectionError(f"Некорректная длина фрейма: {size}")

            frame_len = 4 + size

            # Шаг 2: я накапливаю тело фрейма (size байт) в тот же буфер.
            while len(self._recv_buf) < frame_len:
                chunk = self._socket.recv(frame_len - len(self._recv_buf))
                if not chunk:
                    self._recv_buf.clear()
                    raise ConnectionError("Соединение закрыто ядром при чтении тела фрейма")
                self._recv_buf.extend(chunk)

            # Шаг 3: я извлекаю тело и очищаю использованные байты из буфера.
            body = bytes(self._recv_buf[4:frame_len])
            del self._recv_buf[:frame_len]

            proto = MessageProto()
            proto.ParseFromString(body)
            return proto

        except (ConnectionResetError, BrokenPipeError) as exc:
            self._recv_buf.clear()
            logger.error("Я потерял соединение при получении данных: %s", exc)
            raise
        except socket.timeout:
            # Я НЕ очищаю буфер при таймауте — частично прочитанные данные
            # сохранены и будут дочитаны при следующем вызове.
            raise

    # ===========================================================================
    # Mock-режим
    # ===========================================================================

    def _make_mock_packet(self) -> PerceptionPacket:
        """Я генерирую детерминированный пакет восприятия для тестов без ядра."""
        tick = self._mock_tick
        self._mock_tick += 1

        map_nodes: list[MapNode] = []
        map_edges: list[MapEdge] = []
        # Я добавляю убежище к mock-карте, чтобы тест NeedRefugeException мог
        # строить маршрут через nearest_refuge_path() без ошибок.
        mock_refuge_id = 9999
        if tick == 0:
            map_nodes = [
                MapNode(entity_id=1,              x=0,    y=0),
                MapNode(entity_id=101,             x=3000, y=4000),
                MapNode(entity_id=202,             x=5500, y=2000),
                MapNode(entity_id=203,             x=2500, y=1000),
                MapNode(entity_id=mock_refuge_id,  x=1000, y=500),
            ]
            map_edges = [
                MapEdge(source_id=1, target_id=101,            weight=5000.0),
                MapEdge(source_id=1, target_id=202,            weight=5900.0),
                MapEdge(source_id=1, target_id=203,            weight=2693.0),
                MapEdge(source_id=1, target_id=mock_refuge_id, weight=1118.0),
            ]

        own_state = AgentState(
            id=1,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=1, x=0, y=0),
            resources=Resources(water_quantity=5000, is_transporting=False),
        )
        ally = AgentState(
            id=2,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=203, x=2500, y=1000),
            resources=Resources(water_quantity=3000, is_transporting=False),
        )
        visible_entities = [
            VisibleEntity(
                id=101,
                type=EntityType.BUILDING,
                raw_sensor_data=RawSensorData(temperature=650.0, fieryness=2, floors=3, ground_area=120),
                computed_metrics=ComputedMetrics(path_distance=5000.0, estimated_death_time=999, total_area=360),
                utility_score=0.0,
            ),
            VisibleEntity(
                id=202,
                type=EntityType.CIVILIAN,
                raw_sensor_data=RawSensorData(hp=8500, damage=40, buriedness=15),
                computed_metrics=ComputedMetrics(path_distance=5900.0, estimated_death_time=120, total_area=0),
                utility_score=0.0,
            ),
            VisibleEntity(
                id=203,
                type=EntityType.CIVILIAN,
                raw_sensor_data=RawSensorData(hp=12000, damage=0, buriedness=0),
                computed_metrics=ComputedMetrics(path_distance=2693.0, estimated_death_time=200, total_area=0),
                utility_score=0.0,
            ),
        ]
        return PerceptionPacket(
            tick=tick,
            own_state=own_state,
            visible_entities=visible_entities,
            ally_states=[ally],
            map_nodes=map_nodes,
            map_edges=map_edges,
            # Я передаю refuge_ids только на такте 0 — так же, как ядро RCRS.
            refuge_ids=[mock_refuge_id] if tick == 0 else [],
        )


__all__ = ["RCRSClient"]
