from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from module.network.protocol import (
    KAConnectErrorPayload,
    KAConnectOKPayload,
    KASensePayload,
    URN,
    build_ak_acknowledge,
    build_ak_connect,
    is_shutdown,
    parse_ka_connect_error,
    parse_ka_connect_ok,
    parse_ka_sense,
    recv_message_from_socket,
    send_message_to_socket,
)


@dataclass
class ReceiveResult:
    """Результат чтения разделяет KA_SENSE и SHUTDOWN, чтобы цикл агента корректно завершался."""

    sense: Optional[KASensePayload]
    shutdown: bool


class RCRSClient:
    """Минимальный TCP-клиент RCRS: connect, handshake, sense-loop и отправка AK-команд."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7000, socket_timeout_sec: float = 15.0):
        self.host = host
        self.port = port
        self.socket_timeout_sec = socket_timeout_sec
        self._socket: Optional[socket.socket] = None

    def connect(self) -> None:
        """Открывает TCP-соединение с kernel по тому же формату, что использует Java TCPConnection."""
        if self._socket is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.socket_timeout_sec)
        sock.connect((self.host, int(self.port)))
        self._socket = sock

    def close(self) -> None:
        """Явное закрытие нужно, чтобы корректно завершать агента между эксперементами запуска."""
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def handshake_agent(
        self,
        request_id: int,
        version: int,
        agent_name: str,
        requested_entity_types: Iterable[int],
    ) -> KAConnectOKPayload:
        """Рукопожатие повторяет последовательность AbstractAgent: AK_CONNECT -> KA_CONNECT_OK -> AK_ACKNOWLEDGE."""
        self._ensure_connected()

        connect_message = build_ak_connect(
            request_id=request_id,
            version=version,
            agent_name=agent_name,
            requested_entity_types=requested_entity_types,
        )
        self.send_message(connect_message)

        while True:
            try:
                incoming = self.receive_message()
            except TimeoutError:
                # Kernel может быть занят инициализацией симуляторов; повторяем чтение, не роняя handshake.
                continue

            incoming_urn = int(incoming.urn)
            if incoming_urn == int(URN.ControlMSG.KA_CONNECT_OK):
                connect_ok = parse_ka_connect_ok(incoming)
                if connect_ok.request_id != int(request_id):
                    # Игнорируем ответ не нашего запроса, чтобы не ломать конкурентные подключения.
                    continue

                ack_message = build_ak_acknowledge(request_id=request_id, agent_id=connect_ok.agent_id)
                self.send_message(ack_message)
                return connect_ok

            if incoming_urn == int(URN.ControlMSG.KA_CONNECT_ERROR):
                connect_error: KAConnectErrorPayload = parse_ka_connect_error(incoming)
                if connect_error.request_id == int(request_id):
                    raise RuntimeError(f"KA_CONNECT_ERROR: {connect_error.reason}")

    def wait_for_sense(self, expected_agent_id: int) -> ReceiveResult:
        """Возвращает следующий KA_SENSE для данного агента; остальные control-сообщения пропускаются."""
        self._ensure_connected()

        while True:
            try:
                incoming = self.receive_message()
            except TimeoutError:
                # До старта полноценного тика KA_SENSE может не приходить дольше socket timeout.
                # Это не ошибка протокола, поэтому продолжаем ожидание.
                continue
            if is_shutdown(incoming):
                return ReceiveResult(sense=None, shutdown=True)

            if int(incoming.urn) == int(URN.ControlMSG.KA_SENSE):
                sense_payload = parse_ka_sense(incoming)
                if sense_payload.agent_id != int(expected_agent_id):
                    # Фильтруем тики других агентов, если несколько клиентов читают одну трассу.
                    continue
                return ReceiveResult(sense=sense_payload, shutdown=False)

    def receive_message(self):
        """Низкоуровневое чтение protobuf-кадра скрыто в отдельный метод для изоляции сетевых ошибок."""
        self._ensure_connected()
        assert self._socket is not None
        return recv_message_from_socket(self._socket)

    def send_message(self, message) -> None:
        """Низкоуровневая отправка вынесена отдельно, чтобы команды AK_* и handshake использовали один канал."""
        self._ensure_connected()
        assert self._socket is not None
        send_message_to_socket(self._socket, message)

    def _ensure_connected(self) -> None:
        """Единая проверка соединения предотвращает немые ошибки send/recv при неинициализированном сокете."""
        if self._socket is None:
            raise RuntimeError("RCRSClient не подключён. Сначала вызовите connect().")
