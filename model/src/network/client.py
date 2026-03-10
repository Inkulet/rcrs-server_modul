from __future__ import annotations

"""В этом модуле я реализую каркас TCP-клиента для подключения к RCRS Kernel."""

import logging
import socket
from typing import List, Optional

from world.entities import (
    ComputedMetrics,
    EntityType,
    RawSensorData,
    VisibleEntity,
)


logger = logging.getLogger(__name__)


class RCRSClient:
    """В этом классе я инкапсулирую TCP-соединение и базовый обмен данными."""

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        """Здесь я инициализирую параметры соединения и таймауты."""

        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None

    def connect(self) -> None:
        """Здесь я инициализирую TCP-соединение с ядром симулятора."""

        if self._socket is not None:
            return

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            logger.info("Я установил соединение с RCRS Kernel на %s:%s", self.host, self.port)
        except (ConnectionRefusedError, socket.timeout) as exc:
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
            logger.info("Я закрыл соединение с RCRS Kernel")

    def receive_sense(self) -> List[VisibleEntity]:
        """Здесь я возвращаю мок-данные, имитируя прием сенсорного пакета."""

        if self._socket is None:
            logger.warning("Я пытаюсь получить данные без активного соединения")

        mock_entities = [
            VisibleEntity(
                id=101,
                type=EntityType.BUILDING,
                raw_sensor_data=RawSensorData(
                    temperature=650.0,
                    fieryness=2,
                    floors=3,
                    ground_area=120,
                ),
                computed_metrics=ComputedMetrics(
                    path_distance=30.0,
                    estimated_death_time=999,
                    total_area=360,
                ),
                utility_score=0.0,
            ),
            VisibleEntity(
                id=202,
                type=EntityType.CIVILIAN,
                raw_sensor_data=RawSensorData(
                    hp=8500,
                    damage=40,
                    buriedness=15,
                ),
                computed_metrics=ComputedMetrics(
                    path_distance=55.0,
                    estimated_death_time=120,
                    total_area=0,
                ),
                utility_score=0.0,
            ),
            VisibleEntity(
                id=203,
                type=EntityType.CIVILIAN,
                raw_sensor_data=RawSensorData(
                    hp=12000,
                    damage=0,
                    buriedness=0,
                ),
                computed_metrics=ComputedMetrics(
                    path_distance=25.0,
                    estimated_death_time=200,
                    total_area=0,
                ),
                utility_score=0.0,
            ),
        ]

        logger.debug("Я сформировал мок-данные сенсоров в количестве=%s", len(mock_entities))
        return mock_entities

    def send_command(self, payload: str) -> None:
        """Здесь я отправляю команду ядру симулятора через TCP-сокет."""

        if self._socket is None:
            logger.warning("Я пытаюсь отправить команду без активного соединения")
            return

        try:
            self._socket.sendall(payload.encode("utf-8"))
            logger.debug("Я отправил команду ядру симулятора: %s", payload)
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as exc:
            logger.error("Я потерял соединение при отправке команды: %s", exc)
            raise


__all__ = ["RCRSClient"]
