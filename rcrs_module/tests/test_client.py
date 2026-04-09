from __future__ import annotations

"""В этом модуле я тестирую RCRSClient: mock-режим, подключение, отправку команд."""

import pytest

from network.client import RCRSClient
from world.entities import AgentType, EntityType, PerceptionPacket


# ---------------------------------------------------------------------------
# Mock-режим: подключение и рукопожатие
# ---------------------------------------------------------------------------

class TestMockHandshake:

    def test_mock_connect_succeeds(self) -> None:
        client = RCRSClient(host="localhost", port=7000, mock=True)
        client.connect()  # не бросает исключение

    def test_mock_handshake_returns_agent_id(self) -> None:
        client = RCRSClient(host="localhost", port=7000, mock=True)
        client.connect()
        agent_id = client.handshake("test", AgentType.FIRE_BRIGADE)
        assert agent_id == 1

    def test_mock_disconnect_safe(self) -> None:
        client = RCRSClient(host="localhost", port=7000, mock=True)
        client.connect()
        client.disconnect()  # не бросает
        client.disconnect()  # повторный disconnect тоже безопасен


# ---------------------------------------------------------------------------
# Mock-режим: receive_sense
# ---------------------------------------------------------------------------

class TestMockReceiveSense:

    @pytest.fixture()
    def mock_client(self) -> RCRSClient:
        client = RCRSClient(host="localhost", port=7000, mock=True)
        client.connect()
        client.handshake("test", AgentType.FIRE_BRIGADE)
        return client

    def test_returns_perception_packet(self, mock_client: RCRSClient) -> None:
        packet = mock_client.receive_sense()
        assert isinstance(packet, PerceptionPacket)

    def test_tick_zero_has_map_data(self, mock_client: RCRSClient) -> None:
        packet = mock_client.receive_sense()
        assert packet.tick == 0
        assert len(packet.map_nodes) > 0
        assert len(packet.map_edges) > 0
        assert len(packet.refuge_ids) > 0

    def test_tick_one_has_no_map_data(self, mock_client: RCRSClient) -> None:
        mock_client.receive_sense()  # tick 0
        packet = mock_client.receive_sense()  # tick 1
        assert packet.tick == 1
        assert packet.map_nodes == []
        assert packet.map_edges == []

    def test_tick_increments(self, mock_client: RCRSClient) -> None:
        p0 = mock_client.receive_sense()
        p1 = mock_client.receive_sense()
        p2 = mock_client.receive_sense()
        assert p0.tick == 0
        assert p1.tick == 1
        assert p2.tick == 2

    def test_has_visible_entities(self, mock_client: RCRSClient) -> None:
        packet = mock_client.receive_sense()
        assert len(packet.visible_entities) > 0

    def test_has_ally_states(self, mock_client: RCRSClient) -> None:
        packet = mock_client.receive_sense()
        assert len(packet.ally_states) > 0

    def test_own_state_is_fire_brigade(self, mock_client: RCRSClient) -> None:
        packet = mock_client.receive_sense()
        assert packet.own_state.type == AgentType.FIRE_BRIGADE
        assert packet.own_state.id == 1


# ---------------------------------------------------------------------------
# Mock-режим: send_* не бросают исключений
# ---------------------------------------------------------------------------

class TestMockSendCommands:

    @pytest.fixture()
    def mock_client(self) -> RCRSClient:
        client = RCRSClient(host="localhost", port=7000, mock=True)
        client.connect()
        client.handshake("test", AgentType.FIRE_BRIGADE)
        return client

    def test_send_move_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_move(time=1, path=[1, 2, 3])

    def test_send_rescue_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_rescue(time=1, target_id=10)

    def test_send_extinguish_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_extinguish(time=1, target_id=20, water=10000)

    def test_send_clear_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_clear(time=1, target_id=30)

    def test_send_load_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_load(time=1, target_id=10)

    def test_send_unload_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_unload(time=1)

    def test_send_rest_mock(self, mock_client: RCRSClient) -> None:
        mock_client.send_rest(time=1)


# ---------------------------------------------------------------------------
# Реальный режим: ошибки без соединения
# ---------------------------------------------------------------------------

class TestRealModeErrors:

    def test_send_raw_without_connection_raises(self) -> None:
        client = RCRSClient(host="localhost", port=7000, mock=False)
        with pytest.raises(ConnectionError):
            client._send_raw(b"test")

    def test_recv_proto_without_connection_raises(self) -> None:
        client = RCRSClient(host="localhost", port=7000, mock=False)
        with pytest.raises(ConnectionError):
            client._recv_proto()

    def test_disconnect_without_connect_is_safe(self) -> None:
        client = RCRSClient(host="localhost", port=7000, mock=False)
        client.disconnect()  # не бросает
