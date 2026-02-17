from __future__ import annotations

import socket
import unittest

from module.network.protocol import (
    RCRSProto_pb2,
    URN,
    build_ak_connect,
    parse_ka_sense,
    recv_message_from_socket,
    send_message_to_socket,
)


class NetworkProtocolTests(unittest.TestCase):
    """Протокольные тесты фиксируют формат кадра и разбор KA_SENSE перед интеграцией с kernel."""

    def test_framed_send_receive_roundtrip(self) -> None:
        left, right = socket.socketpair()
        try:
            message = build_ak_connect(
                request_id=11,
                version=2,
                agent_name="py_test_agent",
                requested_entity_types=[int(URN.Entity.FIRE_BRIGADE)],
            )
            send_message_to_socket(left, message)
            received = recv_message_from_socket(right)

            self.assertEqual(int(received.urn), int(URN.ControlMSG.AK_CONNECT))
            request_component = self._get_component(received, int(URN.ComponentControlMSG.RequestID))
            self.assertIsNotNone(request_component)
            assert request_component is not None
            self.assertEqual(int(request_component.intValue), 11)
        finally:
            left.close()
            right.close()

    def test_parse_ka_sense_payload(self) -> None:
        message = RCRSProto_pb2.MessageProto()
        message.urn = int(URN.ControlMSG.KA_SENSE)

        agent_component = RCRSProto_pb2.MessageComponentProto()
        agent_component.entityID = 777
        self._set_component(message, int(URN.ComponentControlMSG.AgentID), agent_component)

        time_component = RCRSProto_pb2.MessageComponentProto()
        time_component.intValue = 15
        self._set_component(message, int(URN.ComponentControlMSG.Time), time_component)

        updates_component = RCRSProto_pb2.MessageComponentProto()
        change = updates_component.changeSet.changes.add()
        change.entityID = 1001
        change.urn = int(URN.Entity.BUILDING)
        self._set_component(message, int(URN.ComponentControlMSG.Updates), updates_component)

        payload = parse_ka_sense(message)
        self.assertEqual(payload.agent_id, 777)
        self.assertEqual(payload.time, 15)
        self.assertEqual(payload.changed_entity_ids, [1001])

    @staticmethod
    def _set_component(message, key: int, component) -> None:
        """Тестовый helper повторяет формат map/repeated контейнера и не завязан на конкретный runtime protobuf."""
        try:
            message.components[int(key)].CopyFrom(component)
            return
        except Exception:  # noqa: BLE001
            pass

        for entry in message.components:
            if int(getattr(entry, "key", -1)) == int(key):
                entry.value.CopyFrom(component)
                return

        entry = message.components.add()
        entry.key = int(key)
        entry.value.CopyFrom(component)

    @staticmethod
    def _get_component(message, key: int):
        """Чтение компонента поддерживает map и ComponentsEntry для стабильной проверки протокола."""
        try:
            if int(key) in message.components:
                return message.components[int(key)]
        except Exception:  # noqa: BLE001
            pass

        for entry in message.components:
            if int(getattr(entry, "key", -1)) == int(key):
                return entry.value
        return None


if __name__ == "__main__":
    unittest.main()
