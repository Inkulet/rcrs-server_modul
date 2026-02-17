from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from module.network.protobuf_runtime import load_rcrs_proto_modules

URN, RCRSProto_pb2 = load_rcrs_proto_modules()


@dataclass
class KAConnectOKPayload:
    """Декодированная структура KA_CONNECT_OK упрощает рукопожатие в клиенте."""

    request_id: int
    agent_id: int
    entities: List[Any]
    config: Dict[str, str]


@dataclass
class KAConnectErrorPayload:
    """Ошибка подключения хранится отдельно, чтобы явно завершать цикл агента с понятной причиной."""

    request_id: int
    reason: str


@dataclass
class KASensePayload:
    """Данные восприятия из KA_SENSE используются как вход для адаптера и функции полезности."""

    agent_id: int
    time: int
    changeset: Any
    changed_entity_ids: List[int]


def recv_message_from_socket(sock: socket.socket) -> Any:
    """Чтение кадра повторяет StreamConnection: 4 байта длины (big-endian) + protobuf payload."""
    header = _recv_exactly(sock, 4)
    size = int.from_bytes(header, byteorder="big", signed=False)
    payload = _recv_exactly(sock, size)

    message = RCRSProto_pb2.MessageProto()
    message.ParseFromString(payload)
    return message


def send_message_to_socket(sock: socket.socket, message: Any) -> None:
    """Запись кадра симметрична серверу, иначе kernel не сможет корректно распарсить сообщение."""
    payload = message.SerializeToString()
    header = len(payload).to_bytes(4, byteorder="big", signed=False)
    sock.sendall(header + payload)


def build_ak_connect(request_id: int, version: int, agent_name: str, requested_entity_types: Iterable[int]) -> Any:
    """AK_CONNECT формирует минимальный обязательный набор компонентов для подключения агента к kernel."""
    message = RCRSProto_pb2.MessageProto()
    message.urn = int(URN.ControlMSG.AK_CONNECT)

    request_component = RCRSProto_pb2.MessageComponentProto()
    request_component.intValue = int(request_id)
    _set_component(message, int(URN.ComponentControlMSG.RequestID), request_component)

    version_component = RCRSProto_pb2.MessageComponentProto()
    version_component.intValue = int(version)
    _set_component(message, int(URN.ComponentControlMSG.Version), version_component)

    name_component = RCRSProto_pb2.MessageComponentProto()
    name_component.stringValue = str(agent_name)
    _set_component(message, int(URN.ComponentControlMSG.Name), name_component)

    requested_component = RCRSProto_pb2.MessageComponentProto()
    requested_component.intList.values.extend(int(urn) for urn in requested_entity_types)
    _set_component(message, int(URN.ComponentControlMSG.RequestedEntityTypes), requested_component)

    return message


def build_ak_acknowledge(request_id: int, agent_id: int) -> Any:
    """AK_ACKNOWLEDGE подтверждает KA_CONNECT_OK; без этого kernel не переводит агента в активный цикл."""
    message = RCRSProto_pb2.MessageProto()
    message.urn = int(URN.ControlMSG.AK_ACKNOWLEDGE)

    request_component = RCRSProto_pb2.MessageComponentProto()
    request_component.intValue = int(request_id)
    _set_component(message, int(URN.ComponentControlMSG.RequestID), request_component)

    agent_component = RCRSProto_pb2.MessageComponentProto()
    agent_component.entityID = int(agent_id)
    _set_component(message, int(URN.ComponentControlMSG.AgentID), agent_component)

    return message


def build_ak_rest(agent_id: int, time: int) -> Any:
    """AK_REST используется как безопасный fallback, когда у агента нет допустимой целевой команды."""
    message = _build_base_command(int(URN.Command.AK_REST), agent_id, time)
    return message


def build_ak_move(agent_id: int, time: int, path: List[int], destination_x: int = -1, destination_y: int = -1) -> Any:
    """AK_MOVE отправляет path и координатный hint, что совместимо со стандартным сообщением AKMove."""
    message = _build_base_command(int(URN.Command.AK_MOVE), agent_id, time)

    path_component = RCRSProto_pb2.MessageComponentProto()
    path_component.entityIDList.values.extend(int(step) for step in path)
    _set_component(message, int(URN.ComponentCommand.Path), path_component)

    x_component = RCRSProto_pb2.MessageComponentProto()
    x_component.intValue = int(destination_x)
    _set_component(message, int(URN.ComponentCommand.DestinationX), x_component)

    y_component = RCRSProto_pb2.MessageComponentProto()
    y_component.intValue = int(destination_y)
    _set_component(message, int(URN.ComponentCommand.DestinationY), y_component)

    return message


def build_ak_extinguish(agent_id: int, time: int, target_id: int, water: int) -> Any:
    """AK_EXTINGUISH соответствует стандартной команде тушения с целевым entity ID и объёмом воды."""
    message = _build_base_command(int(URN.Command.AK_EXTINGUISH), agent_id, time)

    target_component = RCRSProto_pb2.MessageComponentProto()
    target_component.entityID = int(target_id)
    _set_component(message, int(URN.ComponentCommand.Target), target_component)

    water_component = RCRSProto_pb2.MessageComponentProto()
    water_component.intValue = int(water)
    _set_component(message, int(URN.ComponentCommand.Water), water_component)

    return message


def build_ak_rescue(agent_id: int, time: int, target_id: int) -> Any:
    """AK_RESCUE инициирует разбор завала у пострадавшего, когда агент достиг его позиции."""
    message = _build_base_command(int(URN.Command.AK_RESCUE), agent_id, time)

    target_component = RCRSProto_pb2.MessageComponentProto()
    target_component.entityID = int(target_id)
    _set_component(message, int(URN.ComponentCommand.Target), target_component)

    return message


def build_ak_load(agent_id: int, time: int, target_id: int) -> Any:
    """AK_LOAD поднимает пострадавшего в AmbulanceTeam после освобождения из-под завала."""
    message = _build_base_command(int(URN.Command.AK_LOAD), agent_id, time)

    target_component = RCRSProto_pb2.MessageComponentProto()
    target_component.entityID = int(target_id)
    _set_component(message, int(URN.ComponentCommand.Target), target_component)

    return message


def build_ak_unload(agent_id: int, time: int) -> Any:
    """AK_UNLOAD выгружает пострадавшего в убежище, когда AmbulanceTeam находится в refuge."""
    message = _build_base_command(int(URN.Command.AK_UNLOAD), agent_id, time)
    return message


def build_ak_clear(agent_id: int, time: int, target_id: int) -> Any:
    """AK_CLEAR применяет действие PoliceForce к конкретному blockade entity."""
    message = _build_base_command(int(URN.Command.AK_CLEAR), agent_id, time)

    target_component = RCRSProto_pb2.MessageComponentProto()
    target_component.entityID = int(target_id)
    _set_component(message, int(URN.ComponentCommand.Target), target_component)

    return message


def parse_ka_connect_ok(message: Any) -> KAConnectOKPayload:
    """Парсер KA_CONNECT_OK извлекает стартовое состояние мира и ID агента для дальнейшего цикла."""
    _ensure_urn(message, int(URN.ControlMSG.KA_CONNECT_OK))

    request_id = _get_component_int(message, int(URN.ComponentControlMSG.RequestID))
    agent_id = _get_component_entity_id(message, int(URN.ComponentControlMSG.AgentID))
    entities_component = _get_component(message, int(URN.ComponentControlMSG.Entities))
    config_component = _get_component(message, int(URN.ComponentControlMSG.AgentConfig))

    entities: List[Any] = []
    if entities_component is not None and entities_component.HasField("entityList"):
        entities = list(entities_component.entityList.entities)

    config: Dict[str, str] = {}
    if config_component is not None and config_component.HasField("config"):
        config = _extract_string_map(config_component.config.data)

    return KAConnectOKPayload(
        request_id=request_id,
        agent_id=agent_id,
        entities=entities,
        config=config,
    )


def parse_ka_connect_error(message: Any) -> KAConnectErrorPayload:
    """Парсер KA_CONNECT_ERROR возвращает диагностическую причину отказа подключения."""
    _ensure_urn(message, int(URN.ControlMSG.KA_CONNECT_ERROR))

    request_id = _get_component_int(message, int(URN.ComponentControlMSG.RequestID))
    reason_component = _get_component(message, int(URN.ComponentControlMSG.Reason))
    reason = "UNKNOWN"
    if reason_component is not None and reason_component.HasField("stringValue"):
        reason = reason_component.stringValue

    return KAConnectErrorPayload(request_id=request_id, reason=reason)


def parse_ka_sense(message: Any) -> KASensePayload:
    """Парсер KA_SENSE выделяет изменения мира, которые затем адаптируются в структуру матмодели."""
    _ensure_urn(message, int(URN.ControlMSG.KA_SENSE))

    agent_id = _get_component_entity_id(message, int(URN.ComponentControlMSG.AgentID))
    time = _get_component_int(message, int(URN.ComponentControlMSG.Time))

    updates_component = _get_component(message, int(URN.ComponentControlMSG.Updates))
    if updates_component is None or not updates_component.HasField("changeSet"):
        raise ValueError("KA_SENSE не содержит компонента Updates.changeSet")

    changeset = updates_component.changeSet
    changed_ids = [int(change.entityID) for change in changeset.changes]

    return KASensePayload(
        agent_id=agent_id,
        time=time,
        changeset=changeset,
        changed_entity_ids=changed_ids,
    )


def is_shutdown(message: Any) -> bool:
    """SHUTDOWN используется как корректное завершение цикла агента по сигналу kernel."""
    return int(message.urn) == int(URN.ControlMSG.SHUTDOWN)


def _build_base_command(command_urn: int, agent_id: int, time: int) -> Any:
    """Общая часть команд содержит AgentID и Time, как в AbstractCommand Java-реализации."""
    message = RCRSProto_pb2.MessageProto()
    message.urn = int(command_urn)

    agent_component = RCRSProto_pb2.MessageComponentProto()
    agent_component.entityID = int(agent_id)
    _set_component(message, int(URN.ComponentControlMSG.AgentID), agent_component)

    time_component = RCRSProto_pb2.MessageComponentProto()
    time_component.intValue = int(time)
    _set_component(message, int(URN.ComponentControlMSG.Time), time_component)

    return message


def _get_component(message: Any, key: int) -> Optional[Any]:
    """Компонент читается в map/repeated-режиме protobuf, чтобы код работал со старым RCRSProto_pb2."""
    map_component = _get_component_from_map_container(message, key)
    if map_component is not None:
        return map_component

    return _get_component_from_repeated_entries(message, key)


def _set_component(message: Any, key: int, component: Any) -> None:
    """Запись компонента поддерживает оба формата контейнера `components` старого protobuf-кода RCRS."""
    if _set_component_in_map_container(message, key, component):
        return
    _set_component_in_repeated_entries(message, key, component)


def _get_component_from_map_container(message: Any, key: int) -> Optional[Any]:
    """Сначала пробуем map-контейнер, потому что он использует прямой доступ по ключу."""
    try:
        if int(key) in message.components:
            return message.components[int(key)]
    except Exception:  # noqa: BLE001
        return None
    return None


def _get_component_from_repeated_entries(message: Any, key: int) -> Optional[Any]:
    """Fallback на ручной поиск по ComponentsEntry нужен для окружений без map-обертки protobuf."""
    target_key = int(key)
    try:
        for entry in message.components:
            entry_key = getattr(entry, "key", None)
            if entry_key is None:
                continue
            if int(entry_key) != target_key:
                continue
            value = getattr(entry, "value", None)
            if value is None:
                continue
            return value
    except Exception:  # noqa: BLE001
        return None
    return None


def _set_component_in_map_container(message: Any, key: int, component: Any) -> bool:
    """Map-контейнер обновляется напрямую; это совместимо с protobuf-реализациями, где map поддерживается."""
    try:
        message.components[int(key)].CopyFrom(component)
        return True
    except Exception:  # noqa: BLE001
        return False


def _set_component_in_repeated_entries(message: Any, key: int, component: Any) -> None:
    """В repeated-режиме создаём/обновляем ComponentsEntry вручную, сохраняя wire-совместимость map-поля."""
    target_key = int(key)
    for entry in message.components:
        entry_key = getattr(entry, "key", None)
        if entry_key is None:
            continue
        if int(entry_key) == target_key:
            entry.value.CopyFrom(component)
            return

    entry = message.components.add()
    entry.key = target_key
    entry.value.CopyFrom(component)


def _extract_string_map(config_data: Any) -> Dict[str, str]:
    """Config map извлекается через универсальный путь, потому что старый protobuf может вернуть repeated entries."""
    try:
        return {str(key): str(value) for key, value in dict(config_data).items()}
    except Exception:  # noqa: BLE001
        pass

    result: Dict[str, str] = {}
    try:
        for entry in config_data:
            entry_key = getattr(entry, "key", None)
            entry_value = getattr(entry, "value", None)
            if entry_key is None:
                continue
            result[str(entry_key)] = "" if entry_value is None else str(entry_value)
    except Exception:  # noqa: BLE001
        return {}
    return result


def _get_component_int(message: Any, key: int) -> int:
    """Int-компонент обязателен для control-пакетов, поэтому отсутствие считается ошибкой протокола."""
    component = _get_component(message, key)
    if component is None or not component.HasField("intValue"):
        raise ValueError(f"Компонент {key} отсутствует или не содержит intValue")
    return int(component.intValue)


def _get_component_entity_id(message: Any, key: int) -> int:
    """EntityID компоненты обязательны в KA_CONNECT_OK/KA_SENSE и командах AbstractCommand."""
    component = _get_component(message, key)
    if component is None or not component.HasField("entityID"):
        raise ValueError(f"Компонент {key} отсутствует или не содержит entityID")
    return int(component.entityID)


def _ensure_urn(message: Any, expected_urn: int) -> None:
    """Жесткая проверка URN защищает цикл агента от рассинхронизации протокольных состояний."""
    if int(message.urn) != int(expected_urn):
        raise ValueError(f"Ожидался URN={expected_urn}, получен URN={int(message.urn)}")


def _recv_exactly(sock: socket.socket, size: int) -> bytes:
    """Чтение фиксированного числа байт предотвращает частичную десериализацию protobuf-кадра."""
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if part == b"":
            raise ConnectionError("TCP-соединение закрыто удаленной стороной")
        chunks.extend(part)
    return bytes(chunks)
