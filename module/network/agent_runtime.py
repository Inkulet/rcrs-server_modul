from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from module.calculator import UtilityCalculator
from module.data_models import AgentType, VisibleEntity
from module.network.adapter import WorldModelAdapter
from module.network.protocol import (
    URN,
    build_ak_clear,
    build_ak_extinguish,
    build_ak_load,
    build_ak_move,
    build_ak_rescue,
    build_ak_rest,
    build_ak_unload,
)
from module.network.rcrs_client import RCRSClient
from module.network.runtime_config import build_runtime_baseline_config, load_profile_or_baseline
from module.network.snapshot import LiveStateStore
from module.network.world_model import WorldModel
from module.strategy import UtilityBasedTargetSelectionStrategy


@dataclass
class AgentRuntimeConfig:
    """Параметры запуска выделены в dataclass, чтобы CLI и тесты использовали единый контракт."""

    host: str
    port: int
    agent_name: str
    agent_type: AgentType
    request_id: int
    version: int
    snapshot_path: Path
    profile_path: Optional[Path]
    tick_sleep_sec: float = 0.0


class AgentRuntime:
    """Runtime реализует цикл Receive->Parse->Adapt->Think->Act и публикацию snapshot для Live UI."""

    def __init__(self, config: AgentRuntimeConfig):
        self.config = config
        self.client = RCRSClient(host=config.host, port=config.port)
        self.world_model = WorldModel()
        self.adapter = WorldModelAdapter(self.world_model)
        self.snapshot_store = LiveStateStore(snapshot_path=config.snapshot_path)
        self.agent_id: Optional[int] = None
        self.current_target_id: Optional[int] = None

        self._strategy: Optional[UtilityBasedTargetSelectionStrategy] = None

    def run(self) -> None:
        """Основной цикл агента завершает работу по SHUTDOWN или сетевой ошибке, сохраняя причины в stdout."""
        try:
            self.client.connect()
            connect_ok = self.client.handshake_agent(
                request_id=self.config.request_id,
                version=self.config.version,
                agent_name=self.config.agent_name,
                requested_entity_types=[self._requested_entity_urn()],
            )

            self.agent_id = connect_ok.agent_id
            self.world_model.initialize(connect_ok.entities, connect_ok.config)

            baseline_config = build_runtime_baseline_config(self.world_model)
            active_config = load_profile_or_baseline(self.config.profile_path, baseline_config)
            calculator = UtilityCalculator(active_config)
            self._strategy = UtilityBasedTargetSelectionStrategy(calculator)

            print(
                f"[RCRS] Connected: agent_id={self.agent_id}, agent_type={self.config.agent_type.value}, "
                f"entities={len(self.world_model.entities)}"
            )

            while True:
                receive_result = self.client.wait_for_sense(expected_agent_id=self.agent_id)
                if receive_result.shutdown:
                    print("[RCRS] SHUTDOWN received, stopping agent loop.")
                    break

                sense = receive_result.sense
                if sense is None:
                    continue

                self.world_model.apply_changeset(sense.changeset)

                adapted_state = self.adapter.adapt(
                    agent_id=self.agent_id,
                    agent_type=self.config.agent_type,
                    visible_entity_ids=sense.changed_entity_ids,
                )
                adapted_state.observation.agent_state.current_target_id = self.current_target_id

                assert self._strategy is not None
                decision = self._strategy.select_target(
                    agent=adapted_state.observation.agent_state,
                    visible_entities=adapted_state.observation.visible_entities,
                    all_agents=adapted_state.all_agents,
                    refuges=adapted_state.refuges,
                )
                self.current_target_id = decision.selected_target_id

                command_message, command_name, command_reason = self._build_command(
                    tick=sense.time,
                    decision_action=decision.action,
                    decision_target_id=decision.selected_target_id,
                    observation=adapted_state.observation,
                    visible_entities=adapted_state.observation.visible_entities,
                    refuges=adapted_state.refuges,
                )

                self.client.send_message(command_message)

                snapshot_payload = {
                    "agent_state": adapted_state.observation.agent_state,
                    "decision": {
                        "action": decision.action,
                        "selected_target_id": decision.selected_target_id,
                        "reason": decision.reason,
                        "sent_command": command_name,
                        "sent_reason": command_reason,
                    },
                    "utility_matrix": [asdict(row) for row in decision.utility_matrix],
                    "visible_entities": [asdict(entity) for entity in adapted_state.observation.visible_entities],
                    "refuges": [asdict(refuge) for refuge in adapted_state.refuges],
                }
                self.snapshot_store.update_agent_snapshot(
                    tick=sense.time,
                    agent_payload=snapshot_payload,
                    warnings=adapted_state.warnings,
                )

                print(
                    f"[RCRS] t={sense.time}, action={decision.action}, target={decision.selected_target_id}, "
                    f"command={command_name}"
                )
                if self.config.tick_sleep_sec > 0:
                    time.sleep(self.config.tick_sleep_sec)
        finally:
            self.client.close()

    def _build_command(
        self,
        tick: int,
        decision_action: str,
        decision_target_id: Optional[int],
        observation,
        visible_entities: List[VisibleEntity],
        refuges,
    ):
        """Преобразование цели в AK_* команду учитывает тип агента и текущую позицию, чтобы команды были исполнимы."""
        agent_state = observation.agent_state
        agent_id = agent_state.id
        agent_area = agent_state.position.entity_id

        # Ambulance при перевозке и нахождении в refuge должен выгружать пострадавшего раньше любых новых задач.
        if agent_state.type == AgentType.AMBULANCE_TEAM and agent_state.resources.is_transporting:
            if self._is_refuge_area(agent_area):
                return build_ak_unload(agent_id=agent_id, time=tick), "AK_UNLOAD", "Transporting+Refuge"
            nearest_refuge_area = self._choose_nearest_refuge_area(agent_area=agent_area, refuges=refuges)
            if nearest_refuge_area is not None:
                move_path = self.world_model.shortest_path(agent_area, nearest_refuge_area)
                if move_path:
                    return (
                        build_ak_move(agent_id=agent_id, time=tick, path=move_path),
                        "AK_MOVE",
                        "TransportingToRefuge",
                    )
            return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "TransportingNoRefugePath"

        if decision_action in {"NO_TARGET", "ERROR", "SKIP_BUSY"}:
            return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "Fallback"

        if decision_action == "GO_TO_REFUGE":
            if decision_target_id is None:
                return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "NoRefugeTarget"
            move_path = self.world_model.shortest_path(agent_area, decision_target_id)
            if not move_path:
                return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "NoPathToRefuge"
            return (
                build_ak_move(agent_id=agent_id, time=tick, path=move_path),
                "AK_MOVE",
                "GoToRefuge",
            )

        target = next((entity for entity in visible_entities if entity.id == decision_target_id), None)
        if target is None:
            return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "TargetNotVisible"

        target_area = target.position.entity_id

        if agent_state.type == AgentType.FIRE_BRIGADE:
            if target_area is not None and agent_area == target_area:
                water_to_use = max(1, min(500, agent_state.resources.water_quantity))
                return (
                    build_ak_extinguish(agent_id=agent_id, time=tick, target_id=target.id, water=water_to_use),
                    "AK_EXTINGUISH",
                    "AtBuilding",
                )
            move_path = self.world_model.shortest_path(agent_area, target_area)
            if move_path:
                return build_ak_move(agent_id=agent_id, time=tick, path=move_path), "AK_MOVE", "MoveToBuilding"
            return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "NoPathToBuilding"

        if agent_state.type == AgentType.AMBULANCE_TEAM:
            if target_area is not None and agent_area == target_area:
                buriedness = target.raw_sensor_data.buriedness
                if buriedness is not None and buriedness > 0:
                    return build_ak_rescue(agent_id=agent_id, time=tick, target_id=target.id), "AK_RESCUE", "Buriedness>0"
                return build_ak_load(agent_id=agent_id, time=tick, target_id=target.id), "AK_LOAD", "ReadyToLoad"

            move_path = self.world_model.shortest_path(agent_area, target_area)
            if move_path:
                return build_ak_move(agent_id=agent_id, time=tick, path=move_path), "AK_MOVE", "MoveToCivilian"
            return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "NoPathToCivilian"

        if agent_state.type == AgentType.POLICE_FORCE:
            blockade_area = target.raw_sensor_data.position_on_edge or target_area
            if blockade_area is not None and agent_area == blockade_area:
                return build_ak_clear(agent_id=agent_id, time=tick, target_id=target.id), "AK_CLEAR", "AtBlockade"

            move_path = self.world_model.shortest_path(agent_area, blockade_area)
            if move_path:
                return build_ak_move(agent_id=agent_id, time=tick, path=move_path), "AK_MOVE", "MoveToBlockade"
            return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "NoPathToBlockade"

        return build_ak_rest(agent_id=agent_id, time=tick), "AK_REST", "UnsupportedAgentType"

    def _requested_entity_urn(self) -> int:
        """RequestedEntityTypes должен соответствовать типу процесса агента, иначе kernel вернет connect error."""
        if self.config.agent_type == AgentType.FIRE_BRIGADE:
            return int(URN.Entity.FIRE_BRIGADE)
        if self.config.agent_type == AgentType.AMBULANCE_TEAM:
            return int(URN.Entity.AMBULANCE_TEAM)
        if self.config.agent_type == AgentType.POLICE_FORCE:
            return int(URN.Entity.POLICE_FORCE)
        raise ValueError(f"Неизвестный тип агента: {self.config.agent_type}")

    def _is_refuge_area(self, area_entity_id: Optional[int]) -> bool:
        """Проверка refuge-локации нужна для безопасной отправки AK_UNLOAD только в корректной точке."""
        if area_entity_id is None:
            return False
        entity = self.world_model.get_entity(area_entity_id)
        if entity is None:
            return False
        return entity.urn == int(URN.Entity.REFUGE)

    def _choose_nearest_refuge_area(self, agent_area: Optional[int], refuges) -> Optional[int]:
        """Для транспортировки выбираем ближайшее убежище детерминированно по graph-distance и entity_id."""
        if agent_area is None:
            return None

        candidate_refuge_ids: List[int] = []
        for refuge in refuges:
            refuge_id = refuge.entity_id
            if refuge_id is None:
                continue
            candidate_refuge_ids.append(int(refuge_id))

        if not candidate_refuge_ids:
            return None

        def refuge_sort_key(refuge_id: int):
            distance = self.world_model.shortest_distance(agent_area, refuge_id)
            safe_distance = float("inf") if distance is None else float(distance)
            return (safe_distance, refuge_id)

        ordered_refuges = sorted(candidate_refuge_ids, key=refuge_sort_key)
        for refuge_id in ordered_refuges:
            path = self.world_model.shortest_path(agent_area, refuge_id)
            if path:
                return refuge_id
        return None
