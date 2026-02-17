from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from module.data_models import AgentOperationalState, AgentState, AgentType, Position, Resources
from module.network.snapshot import LiveStateStore


class LiveStateStoreTests(unittest.TestCase):
    """Проверяем, что snapshot store устойчиво принимает dataclass payload из agent runtime."""

    def test_update_agent_snapshot_accepts_dataclass_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "live_state.json"
            store = LiveStateStore(snapshot_path=snapshot_path)

            payload = {
                "agent_state": AgentState(
                    id=101,
                    type=AgentType.FIRE_BRIGADE,
                    position=Position(entity_id=256, x=10, y=20),
                    state=AgentOperationalState.IDLE,
                    resources=Resources(water_quantity=3000, is_transporting=False),
                ),
                "decision": {"action": "NO_TARGET", "selected_target_id": None},
            }

            store.update_agent_snapshot(tick=1, agent_payload=payload, warnings=["test-warning"])

            loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded.get("mode"), "live")
            self.assertEqual(loaded.get("tick"), 1)

            agents = loaded.get("agents", {})
            self.assertIn("101", agents)

            snapshot = agents["101"]
            self.assertEqual(snapshot.get("tick"), 1)
            self.assertEqual(snapshot.get("warnings"), ["test-warning"])

            saved_agent_state = snapshot.get("payload", {}).get("agent_state", {})
            self.assertEqual(saved_agent_state.get("id"), 101)
            self.assertEqual(saved_agent_state.get("type"), AgentType.FIRE_BRIGADE.value)


if __name__ == "__main__":
    unittest.main()
