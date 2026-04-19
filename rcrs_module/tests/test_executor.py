from __future__ import annotations

"""В этом модуле я фиксирую регрессии в action.executor для полиции."""

from action.executor import dispatch_action
from conftest import make_agent, make_blockade
from world.cache import WorldModel
from world.entities import AgentType


class _ClientStub:
    def __init__(self) -> None:
        self.clear_area_calls: list[tuple[int, int, int]] = []
        self.move_calls: list[tuple[int, list[int], int, int]] = []

    def send_clear_area(self, time: int, dest_x: int, dest_y: int) -> None:
        self.clear_area_calls.append((time, dest_x, dest_y))

    def send_move(
        self,
        time: int,
        path: list[int],
        dest_x: int = -1,
        dest_y: int = -1,
    ) -> None:
        self.move_calls.append((time, path, dest_x, dest_y))


def test_police_does_not_clear_unconfirmed_blockade_target() -> None:
    wm = WorldModel()
    wm.road_graph.add_node(1, x=0, y=0)

    blockade = make_blockade(entity_id=77, repair_cost=900)
    blockade = blockade.model_copy(
        update={
            "raw_sensor_data": blockade.raw_sensor_data.model_copy(
                update={"position_on_edge": 1},
            ),
        },
    )
    wm.tasks[77] = blockade

    client = _ClientStub()
    agent = make_agent(
        agent_id=5,
        agent_type=AgentType.POLICE_FORCE,
        x=0,
        y=0,
        entity_id=1,
    )

    target_valid, unreachable, working, attempted = dispatch_action(
        client=client,
        agent_type=AgentType.POLICE_FORCE,
        agent_state=agent,
        tick=3,
        target_id=77,
        agent_node_id=1,
        world_model=wm,
    )

    assert target_valid is False
    assert unreachable is False
    assert working is False
    assert attempted is None
    assert client.clear_area_calls == []
