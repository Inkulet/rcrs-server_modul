from __future__ import annotations

"""В этом модуле я фиксирую регрессии в action.executor."""

from action.executor import dispatch_action
from conftest import make_agent, make_blockade, make_civilian
from world.cache import WorldModel
from world.entities import AgentType


class _ClientStub:
    def __init__(self) -> None:
        self.clear_area_calls: list[tuple[int, int, int]] = []
        self.move_calls: list[tuple[int, list[int], int, int]] = []
        self.load_calls: list[tuple[int, int]] = []

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

    def send_load(self, time: int, target_id: int) -> None:
        self.load_calls.append((time, target_id))


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


def test_police_falls_back_to_blockade_geometry_when_path_aim_misses() -> None:
    wm = WorldModel()
    wm.road_graph.add_node(1, x=0, y=0)
    wm.road_graph.add_node(2, x=10000, y=10000)
    wm.road_graph.add_node(3, x=20000, y=10000)
    wm.road_graph.add_edge(1, 2, weight=10000.0)
    wm.road_graph.add_edge(2, 3, weight=10000.0)

    local_blockade = make_blockade(entity_id=77, repair_cost=900)
    local_blockade = local_blockade.model_copy(
        update={
            "entity_x": 9000,
            "entity_y": 0,
            "raw_sensor_data": local_blockade.raw_sensor_data.model_copy(
                update={
                    "position_on_edge": 2,
                    "apexes": [8500, -500, 9500, -500, 9500, 500, 8500, 500],
                },
            ),
        },
    )
    wm.tasks[77] = local_blockade
    wm.blockades_by_node[2] = {77}

    far_target = make_blockade(entity_id=88, repair_cost=1200)
    far_target = far_target.model_copy(
        update={
            "entity_x": 20000,
            "entity_y": 10000,
            "raw_sensor_data": far_target.raw_sensor_data.model_copy(
                update={"position_on_edge": 3},
            ),
        },
    )
    wm.tasks[88] = far_target

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
        tick=4,
        target_id=88,
        agent_node_id=1,
        world_model=wm,
    )

    assert target_valid is True
    assert unreachable is False
    assert working is True
    assert attempted == 77
    assert len(client.clear_area_calls) == 1
    _time, dest_x, dest_y = client.clear_area_calls[0]
    assert dest_x > 9000
    assert abs(dest_y) < 2000


def test_police_does_not_clear_when_clear_ray_misses_blockade_polygon() -> None:
    wm = WorldModel()
    wm.road_graph.add_node(1, x=0, y=0)

    blockade = make_blockade(entity_id=79, repair_cost=900)
    blockade = blockade.model_copy(
        update={
            "entity_x": 0,
            "entity_y": 9000,
            "raw_sensor_data": blockade.raw_sensor_data.model_copy(
                update={
                    "position_on_edge": 1,
                    "apexes": [8500, -500, 9500, -500, 9500, 500, 8500, 500],
                },
            ),
        },
    )
    wm.tasks[79] = blockade

    client = _ClientStub()
    agent = make_agent(
        agent_id=9,
        agent_type=AgentType.POLICE_FORCE,
        x=0,
        y=0,
        entity_id=1,
    )

    target_valid, unreachable, working, attempted = dispatch_action(
        client=client,
        agent_type=AgentType.POLICE_FORCE,
        agent_state=agent,
        tick=8,
        target_id=79,
        agent_node_id=1,
        world_model=wm,
    )

    assert target_valid is False
    assert unreachable is False
    assert working is False
    assert attempted is None
    assert client.clear_area_calls == []


def test_ambulance_does_not_chase_or_load_civilian_already_in_refuge() -> None:
    wm = WorldModel()
    wm.road_graph.add_node(1, x=0, y=0)
    wm.road_graph.add_node(5, x=5000, y=0)
    wm.road_graph.add_edge(1, 5, weight=5000.0)
    wm.refuge_ids = [5]

    civilian = make_civilian(entity_id=91, hp=9000, damage=20, buriedness=0)
    civilian = civilian.model_copy(
        update={
            "raw_sensor_data": civilian.raw_sensor_data.model_copy(
                update={"position_on_edge": 5},
            ),
        },
    )
    wm.tasks[91] = civilian

    client = _ClientStub()
    agent = make_agent(
        agent_id=6,
        agent_type=AgentType.AMBULANCE_TEAM,
        x=0,
        y=0,
        entity_id=1,
    )

    target_valid, unreachable, working, attempted = dispatch_action(
        client=client,
        agent_type=AgentType.AMBULANCE_TEAM,
        agent_state=agent,
        tick=7,
        target_id=91,
        agent_node_id=1,
        world_model=wm,
    )

    assert target_valid is False
    assert unreachable is False
    assert working is False
    assert attempted is None
    assert 91 not in wm.tasks
    assert client.move_calls == []
    assert client.load_calls == []
