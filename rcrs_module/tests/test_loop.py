from __future__ import annotations

from agent.loop import (
    _prioritize_role_tasks,
    _same_role_claim_exclusions,
    _search_partition,
)
from world.cache import WorldModel
from world.entities import AgentType

from conftest import make_agent, make_building, make_civilian


def test_search_partition_uses_same_role_roster() -> None:
    world_model = WorldModel()
    world_model.set_agent(make_agent(agent_id=9, agent_type=AgentType.FIRE_BRIGADE))
    world_model.set_agent(make_agent(agent_id=3, agent_type=AgentType.FIRE_BRIGADE))
    world_model.set_agent(make_agent(agent_id=8, agent_type=AgentType.AMBULANCE_TEAM))

    partition_index, partition_count = _search_partition(
        make_agent(agent_id=5, agent_type=AgentType.FIRE_BRIGADE),
        world_model,
    )

    assert (partition_index, partition_count) == (1, 3)


def test_ambulance_prioritizes_transport_ready_civilians() -> None:
    refuge_ids = [999]
    ready = make_civilian(entity_id=10, buriedness=0, damage=20)
    ready.raw_sensor_data = ready.raw_sensor_data.model_copy(update={"position_on_edge": 10})
    buried = make_civilian(entity_id=11, buriedness=5, damage=20)
    buried.raw_sensor_data = buried.raw_sensor_data.model_copy(update={"position_on_edge": 11})
    burning_building = make_building(entity_id=20)

    prioritized = _prioritize_role_tasks(
        AgentType.AMBULANCE_TEAM,
        [buried, burning_building, ready],
        refuge_ids,
    )

    assert [entity.id for entity in prioritized] == [10]


def test_ambulance_keeps_original_candidates_without_transport_ready() -> None:
    refuge_ids = [999]
    buried = make_civilian(entity_id=11, buriedness=5, damage=20)
    buried.raw_sensor_data = buried.raw_sensor_data.model_copy(update={"position_on_edge": 11})
    burning_building = make_building(entity_id=20)

    original = [buried, burning_building]
    prioritized = _prioritize_role_tasks(
        AgentType.AMBULANCE_TEAM,
        original,
        refuge_ids,
    )

    assert prioritized == original


def test_same_role_claim_exclusions_only_block_lower_id_same_role() -> None:
    claims = {
        101: (5, 1, 3),
        102: (5, 1, 8),
        103: (5, 2, 2),
        104: (5, 1, 0),
    }

    excluded = _same_role_claim_exclusions(
        claims,
        own_role=1,
        agent_id=5,
    )

    assert excluded == {101}
