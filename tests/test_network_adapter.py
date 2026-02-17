from __future__ import annotations

import unittest

from module.data_models import AgentType, EntityType
from module.network.adapter import WorldModelAdapter
from module.network.protocol import URN
from module.network.world_model import ServerEntity, WorldModel


class WorldModelAdapterTests(unittest.TestCase):
    """Проверяем, что адаптер сохраняет контракт схемы 2.2 и не пропускает невалидные данные."""

    def test_adapt_maps_entities_and_metrics(self) -> None:
        world = WorldModel()

        world.entities = {
            1: ServerEntity(
                entity_id=1,
                urn=int(URN.Entity.ROAD),
                properties={
                    int(URN.Property.X): 0,
                    int(URN.Property.Y): 0,
                    int(URN.Property.EDGES): [{"neighbour": 2}],
                },
            ),
            2: ServerEntity(
                entity_id=2,
                urn=int(URN.Entity.ROAD),
                properties={
                    int(URN.Property.X): 100,
                    int(URN.Property.Y): 0,
                    int(URN.Property.EDGES): [{"neighbour": 1}, {"neighbour": 3}],
                },
            ),
            3: ServerEntity(
                entity_id=3,
                urn=int(URN.Entity.REFUGE),
                properties={
                    int(URN.Property.X): 200,
                    int(URN.Property.Y): 0,
                    int(URN.Property.EDGES): [{"neighbour": 2}],
                },
            ),
            10: ServerEntity(
                entity_id=10,
                urn=int(URN.Entity.FIRE_BRIGADE),
                properties={
                    int(URN.Property.POSITION): 1,
                    int(URN.Property.WATER_QUANTITY): 1500,
                },
            ),
            20: ServerEntity(
                entity_id=20,
                urn=int(URN.Entity.BUILDING),
                properties={
                    int(URN.Property.X): 100,
                    int(URN.Property.Y): 100,
                    int(URN.Property.FIERYNESS): 2,
                    int(URN.Property.TEMPERATURE): 500,
                    int(URN.Property.FLOORS): 2,
                    int(URN.Property.BUILDING_AREA_GROUND): 300,
                },
            ),
            30: ServerEntity(
                entity_id=30,
                urn=int(URN.Entity.CIVILIAN),
                properties={
                    int(URN.Property.POSITION): 2,
                    int(URN.Property.HP): 8000,
                    int(URN.Property.DAMAGE): 20,
                    int(URN.Property.BURIEDNESS): 30,
                },
            ),
            40: ServerEntity(
                entity_id=40,
                urn=int(URN.Entity.BLOCKADE),
                properties={
                    int(URN.Property.POSITION): 2,
                    int(URN.Property.REPAIR_COST): 250,
                },
            ),
        }

        adapter = WorldModelAdapter(world)
        adapted = adapter.adapt(agent_id=10, agent_type=AgentType.FIRE_BRIGADE, visible_entity_ids=[20, 30, 40])

        self.assertEqual(adapted.observation.agent_state.id, 10)
        self.assertEqual(adapted.observation.agent_state.position.entity_id, 1)
        self.assertEqual(adapted.observation.agent_state.resources.water_quantity, 1500)

        entity_by_id = {entity.id: entity for entity in adapted.observation.visible_entities}
        self.assertIn(20, entity_by_id)
        self.assertIn(30, entity_by_id)
        self.assertIn(40, entity_by_id)

        self.assertEqual(entity_by_id[20].type, EntityType.BUILDING)
        self.assertEqual(entity_by_id[20].raw_sensor_data.fieryness, 2)
        self.assertAlmostEqual(entity_by_id[20].computed_metrics.total_area or 0.0, 600.0)

        self.assertEqual(entity_by_id[30].type, EntityType.CIVILIAN)
        self.assertEqual(entity_by_id[30].raw_sensor_data.hp, 8000)
        self.assertAlmostEqual(entity_by_id[30].computed_metrics.estimated_death_time or 0.0, 400.0)

        self.assertEqual(entity_by_id[40].type, EntityType.BLOCKADE)
        self.assertEqual(entity_by_id[40].raw_sensor_data.repair_cost, 250)

        self.assertTrue(any(refuge.entity_id == 3 for refuge in adapted.refuges))

    def test_negative_values_are_validated(self) -> None:
        world = WorldModel()
        world.entities = {
            1: ServerEntity(
                entity_id=1,
                urn=int(URN.Entity.ROAD),
                properties={int(URN.Property.X): 0, int(URN.Property.Y): 0},
            ),
            10: ServerEntity(
                entity_id=10,
                urn=int(URN.Entity.AMBULANCE_TEAM),
                properties={int(URN.Property.POSITION): 1},
            ),
            30: ServerEntity(
                entity_id=30,
                urn=int(URN.Entity.CIVILIAN),
                properties={
                    int(URN.Property.POSITION): 1,
                    int(URN.Property.HP): -100,
                    int(URN.Property.DAMAGE): -1,
                    int(URN.Property.BURIEDNESS): -3,
                },
            ),
        }

        adapter = WorldModelAdapter(world)
        adapted = adapter.adapt(agent_id=10, agent_type=AgentType.AMBULANCE_TEAM, visible_entity_ids=[30])

        civilian = adapted.observation.visible_entities[0]
        self.assertIsNone(civilian.raw_sensor_data.hp)
        self.assertIsNone(civilian.raw_sensor_data.damage)
        self.assertIsNone(civilian.raw_sensor_data.buriedness)
        self.assertGreater(len(adapted.warnings), 0)


if __name__ == "__main__":
    unittest.main()
