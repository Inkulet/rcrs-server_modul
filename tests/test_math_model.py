from __future__ import annotations

import math
import unittest

from module.calculator import UtilityCalculator
from module.config import ModelConfig, ModelConstants, UtilityWeights
from module.data_models import (
    AgentOperationalState,
    AgentState,
    AgentType,
    ComputedMetrics,
    EntityType,
    Position,
    RawSensorData,
    Resources,
    VisibleEntity,
)
from module.strategy import UtilityBasedTargetSelectionStrategy


def build_test_config(c_switch: float = 0.0) -> ModelConfig:
    """Тестовый конфиг сохраняет формулу модели и дает детерминированные ожидания в unit-тестах."""
    return ModelConfig(
        weights_by_agent={
            AgentType.FIRE_BRIGADE: UtilityWeights(w_c=2.0, w_d=3.0, w_e=4.0, w_n=5.0),
            AgentType.AMBULANCE_TEAM: UtilityWeights(w_c=2.0, w_d=3.0, w_e=4.0, w_n=5.0),
            AgentType.POLICE_FORCE: UtilityWeights(w_c=2.0, w_d=3.0, w_e=4.0, w_n=5.0),
        },
        constants=ModelConstants(
            max_map_distance=100.0,
            max_buriedness=100.0,
            max_total_area=1000.0,
            max_repair_cost=200.0,
            temperature_max=100.0,
            social_radius=50.0,
            ambulance_clear_rate=10.0,
            travel_speed=10.0,
            epsilon=1e-6,
            c_switch=c_switch,
        ),
    )


class UtilityModelTests(unittest.TestCase):
    """Покрываем базовые формулы 2.2 и критические фильтры раздела 4."""

    def setUp(self) -> None:
        self.config = build_test_config()
        self.calculator = UtilityCalculator(self.config)

    def test_ttl_damage_zero_returns_infinity(self) -> None:
        """По 2.2 при Damage=0 должен получаться TTL=inf."""
        ttl = self.calculator.calculate_ttl(hp=5000, damage=0)
        self.assertTrue(math.isinf(ttl))

    def test_fire_utility_weighted_sum_matches_formula(self) -> None:
        """Проверяем U_ij = w_c*f_urgency - (w_d*f_dist + w_e*f_effort + w_n*f_social)."""
        agent = AgentState(
            id=1,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=10, x=0, y=0),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=1000, is_transporting=False),
        )
        target = VisibleEntity(
            id=100,
            type=EntityType.BUILDING,
            position=Position(entity_id=100, x=10, y=0),
            raw_sensor_data=RawSensorData(temperature=50.0, fieryness=1, ground_area=100, floors=1),
            computed_metrics=ComputedMetrics(path_distance=20.0, total_area=100.0),
        )

        result = self.calculator.calculate_utility_breakdown(
            agent=agent,
            entity=target,
            all_agents=[agent],
            all_visible_entities=[target],
            refuges=[],
        )

        # f_urgency=0.5, f_dist=0.2, f_effort=0.1, f_social=0.0 -> U=2*0.5-(3*0.2+4*0.1)=0
        self.assertAlmostEqual(result.f_urgency, 0.5, places=6)
        self.assertAlmostEqual(result.f_dist, 0.2, places=6)
        self.assertAlmostEqual(result.f_effort, 0.1, places=6)
        self.assertAlmostEqual(result.f_social, 0.0, places=6)
        self.assertAlmostEqual(result.utility_score, 0.0, places=6)

    def test_fire_prefilter_excludes_fieryness_8(self) -> None:
        """Фильтр раздела 4 обязан исключать здания с Fieryness in {4,5,6,7,8}."""
        agent = AgentState(
            id=1,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=10, x=0, y=0),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=1000, is_transporting=False),
        )
        target = VisibleEntity(
            id=101,
            type=EntityType.BUILDING,
            position=Position(entity_id=101, x=10, y=0),
            raw_sensor_data=RawSensorData(temperature=300.0, fieryness=8, ground_area=100, floors=1),
            computed_metrics=ComputedMetrics(path_distance=10.0, total_area=100.0),
        )

        passed, reason = self.calculator.prefilter_candidate(agent, target)
        self.assertFalse(passed)
        self.assertIn("Fieryness_i", reason)

    def test_ambulance_prefilter_excludes_unreachable_by_ttl(self) -> None:
        """Если TTL <= T_travel + T_work, цель должна быть исключена по разделу 4."""
        agent = AgentState(
            id=2,
            type=AgentType.AMBULANCE_TEAM,
            position=Position(entity_id=20, x=0, y=0),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=0, is_transporting=False),
        )
        target = VisibleEntity(
            id=201,
            type=EntityType.CIVILIAN,
            position=Position(entity_id=201, x=100, y=100),
            raw_sensor_data=RawSensorData(hp=100, damage=100, buriedness=120),
            computed_metrics=ComputedMetrics(path_distance=3000.0),
        )

        passed, reason = self.calculator.prefilter_candidate(agent, target)
        self.assertFalse(passed)
        self.assertIn("TTL_i <= T_travel + T_work", reason)

    def test_ambulance_prefilter_excludes_stable_casualty(self) -> None:
        """По разделу 4 случай Damage=0 и Buriedness=0 должен быть исключен до расчета U_ij."""
        agent = AgentState(
            id=2,
            type=AgentType.AMBULANCE_TEAM,
            position=Position(entity_id=20, x=0, y=0),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=0, is_transporting=False),
        )
        target = VisibleEntity(
            id=202,
            type=EntityType.CIVILIAN,
            position=Position(entity_id=202, x=10, y=10),
            raw_sensor_data=RawSensorData(hp=4000, damage=0, buriedness=0),
            computed_metrics=ComputedMetrics(path_distance=1.0),
        )

        passed, reason = self.calculator.prefilter_candidate(agent, target)
        self.assertFalse(passed)
        self.assertIn("Damage_i = 0 и Buriedness_i = 0", reason)

    def test_fire_urgency_is_clipped_to_unit_interval(self) -> None:
        """Температура выше T_max не должна выводить f_urgency за пределы [0,1]."""
        agent = AgentState(
            id=1,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=10, x=0, y=0),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=1000, is_transporting=False),
        )
        target = VisibleEntity(
            id=102,
            type=EntityType.BUILDING,
            position=Position(entity_id=102, x=10, y=0),
            raw_sensor_data=RawSensorData(temperature=10000.0, fieryness=1, ground_area=100, floors=1),
            computed_metrics=ComputedMetrics(path_distance=10.0, total_area=100.0),
        )

        result = self.calculator.calculate_utility_breakdown(
            agent=agent,
            entity=target,
            all_agents=[agent],
            all_visible_entities=[target],
            refuges=[],
        )
        self.assertLessEqual(result.f_urgency, 1.0)
        self.assertGreaterEqual(result.f_urgency, 0.0)

    def test_police_urgency_excludes_self_reference(self) -> None:
        """У блокады-кандидата не должно быть приоритета 1.0 из-за расстояния до самой себя."""
        agent = AgentState(
            id=3,
            type=AgentType.POLICE_FORCE,
            position=Position(entity_id=30, x=0, y=0),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=0, is_transporting=False),
        )
        blockade = VisibleEntity(
            id=301,
            type=EntityType.BLOCKADE,
            position=Position(entity_id=3010, x=0, y=0),
            raw_sensor_data=RawSensorData(repair_cost=100),
            computed_metrics=ComputedMetrics(path_distance=10.0),
        )
        refuges = [Position(entity_id=9001, x=100, y=0)]

        result = self.calculator.calculate_utility_breakdown(
            agent=agent,
            entity=blockade,
            all_agents=[agent],
            all_visible_entities=[blockade],
            refuges=refuges,
        )

        self.assertGreater(result.f_urgency, 0.0)
        self.assertLess(result.f_urgency, 0.1)


class StrategyTests(unittest.TestCase):
    """Проверяем обязательные ветки алгоритма выбора цели из раздела 4."""

    def test_fire_agent_without_water_goes_to_refuge(self) -> None:
        """При Water=0 стратегия должна вернуть GoToRefuge, а не выбирать здание для тушения."""
        calculator = UtilityCalculator(build_test_config())
        strategy = UtilityBasedTargetSelectionStrategy(calculator)

        fire_agent = AgentState(
            id=10,
            type=AgentType.FIRE_BRIGADE,
            position=Position(entity_id=1000, x=50, y=50),
            state=AgentOperationalState.IDLE,
            resources=Resources(water_quantity=0, is_transporting=False),
        )
        burning_building = VisibleEntity(
            id=777,
            type=EntityType.BUILDING,
            position=Position(entity_id=777, x=80, y=80),
            raw_sensor_data=RawSensorData(temperature=70.0, fieryness=2, ground_area=100, floors=1),
            computed_metrics=ComputedMetrics(path_distance=42.0, total_area=100.0),
        )
        refuges = [
            Position(entity_id=9001, x=60, y=60),
            Position(entity_id=9002, x=300, y=300),
        ]

        result = strategy.select_target(
            agent=fire_agent,
            visible_entities=[burning_building],
            all_agents=[fire_agent],
            refuges=refuges,
        )

        self.assertEqual(result.action, "GO_TO_REFUGE")
        self.assertEqual(result.selected_target_id, 9001)

    def test_busy_state_skips_target_selection(self) -> None:
        """Статусы Loading/Unloading/Transporting должны пропускать этап выбора цели."""
        calculator = UtilityCalculator(build_test_config())
        strategy = UtilityBasedTargetSelectionStrategy(calculator)

        ambulance_agent = AgentState(
            id=11,
            type=AgentType.AMBULANCE_TEAM,
            position=Position(entity_id=1100, x=0, y=0),
            state=AgentOperationalState.LOADING,
            resources=Resources(water_quantity=0, is_transporting=False),
        )
        candidate = VisibleEntity(
            id=901,
            type=EntityType.CIVILIAN,
            position=Position(entity_id=901, x=5, y=5),
            raw_sensor_data=RawSensorData(hp=5000, damage=20, buriedness=10),
            computed_metrics=ComputedMetrics(path_distance=1.0),
        )

        result = strategy.select_target(
            agent=ambulance_agent,
            visible_entities=[candidate],
            all_agents=[ambulance_agent],
            refuges=[],
        )

        self.assertEqual(result.action, "SKIP_BUSY")
        self.assertEqual(result.utility_matrix, [])


if __name__ == "__main__":
    unittest.main()
