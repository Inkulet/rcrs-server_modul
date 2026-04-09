from __future__ import annotations

"""В этом модуле я проверяю корректность агрегатора U_ij по формуле диплома."""

# Формула диплома (стр. 105):
#   U_ij = w_c * f_urgency - w_d * f_dist + w_e * f_effort - w_n * f_social
#
# Ключевые инварианты:
# - Знаки: f_urgency (+), f_dist (−), f_effort (+), f_social (−) — антироевой механизм
# - f_social вычитается: чем больше союзников рядом с целью, тем ниже полезность
# - Сумма весов = 1 (w_c + w_d + w_e + w_n = 1)
# - Результат не ограничен [0,1] — может быть отрицательным при большом f_dist

import pytest
from unittest.mock import patch

from decision.utility.aggregator import UtilityAggregator
from world.entities import AgentType, Position

from conftest import make_agent, make_building, make_civilian, make_world_with_graph


W_C, W_D, W_E, W_N = 0.4, 0.2, 0.2, 0.2


def make_aggregator() -> UtilityAggregator:
    """Я создаю агрегатор с весами из диплома."""
    return UtilityAggregator(w_c=W_C, w_d=W_D, w_e=W_E, w_n=W_N)


def make_target_pos(entity_id: int = 2) -> Position:
    """Я создаю позицию цели для передачи в агрегатор."""
    return Position(entity_id=entity_id, x=500, y=0)


# ===========================================================================
# Тесты формулы агрегатора
# ===========================================================================


class TestAggregatorFormula:
    """Я проверяю, что агрегатор вычисляет именно формулу диплома, а не другую."""

    def test_formula_sign_convention(self) -> None:
        """Я проверяю знаки в формуле: f_urgency (+), f_dist (−), f_effort (+), f_social (−).

        f_social вычитается — это реализует антироевой механизм (Критерий 4 диплома):
        чем больше союзников уже работают с целью, тем ниже её полезность для данного агента.
        Я патчу все четыре фактора и вручную считаю ожидаемый результат.
        """
        aggregator = make_aggregator()
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        entity = make_building()
        wm = make_world_with_graph()

        with (
            patch("decision.utility.aggregator.compute_urgency", return_value=0.8) as m_u,
            patch("decision.utility.aggregator.compute_effort", return_value=0.5) as m_e,
            patch("decision.utility.aggregator.distance_factor_precomputed", return_value=0.3) as m_d,
            patch("decision.utility.aggregator.social_factor", return_value=0.2) as m_s,
        ):
            result = aggregator.calculate_utility(
                agent_state=agent,
                entity=entity,
                world_model=wm,
                target_position=make_target_pos(),
            )

        # U = 0.4*0.8 - 0.2*0.3 + 0.2*0.5 - 0.2*0.2 = 0.32 - 0.06 + 0.10 - 0.04 = 0.32
        expected = W_C * 0.8 - W_D * 0.3 + W_E * 0.5 - W_N * 0.2
        assert abs(result - expected) < 1e-9

    def test_weights_sum_to_one(self) -> None:
        """Я проверяю инвариант диплома: w_c + w_d + w_e + w_n = 1.0."""
        agg = make_aggregator()
        total = agg.w_c + agg.w_d + agg.w_e + agg.w_n
        assert abs(total - 1.0) < 1e-9

    def test_increasing_urgency_increases_utility(self) -> None:
        """Я проверяю: чем выше f_urgency, тем выше U_ij — f_urgency — это Benefit."""
        aggregator = make_aggregator()
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        wm = make_world_with_graph()

        with (
            patch("decision.utility.aggregator.compute_effort", return_value=0.0),
            patch("decision.utility.aggregator.distance_factor_precomputed", return_value=0.0),
            patch("decision.utility.aggregator.social_factor", return_value=0.0),
        ):
            with patch("decision.utility.aggregator.compute_urgency", return_value=0.2):
                u_low = aggregator.calculate_utility(
                    agent, make_building(), wm, make_target_pos()
                )
            with patch("decision.utility.aggregator.compute_urgency", return_value=0.9):
                u_high = aggregator.calculate_utility(
                    agent, make_building(), wm, make_target_pos()
                )

        assert u_high > u_low

    def test_increasing_distance_decreases_utility(self) -> None:
        """Я проверяю: чем больше f_dist, тем ниже U_ij — расстояние — это Cost."""
        aggregator = make_aggregator()
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        wm = make_world_with_graph()

        with (
            patch("decision.utility.aggregator.compute_urgency", return_value=0.5),
            patch("decision.utility.aggregator.compute_effort", return_value=0.0),
            patch("decision.utility.aggregator.social_factor", return_value=0.0),
        ):
            with patch("decision.utility.aggregator.distance_factor_precomputed", return_value=0.1):
                u_near = aggregator.calculate_utility(
                    agent, make_building(), wm, make_target_pos()
                )
            with patch("decision.utility.aggregator.distance_factor_precomputed", return_value=0.9):
                u_far = aggregator.calculate_utility(
                    agent, make_building(), wm, make_target_pos()
                )

        assert u_near > u_far

    def test_increasing_effort_increases_utility(self) -> None:
        """Я проверяю: чем выше f_effort, тем выше U_ij — трудоёмкость — это Benefit."""
        aggregator = make_aggregator()
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        wm = make_world_with_graph()

        with (
            patch("decision.utility.aggregator.compute_urgency", return_value=0.0),
            patch("decision.utility.aggregator.distance_factor_precomputed", return_value=0.0),
            patch("decision.utility.aggregator.social_factor", return_value=0.0),
        ):
            with patch("decision.utility.aggregator.compute_effort", return_value=0.1):
                u_easy = aggregator.calculate_utility(
                    agent, make_building(), wm, make_target_pos()
                )
            with patch("decision.utility.aggregator.compute_effort", return_value=0.9):
                u_hard = aggregator.calculate_utility(
                    agent, make_building(), wm, make_target_pos()
                )

        assert u_hard > u_easy

    def test_zero_division_returns_zero(self) -> None:
        """Я проверяю защиту: если внутри возникает ZeroDivisionError, возвращаю 0.0."""
        aggregator = make_aggregator()
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        wm = make_world_with_graph()

        with patch("decision.utility.aggregator.compute_urgency", side_effect=ZeroDivisionError):
            result = aggregator.calculate_utility(
                agent, make_building(), wm, make_target_pos()
            )

        assert result == 0.0

    def test_all_factors_zero_gives_zero_utility(self) -> None:
        """Я проверяю: все факторы 0 → U = 0."""
        aggregator = make_aggregator()
        agent = make_agent(agent_type=AgentType.FIRE_BRIGADE)
        wm = make_world_with_graph()

        with (
            patch("decision.utility.aggregator.compute_urgency", return_value=0.0),
            patch("decision.utility.aggregator.compute_effort", return_value=0.0),
            patch("decision.utility.aggregator.distance_factor_precomputed", return_value=0.0),
            patch("decision.utility.aggregator.social_factor", return_value=0.0),
        ):
            result = aggregator.calculate_utility(
                agent, make_building(), wm, make_target_pos()
            )

        assert result == 0.0
