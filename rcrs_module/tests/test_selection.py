from __future__ import annotations

"""В этом модуле я проверяю механизм гистерезиса при выборе целевой задачи (UC-5)."""

# Гистерезис предотвращает бесконечное переключение между задачами с близкими U_ij.
# Переключение происходит только если U_best > U_current + C_switch.

import pytest

from action.selection import TargetSelector


def make_selector(c_switch: float = 0.1) -> TargetSelector:
    """Я создаю TargetSelector с заданным порогом переключения."""
    return TargetSelector(c_switch=c_switch)


# ===========================================================================
# Тесты базового выбора
# ===========================================================================


class TestBasicSelection:
    """Я проверяю выбор без гистерезиса (нет текущей цели или пустой словарь)."""

    def test_empty_utilities_returns_none(self) -> None:
        """Я проверяю: нет задач → режим ожидания → None."""
        selector = make_selector()
        assert selector.select_best_target(None, {}) is None

    def test_single_task_selected(self) -> None:
        """Я проверяю: одна задача → она и выбирается независимо от значения."""
        selector = make_selector()
        assert selector.select_best_target(None, {42: 0.5}) == 42

    def test_best_task_selected_from_many(self) -> None:
        """Я проверяю: из нескольких задач выбирается та, у которой max U_ij."""
        selector = make_selector()
        utilities = {1: 0.2, 2: 0.8, 3: 0.5}
        assert selector.select_best_target(None, utilities) == 2

    def test_no_current_target_selects_best(self) -> None:
        """Я проверяю: current_target=None → выбирается лучшая без проверки гистерезиса."""
        selector = make_selector(c_switch=0.5)
        utilities = {10: 0.3, 20: 0.6}
        assert selector.select_best_target(None, utilities) == 20


# ===========================================================================
# Тесты механизма гистерезиса
# ===========================================================================


class TestHysteresis:
    """Я проверяю, что переключение цели происходит только при достаточном преимуществе."""

    def test_no_switch_when_improvement_below_threshold(self) -> None:
        """Я проверяю: U_best - U_current < C_switch → текущая цель сохраняется."""
        selector = make_selector(c_switch=0.2)
        # current=1 (U=0.5), best=2 (U=0.6) → улучшение 0.1 < 0.2 → не переключаемся
        utilities = {1: 0.5, 2: 0.6}
        result = selector.select_best_target(current_target_id=1, utilities_dict=utilities)
        assert result == 1

    def test_no_switch_when_improvement_equals_threshold(self) -> None:
        """Я проверяю граничное условие: улучшение == C_switch → не переключаемся."""
        selector = make_selector(c_switch=0.1)
        # current=1 (U=0.5), best=2 (U=0.6) → улучшение ровно 0.1 → не переключаемся
        utilities = {1: 0.5, 2: 0.6}
        result = selector.select_best_target(current_target_id=1, utilities_dict=utilities)
        assert result == 1

    def test_switch_when_improvement_above_threshold(self) -> None:
        """Я проверяю: U_best - U_current > C_switch → переключаемся на новую цель."""
        selector = make_selector(c_switch=0.1)
        # current=1 (U=0.5), best=2 (U=0.7) → улучшение 0.2 > 0.1 → переключаемся
        utilities = {1: 0.5, 2: 0.7}
        result = selector.select_best_target(current_target_id=1, utilities_dict=utilities)
        assert result == 2

    def test_current_target_remains_when_still_best(self) -> None:
        """Я проверяю: текущая цель уже лучшая → гистерезис удерживает её."""
        selector = make_selector(c_switch=0.1)
        utilities = {1: 0.9, 2: 0.3, 3: 0.1}
        result = selector.select_best_target(current_target_id=1, utilities_dict=utilities)
        assert result == 1

    def test_current_target_disappeared_selects_new_best(self) -> None:
        """Я проверяю: текущая цель исчезла из словаря → выбирается новая лучшая без гистерезиса."""
        selector = make_selector(c_switch=0.5)
        # Задача 99 была текущей, но теперь её нет в словаре
        utilities = {10: 0.3, 20: 0.8}
        result = selector.select_best_target(current_target_id=99, utilities_dict=utilities)
        assert result == 20

    def test_negative_utilities_handled_correctly(self) -> None:
        """Я проверяю: отрицательные значения U_ij не ломают выбор."""
        selector = make_selector(c_switch=0.1)
        utilities = {1: -0.5, 2: -0.1, 3: -0.9}
        # best = 2 (наименее отрицательный), current нет → выбираем 2
        result = selector.select_best_target(None, utilities)
        assert result == 2

    def test_large_c_switch_prevents_all_switches(self) -> None:
        """Я проверяю: очень большой C_switch → никогда не переключаемся."""
        selector = make_selector(c_switch=999.0)
        utilities = {1: 0.1, 2: 0.99}
        # Текущая цель 1, новая лучшая 2 — но порог огромный → остаёмся на 1
        result = selector.select_best_target(current_target_id=1, utilities_dict=utilities)
        assert result == 1
