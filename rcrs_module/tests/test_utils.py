from __future__ import annotations

"""В этом модуле я тестирую утилитарную функцию _clamp_to_unit из слоя utility."""

from decision.utility._utils import _clamp_to_unit


class TestClampToUnit:
    """Я проверяю ограничение значения диапазоном [0, 1]."""

    def test_negative_returns_zero(self) -> None:
        assert _clamp_to_unit(-0.5) == 0.0

    def test_large_negative_returns_zero(self) -> None:
        assert _clamp_to_unit(-1000.0) == 0.0

    def test_above_one_returns_one(self) -> None:
        assert _clamp_to_unit(1.5) == 1.0

    def test_large_positive_returns_one(self) -> None:
        assert _clamp_to_unit(9999.0) == 1.0

    def test_zero_stays_zero(self) -> None:
        assert _clamp_to_unit(0.0) == 0.0

    def test_one_stays_one(self) -> None:
        assert _clamp_to_unit(1.0) == 1.0

    def test_mid_value_unchanged(self) -> None:
        assert _clamp_to_unit(0.42) == 0.42

    def test_boundary_just_below_zero(self) -> None:
        assert _clamp_to_unit(-1e-10) == 0.0

    def test_boundary_just_above_one(self) -> None:
        assert _clamp_to_unit(1.0 + 1e-10) == 1.0
