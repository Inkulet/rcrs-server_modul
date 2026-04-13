from __future__ import annotations


def _clamp_to_unit(value: float) -> float:
    if value < 0.0:
        return 0.0

    if value > 1.0:
        return 1.0

    return value


__all__ = ["_clamp_to_unit"]
