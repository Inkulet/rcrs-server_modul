from __future__ import annotations

"""В этом модуле я выношу утилиты, общие для всех файлов слоя utility.

Я избегаю дублирования кода: _clamp_to_unit раньше была скопирована в
urgency.py и effort.py. Теперь обе функции импортируют её отсюда.
"""


def _clamp_to_unit(value: float) -> float:
    """Здесь я ограничиваю значение диапазоном [0, 1], чтобы сохранить нормировку."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = ["_clamp_to_unit"]
