from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Tuple


def load_rcrs_proto_modules() -> Tuple[ModuleType, ModuleType]:
    """Загружает URN и protobuf-модули платформы RCRS из rcrs-server с контролируемым fallback."""
    # Генерированный RCRSProto_pb2.py старого формата требует python-реализацию protobuf,
    # иначе импорт падает на новых версиях пакета protobuf.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    platform_python_dir = _resolve_platform_python_dir()
    platform_python_path = str(platform_python_dir)
    if platform_python_path not in sys.path:
        sys.path.insert(0, platform_python_path)

    try:
        urn_module = importlib.import_module("URN")
        proto_module = importlib.import_module("RCRSProto_pb2")
        return urn_module, proto_module
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "Не удалось загрузить URN/RCRSProto_pb2 из rcrs-server/scripts/platforms/python. "
            "Проверьте наличие rcrs-server и совместимость protobuf."
        ) from error


def _resolve_platform_python_dir() -> Path:
    """Путь можно переопределить через env, чтобы запуск работал и вне исходной структуры репозитория."""
    override = os.getenv("RCRS_PLATFORM_PYTHON_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
        raise RuntimeError(f"RCRS_PLATFORM_PYTHON_DIR указывает на несуществующий путь: {path}")

    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "rcrs-server" / "scripts" / "platforms" / "python"
    if default_path.exists() and default_path.is_dir():
        return default_path

    raise RuntimeError(
        "Не найден каталог rcrs-server/scripts/platforms/python. "
        "Укажите путь через переменную окружения RCRS_PLATFORM_PYTHON_DIR."
    )
