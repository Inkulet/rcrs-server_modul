from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import fcntl


class LiveStateStore:
    """Store обновляет общий live_state.json атомарно, чтобы UI видел согласованные снимки между процессами агентов."""

    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path
        self.lock_path = snapshot_path.with_name(f"{snapshot_path.name}.lock")
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    def update_agent_snapshot(
        self,
        tick: int,
        agent_payload: Dict[str, Any],
        warnings: Optional[List[str]] = None,
    ) -> None:
        """Обновляем запись одного агента по ID, не стирая остальные процессы в том же файле snapshot."""
        with self._acquire_lock():
            current = self._read_snapshot()
            agents = current.get("agents", {})

            # Сначала приводим payload к JSON-safe структуре: runtime передает dataclass объекты,
            # а прямой .get() по dataclass ломает обновление snapshot.
            safe_payload = self._to_json_safe(agent_payload)
            if not isinstance(safe_payload, dict):
                safe_payload = {}

            agent_state = safe_payload.get("agent_state", {})
            agent_id = "unknown"
            if isinstance(agent_state, dict):
                agent_id = str(agent_state.get("id", "unknown"))

            agents[agent_id] = {
                "tick": int(tick),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "payload": safe_payload,
                "warnings": list(warnings or []),
            }

            current["mode"] = "live"
            current["tick"] = int(tick)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            current["agents"] = agents

            self._atomic_write(current)

    def _read_snapshot(self) -> Dict[str, Any]:
        """Чтение с обработкой JSON-ошибок позволяет восстановиться после аварийного прерывания записи."""
        if not self.snapshot_path.exists():
            return {}

        try:
            content = self.snapshot_path.read_text(encoding="utf-8")
            if not content.strip():
                return {}
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                return loaded
            return {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _atomic_write(self, payload: Dict[str, Any]) -> None:
        """Атомарная запись через временный файл исключает чтение частично записанного JSON в UI."""
        directory = self.snapshot_path.parent
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)

        fd, temp_path = tempfile.mkstemp(prefix="live_state_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(serialized)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.snapshot_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @contextmanager
    def _acquire_lock(self):
        """File-lock сериализует read-modify-write между процессами агентов и исключает потерю snapshot-обновлений."""
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def _to_json_safe(self, value: Any) -> Any:
        """Сериализация преобразует dataclass/enum объекты в примитивы, чтобы snapshot оставался JSON-валидным."""
        if is_dataclass(value):
            return self._to_json_safe(asdict(value))

        if isinstance(value, dict):
            return {str(key): self._to_json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._to_json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [self._to_json_safe(item) for item in value]

        enum_value = getattr(value, "value", None)
        if enum_value is not None and not isinstance(value, (str, int, float, bool)):
            return self._to_json_safe(enum_value)

        if isinstance(value, float):
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"

        return value
