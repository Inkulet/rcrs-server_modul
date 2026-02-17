from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    """Параметры launcher-скрипта позволяют массово поднять три типа агентов одной командой."""
    parser = argparse.ArgumentParser(description="Run FIRE/AMBULANCE/POLICE python bridge agents")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument(
        "--snapshot-path",
        default=str(Path(__file__).resolve().parents[0] / "data" / "live_state.json"),
    )
    parser.add_argument(
        "--state-path",
        default=str(Path(__file__).resolve().parents[0] / "data" / "run_agents_state.json"),
        help="JSON-состояние процесса для устойчивого stop/recover в UI после перезапуска Streamlit",
    )
    parser.add_argument("--profile-path", default=None)
    parser.add_argument("--tick-sleep", type=float, default=0.0)
    return parser.parse_args()


def _write_state_file(
    state_path: Path,
    parent_pid: int,
    parent_command: List[str],
    children: List[Dict[str, Any]],
) -> None:
    """Атомарная запись state-файла нужна, чтобы UI читал только целостный JSON без полумер."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "parent_pid": int(parent_pid),
        "parent_command": list(parent_command),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "children": children,
    }
    tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)


def _cleanup_state_file_if_owned(state_path: Path, parent_pid: int) -> None:
    """Удаляем state только своего процесса, чтобы параллельный запуск не потерял актуальные PID-данные."""
    if not state_path.exists():
        return
    try:
        raw_data = json.loads(state_path.read_text(encoding="utf-8"))
        file_parent_pid = int(raw_data.get("parent_pid"))
    except Exception:  # noqa: BLE001
        try:
            state_path.unlink()
        except OSError:
            pass
        return

    if file_parent_pid != int(parent_pid):
        return
    try:
        state_path.unlink()
    except OSError:
        pass


def main() -> None:
    """Launcher управляет жизненным циклом дочерних процессов и завершает их единым Ctrl+C."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    state_path = Path(args.state_path).expanduser().resolve()
    parent_pid = os.getpid()
    parent_command = [sys.executable, "-m", "module.run_agents", *sys.argv[1:]]

    agents = [
        ("FIRE_BRIGADE", "python_fire_1", 101),
        ("AMBULANCE_TEAM", "python_ambulance_1", 201),
        ("POLICE_FORCE", "python_police_1", 301),
    ]

    processes: List[subprocess.Popen] = []
    child_state: List[Dict[str, Any]] = []
    try:
        try:
            _write_state_file(
                state_path=state_path,
                parent_pid=parent_pid,
                parent_command=parent_command,
                children=child_state,
            )
        except Exception as error:  # noqa: BLE001
            print(f"[LAUNCHER] warning: cannot write state file {state_path}: {error}", flush=True)

        for agent_type, agent_name, request_id in agents:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "module.main_agent",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--agent-type",
                agent_type,
                "--agent-name",
                agent_name,
                "--request-id",
                str(request_id),
                "--snapshot-path",
                str(Path(args.snapshot_path).expanduser().resolve()),
                "--tick-sleep",
                str(args.tick_sleep),
            ]
            if args.profile_path:
                cmd.extend(["--profile-path", str(Path(args.profile_path).expanduser().resolve())])

            # Запуск через `-m module.main_agent` сохраняет корректный импорт package `module`
            # даже если launcher стартует не из корня репозитория.
            process = subprocess.Popen(cmd, cwd=str(project_root))
            processes.append(process)
            child_state.append(
                {
                    "agent_type": agent_type,
                    "agent_name": agent_name,
                    "request_id": int(request_id),
                    "pid": int(process.pid),
                    "command": cmd,
                }
            )
            try:
                _write_state_file(
                    state_path=state_path,
                    parent_pid=parent_pid,
                    parent_command=parent_command,
                    children=child_state,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[LAUNCHER] warning: cannot update state file {state_path}: {error}", flush=True)

            print(f"[LAUNCHER] started {agent_name} (pid={process.pid})", flush=True)

        exit_codes: List[int] = []
        for process in processes:
            exit_codes.append(int(process.wait()))

        # Лаунчер верхнего уровня должен видеть ошибку, если хотя бы один агент завершился аварийно.
        if any(exit_code != 0 for exit_code in exit_codes):
            sys.exit(1)
    except KeyboardInterrupt:
        print("[LAUNCHER] Ctrl+C received, terminating agents...", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        _cleanup_state_file_if_owned(state_path=state_path, parent_pid=parent_pid)


if __name__ == "__main__":
    main()
