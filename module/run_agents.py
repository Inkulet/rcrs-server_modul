from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    """Параметры launcher-скрипта позволяют массово поднять три типа агентов одной командой."""
    parser = argparse.ArgumentParser(description="Run FIRE/AMBULANCE/POLICE python bridge agents")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument(
        "--snapshot-path",
        default=str(Path(__file__).resolve().parents[0] / "data" / "live_state.json"),
    )
    parser.add_argument("--profile-path", default=None)
    parser.add_argument("--tick-sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """Launcher управляет жизненным циклом дочерних процессов и завершает их единым Ctrl+C."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    agents = [
        ("FIRE_BRIGADE", "python_fire_1", 101),
        ("AMBULANCE_TEAM", "python_ambulance_1", 201),
        ("POLICE_FORCE", "python_police_1", 301),
    ]

    processes: List[subprocess.Popen] = []
    try:
        for agent_type, agent_name, request_id in agents:
            cmd = [
                sys.executable,
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
            print(f"[LAUNCHER] started {agent_name} (pid={process.pid})")

        exit_codes: List[int] = []
        for process in processes:
            exit_codes.append(int(process.wait()))

        # Лаунчер верхнего уровня должен видеть ошибку, если хотя бы один агент завершился аварийно.
        if any(exit_code != 0 for exit_code in exit_codes):
            sys.exit(1)
    except KeyboardInterrupt:
        print("[LAUNCHER] Ctrl+C received, terminating agents...")
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


if __name__ == "__main__":
    main()
