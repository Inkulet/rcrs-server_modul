from __future__ import annotations

import argparse
import sys
from pathlib import Path

from module.data_models import AgentType
from module.network.agent_runtime import AgentRuntime, AgentRuntimeConfig


def parse_args() -> argparse.Namespace:
    """CLI-параметры вынесены отдельно для удобного запуска разных типов агентов из терминала."""
    parser = argparse.ArgumentParser(description="RCRS bridge agent: Receive -> Adapt -> Think -> Act")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес kernel (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7000, help="TCP-порт kernel (по умолчанию 7000)")
    parser.add_argument(
        "--agent-type",
        required=True,
        choices=[agent_type.value for agent_type in AgentType],
        help="Тип запускаемого агента",
    )
    parser.add_argument("--agent-name", default=None, help="Имя агента в AK_CONNECT")
    parser.add_argument("--request-id", type=int, default=1, help="RequestID для AK_CONNECT")
    parser.add_argument("--version", type=int, default=2, help="Версия протокола AK_CONNECT")
    parser.add_argument(
        "--snapshot-path",
        default=str(Path(__file__).resolve().parents[0] / "data" / "live_state.json"),
        help="Путь к live_state.json для Streamlit Live Mode",
    )
    parser.add_argument(
        "--profile-path",
        default=None,
        help="Опциональный JSON-профиль весов/констант матмодели (формулы не меняются)",
    )
    parser.add_argument("--tick-sleep", type=float, default=0.0, help="Пауза между тиками после отправки команды")
    return parser.parse_args()


def main() -> None:
    """Точка входа создаёт runtime и запускает непрерывный цикл обмена с kernel."""
    args = parse_args()

    agent_type = AgentType(args.agent_type)
    agent_name = args.agent_name or f"python_{agent_type.value.lower()}"

    runtime_config = AgentRuntimeConfig(
        host=args.host,
        port=int(args.port),
        agent_name=agent_name,
        agent_type=agent_type,
        request_id=int(args.request_id),
        version=int(args.version),
        snapshot_path=Path(args.snapshot_path).expanduser().resolve(),
        profile_path=Path(args.profile_path).expanduser().resolve() if args.profile_path else None,
        tick_sleep_sec=float(args.tick_sleep),
    )

    runtime = AgentRuntime(runtime_config)
    try:
        runtime.run()
    except KeyboardInterrupt:
        # При штатной остановке launcher посылает SIGINT; завершаемся без traceback для чистого UX.
        print("[RCRS] Остановка агента по сигналу прерывания.")
        sys.exit(0)
    except Exception as error:  # noqa: BLE001
        # Явный перехват нужен для понятной диагностики запуска без многословного traceback в консоли пользователя.
        print(f"[RCRS] Критическая ошибка запуска агента: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
