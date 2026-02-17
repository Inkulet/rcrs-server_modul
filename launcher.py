from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


class AnsiColor:
    """ANSI-коды позволяют визуально отделять статусы запуска в одном общем терминале."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def log_info(message: str) -> None:
    """Информационный лог фиксирует текущий этап оркестрации."""
    print(f"{AnsiColor.CYAN}[INFO]{AnsiColor.RESET} {message}", flush=True)


def log_ok(message: str) -> None:
    """Успешное завершение шага печатается зелёным для быстрого контроля состояния системы."""
    print(f"{AnsiColor.GREEN}[OK]{AnsiColor.RESET} {message}", flush=True)


def log_wait(message: str) -> None:
    """Ожидание внешней готовности (например, порта kernel) печатаем жёлтым, чтобы не путать с ошибками."""
    print(f"{AnsiColor.YELLOW}[WAIT]{AnsiColor.RESET} {message}", flush=True)


def log_error(message: str) -> None:
    """Критические ошибки подсвечиваются красным и приводят к контролируемому shutdown."""
    print(f"{AnsiColor.RED}[ERROR]{AnsiColor.RESET} {message}", flush=True)


@dataclass
class ManagedProcess:
    """Сохраняем метаданные процесса, чтобы завершать все подпроцессы единообразно."""

    name: str
    command: Sequence[str]
    cwd: Path
    process: subprocess.Popen


def parse_args() -> argparse.Namespace:
    """CLI позволяет переиспользовать launcher на разных картах/портах без редактирования кода."""
    project_root = Path(__file__).resolve().parent
    default_map_path = project_root / "rcrs-server" / "maps" / "test" / "map"
    default_config_path = project_root / "rcrs-server" / "maps" / "test" / "config"

    parser = argparse.ArgumentParser(description="Unified launcher for RCRS server + Python agents + Streamlit UI")
    parser.add_argument("--host", default=None, help="Host kernel-сервера (по умолчанию из config)")
    parser.add_argument("--kernel-port", type=int, default=None, help="TCP-порт kernel-сервера (по умолчанию из config)")
    parser.add_argument("--kernel-timeout", type=float, default=90.0, help="Максимум секунд ожидания готовности kernel")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Интервал опроса готовности порта kernel")
    parser.add_argument("--map-path", default=str(default_map_path), help="Путь к карте RCRS")
    parser.add_argument("--config-path", default=str(default_config_path), help="Путь к конфигу карты RCRS")
    parser.add_argument("--rcrs-log-dir", default=None, help="Каталог логов RCRS (если не задан, создается launcher-лог)")
    parser.add_argument("--snapshot-path", default=None, help="Опциональный путь к live_state.json для module.run_agents")
    parser.add_argument("--profile-path", default=None, help="Опциональный JSON-профиль матмодели для module.run_agents")
    parser.add_argument("--tick-sleep", type=float, default=0.0, help="Пауза между тиками для module.run_agents")
    parser.add_argument("--ui-host", default="127.0.0.1", help="Host для Streamlit UI")
    parser.add_argument("--ui-port", type=int, default=8501, help="Порт для Streamlit UI")
    parser.add_argument("--ui-timeout", type=float, default=45.0, help="Максимум секунд ожидания готовности UI")
    parser.add_argument("--browser-delay", type=float, default=2.0, help="Задержка перед открытием вкладки браузера")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер автоматически")
    parser.add_argument("--dry-run", action="store_true", help="Показать команды запуска без старта процессов")
    return parser.parse_args()


class Launcher:
    """Оркестратор поднимает все компоненты дипломного стенда в одном терминале."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).resolve().parent
        self.scripts_dir = self.project_root / "rcrs-server" / "scripts"
        self.start_script = self.scripts_dir / "start.sh"
        self.app_path = self.project_root / "app.py"

        self.map_path = self._resolve_path(args.map_path, base_dir=self.project_root)
        self.config_path = self._resolve_path(args.config_path, base_dir=self.project_root)
        self.kernel_host, self.kernel_port = self._resolve_kernel_endpoint()
        self.rcrs_log_dir = self._resolve_rcrs_log_dir()
        self.kernel_log_path = self.rcrs_log_dir / "kernel.log"
        self._kernel_ready_marker = "Listening for connections"

        self.processes: List[ManagedProcess] = []
        self._shutting_down = False
        self._kernel_was_ready = False
        self._server_exit_noted = False
        self._kernel_port_failures = 0
        self._kernel_port_fail_threshold = 5

    def run(self) -> None:
        """Полный сценарий запуска: kernel -> wait -> agents -> streamlit."""
        self._validate_environment()
        log_info(f"Kernel endpoint: {self.kernel_host}:{self.kernel_port}")
        log_info(f"RCRS log dir: {self.rcrs_log_dir}")

        if self.args.dry_run:
            self._print_plan()
            return

        self._start_kernel()
        self._wait_kernel_ready()
        self._start_agents()
        self._start_streamlit()
        self._open_browser_if_enabled()
        log_ok("Все компоненты запущены. Для остановки нажмите Ctrl+C.")
        self._monitor_processes()

    def shutdown(self) -> None:
        """Graceful shutdown удаляет все подпроцессы, чтобы не оставлять висящие java/python процессы."""
        if self._shutting_down:
            return
        self._shutting_down = True

        alive = [managed for managed in self.processes if managed.process.poll() is None]
        if not alive:
            return

        log_wait("Останавливаем процессы...")
        self._signal_all(alive, signal.SIGINT)
        if self._wait_for_exit(timeout_seconds=4.0):
            log_ok("Все процессы остановлены через SIGINT.")
            return

        self._signal_all(alive, signal.SIGTERM)
        if self._wait_for_exit(timeout_seconds=4.0):
            log_ok("Все процессы остановлены через SIGTERM.")
            return

        self._signal_all(alive, signal.SIGKILL)
        self._wait_for_exit(timeout_seconds=2.0)
        log_ok("Принудительная остановка завершена (SIGKILL).")

    def _validate_environment(self) -> None:
        """Ранние проверки дают диагностируемую ошибку до старта подпроцессов."""
        if not self.start_script.exists():
            raise FileNotFoundError(f"Не найден серверный скрипт: {self.start_script}")
        if not os.access(self.start_script, os.X_OK):
            raise PermissionError(f"Скрипт не исполняемый: {self.start_script}")
        if not self.map_path.exists():
            raise FileNotFoundError(f"Не найдена карта: {self.map_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Не найден конфиг карты: {self.config_path}")
        if not self.app_path.exists():
            raise FileNotFoundError(f"Не найден UI entrypoint: {self.app_path}")
        if not (self.project_root / "module").exists():
            raise FileNotFoundError("Не найдена папка module/")
        self.rcrs_log_dir.mkdir(parents=True, exist_ok=True)

    def _print_plan(self) -> None:
        """Dry-run нужен для безопасной проверки путей и итоговых команд перед фактическим запуском."""
        kernel_cmd = self._build_kernel_command()
        agents_cmd = self._build_agents_command()
        streamlit_cmd = self._build_streamlit_command()

        log_info("DRY RUN: команды запуска")
        print(f"  [1] cwd={self.scripts_dir} :: {' '.join(kernel_cmd)}")
        print(f"  [2] cwd={self.project_root} :: {' '.join(agents_cmd)}")
        print(f"  [3] cwd={self.project_root} :: {' '.join(streamlit_cmd)}")

    def _start_kernel(self) -> None:
        """Kernel стартуем первым, так как агенты и UI Live Mode зависят от его состояния."""
        command = self._build_kernel_command()
        managed = self._spawn_process(name="RCRS Server", command=command, cwd=self.scripts_dir)
        log_ok(f"RCRS Server запущен (pid={managed.process.pid})")

    def _wait_kernel_ready(self) -> None:
        """Проверка порта 7000 синхронизирует оркестрацию и исключает ранний запуск агентов."""
        log_wait(
            f"Ожидание готовности kernel на {self.kernel_host}:{self.kernel_port} "
            f"(маркер в {self.kernel_log_path}) ..."
        )
        deadline = time.monotonic() + max(1.0, float(self.args.kernel_timeout))

        while time.monotonic() < deadline:
            server = self._get_process("RCRS Server")
            if server is None:
                raise RuntimeError("Процесс RCRS Server не найден во внутреннем реестре.")
            exit_code = server.process.poll()
            if exit_code is not None:
                raise RuntimeError(f"RCRS Server завершился до готовности порта (code={exit_code})")

            marker_ready = self._kernel_startup_marker_ready()
            port_ready = self._is_port_open(self.kernel_host, self.kernel_port)
            if marker_ready and port_ready:
                self._kernel_was_ready = True
                log_ok("Kernel готов, запускаем агентов.")
                return
            time.sleep(max(0.1, float(self.args.poll_interval)))

        raise TimeoutError(
                f"Kernel не стал готов за {self.args.kernel_timeout:.1f} сек. "
                f"Проверьте логи: {self.rcrs_log_dir}"
            )

    def _start_agents(self) -> None:
        """Агенты стартуют вторым этапом, когда гарантирован доступ к kernel."""
        command = self._build_agents_command()
        managed = self._spawn_process(name="Python Agents", command=command, cwd=self.project_root)
        log_ok(f"Python Agents запущены (pid={managed.process.pid})")

    def _start_streamlit(self) -> None:
        """UI запускается после агентов, чтобы сразу показывать живые snapshot-данные."""
        command = self._build_streamlit_command()
        managed = self._spawn_process(name="Streamlit UI", command=command, cwd=self.project_root)
        log_ok(f"Streamlit UI запущен (pid={managed.process.pid})")

    def _open_browser_if_enabled(self) -> None:
        """Автооткрытие браузера упрощает UX единым запуском `python launcher.py`."""
        if bool(self.args.no_browser):
            log_info("Автооткрытие браузера отключено флагом --no-browser.")
            return

        ui_url = f"http://{self.args.ui_host}:{self.args.ui_port}"
        log_wait(f"Ожидание готовности Streamlit на {ui_url} ...")
        deadline = time.monotonic() + max(1.0, float(self.args.ui_timeout))

        while time.monotonic() < deadline:
            ui_process = self._get_process("Streamlit UI")
            if ui_process is None:
                raise RuntimeError("Процесс Streamlit UI не найден во внутреннем реестре.")
            exit_code = ui_process.process.poll()
            if exit_code is not None:
                raise RuntimeError(f"Streamlit UI завершился до открытия браузера (code={exit_code})")

            if self._is_port_open(str(self.args.ui_host), int(self.args.ui_port)):
                break
            time.sleep(0.3)
        else:
            log_error(f"UI не открыл порт {self.args.ui_port} за {self.args.ui_timeout:.1f} сек.")
            log_info(f"Откройте вручную: {ui_url}")
            return

        browser_delay = max(0.0, float(self.args.browser_delay))
        if browser_delay > 0:
            log_wait(f"UI готов. Открываем браузер через {browser_delay:.1f} сек...")
            time.sleep(browser_delay)

        try:
            opened = webbrowser.open(ui_url, new=2)
            if opened:
                log_ok(f"Браузер открыт: {ui_url}")
            else:
                log_wait("Система не подтвердила автооткрытие браузера.")
                log_info(f"Откройте вручную: {ui_url}")
        except Exception as error:  # noqa: BLE001
            log_error(f"Не удалось открыть браузер автоматически: {error}")
            log_info(f"Откройте вручную: {ui_url}")

    def _monitor_processes(self) -> None:
        """Мониторинг фиксирует аварийное завершение любого компонента и останавливает весь стенд."""
        while True:
            for managed in self.processes:
                if managed.name == "RCRS Server":
                    continue
                exit_code = managed.process.poll()
                if exit_code is None:
                    continue
                raise RuntimeError(f"{managed.name} неожиданно завершился (code={exit_code})")

            self._monitor_kernel_liveness()
            time.sleep(1.0)

    def _monitor_kernel_liveness(self) -> None:
        """Kernel контролируется по порту, так как `start.sh` может завершиться, оставив Java-процессы живыми."""
        if not self._kernel_was_ready:
            return

        server_process = self._get_process("RCRS Server")
        if server_process is not None:
            server_exit_code = server_process.process.poll()
            if server_exit_code is not None and not self._server_exit_noted:
                self._server_exit_noted = True
                log_wait(
                    f"Процесс start.sh завершился (code={server_exit_code}). "
                    f"Проверяем состояние kernel по порту {self.kernel_port}."
                )

        kernel_is_open = self._is_port_open(self.kernel_host, self.kernel_port)
        if kernel_is_open:
            self._kernel_port_failures = 0
            return

        self._kernel_port_failures += 1
        if self._kernel_port_failures == 1:
            log_wait(
                f"Kernel порт {self.kernel_port} временно недоступен. "
                f"Повторяем проверку ({self._kernel_port_fail_threshold} попыток)."
            )
        if self._kernel_port_failures >= self._kernel_port_fail_threshold:
            raise RuntimeError(
                f"Kernel порт {self.kernel_port} недоступен "
                f"{self._kernel_port_failures} проверок подряд."
            )

    def _spawn_process(self, name: str, command: Sequence[str], cwd: Path) -> ManagedProcess:
        """Каждый подпроцесс запускается в отдельной process-group для надежного группового shutdown."""
        log_info(f"Запуск {name}: {' '.join(command)}")

        popen_kwargs = {"cwd": str(cwd)}
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(command, **popen_kwargs)
        managed = ManagedProcess(name=name, command=command, cwd=cwd, process=process)
        self.processes.append(managed)
        return managed

    def _build_kernel_command(self) -> List[str]:
        """Старт server идёт через start.sh из scripts/, чтобы функции окружения RCRS отработали корректно."""
        # Важно передавать относительные аргументы: kill.sh внутри rcrs-server фильтрует процессы
        # по абсолютному пути репозитория и может убить сам start.sh, если в argv есть '/.../rcrs-server/...'.
        map_arg = self._to_relative_arg(self.map_path, self.scripts_dir)
        config_arg = self._to_relative_arg(self.config_path, self.scripts_dir)
        log_arg = self._to_relative_arg(self.rcrs_log_dir, self.scripts_dir)
        return [
            "./start.sh",
            "-m",
            map_arg,
            "-c",
            config_arg,
            "-l",
            log_arg,
        ]

    def _kernel_startup_marker_ready(self) -> bool:
        """Маркер в kernel.log защищает от ложной готовности, когда порт занят не тем процессом."""
        if not self.kernel_log_path.exists():
            return False
        try:
            content = self.kernel_log_path.read_text(encoding="utf-8", errors="ignore")
            return self._kernel_ready_marker in content
        except OSError:
            return False

    def _build_agents_command(self) -> List[str]:
        """Используем sys.executable, чтобы агенты гарантированно шли из текущего venv."""
        command = [
            sys.executable,
            "-m",
            "module.run_agents",
            "--host",
            str(self.kernel_host),
            "--port",
            str(self.kernel_port),
            "--tick-sleep",
            str(self.args.tick_sleep),
        ]
        if self.args.snapshot_path:
            snapshot_path = self._resolve_path(self.args.snapshot_path, base_dir=self.project_root)
            command.extend(["--snapshot-path", str(snapshot_path)])
        if self.args.profile_path:
            profile_path = self._resolve_path(self.args.profile_path, base_dir=self.project_root)
            command.extend(["--profile-path", str(profile_path)])
        return command

    def _build_streamlit_command(self) -> List[str]:
        """UI запускаем через `python -m streamlit`, чтобы не зависеть от PATH системного shell."""
        return [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(self.app_path),
            "--server.address",
            str(self.args.ui_host),
            "--server.port",
            str(self.args.ui_port),
            "--server.headless",
            "true",
        ]

    def _signal_all(self, processes: List[ManagedProcess], sig: signal.Signals) -> None:
        """Сигналы отправляются всей process-group, иначе дочерние Java/Python процессы могут остаться жить."""
        for managed in processes:
            if managed.process.poll() is not None:
                continue
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(managed.process.pid), sig)
                else:
                    if sig == signal.SIGKILL:
                        managed.process.kill()
                    elif sig == signal.SIGTERM:
                        managed.process.terminate()
                    else:
                        managed.process.send_signal(signal.CTRL_BREAK_EVENT)
            except ProcessLookupError:
                continue
            except Exception as error:  # noqa: BLE001
                log_error(f"Не удалось отправить {sig.name} процессу {managed.name}: {error}")

    def _wait_for_exit(self, timeout_seconds: float) -> bool:
        """Ожидание после сигнала нужно, чтобы дать скриптам возможность выполнить свои cleanup-hooks."""
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            if all(managed.process.poll() is not None for managed in self.processes):
                return True
            time.sleep(0.2)
        return all(managed.process.poll() is not None for managed in self.processes)

    def _get_process(self, name: str) -> ManagedProcess | None:
        """Поиск процесса по имени централизован, чтобы контроль состояния был читаемым."""
        for managed in self.processes:
            if managed.name == name:
                return managed
        return None

    @staticmethod
    def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
        """Проверка TCP-порта подтверждает готовность kernel к приему AK_CONNECT от агентов."""
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def _resolve_path(raw_path: str, base_dir: Path) -> Path:
        """Относительные пути интерпретируются от корня проекта для предсказуемого поведения launcher."""
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (base_dir / path).resolve()

    @staticmethod
    def _to_relative_arg(target_path: Path, from_dir: Path) -> str:
        """Относительный аргумент минимизирует риск self-kill в штатных скриптах rcrs-server."""
        try:
            return os.path.relpath(str(target_path), start=str(from_dir))
        except Exception:  # noqa: BLE001
            return str(target_path)

    def _resolve_rcrs_log_dir(self) -> Path:
        """Отдельный лог-каталог на запуск исключает чтение устаревших логов и упрощает диагностику."""
        if self.args.rcrs_log_dir:
            return self._resolve_path(str(self.args.rcrs_log_dir), base_dir=self.project_root)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return (self.project_root / "rcrs-server" / "logs" / f"launcher-{timestamp}").resolve()

    def _resolve_kernel_endpoint(self) -> Tuple[str, int]:
        """Host/port берём из config, чтобы launcher и rcrs-server использовали одинаковые сетевые параметры."""
        config_host, config_port = self._read_kernel_endpoint_from_config_dir(self.config_path)

        host = str(self.args.host).strip() if self.args.host else None
        port = int(self.args.kernel_port) if self.args.kernel_port is not None else None

        resolved_host = host or config_host or "127.0.0.1"
        resolved_port = port if port is not None else (config_port if config_port is not None else 7000)
        return resolved_host, int(resolved_port)

    def _read_kernel_endpoint_from_config_dir(self, config_dir: Path) -> Tuple[Optional[str], Optional[int]]:
        """Разбор kernel.cfg с include-цепочкой извлекает реальные kernel.host/kernel.port без ручных констант."""
        kernel_cfg = config_dir / "kernel.cfg"
        if not kernel_cfg.exists():
            return None, None

        values: Dict[str, str] = {}
        visited: Set[Path] = set()
        self._parse_cfg_recursive(kernel_cfg, values, visited)

        host_value = values.get("kernel.host")
        port_value = values.get("kernel.port")

        host = host_value.strip() if host_value is not None and host_value.strip() else None
        port: Optional[int]
        try:
            port = int(port_value) if port_value is not None else None
        except (TypeError, ValueError):
            port = None

        return host, port

    def _parse_cfg_recursive(self, config_file: Path, output: Dict[str, str], visited: Set[Path]) -> None:
        """Рекурсивный парсер понимает `!include` и формирует итоговые значения, как это делает launcher Java."""
        resolved_file = config_file.resolve()
        if resolved_file in visited:
            return
        if not resolved_file.exists():
            return
        visited.add(resolved_file)

        try:
            lines = resolved_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue

            if stripped.startswith("!include"):
                include_target = stripped[len("!include") :].strip()
                if not include_target:
                    continue
                include_file = (resolved_file.parent / include_target).resolve()
                self._parse_cfg_recursive(include_file, output, visited)
                continue

            if ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            normalized_key = key.strip()
            if not normalized_key:
                continue

            normalized_value = value.strip()
            if "#" in normalized_value:
                normalized_value = normalized_value.split("#", 1)[0].strip()
            output[normalized_key] = normalized_value


def main() -> None:
    """Точка входа объединяет обработку Ctrl+C и аварийных ошибок в одном месте."""
    args = parse_args()
    launcher = Launcher(args)

    try:
        launcher.run()
    except KeyboardInterrupt:
        log_wait("Получен Ctrl+C, выполняем graceful shutdown...")
    except Exception as error:  # noqa: BLE001
        log_error(str(error))
        launcher.shutdown()
        sys.exit(1)
    else:
        launcher.shutdown()
    finally:
        # При Ctrl+C всегда завершить процессы; повторный вызов безопасен за счет _shutting_down guard.
        launcher.shutdown()


if __name__ == "__main__":
    main()
