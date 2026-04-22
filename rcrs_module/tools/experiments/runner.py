"""Оркестратор одного прогона: сервер → агенты → сбор метрик → очистка."""

from __future__ import annotations

import configparser
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .configs import RunSpec
from .metrics import AgentsOutcome, KernelOutcome, RunMetrics, collect_agents_outcome, parse_kernel_logs


logger = logging.getLogger("experiments.runner")


# ---------------------------------------------------------------------------
# Конфигурация стенда
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    repo_root: Path
    server_dir: Path             # <repo>/rcrs-server
    module_dir: Path             # <repo>/rcrs_module
    agent_cfg_path: Path         # <repo>/rcrs_module/agent.cfg
    launch_agents_sh: Path       # <repo>/rcrs_module/launch_agents.sh
    kernel_scripts_dir: Path     # <repo>/rcrs-server/scripts
    kernel_log_dir: Path         # <repo>/rcrs-server/logs/log
    map_dir: Path                # <repo>/rcrs-server/maps/kobe/map
    config_dir: Path             # <repo>/rcrs-server/maps/kobe/config
    agents_log_dir: Path         # <repo>/log  (куда пишут summary_*.json и т.д.)
    kernel_host: str = "127.0.0.1"
    kernel_port: int = 7000


def detect_env(repo_root: Path, map_name: str = "kobe") -> EnvConfig:
    server_dir = repo_root / "rcrs-server"
    module_dir = repo_root / "rcrs_module"
    return EnvConfig(
        repo_root=repo_root,
        server_dir=server_dir,
        module_dir=module_dir,
        agent_cfg_path=module_dir / "agent.cfg",
        launch_agents_sh=module_dir / "launch_agents.sh",
        kernel_scripts_dir=server_dir / "scripts",
        kernel_log_dir=server_dir / "logs" / "log",
        map_dir=server_dir / "maps" / map_name / "map",
        config_dir=server_dir / "maps" / map_name / "config",
        agents_log_dir=repo_root / "log",
    )


# ---------------------------------------------------------------------------
# Работа с agent.cfg
# ---------------------------------------------------------------------------

def write_agent_cfg(env: EnvConfig, spec: RunSpec) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(env.agent_cfg_path)

    cfg.setdefault("logging", {})
    cfg["logging"]["enabled"] = "true"
    cfg["logging"]["level"] = "WARNING"
    cfg["logging"]["log_dir"] = str(env.agents_log_dir)

    cfg.setdefault("metrics", {})
    cfg["metrics"]["enabled"] = "true"

    cfg.setdefault("csv_metrics", {})
    cfg["csv_metrics"]["tick_csv_enabled"] = "true"
    cfg["csv_metrics"]["events_csv_enabled"] = "true"
    cfg["csv_metrics"]["summary_json_enabled"] = "true"

    cfg.setdefault("communication", {})
    cfg["communication"]["radio_enabled"] = "true" if spec.radio_enabled else "false"

    cfg.setdefault("weights", {})
    cfg["weights"]["fire"] = _fmt_weights(spec.weights_fire)
    cfg["weights"]["ambulance"] = _fmt_weights(spec.weights_ambulance)
    cfg["weights"]["police"] = _fmt_weights(spec.weights_police)

    cfg.setdefault("run_table", {})
    cfg["run_table"]["enabled"] = "false"

    with open(env.agent_cfg_path, "w", encoding="utf-8") as fh:
        cfg.write(fh)


def _fmt_weights(w) -> str:
    if w is None:
        return ""
    return ",".join(f"{x:.6g}" for x in w)


# ---------------------------------------------------------------------------
# Инфраструктура процессов
# ---------------------------------------------------------------------------

class ProcessGroup:
    # Минимальная обёртка над Popen, запускающая дочерние процессы в отдельной
    # группе (setsid) и корректно убивающая всю группу при cleanup.

    def __init__(self, name: str, args: list[str], cwd: Path, log_file: Optional[Path] = None) -> None:
        self.name = name
        self.args = args
        self.cwd = cwd
        self.log_file = log_file
        self.proc: Optional[subprocess.Popen] = None
        self._fh = None

    def start(self) -> None:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.log_file, "w", encoding="utf-8")
            stdout = self._fh
            stderr = subprocess.STDOUT
        logger.info("start [%s]: %s (cwd=%s)", self.name, " ".join(self.args), self.cwd)
        self.proc = subprocess.Popen(
            self.args,
            cwd=str(self.cwd),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,  # отдельная process group
        )

    def poll(self) -> Optional[int]:
        return self.proc.poll() if self.proc else None

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if not self.proc:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self, grace: float = 5.0) -> None:
        if not self.proc:
            return
        if self.proc.poll() is not None:
            self._close_log()
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            pgid = None
        logger.info("stop [%s] pid=%s pgid=%s", self.name, self.proc.pid, pgid)
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + grace
        while time.time() < deadline and self.proc.poll() is None:
            time.sleep(0.2)
        if self.proc.poll() is None and pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._close_log()

    def _close_log(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


# ---------------------------------------------------------------------------
# Запуск Kernel-а
# ---------------------------------------------------------------------------

def _wait_port_open(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def _wait_kernel_shutdown(kernel_log_dir: Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    target_file = kernel_log_dir / "kernel.log"
    needle = "Kernel has shut down"
    while time.time() < deadline:
        if target_file.exists():
            try:
                data = target_file.read_text(encoding="utf-8", errors="replace")
                if needle in data:
                    return True
            except OSError:
                pass
        time.sleep(2.0)
    return False


# ---------------------------------------------------------------------------
# Один прогон
# ---------------------------------------------------------------------------

@dataclass
class RunPaths:
    run_dir: Path
    kernel_log_copy: Path
    agents_log_copy: Path
    agent_cfg_copy: Path


def _prepare_run_dir(root: Path, run_name: str) -> RunPaths:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        kernel_log_copy=run_dir / "kernel_logs",
        agents_log_copy=run_dir / "agents_logs",
        agent_cfg_copy=run_dir / "agent.cfg",
    )


def _purge_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for item in path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError:
            pass


_AGENT_ARTEFACT_PATTERNS = (
    "summary_*.json",
    "tick_*.csv",
    "events_*.csv",
    "agent_debug_*.log",
    "time.md",
    "runs.md",
)


def _purge_agent_artefacts(path: Path) -> None:
    # Точечно удаляет per-run файлы агентов в log_dir, не трогая подкаталоги
    # (в частности experiments/, где лежат результаты предыдущих прогонов).
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for pat in _AGENT_ARTEFACT_PATTERNS:
        for f in path.glob(pat):
            try:
                if f.is_file():
                    f.unlink()
            except OSError:
                pass


def _copy_tree(src: Path, dst: Path, exclude: Optional[Path] = None) -> None:
    if not src.exists():
        return
    src = src.resolve()
    dst = dst.resolve()
    # Защита от рекурсии: dst не должен лежать внутри src.
    try:
        dst.relative_to(src)
        # dst внутри src — копируем всё, кроме exclude и самого dst.
        if exclude is None:
            exclude = dst
    except ValueError:
        pass

    excl_resolved: Optional[Path] = None
    if exclude is not None:
        try:
            excl_resolved = exclude.resolve()
        except OSError:
            excl_resolved = exclude

    def _ignore(directory: str, names: list[str]) -> list[str]:
        if excl_resolved is None:
            return []
        result: list[str] = []
        for n in names:
            p = Path(directory) / n
            try:
                p_res = p.resolve()
            except OSError:
                continue
            if p_res == excl_resolved:
                result.append(n)
            else:
                try:
                    p_res.relative_to(excl_resolved)
                    result.append(n)
                except ValueError:
                    pass
        return result

    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_ignore)
    except OSError as exc:
        logger.warning("copy %s → %s failed: %s", src, dst, exc)


def _kill_stale_agents() -> None:
    # Между прогонами подстраховываемся: убиваем все python-процессы запуска
    # агента, чтобы они не подцеплялись к новому ядру.
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "main.py --agent-type"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


class ExperimentRunner:

    def __init__(self, env: EnvConfig, output_root: Path, timeout_minutes: float = 20.0) -> None:
        self.env = env
        self.output_root = output_root
        self.timeout_seconds = timeout_minutes * 60.0
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, spec: RunSpec) -> RunMetrics:
        logger.info("=== RUN %s (%s) ===", spec.name, spec.description)
        paths = _prepare_run_dir(self.output_root, spec.name)

        # 1. Подготовить agent.cfg под конфигурацию прогона.
        write_agent_cfg(self.env, spec)
        shutil.copy(self.env.agent_cfg_path, paths.agent_cfg_copy)

        # 2. Очистить логи предыдущего прогона.
        # В kernel_log_dir смело удаляем всё — это отдельный каталог ядра.
        # Для agents_log_dir точечно убираем только per-run файлы агентов,
        # чтобы не снести /log/experiments/<previous-run>/ с уже собранными артефактами.
        _purge_dir(self.env.kernel_log_dir)
        _purge_agent_artefacts(self.env.agents_log_dir)
        _kill_stale_agents()

        t0 = time.time()
        # ВАЖНО: start.sh делает self-kill через ps-grep по имени rcrs-server,
        # поэтому пути передаём относительные — чтобы "/rcrs-server" не попадал
        # в argv самого start.sh (иначе kill.sh убивает его сразу после запуска).
        try:
            rel_map = self.env.map_dir.relative_to(self.env.kernel_scripts_dir.parent).as_posix()
            rel_config = self.env.config_dir.relative_to(self.env.kernel_scripts_dir.parent).as_posix()
            map_arg = f"../{rel_map}"
            config_arg = f"../{rel_config}"
        except ValueError:
            map_arg = str(self.env.map_dir)
            config_arg = str(self.env.config_dir)
        kernel = ProcessGroup(
            name="kernel",
            args=[
                "bash",
                "start.sh",
                "-m", map_arg,
                "-c", config_arg,
                "-g",  # без GUI
            ],
            cwd=self.env.kernel_scripts_dir,
            log_file=paths.run_dir / "kernel_stdout.log",
        )
        agents = ProcessGroup(
            name="agents",
            args=[
                "bash",
                str(self.env.launch_agents_sh),
                self.env.kernel_host,
                str(self.env.kernel_port),
            ],
            cwd=self.env.module_dir,
            log_file=paths.run_dir / "agents_stdout.log",
        )

        error: Optional[str] = None
        try:
            kernel.start()
            # Ждём, пока ядро начнёт слушать на порту.
            if not _wait_port_open(self.env.kernel_host, self.env.kernel_port, timeout=90.0):
                raise RuntimeError("kernel port not open within 90s")

            # Запускаем агентов.
            agents.start()
            time.sleep(3.0)

            # Ждём либо "Kernel has shut down", либо завершение kernel-процесса.
            deadline = t0 + self.timeout_seconds
            shut_down = False
            while time.time() < deadline:
                if kernel.poll() is not None:
                    break
                if _wait_kernel_shutdown(self.env.kernel_log_dir, timeout=5.0):
                    shut_down = True
                    break
            if not shut_down and kernel.poll() is None:
                error = f"timeout after {self.timeout_seconds:.0f}s"
                logger.warning("[%s] %s", spec.name, error)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] run failed: %s", spec.name, exc)
        finally:
            agents.stop(grace=5.0)
            kernel.stop(grace=10.0)
            _kill_stale_agents()

        wall = time.time() - t0

        # Ждём дописи финальных summary_*.json — агенты его пишут в finally.
        time.sleep(2.0)

        # 3. Копируем логи прогона рядом с его папкой.
        _copy_tree(self.env.kernel_log_dir, paths.kernel_log_copy)
        _copy_tree(self.env.agents_log_dir, paths.agents_log_copy)

        # 4. Парсим score и метрики агентов.
        kernel_outcome: KernelOutcome = parse_kernel_logs(paths.kernel_log_copy)
        agents_outcome: AgentsOutcome = collect_agents_outcome(paths.agents_log_copy)

        metrics = RunMetrics(
            kernel=kernel_outcome,
            agents=agents_outcome,
            wall_clock_s=wall,
            error=error,
        )

        # 5. Пишем run_metrics.json рядом.
        try:
            import json
            (paths.run_dir / "run_metrics.json").write_text(
                json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("write run_metrics.json failed: %s", exc)

        logger.info(
            "=== DONE %s: score=%s, wall=%.1fs, error=%s ===",
            spec.name, kernel_outcome.score, wall, error,
        )
        return metrics
