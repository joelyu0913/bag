import asyncio
import importlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from multiprocessing import Pipe, Process
from typing import Any

import numpy as np

from sim.env import Env, RunStage


def import_attr(full_name) -> None:
    mod, attr = full_name.rsplit(".", 1)
    mod = importlib.import_module(mod)
    return getattr(mod, attr)


CONN_TIMEOUT_SECS = 30
HEARTBEAT_SECS = 1
TASK_POLL_SECS = 0.01


class WorkerError(Exception):
    pass


def run_worker_process(
    worker_name: str,
    conn,
    config: dict[str, Any],
    run_options: dict[str, Any],
    mod_deps: dict[str, list[str]],
) -> None:

    env = Env(config)
    env.live = run_options.get("live", False)
    env.prod = run_options.get("prod", False)
    logging.info(f"user_mode: {env.user_mode}")
    logging.info(f"sys_cache: {env.cache_dir.sys_dir}")
    logging.info(f"user_cache: {env.cache_dir.user_dir}")
    logging.info(f"daily: {env.daily}")
    logging.info(f"live: {env.live}")
    logging.info(f"prod: {env.prod}")
    logging.info(f"univ_start_datetime: {env.univ_start_datetime}")
    logging.info(f"univ_end_datetime: {env.univ_end_datetime}")
    logging.info(f"sim_start_datetime: {env.sim_start_datetime}")
    logging.info(f"sim_end_datetime: {env.sim_end_datetime}")
    mod_configs = {}
    for mod_config in config["modules"]:
        mod_configs[mod_config["name"]] = mod_config

    always_run_mods = set(config.get("always_run_modules", []))

    def run_mod(mod_name: str) -> None:
        # use_rerun_manager = env.rerun_manager is not None 
        use_rerun_manager = env.rerun_manager is not None and not run_options["post"]
        try:
            mod_config = mod_configs[mod_name]
            # check sys/user mod
            if env.user_mode and mod_config.get("sys", False):
                return

            stages = mod_config.get("stages", ["intraday"])
            stages_to_run = []
            for s in stages:
                stage_val = RunStage.parse(s)
                if stage_val in run_options["stages"]:
                    stages_to_run.append(stage_val)
            if len(stages_to_run) == 0:
                return

            if use_rerun_manager:
                if mod_name not in always_run_mods and env.rerun_manager.can_skip_run(
                    mod_name, mod_deps[mod_name]
                ):
                    logging.debug(f"Skip module {mod_name}: already built")
                    return

            logging.debug(f"Running module {mod_name} on {worker_name}")
            logging.debug(f"Running module {mod_name} on {worker_name}")
            logging.debug(f"Making module {mod_name} on {worker_name}")
            mod_cls = import_attr(mod_config["class"])
            mod = mod_cls(mod_name, mod_config, env)
            if use_rerun_manager:
                env.rerun_manager.record_before_run(mod_name)

            for stage in stages_to_run:
                logging.info(f"Running module {mod_name} - {stage} on {worker_name}")
                mod.stage = stage
                with np.errstate(divide="ignore", invalid="ignore"):
                    mod.run()
            logging.info(f"Finished module {mod_name} on {worker_name}")

            if use_rerun_manager:
                env.rerun_manager.record_run(mod_name)
        except Exception:
            logging.exception(f"Failed to run {mod_name} on {worker_name}")
            raise

    logging.debug(f"Worker process {worker_name} started")
    conn_timeout_tm = datetime.now().timestamp() + CONN_TIMEOUT_SECS
    next_heartbeat_tm = None
    fut = None
    mod_name = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            busy = False
            last_poll_tm = datetime.now().timestamp()
            if conn.poll():
                msg = conn.recv()
                if msg[0] == "run":
                    assert fut is None
                    mod_name = msg[1]
                    fut = executor.submit(run_mod, mod_name)
                elif msg[0] == "stop":
                    break
                elif msg[0] == "hb":
                    pass
                else:
                    logging.warning(f"Unknown message: {msg}, worker: {worker_name}")
                conn_timeout_tm = datetime.now().timestamp() + CONN_TIMEOUT_SECS
                busy = True
            if fut is not None:
                done = True
                try:
                    fut.result(timeout=TASK_POLL_SECS)
                    conn.send(("done", mod_name))
                except TimeoutError:
                    done = False
                except Exception as e:
                    conn.send(("error", mod_name, str(e)))
                if done:
                    fut = None
                    mod_name = None
                    busy = True
            now_tm = datetime.now().timestamp()
            if next_heartbeat_tm is None or now_tm > next_heartbeat_tm:
                conn.send(("hb",))
                next_heartbeat_tm = now_tm + HEARTBEAT_SECS
            if last_poll_tm > conn_timeout_tm:
                logging.error(f"Worker parent process not responsive, worker: {worker_name}")
                if fut:
                    fut.cancel()
                break
            if not busy:
                time.sleep(TASK_POLL_SECS)
    logging.debug(f"Worker child process {worker_name} exiting")


class SubprocessWorker(object):
    def __init__(self, name: str, loop: asyncio.BaseEventLoop):
        self.name = name
        self.loop = loop
        self.run_fut = None
        self.active_mod = None
        self.proc = None
        self.conn = None
        self.stopped = False
        self.error = False

    @property
    def busy(self) -> bool:
        return self.run_fut is not None

    def start(
        self,
        config: dict[str, Any],
        run_options: dict[str, Any],
        mod_deps: dict[str, list[str]],
    ) -> None:
        conn1, conn2 = Pipe()
        self.conn = conn1
        self.proc = Process(
            target=run_worker_process,
            args=(self.name, conn2, config, run_options, mod_deps),
            name=self.name,
        )
        self.proc.start()
        self.loop.create_task(self._send_heartbeat())

    def poll(self) -> bool:
        try:
            if self.conn.poll():
                msg = self.conn.recv()
                if msg[0] == "done":
                    if self.run_fut:
                        self.run_fut.set_result(self.active_mod)
                elif msg[0] == "error":
                    if self.run_fut:
                        self.run_fut.set_exception(WorkerError(msg[2]))
                elif msg[0] == "hb":
                    pass
                else:
                    logging.warning(f"Unknown message: {msg}, worker: {self.name}")
                return True
            elif not self.proc.is_alive:
                self.on_error(f"Worker child process exited, worker: {self.name}")
        except Exception:
            logging.exception(f"Worker {self.name} failed")
            raise
        return False

    def run_module(self, mod: str) -> asyncio.Future:
        assert self.run_fut is None
        assert not self.error
        self.conn.send(("run", mod))
        self.active_mod = mod
        self.run_fut = self.loop.create_future()
        self.run_fut.add_done_callback(self._on_run_done)
        return self.run_fut

    def _on_run_done(self, fut: asyncio.Future) -> None:
        self.run_fut = None
        self.active_mod = None
        if fut.exception():
            self.error = True

    def stop(self, kill=False) -> None:
        if not kill:
            assert self.run_fut is None
        logging.debug(f"Stopping worker {self.name} (kill: {kill})")
        self.stopped = True
        self.conn.send(("stop",))
        if kill:
            self.proc.kill()
        else:
            self.proc.terminate()

    async def _send_heartbeat(self) -> None:
        while not self.stopped:
            t_start = time.time()
            await asyncio.sleep(HEARTBEAT_SECS)
            if not self.stopped:
                t_delay = time.time() - t_start - HEARTBEAT_SECS
                if t_delay > 2:
                    logging.warning(f"worker {self.name} heartbeat delay: {t_delay:.2f}")
                self.conn.send(("hb",))

    def on_error(self, msg) -> None:
        logging.error(msg)
        self.error = True
        if self.run_fut is not None:
            self.run_fut.set_exception(WorkerError(msg))
