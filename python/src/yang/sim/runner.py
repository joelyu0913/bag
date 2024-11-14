import asyncio
import functools
import logging
from collections import deque
from typing import Any

from yang.sim.scheduler import Scheduler
from yang.sim.worker import SubprocessWorker

IDLE_SLEEP_SECS = 0.002


class Runner(object):
    def run(
        self,
        config: dict,
        run_options: dict[str, Any],
        num_workers: int = 1,
        config_path: str = None,
    ) -> None:
        mods = []
        mod_deps = {}
        run_set = set(config.get("run_modules", []))
        for mod_config in config["modules"]:
            if mod_config["lang"] != "py":
                continue
            name = mod_config["name"]
            if len(run_set) > 0 and name not in run_set:
                continue

            mods.append(name)
            mod_deps[name] = list(set(mod_config.get("deps", []) + ["base"]))

        scheduler = Scheduler(mods, mod_deps)
        loop = asyncio.get_event_loop()
        num_workers = min(num_workers, len(mods))
        workers = [SubprocessWorker(f"worker_{i}", loop) for i in range(num_workers)]
        for i in range(num_workers):
            workers[i].start(config, run_options, mod_deps, config_path=config_path)

        idle_workers = deque(workers)
        logging.info(f"Running ({num_workers} threads)")

        def on_mod_done(worker, mod_name, fut):
            try:
                fut.result()
                scheduler.on_task_finished(mod_name)
                idle_workers.append(worker)
            except Exception as e:
                logging.exception("Failed to run %s", mod_name)
                scheduler.on_task_error(mod_name)
                worker.stop()

        async def run():
            while True:
                idle = True
                if len(idle_workers) > 0 and scheduler.has_ready_tasks:
                    mod_name = scheduler.next_ready_task()
                    worker = idle_workers.popleft()
                    fut = worker.run_module(mod_name)
                    fut.add_done_callback(functools.partial(on_mod_done, worker, mod_name))
                    idle = False

                for worker in workers:
                    if not worker.stopped and worker.poll():
                        idle = False

                if idle:
                    await asyncio.sleep(IDLE_SLEEP_SECS)
                if scheduler.error:
                    break
                if len(idle_workers) < len(workers):
                    continue
                if not scheduler.has_ready_tasks:
                    break

            while True:
                busy = False
                for worker in workers:
                    if worker.busy:
                        worker.poll()
                        busy = True
                    elif not worker.stopped:
                        worker.stop()
                if not busy:
                    break
                await asyncio.sleep(0.1)

        try:
            loop.run_until_complete(run())
        finally:
            for worker in workers:
                if not worker.stopped:
                    worker.stop(kill=True)

        if scheduler.error:
            raise RuntimeError(f"Runner failed, failed tasks: {scheduler.error}")
        if not scheduler.finished and not scheduler.error:
            logging.fatal(f"Found dependency cycle containing {scheduler.get_pending_tasks()}")
            raise RuntimeError("Runner failed")
