import os
import asyncio
import functools
import logging
from collections import deque
from typing import Any

from sim.scheduler import Scheduler
from sim.worker import SubprocessWorker
import yaml
import pandas as pd

IDLE_SLEEP_SECS = 0.002

def create_env(config):
    # check existing meta
    meta = {}
    meta["daily"] = config.get('daily', True)
    meta["max_univ_size"] = config['max_univ_size']
    meta["univ_start_datetime"] = config['univ_start_date'] if 'univ_start_date' in config else config['univ_start_datetime']
    meta["univ_end_datetime"] = config['univ_end_date'] if 'univ_end_date' in config else config['univ_end_datetime']
    meta["univ_indices"] = config.get('univ_indices', {})
    meta["univ_indices_id_start"] = config['univ_indices_id_start']
    meta["intraday_times"] = config.get('intraday_times', {})
    meta["taq_times"] =  config['taq']['times']
    meta["days_per_year"] = config['days_per_year']
    meta["short_book_size"] = config['short_book_size']
    meta["benchmark_index"] = config['benchmark_index']
    dir_env = f"{config['sys_cache']}/env/"
    os.makedirs(dir_env, exist_ok=True)
    yaml.safe_dump(meta, open(f"{dir_env}/meta.yml", 'w'), sort_keys=False)
    df = pd.read_csv(config['trade_dates'], header = None)
    df = df[(int(config['univ_start_date']) <= df[0]) & (df[0] <= int(config['univ_end_date']))]
    df.to_csv(f'{dir_env}/trade_dates', header=False, index=False)
    
    

#   env->set_live(options.live);
#   env->set_prod(options.prod);

#   env->Initialize(config);
#   if (!env->user_mode() && options.stages.count(RunStage::PREPARE)) {
#     env->Build();
#   } else {
#     env->Load();
#   }

class Runner(object):
    def run(
        self,
        config: dict,
        run_options: dict[str, Any],
        num_workers: int = 1,
    ) -> None:
        mods = []
        mod_deps = {}
        run_set = set(config.get("run_modules", []))
        for mod_config in config["modules"]:
            name = mod_config["name"]
            if len(run_set) > 0 and name not in run_set:
                continue

            mods.append(name)
            mod_deps[name] = list(set(mod_config.get("deps", []) + ["base"]))
        
        create_env(config)

        scheduler = Scheduler(mods, mod_deps)
        loop = asyncio.get_event_loop()
        num_workers = min(num_workers, len(mods))
        workers = [SubprocessWorker(f"worker_{i}", loop) for i in range(num_workers)]
        for i in range(num_workers):
            workers[i].start(config, run_options, mod_deps)
        return

        idle_workers = deque(workers)
        logging.info(f"Running ({num_workers} threads)")
        return

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
