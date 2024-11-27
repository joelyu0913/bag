#!/usr/bin/env python3

import os

if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import fnmatch
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from typing import Optional

import click
import yaml

from cli.util import configure_logging, run_cmd, run_with_tty


def setup_logger(level: str, log_file: Optional[str]) -> None:
    configure_logging(level, log_file)


@click.command()
@click.option("--cfg", required=True, multiple=True, help="config files")
@click.option(
    "--run-dir", default="tmp/run", help="directory to store runtime temp files", show_default=True
)
@click.option("--log-dir", default="", help="directory to store log files, use 'no' to disable")
@click.option(
    "--log-level",
    "-l",
    default="info",
    type=click.Choice(["trace", "debug", "info", "warn", "error"]),
    help="logging level: trace, debug, info, warn, error",
    show_default=True,
)
@click.option(
    "--stage",
    default="all",
    type=click.Choice(["all", "open", "prepare", "intraday", "eod"]),
    help="Stage to run",
    show_default=True,
)
@click.option("--num-threads", "-t", default=1, help="number of threads to use", show_default=True)
@click.option("--module", "-m", multiple=True, type=str, help="modules to run")
@click.option("--start-date", default=None, type=int, help="set sim_start_date")
@click.option("--end-date", default=None, type=int, help="set sim_end_date")
@click.option("--post", is_flag=True, help="Run post processing only", show_default=True)
@click.option("--live", is_flag=True, help="live mode", show_default=True)
@click.option("--prod", is_flag=True, help="production mode", show_default=True)
@click.option(
    "--always-run", multiple=True, type=str, help="set always_run_modules, support wildcard *"
)
def main(
    cfg: str,
    run_dir: str,
    log_dir: str,
    log_level: str,
    stage: str,
    num_threads: int,
    module: tuple[str],
    start_date: int,
    end_date: int,
    post: bool,
    live: bool,
    prod: bool,
    always_run: tuple[str],
) -> None:
    from cli.cfg import CfgContext
    from cli.scheduler import Scheduler

    current_run_dir = os.path.join(run_dir, datetime.now().strftime("run.%Y%m%d_%H%M%S"))
    os.makedirs(current_run_dir, exist_ok=True)
    current_link = os.path.join(run_dir, "current")
    if os.path.exists and os.path.islink(current_link):
        os.remove(current_link)
    os.symlink(os.path.basename(current_run_dir), current_link)

    if log_dir == "no":
        log_dir = None
    elif log_dir == "":
        log_dir = current_run_dir
    if log_dir is None:
        setup_logger(log_level, None)
    else:
        os.makedirs(log_dir, exist_ok=True)
        setup_logger(log_level, f"{log_dir}/py.log")
        logging.info(f"Output to log dir {log_dir}")

    cfg_ctx = CfgContext(prod_mode=prod)
    for cfg_file in cfg:
        logging.info(f"Using cfg file: {cfg_file}")
        cfg_ctx.use(cfg_file)
    logging.info("Loaded cfg")
    gen_cfg = cfg_ctx.to_cfg()
    logging.info("Generated yaml cfg")
    if start_date is not None:
        gen_cfg["sim_start_date"] = start_date
    if end_date is not None:
        gen_cfg["sim_end_date"] = end_date

    modules = gen_cfg.get("modules", [])

    always_run_modules = []
    always_run_patterns = []
    for pattern in always_run:
        if "*" in pattern:
            always_run_patterns.append(re.compile(fnmatch.translate(pattern)))
        else:
            always_run_modules.append(pattern)
    
    if len(always_run_patterns) > 0:
        for m in modules:
            if any(pat.fullmatch(m["name"]) for pat in always_run_patterns):
                always_run_modules.append(m["name"])

    if len(always_run_modules) > 0:
        gen_cfg["always_run_modules"] = list(
            set(gen_cfg.get("always_run_modules", []) + always_run_modules)
        )

    cfg_path = f"{current_run_dir}/cfg.yml"
    base_cfg_path = f"{current_run_dir}/cfg.base.yml"

    def run_modules(mods: list[str], post: bool = False) -> None:
        from sim import Runner
        from sim.env import RunStage
        if len(mods) == 0:
            return
        scheduler = Scheduler(gen_cfg, mods)

        while not scheduler.finished:
            num_pending = scheduler.num_pending
            if num_pending == 0:
                continue
            ready_mods = scheduler.pop_ready_modules()
            if not ready_mods:
                continue
            logging.info(f"Running next: {len(ready_mods)} / {num_pending} ")
            shutil.copy(base_cfg_path, cfg_path)
            with open(cfg_path, "a") as f:
                yaml.dump({"run_modules": ready_mods}, f, Dumper=yaml.CDumper)
            logging.info(f"Generated config file {cfg_path}")
            gen_cfg["run_modules"] = ready_mods

            options = {
                "live": live,
                "prod": prod,
                "post": post,
            }
            if stage == "all":
                options["stages"] = [
                    RunStage.PREPARE,
                    RunStage.OPEN,
                    RunStage.INTRADAY,
                    RunStage.EOD,
                ]
            else:
                stage_val = RunStage.parse(stage)
                if stage_val == RunStage.ERROR:
                    raise RuntimeError(f"invalid stage {stage}")
                options["stages"] = [stage_val]
            runner = Runner()
            runner.run(
                gen_cfg,
                options,
                num_threads if not post else 1,
                # config_path=cfg_path,
            )

    skip_mods = set(gen_cfg.get("skip_modules", []))
    mods_to_run = []
    post_mods_to_run = []
    only_run_modules = module
    for mod in modules:
        if len(only_run_modules) > 0 and mod["name"] not in only_run_modules:
            continue
        if mod["name"] in skip_mods:
            continue
        if mod["post"]:
            post_mods_to_run.append(mod["name"])
        else:
            mods_to_run.append(mod["name"])
    if len(mods_to_run) == 0 and len(post_mods_to_run) == 0:
        logging.info("No modules to run")
        return

    with open(base_cfg_path, "w") as f:
        yaml.dump(gen_cfg, f, Dumper=yaml.CDumper)
    logging.info(f"Dumped {base_cfg_path}")
    if not post:
        run_modules(mods_to_run)
    run_modules(post_mods_to_run, True)


if __name__ == "__main__":
    main()
