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

from .util import configure_logging, run_cmd, run_with_tty


def setup_logger(level: str, log_file: Optional[str]) -> None:
    configure_logging(level, log_file)

    from yang.util.ext import configure_cpp_logging

    configure_cpp_logging("[%C-%m-%d %T.%e] [%P:%t] [%s:%#] %^[%l]%$ %v", level)


@click.command()
@click.option("--cfg", required=True, multiple=True, help="config files")
@click.option(
    "--run-dir", default="tmp/run", help="directory to store runtime temp files", show_default=True
)
@click.option(
    "--cpp-runner",
    default=None,
    help="path to cpp runner",
    show_default=True,
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
@click.option("--ycfgrc", default=".ycfgrc", help="default ycfg config file", show_default=True)
@click.option("--post", is_flag=True, help="Run post processing only", show_default=True)
@click.option("--debug", is_flag=True, help="Run debug cpp binary", show_default=True)
@click.option("--live", is_flag=True, help="live mode", show_default=True)
@click.option("--prod", is_flag=True, help="production mode", show_default=True)
@click.option(
    "--always-run", multiple=True, type=str, help="set always_run_modules, support wildcard *"
)
def main(
    cfg: str,
    run_dir: str,
    cpp_runner: str,
    log_dir: str,
    log_level: str,
    stage: str,
    num_threads: int,
    module: tuple[str],
    start_date: int,
    end_date: int,
    ycfgrc: str,
    post: bool,
    debug: bool,
    live: bool,
    prod: bool,
    always_run: tuple[str],
) -> None:
    from yang.cli.cfg import CfgContext
    from yang.cli.scheduler import Scheduler

    cpp_runner = os.path.realpath("bin/runner")
    if not os.path.exists(cpp_runner):
        raise RuntimeError("Missing cpp runner")
    if "bazel-" in cpp_runner:
        cpp_runner_path = os.path.realpath(cpp_runner)
        if "fastbuild/bin" in cpp_runner_path and not debug:
            raise RuntimeError("Using debug build cpp runner without --debug option")
        if "opt/bin" in cpp_runner_path and debug:
            raise RuntimeError("Using opt build cpp runner with --debug option")

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
    if os.path.exists(ycfgrc):
        logging.info(f"Using ycfgrc file: {ycfgrc}")
        cfg_ctx.use(ycfgrc)
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

    def run_cpp(cfg: dict, cfg_path: str, post: bool) -> None:
        cpp_cmd = [
            cpp_runner,
            "--config",
            cfg_path,
            "--log_level",
            log_level,
            "--stage",
            stage,
            "--num_threads",
            num_threads if not post else 1,
        ]
        if live:
            cpp_cmd.append("--live")
        if prod:
            cpp_cmd.append("--prod")
        if log_dir:
            log_file = f"{log_dir}/cpp.log"
            if post:
                log_file += "_post"
            cpp_cmd += ["--log_file", log_file]
        cpp_cmd = list(map(str, cpp_cmd))
        logging.info(f"Running cpp runner: {' '.join(cpp_cmd)}")
        if sys.stdout.isatty():
            cpp_process = run_with_tty(cpp_cmd)
        else:
            cpp_process = run_cmd(cpp_cmd)
        if cpp_process.returncode != 0:
            raise RuntimeError(
                f"cpp runner failed with exit code: {cpp_process.returncode}, "
                + f"cmd: {' '.join(cpp_cmd)}"
            )

    def run_py(cfg: dict, cfg_path: str, post: bool) -> None:
        from yang.sim import Runner
        from yang.sim.env import RunStage

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
            cfg,
            options,
            num_threads if not post else 1,
            config_path=cfg_path,
        )

    cfg_path = f"{current_run_dir}/cfg.yml"
    base_cfg_path = f"{current_run_dir}/cfg.base.yml"

    def run_modules(mods: list[str], post: bool = False) -> None:
        if len(mods) == 0:
            return
        scheduler = Scheduler(gen_cfg, mods)

        run_funcs = {"cpp": run_cpp, "py": run_py}
        while not scheduler.finished:
            for lang in ("cpp", "py"):
                num_pending = scheduler.num_pending
                if num_pending == 0:
                    continue
                ready_mods = scheduler.pop_ready_modules(lang)
                if not ready_mods:
                    continue
                logging.info(f"Running next: {len(ready_mods)} / {num_pending} ({lang})")
                shutil.copy(base_cfg_path, cfg_path)
                with open(cfg_path, "a") as f:
                    yaml.dump({"run_modules": ready_mods}, f, Dumper=yaml.CDumper)
                logging.info(f"Generated config file {cfg_path}")
                gen_cfg["run_modules"] = ready_mods
                run_funcs[lang](gen_cfg, cfg_path, post)

    skip_mods = set(gen_cfg.get("skip_modules", []))
    mods_to_run = []
    post_mods_to_run = []
    for mod in modules:
        if len(module) > 0 and mod["name"] not in module:
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
