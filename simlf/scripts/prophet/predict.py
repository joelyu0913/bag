#!/usr/bin/env python3

import os

if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import logging

import click
import numpy as np
import torch
import yaml
from prophet.util import import_attr
from prophet.yang import (
    DateTimeIndex,
    collect_valid_indices,
    collect_valid_indices_grf,
    get_feature_paths,
    load_blocks_grf,
    load_features,
    load_features_grf,
    load_univ,
    predict_ml,
    predict_nn,
)

from cli.util import configure_logging
from data import Array
from sim import Env


def run(
    workdir: str,
    config: str = "",
    start_datetime: int = None,
    end_datetime: int = None,
    data_dir: str = "",
    sys_data_dir: str = "",
    user_data_dir: str = "",
    output: str = "",
    update_output: bool = True,
    inds: str = "",
    no_gpu: bool = False,
    grf: bool = False,
    univ_mask: str = "",
):
    logging.info("workdir: %s", workdir)
    logging.info("config: %s", config)

    config_path = config
    inds_path = inds
    if inds_path == "":
        inds_path = os.path.join(workdir, "inds.yml")

    if config_path != "":
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    with open(os.path.join(workdir, "config.yml")) as f:
        train_config = yaml.safe_load(f)
    data_config = train_config["data"]
    model_config = train_config["model"]
    seq_len = model_config.get("seq_len", 1)
    logging.info("seq_len: %s", seq_len)

    env = Env(
        {
            "user_cache": user_data_dir or data_dir,
            "sys_cache": sys_data_dir or data_dir,
        }
    )
    dates = DateTimeIndex(env.dates, env.daily)

    if start_datetime is None:
        start_di = 0
        start_date = dates[0]
    else:
        start_di = dates.lower_bound(start_datetime)
        start_date = start_datetime
    if end_datetime is None:
        end_di = len(dates)
        end_date = dates[-1]
    else:
        end_di = dates.upper_bound(end_datetime)
        end_date = end_datetime
    if not env.daily:
        start_date = start_date // 10000
        end_date = end_date // 10000
    preprocess_config = train_config["preprocess"]
    preprocess_lookback = preprocess_config.get("lookback", 0)
    logging.info("preprocess_lookback: %s", preprocess_lookback)
    logging.info("di: %s, start_date: %s, end_date: %s", (start_di, end_di), start_date, end_date)

    with open(inds_path) as f:
        feature_yml = yaml.safe_load(f)
        if isinstance(feature_yml[0], dict):
            feature_yml = [feature["fi"] for feature in feature_yml]

    features = get_feature_paths(feature_yml)
    logging.info("features: %s", len(features))

    # Preprocess
    preprocess_config = train_config["preprocess"]

    model_type = model_config.pop("type")
    if "prophet.nn" in model_type:
        model_config["input_size"] = len(features)
    model_cls = import_attr(model_type)

    all_roll_dirs = []
    for d in os.listdir(workdir):
        if not d.startswith("roll_"):
            continue
        _, pred_start_date, pred_end_date = d.split("_")
        pred_start_date = int(pred_start_date)
        pred_end_date = int(pred_end_date)
        all_roll_dirs.append((pred_start_date, pred_end_date, os.path.join(workdir, d)))
    assert len(all_roll_dirs) > 0
    all_roll_dirs = sorted(all_roll_dirs)

    roll_dirs = []
    for roll_dir in all_roll_dirs:
        pred_start_date, pred_end_date, _ = roll_dir
        if pred_end_date < start_date or pred_start_date > end_date:
            continue
        roll_dirs.append(roll_dir)

    if len(roll_dirs) == 0 or roll_dirs[-1][1] < end_date:
        logging.error(
            f"missing roll dir for latest dates, using last roll dir {all_roll_dirs[-1][2]}"
        )
        _, last_end_date, last_roll_dir = all_roll_dirs[-1]
        roll_dirs.append((max(start_date, last_end_date + 1), end_date, last_roll_dir))
    logging.info("roll_dirs: %d", len(roll_dirs))

    sigs = []
    sig_start_di = None
    sig_end_di = None
    feature_paths = [env.cache_dir.get_read_path(p) for p in features]
    pred_batch_size = 90
    if preprocess_lookback == 0:
        pred_batch_size = 30
    if env.univ_size < 1000:
        pred_batch_size = int(1000 / env.univ_size * pred_batch_size)
    for i in range(len(roll_dirs)):
        pred_start_date, pred_end_date, roll_workdir = roll_dirs[i]
        pred_start_date = max(start_date, pred_start_date)
        pred_end_date = min(pred_end_date, end_date)
        if pred_end_date < pred_start_date:
            break
        if i == len(roll_dirs) - 1:
            pred_end_date = end_date
        roll_start_di = dates.lower_bound_date(pred_start_date)
        roll_end_di = dates.upper_bound_date(pred_end_date)
        logging.info("----roll----")
        logging.info("roll_workdir: %s", roll_workdir)
        logging.info("pred_start_date: %s", pred_start_date)
        logging.info("pred_end_date: %s", pred_end_date)
        logging.info("roll_start_di: %s", roll_start_di)
        logging.info("roll_end_di: %s", roll_end_di)
        for pred_start_di in range(roll_start_di, roll_end_di, pred_batch_size):
            pred_end_di = min(pred_start_di + pred_batch_size, roll_end_di)
            logging.info("pred_start_di: %s", pred_start_di)
            logging.info("pred_end_di: %s", pred_end_di)

            pred_data_start_di = max(0, pred_start_di - seq_len + 1 - preprocess_lookback)
            logging.info("pred_data_start_di: %s", pred_data_start_di)

            if grf:
                assert seq_len == 1
                assert preprocess_lookback == 0
                # Load data
                X, X_indices, blocks = load_features_grf(
                    feature_paths,
                    start_di=pred_start_di,
                    end_di=pred_end_di,
                )
                logging.info("X: %s", X.shape)
                logging.info("X_indices: %s", X_indices.shape)
                logging.info("blocks: %s (last: %d)", blocks.shape, blocks[-1])

                # Compute valid indices [(si, ti)]
                indices = collect_valid_indices_grf(
                    X,
                    None,
                    blocks,
                    0,
                    pred_end_di - pred_start_di,
                    train_config["data"]["min_valid_ratio"],
                )

                from prophet.preprocess_grf import Pipeline as PipelineGrf

                X_pipeline = PipelineGrf(preprocess_config.get("features", []))
                logging.info("X pipeline: %s", " ".join([t for t, _ in X_pipeline.processors]))
                X_pipeline.load_state_dict(torch.load(os.path.join(roll_workdir, "X_pipeline.pkl")))
                X = X_pipeline.transform(X, blocks)
            else:
                univ_mask = univ_mask or data_config.get("univ")
                filter_univ = load_univ(
                    env.cache_dir,
                    univ_mask,
                    pred_data_start_di,
                    pred_end_di,
                    len(env.univ),
                )

                # Load data
                X = load_features(
                    feature_paths,
                    start_di=pred_data_start_di,
                    end_di=pred_end_di,
                    univ_size=len(env.univ),
                    shape_hint=(-1, env.max_univ_size),
                    dtype_hint=np.float32,
                    filter_mask=filter_univ,
                )
                logging.info("X: %s", X.shape)

                # Compute valid indices [(si, ti)]
                indices = collect_valid_indices(
                    X,
                    None,
                    pred_start_di - pred_data_start_di,
                    pred_end_di - pred_data_start_di,
                    seq_len,
                    train_config["data"]["min_valid_ratio"],
                )

                from prophet.preprocess import Pipeline

                X_pipeline = Pipeline(
                    preprocess_config.get("features", []), preprocess_config["axis"]
                )
                logging.info("X pipeline: %s", " ".join([t for t, _ in X_pipeline.processors]))
                X_pipeline.load_state_dict(torch.load(os.path.join(roll_workdir, "X_pipeline.pkl")))
                X = X_pipeline.transform(X)

                Y_pipeline = Pipeline(
                    preprocess_config.get("target", []), preprocess_config["axis"]
                )
                logging.info("Y pipeline: %s", " ".join([t for t, _ in Y_pipeline.processors]))
                Y_pipeline.load_state_dict(torch.load(os.path.join(roll_workdir, "Y_pipeline.pkl")))

            if indices is None:
                logging.error("No valid indices")
            else:
                logging.info("indices: %s", indices.shape)
                gc.collect()

            predict_kwargs = {}
            if model_type == "prophet.ml.LGBM":
                predict_kwargs["num_threads"] = config.get("num_threads", 1)

            if grf:
                sig = np.full(len(X_indices), np.nan)
                if indices is not None:
                    model = model_cls.load(os.path.join(roll_workdir, "model.pkl"))
                    Y = np.full(len(X), np.nan)
                    X = X[indices]
                    gc.collect()
                    Y[indices] = predict_ml(model, X, None, **predict_kwargs)[:, 0]
                    sig[X_indices] = Y
                sigs.append(sig)
            else:
                if indices is None:
                    Y = np.full((X.shape[0], X.shape[1], 1), np.nan)
                elif "prophet.nn" in model_type:
                    model = model_cls(model_config)
                    torch.set_num_threads(config.get("num_threads", 1))
                    if not no_gpu and torch.cuda.is_available():
                        gpu = config.get("gpu", 0)
                        device = f"cuda:{gpu}"
                    else:
                        device = "cpu"
                    logging.info("device: %s", device)
                    # logging.info("model: \n%s", model)
                    model.eval()
                    model_path = os.path.join(roll_workdir, "data/model_best.pkl")
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model = model.to(device)

                    Y = predict_nn(
                        model,
                        X,
                        indices,
                        seq_len,
                        device=device,
                        batch_size=config.get("batch_size", 512),
                        data_workers=config.get("data_workers", 4),
                    )
                else:
                    model = model_cls.load(os.path.join(roll_workdir, "model.pkl"))
                    Y = predict_ml(model, X, indices, **predict_kwargs)
                Y = Y_pipeline.inverse_transform(Y)
                sigs.append(Y[:, pred_start_di - pred_data_start_di :, 0].transpose((1, 0)))

            if sig_start_di is None:
                sig_start_di = pred_start_di
            sig_end_di = pred_end_di
            gc.collect()

    if sig_start_di is None:
        return

    if output == "":
        output_path = os.path.join(workdir, "sig_pred")
    else:
        output_path = output
    logging.info("writing to %s", output_path)

    logging.info("sig_start_di: %s", sig_start_di)
    logging.info("sig_end_di: %s", sig_end_di)

    sig = np.concatenate(sigs)
    if grf:
        blocks = load_blocks_grf(feature_paths[0])[1:]
        start_off = 0 if sig_start_di == 0 else blocks[sig_start_di - 1]
        if update_output:
            output_blocks = Array.mmap(
                output_path + ".blocks",
                writable=True,
                shape=blocks.shape,
                dtype=np.uint64,
                null_value=0,
            )
            output_blocks[sig_start_di:sig_end_di] = blocks[sig_start_di:sig_end_di]

            # block size =0 will fail
            output_array = Array.mmap(
                output_path,
                writable=True,
                shape=(int(blocks[-1]),),
                dtype=np.float32,
            )
            output_array.data[start_off : blocks[sig_end_di - 1]] = sig
        else:
            Array(blocks).save(output_path + ".blocks")
            sig = np.concatenate((np.full((start_off,), np.nan), sig))
            Array(sig.astype(np.float32)).save(output_path)
    else:
        univ_size = len(env.univ)
        if update_output:
            output_array = Array.mmap(
                output_path,
                writable=True,
                shape=(env.max_dates_size, env.max_univ_size),
                dtype=np.float32,
            )
            output_array.data[sig_start_di:sig_end_di, :univ_size] = sig
            output_array.data[sig_start_di:sig_end_di, univ_size:].fill(np.nan)
        else:
            sig_out = np.full((env.dates_size, env.max_univ_size), np.nan, dtype=np.float32)
            sig_out[sig_start_di:sig_end_di, :univ_size] = sig
            Array(sig_out).save(output_path)
    logging.info("Done")


@click.command()
@click.option("--workdir", required=True)
@click.option("--config", default="")
@click.option("--start-date", type=int, default=None)
@click.option("--end-date", type=int, default=None)
@click.option("--start-datetime", type=int, default=None)
@click.option("--end-datetime", type=int, default=None)
@click.option("--data-dir", default="")
@click.option("--user-data-dir", default="")
@click.option("--sys-data-dir", default="")
@click.option("--output", type=str, default="")
@click.option("--update-output", is_flag=True)
@click.option("--inds", default="")
@click.option("--no-gpu", is_flag=True)
@click.option("--grf", is_flag=True)
@click.option("--univ", default="")
def main(
    workdir: str,
    config: str,
    start_date: int,
    end_date: int,
    start_datetime: int,
    end_datetime: int,
    data_dir: str,
    user_data_dir: str,
    sys_data_dir: str,
    output: str,
    update_output: bool,
    inds: str,
    no_gpu: bool,
    grf: bool,
    univ: str,
):
    configure_logging("info")
    run(
        workdir=workdir,
        config=config,
        start_datetime=start_datetime or start_date,
        end_datetime=end_datetime or end_date,
        data_dir=data_dir,
        user_data_dir=user_data_dir,
        sys_data_dir=sys_data_dir,
        output=output,
        update_output=update_output,
        inds=inds,
        no_gpu=no_gpu,
        grf=grf,
        univ_mask=univ,
    )


if __name__ == "__main__":
    main()
