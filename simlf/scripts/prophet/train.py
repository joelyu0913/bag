#!/usr/bin/env python3

import os

if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import logging
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
import yaml
from prophet.util import import_attr
from prophet.yang import (
    DateTimeIndex,
    IndexTsDataset,
    collect_valid_indices,
    collect_valid_indices_grf,
    flatten_indices,
    generate_roll_dates,
    generate_sample_weights,
    get_feature_paths,
    load_univ,
    load_xy,
    load_xy_grf,
)

from cli.util import configure_logging
from sim import Env


def train_nn(model, X, Y, seq_len, phase_indices, workdir, config, resume):
    import prophet.train

    logging.info("seq_len: %s", seq_len)
    logging.info("%s", model)

    # Train
    train_dataset = IndexTsDataset(phase_indices["train"], X, Y, seq_len, time_tag=True)
    valid_dataset = IndexTsDataset(phase_indices["valid"], X, Y, seq_len, time_tag=True)

    trainer = prophet.train.Trainer(
        config["train"], model, (train_dataset, valid_dataset), workdir=workdir
    )
    if resume:
        trainer.resume()
    trainer.train()
    trainer.eval()
    trainer.close()


def train_ml(model, X, Y, phase_indices, workdir, tsample_weights, grf=False):
    from prophet.metric import IC

    kwargs = {}
    if grf:
        indices = phase_indices["train"]
    else:
        indices = flatten_indices(phase_indices["train"], Y.shape[1])
        if tsample_weights is not None:
            kwargs["sample_weight"] = generate_sample_weights(
                phase_indices["train"], tsample_weights
            )
    X = X.reshape(-1, X.shape[-1])
    Y = Y.reshape(-1)
    X = X[indices]
    Y = Y[indices]
    model.fit(X, Y, **kwargs)
    model.save(os.path.join(workdir, "model.pkl"))
    pred = model.predict(X)
    logging.info("train cos: %s", IC(pred, Y))


def make_pipeline(preprocess_config):
    from prophet.preprocess import Pipeline

    X_pipeline = Pipeline(preprocess_config.get("features", []), preprocess_config["axis"])
    Y_pipeline = Pipeline(preprocess_config.get("target", []), preprocess_config["axis"])
    return X_pipeline, Y_pipeline


def make_pipeline_grf(preprocess_config):
    from prophet.preprocess_grf import Pipeline as PipelineGrf

    X_pipeline = PipelineGrf(preprocess_config.get("features", []))
    return X_pipeline


def prepare_data(
    phase_di, cache_dir, data_config, preprocess_config, features, seq_len, univ, univ_mask
):
    logging.info("prepare_data: %s", phase_di)
    preprocess_lookback = preprocess_config["lookback"]
    y_len = data_config.get("target_len", 1)
    logging.info("preprocess_lookback: %s", preprocess_lookback)
    logging.info("y_len: %s", y_len)
    data_start_di = max(phase_di["train"][0] - seq_len + 1 - preprocess_lookback - y_len + 1, 0)
    data_end_di = phase_di["valid"][1]

    univ_mask = univ_mask or data_config.get("univ")
    filter_univ = load_univ(cache_dir, univ_mask, data_start_di, data_end_di, len(univ))

    target = data_config["target"]
    # Load data
    X, Y = load_xy(
        cache_dir,
        x_paths=features,
        y_path=target,
        start_di=data_start_di,
        end_di=data_end_di,
        univ_size=len(univ),
        filter_mask=filter_univ,
    )

    logging.info("X: %s", X.shape)
    logging.info("Y: %s", Y.shape)
    assert X.shape[2] == len(features)
    assert X.shape[:2] == Y.shape[:2]

    valid_ratio = data_config["min_valid_ratio"]
    logging.info("collect indices, valid_ratio: %s", valid_ratio)
    # Compute valid indices [(si, ti)]
    indices = collect_valid_indices(
        X,
        Y,
        0,
        data_end_di - data_start_di,
        seq_len,
        valid_ratio=valid_ratio,
        check_y_seq=data_config.get("check_y_seq", False),
    )
    logging.info("indices: %d", len(indices))
    if indices is None:
        logging.error("no valid indices found")
        indices = []

    # Preprocess
    train_len = phase_di["train"][1] - data_start_di

    X_pipeline, Y_pipeline = make_pipeline(preprocess_config)

    logging.info("X pipeline: %s", [t for t, _ in X_pipeline.processors])
    X_pipeline.fit(X[:, :train_len, :])
    X = X_pipeline.transform(X)

    logging.info("Y pipeline: %s", [t for t, _ in Y_pipeline.processors])
    Y_pipeline.fit(Y[:, :train_len, :])
    Y = Y_pipeline.transform(Y)
    gc.collect()
    return dict(
        data_start_di=data_start_di,
        data_end_di=data_end_di,
        X=X,
        Y=Y,
        X_pipeline=X_pipeline,
        Y_pipeline=Y_pipeline,
        indices=indices,
    )


def prepare_data_grf(phase_di, cache_dir, data_config, preprocess_config, features):
    logging.info("prepare_data: %s", phase_di)
    y_len = data_config.get("target_len", 1)
    logging.info("y_len: %s", y_len)
    data_start_di = max(phase_di["train"][0] - y_len + 1, 0)
    data_end_di = phase_di["valid"][1]

    target = data_config["target"]
    # Load data
    X, Y, orig_mask, blocks = load_xy_grf(
        cache_dir,
        x_paths=features,
        y_path=target,
        start_di=data_start_di,
        end_di=data_end_di,
    )

    logging.info("X: %s", X.shape)
    logging.info("Y: %s", Y.shape)
    logging.info("orig_mask: %s", orig_mask.shape)
    logging.info("blocks: %s", blocks.shape)
    assert X.shape[1] == len(features)
    assert X.shape[:1] == Y.shape[:1]

    valid_ratio = data_config["min_valid_ratio"]
    logging.info("collect indices, valid_ratio: %s", valid_ratio)
    # Compute valid indices [(si, ti)]
    indices = collect_valid_indices_grf(
        X,
        Y,
        blocks,
        0,
        data_end_di - data_start_di,
        valid_ratio=valid_ratio,
    )
    logging.info("indices: %d", len(indices))

    # Preprocess
    train_len = blocks[phase_di["train"][1] - data_start_di]

    X_pipeline = make_pipeline_grf(preprocess_config)
    assert not X_pipeline.requires_fit()

    logging.info("X pipeline: %s", [t for t, _ in X_pipeline.processors])
    X = X_pipeline.transform(X, blocks)

    gc.collect()
    return dict(
        data_start_di=data_start_di,
        data_end_di=data_end_di,
        X=X,
        Y=Y,
        X_pipeline=X_pipeline,
        indices=indices,
        blocks=blocks,
    )


@click.command()
@click.option("--workdir", required=True)
@click.option("--config", required=True, multiple=True)
@click.option("--inds", required=True, multiple=True)
@click.option("--data-dir", default="")
@click.option("--user-data-dir", default="")
@click.option("--sys-data-dir", default="")
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--start-date", type=int, default=None)
@click.option("--end-date", type=int, default=None)
@click.option("--random-state", type=int, default=None)
@click.option("--grf", is_flag=True)
@click.option("--univ", default="")
def main(
    workdir: str,
    config: list[str],
    inds: list[str],
    data_dir: str,
    user_data_dir: str,
    sys_data_dir: str,
    resume: bool,
    start_date: int,
    end_date: int,
    random_state: int,
    grf: bool,
    univ: str,
):
    configure_logging("info")
    logging.info("workdir: %s", workdir)
    logging.info("config: %s", config)
    logging.info("inds: %s", inds)
    logging.info("univ: %s", univ)
    run(
        workdir=workdir,
        config=config,
        inds=inds,
        data_dir=data_dir,
        user_data_dir=user_data_dir,
        sys_data_dir=sys_data_dir,
        resume=resume,
        start_date=start_date,
        end_date=end_date,
        random_state=random_state,
        grf=grf,
        univ=univ,
    )


def run(
    workdir: str,
    config: list[str],
    inds: list[str],
    data_dir: str,
    user_data_dir: str,
    sys_data_dir: str,
    resume: bool,
    start_date: int,
    end_date: int,
    random_state: int,
    grf: bool,
    univ: str,
):
    config_paths = config
    inds_paths = inds

    # Load config
    config = {}
    for path in config_paths:
        with open(path) as f:
            config = {**config, **yaml.safe_load(f)}

    data_config = config["data"]
    model_config = config["model"]
    if random_state is not None:
        model_config["random_state"] = random_state

    # Generate rolling dates
    env = Env(
        {
            "user_cache": user_data_dir or data_dir,
            "sys_cache": sys_data_dir or data_dir,
        }
    )
    dates = DateTimeIndex(env.dates, env.daily)
    if start_date is None:
        start_date = dates.get_date(0)
    end_date = os.environ.get("END_DATE", end_date)
    if end_date is None:
        end_date = dates.get_date(-1)

    feature_yml = []
    for path in inds_paths:
        with open(path) as f:
            feature_yml += yaml.safe_load(f)

    if isinstance(feature_yml[0], dict):
        feature_yml = [feature["fi"] for feature in feature_yml]

    features = get_feature_paths(feature_yml)
    target = data_config["target"]
    logging.info("features: %s", len(features))
    logging.info("target: %s", target)

    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "config.yml"), "w") as f:
        yaml.safe_dump(config, f)
    with open(os.path.join(workdir, "inds.yml"), "w") as f:
        yaml.safe_dump(feature_yml, f)

    roll_dates = generate_roll_dates(start_date, end_date, data_config["roll_years"])
    if resume:
        all_roll_dates = roll_dates
        roll_dates = []
        for roll_info in all_roll_dates:
            roll_workdir = os.path.join(
                workdir, f"roll_{roll_info['pred'][0]}_{roll_info['pred'][1]}"
            )
            if resume and os.path.exists(os.path.join(roll_workdir, ".done")):
                logging.info("found .done, skip %s", roll_info)
                continue
            roll_dates.append(roll_info)
    if len(roll_dates) == 0:
        return

    seq_len = config["model"].get("seq_len", 1)
    preprocess_config = config["preprocess"]
    preprocess_lookback = preprocess_config.get("lookback", 0)
    logging.info("preprocess_lookback: %s", preprocess_lookback)
    if grf:
        prepare_data_func = lambda phase_di: prepare_data_grf(
            phase_di,
            env.cache_dir,
            data_config,
            preprocess_config,
            features,
        )
        can_preload_data = True
        assert seq_len == 1
        assert preprocess_lookback == 0
    else:
        prepare_data_func = lambda phase_di: prepare_data(
            phase_di,
            env.cache_dir,
            data_config,
            preprocess_config,
            features,
            seq_len,
            env.univ,
            univ,
        )
        X_pipeline, Y_pipeline = make_pipeline(preprocess_config)
        can_preload_data = not X_pipeline.requires_fit() and not Y_pipeline.requires_fit()

    if can_preload_data:
        start_di = dates.lower_bound_date(roll_dates[0]["train"][0])
        end_di = dates.upper_bound_date(roll_dates[-1]["valid"][1])
        preload_data = prepare_data_func({"train": (start_di, end_di), "valid": (end_di, end_di)})
    else:
        preload_data = None

    model_type = model_config.pop("type")
    if "prophet.nn" in model_type:
        model_config["input_size"] = len(features)
        if random_state is not None:
            torch.manual_seed(random_state)
    model_cls = import_attr(model_type)
    # Work on roll_dates[0]
    for ri in range(len(roll_dates)):
        logging.info("----roll----")
        logging.info(f"ri: {ri}, dates: {roll_dates[ri]}")
        roll_workdir = os.path.join(
            workdir, f"roll_{roll_dates[ri]['pred'][0]}_{roll_dates[ri]['pred'][1]}"
        )

        phase_di = {
            p: (dates.lower_bound_date(d[0]), dates.upper_bound_date(d[1]))
            for p, d in roll_dates[ri].items()
        }
        del phase_di["pred"]
        logging.info("di: %s", phase_di)
        y_len = data_config.get("target_len", 1)
        logging.info("y_len: %s", y_len)
        start_di = max(phase_di["train"][0] - seq_len + 1 - preprocess_lookback - y_len + 1, 0)
        end_di = phase_di["valid"][1] - y_len + 1

        if preload_data is None:
            all_data = prepare_data_func(phase_di)
        else:
            all_data = preload_data
        data_start_di = all_data["data_start_di"]
        data_end_di = all_data["data_end_di"]
        logging.info("start_di: %s", start_di)
        logging.info("end_di: %s", end_di)
        logging.info("data_start_di: %s", data_start_di)
        logging.info("data_end_di: %s", data_end_di)

        all_indices = all_data["indices"]
        phase_indices = {}
        if grf:
            blocks = all_data["blocks"]
            X = all_data["X"][blocks[start_di - data_start_di] : blocks[end_di - data_start_di]]
            Y = all_data["Y"][blocks[start_di - data_start_di] : blocks[end_di - data_start_di]]
            for phase, (di0, di1) in phase_di.items():
                indices = []
                di0_off = blocks[max(di0 - y_len + 1 - data_start_di, 0)]
                di1_off = blocks[max(di1 - y_len + 1 - data_start_di, 0)]
                for t in all_indices:
                    if di0_off <= t < di1_off:
                        indices.append(t - blocks[start_di - data_start_di])
                phase_indices[phase] = np.array(indices)
        else:
            X = all_data["X"][:, start_di - data_start_di : end_di - data_start_di]
            Y = all_data["Y"][:, start_di - data_start_di : end_di - data_start_di]
            for phase, (di0, di1) in phase_di.items():
                indices = []
                di0_off = max(di0 - y_len + 1 - data_start_di, 0)
                di1_off = max(di1 - y_len + 1 - data_start_di, 0)
                for s, t in all_indices:
                    if di0_off <= t < di1_off:
                        indices.append((s, t - (start_di - data_start_di)))
                phase_indices[phase] = np.array(indices)

        logging.info("X: %s", X.shape)
        logging.info("Y: %s", Y.shape)
        logging.info(
            "Indices: %s", [(p, i.shape if i is not None else i) for p, i in phase_indices.items()]
        )
        if len(phase_indices["train"]) == 0:
            continue
        os.makedirs(roll_workdir, exist_ok=True)
        with open(os.path.join(roll_workdir, "dates.yml"), "w") as f:
            yaml.safe_dump(roll_dates[ri], f)
        X_pipeline = all_data["X_pipeline"]
        Y_pipeline = all_data.get("Y_pipeline")
        torch.save(X_pipeline.state_dict(), os.path.join(roll_workdir, "X_pipeline.pkl"))
        if Y_pipeline is not None:
            torch.save(Y_pipeline.state_dict(), os.path.join(roll_workdir, "Y_pipeline.pkl"))

        sample_weight_func_name = data_config.get("sample_weight_func")
        if sample_weight_func_name:
            assert not grf, "sample_weight_func not supported for grf"
            logging.info("sample_weight_func: %s", sample_weight_func_name)
            sample_weight_func = import_attr(sample_weight_func_name)
            tsample_weights = np.array(
                sample_weight_func(phase_di["train"][1] - phase_di["train"][0])
            )
        else:
            tsample_weights = None

        model = model_cls(model_config)

        if isinstance(model, nn.Module):
            assert not grf, "pytorch model not supported for grf"
            # optimize memory access pattern
            X_strides = X.strides
            if not (X_strides[0] > X_strides[1] > X_strides[2]):
                X = np.copy(X, order="c")
            train_nn(
                model,
                X=X,
                Y=Y,
                seq_len=seq_len,
                phase_indices=phase_indices,
                workdir=roll_workdir,
                config=config,
                resume=resume,
            )
        else:
            logging.info("model: %s %s", model_type, model_config)
            # optimize memory access pattern
            if np.argmin(X.strides) != 2:
                X = np.copy(X, order="c")
            train_ml(
                model,
                X=X,
                Y=Y,
                phase_indices=phase_indices,
                workdir=roll_workdir,
                tsample_weights=tsample_weights,
                grf=grf,
            )

        with open(os.path.join(roll_workdir, ".done"), "w") as f:
            f.write(datetime.now().strftime("%Y%m%d-%H%M%S") + "\n")


if __name__ == "__main__":
    main()
