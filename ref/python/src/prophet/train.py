from .metric import METRIC_MAP, METRIC_ACC_MAP
from .util import in_notebook, import_attr
from tqdm import tqdm
from typing import Optional
import copy
import logging
import numpy as np
import os
import pandas as pd
import psutil
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data
import uuid
import yaml


def make_loss(type, **kwargs):
    return import_attr(f"torch.nn.{type}")(**kwargs)


def make_optimizer(type, **kwargs) -> torch.optim.Optimizer:
    return import_attr(f"torch.optim.{type}")(**kwargs)


def make_lr_scheduler(type, num_epochs, num_steps_per_epoch, **kwargs):
    args = kwargs
    if type == "OneCycleLR":
        args["epochs"] = num_epochs
        args["steps_per_epoch"] = num_steps_per_epoch
    scheduler = import_attr(f"torch.optim.lr_scheduler.{type}")(**args)
    lr_step = "epoch"
    if type == "OneCycleLR" or type == "CyclicLR":
        lr_step = "batch"
    return scheduler, lr_step


class Trainer(object):
    def __init__(
        self, config: dict, model: nn.Module, datasets: list[torch.utils.data.Dataset], workdir: str
    ):
        self.config = config

        if torch.cuda.is_available() and config.get("use_gpu", False):
            gpus = config.get("gpus", [0])
            device = f"cuda:{gpus[0]}"
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus)
        else:
            device = "cpu"
        model = model.to(device)
        self.device = device
        self.model = model
        self.current_epoch = -1
        self.num_epochs = config["epochs"]
        self.best_epoch = -1
        self.num_bad_epochs = 0
        self.early_stopping = config["early_stopping"]

        self.batch_size = config.get("batch_size", 256)
        data_workers = config.get("data_workers", 4)
        self.train_loader = torch.utils.data.DataLoader(
            datasets[0], batch_size=self.batch_size, shuffle=True, num_workers=data_workers
        )
        self.valid_loader = torch.utils.data.DataLoader(
            datasets[1], batch_size=self.batch_size, shuffle=False, num_workers=data_workers
        )
        if len(datasets) > 2:
            self.test_loader = torch.utils.data.DataLoader(
                datasets[2], batch_size=self.batch_size, shuffle=False, num_workers=data_workers
            )
        else:
            self.test_loader = None

        self.loss = make_loss(**config["loss"])
        self.metrics = config.get("metrics", [])
        self.metric_hist = {
            t: {name: [] for name in self.metrics + ["loss"]} for t in ("train", "valid")
        }
        self.optimizer = make_optimizer(params=model.parameters(), **config["optimizer"])
        lr_scheduler_config = config.get(
            "lr_scheduler", dict(type="StepLR", step_size=self.num_epochs, gamma=1)
        )
        self.lr_scheduler, self.lr_step = make_lr_scheduler(
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            num_steps_per_epoch=len(self.train_loader),
            **lr_scheduler_config,
        )

        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        with open(os.path.join(workdir, "train.yml"), "w") as f:
            yaml.safe_dump(self.config, f)

        self.process = psutil.Process(os.getpid())
        self.logger = None
        self.init_logger()

    def init_logger(self) -> None:
        self.logger = logging.getLogger(str(uuid.uuid4()))
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)-15s [%(module)s:%(lineno)s] [%(levelname)s] %(message)s"
        )

        ch = logging.StreamHandler(stream=sys.stdout if in_notebook() else None)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        log_file = os.path.join(self.workdir, "log.txt")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def train(self, show_progress: bool = True) -> None:
        while self.train_one(show_progress):
            pass

    def train_one(self, show_progress: bool = True) -> bool:
        if self.current_epoch >= self.num_epochs - 1:
            return False
        if self.num_bad_epochs >= self.early_stopping["patience"]:
            return False

        torch.set_num_threads(self.config.get("num_threads", 1))
        self.current_epoch += 1

        epoch = self.current_epoch
        self.logger.info(
            f"[Epoch %d] Start (LR: %.5f, RAM: %.f MB)",
            epoch,
            self.lr_scheduler.get_last_lr()[0],
            self.process.memory_info().rss / 1024 / 1024,
        )

        start_time = time.time()
        train_metric_acc = {m: METRIC_ACC_MAP[m]() for m in self.metrics}
        train_losses = []

        self.model.train()

        def train_steps(update_progress=None):
            for item in self.train_loader:
                x = item[0].to(self.device)
                y = item[1].to(self.device)
                tag = item[2] if len(item) > 2 else None

                self.optimizer.zero_grad()
                pred_y = self.model(x)
                loss = self.loss(pred_y, y)
                loss.backward()
                if "clip_norm" in self.config:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config["clip_norm"], norm_type=2
                    )
                self.optimizer.step()
                if self.lr_step == "batch":
                    self.lr_scheduler.step()
                train_losses.append(loss.item())
                for acc in train_metric_acc.values():
                    acc.add(y, pred_y, tag=tag)
                if update_progress is not None:
                    update_progress(np.mean(train_losses))

        pg_format = (
            "{postfix[0]} {postfix[1][value]:>8.6f}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        if show_progress:
            with tqdm(
                total=len(self.train_loader),
                bar_format=pg_format,
                postfix=["train_loss", dict(value=0)],
                file=sys.stdout if in_notebook() else None,
            ) as t:

                def update(train_loss):
                    t.postfix[1]["value"] = train_loss
                    t.update()

                train_steps(update)
        else:
            train_steps()

        self.model.eval()

        train_metric_vals = {m: v.get() for m, v in train_metric_acc.items()}
        train_metric_vals["loss"] = np.mean(train_losses)
        valid_metric_vals = self._compute_metrics(self.model, self.valid_loader)
        print_metrics = ["loss"] + self.metrics
        self.logger.info(
            "[Epoch %d] train %s",
            epoch,
            ", ".join(f"{m}: {train_metric_vals[m]:.3g}" for m in print_metrics),
        )
        self.logger.info(
            "[Epoch %d] valid %s",
            epoch,
            ", ".join(f"{m}: {valid_metric_vals[m]:.3g}" for m in print_metrics),
        )

        if self.lr_step == "epoch":
            self.lr_scheduler.step()

        for m, v in train_metric_vals.items():
            self.metric_hist["train"][m].append(v)
        for m, v in valid_metric_vals.items():
            self.metric_hist["valid"][m].append(v)

        es_metric = self.early_stopping["metric"]
        es_mode = self.early_stopping["mode"].lower()
        assert es_mode == "max" or es_mode == "min"
        valid_val = valid_metric_vals[es_metric]
        if self.best_epoch >= 0:
            best_val = self.metric_hist["valid"][es_metric][self.best_epoch]
        elif es_mode == "max":
            best_val = -np.inf
        else:
            best_val = np.inf
        if (es_mode == "max" and valid_val > best_val) or (
            es_mode == "min" and valid_val < best_val
        ):
            self.logger.info(
                "[Epoch %d] Found better model, %s %.3g -> %.3g",
                epoch,
                es_metric,
                best_val,
                valid_val,
            )
            self.num_bad_epochs = 0
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1
            self.logger.info(
                "[Epoch %d] Bad epoch %d, %s %.3g -> %.3g",
                epoch,
                self.num_bad_epochs,
                es_metric,
                best_val,
                valid_val,
            )
        self.checkpoint()

        end_time = time.time()
        self.logger.info(
            "[Epoch %d] Stop after %.2f secs (RAM: %.f MB)",
            epoch,
            end_time - start_time,
            self.process.memory_info().rss / 1024 / 1024,
        )

        if self.num_bad_epochs >= self.early_stopping["patience"]:
            self.logger.info("[Epoch %d] Early stop, best epoch: %d", epoch, self.best_epoch)
            return False
        return True

    def _compute_metrics(
        self, model: nn.Module, loader: torch.utils.data.DataLoader
    ) -> dict[str, float]:
        model.eval()
        metric_acc = {m: METRIC_ACC_MAP[m]() for m in self.metrics}
        losses = []
        for item in loader:
            x = item[0].to(self.device)
            y = item[1].to(self.device)
            tag = item[2] if len(item) > 2 else None
            pred_y = model(x)
            loss = self.loss(pred_y, y)
            losses.append(loss.item())
            for acc in metric_acc.values():
                acc.add(y, pred_y, tag=tag)
        results = {m: acc.get() for m, acc in metric_acc.items()}
        results["loss"] = np.mean(losses)
        return results

    def load_model(self, epoch: int) -> nn.Module:
        model_path = os.path.join(self.workdir, "data", f"model_{epoch}.pkl")
        if os.path.exists(model_path):
            model = copy.deepcopy(self.model)
            model.eval()
            model.load_state_dict(torch.load(model_path))
            model = model.to(self.device)
            return model
        else:
            return None

    def eval(
        self,
        dataset: torch.utils.data.Dataset = None,
        use_best: bool = True,
        epoch: Optional[int] = None,
    ) -> dict[str, float]:
        if dataset is None:
            loader = self.test_loader
        else:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=16
            )
        if loader is None:
            return None

        if epoch is None:
            if use_best:
                model = self.load_model(self.best_epoch)
                epoch = self.best_epoch
            else:
                model = self.model
                epoch = self.current_epoch
        else:
            model = self.load_model(epoch)
        metric_vals = self._compute_metrics(model, loader)
        print_metrics = ["loss"] + self.metrics
        self.logger.info(
            "[Eval] test %s (Epoch %d)",
            ", ".join(f"{m}: {metric_vals[m]:.3g}" for m in print_metrics),
            epoch,
        )
        return metric_vals

    def checkpoint(self):
        data_dir = os.path.join(self.workdir, "data")
        os.makedirs(data_dir, exist_ok=True)

        model_path = os.path.join(data_dir, f"model_{self.current_epoch}.pkl")
        best_model_path = os.path.join(data_dir, f"model_best.pkl")
        torch.save(self.model.state_dict(), model_path)
        if self.current_epoch == self.best_epoch:
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            os.symlink(os.path.basename(model_path), best_model_path)

        states = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(states, os.path.join(data_dir, f"states_{self.current_epoch}.pkl"))

        for m in ("train", "valid"):
            df = pd.DataFrame(self.metric_hist[m])
            df.to_parquet(os.path.join(data_dir, f"{m}_metrics.parquet"))

        checkpoint = dict(
            current_epoch=self.current_epoch,
            best_epoch=self.best_epoch,
            num_bad_epochs=self.num_bad_epochs,
        )
        with open(os.path.join(self.workdir, "checkpoint.yml"), "w") as f:
            yaml.safe_dump(checkpoint, f)

    def resume(self) -> None:
        checkpoint_path = os.path.join(self.workdir, "checkpoint.yml")
        if not os.path.exists(checkpoint_path):
            return
        with open(checkpoint_path) as f:
            checkpoint = yaml.safe_load(f)
        self.current_epoch = checkpoint["current_epoch"]
        self.best_epoch = checkpoint["best_epoch"]
        self.num_bad_epochs = checkpoint["num_bad_epochs"]

        data_dir = os.path.join(self.workdir, "data")

        self.model.eval()
        self.model.load_state_dict(
            torch.load(os.path.join(data_dir, f"model_{self.current_epoch}.pkl"))
        )

        states = torch.load(os.path.join(data_dir, f"states_{self.current_epoch}.pkl"))
        self.optimizer.load_state_dict(states["optimizer"])
        self.lr_scheduler.load_state_dict(states["lr_scheduler"])

        for m in ("train", "valid"):
            df = pd.read_parquet(os.path.join(data_dir, f"{m}_metrics.parquet"))
            for key in self.metric_hist[m]:
                self.metric_hist[m][key] = list(df[key])

        self.logger.info("Resuming from epoch %d", self.current_epoch)

    def close(self) -> None:
        if self.logger:
            for handler in list(self.logger.handlers):
                handler.close()
                self.logger.removeHandler(handler)
            self.logger = None
