import ast
import os
import time
from pathlib import Path
from pprint import pformat
from textwrap import indent
from loguru import logger
import warnings
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from yolox.data import DataPrefetcher
from yolox.utils import (
    MlflowLogger,
    ModelEMA,
    adjust_status,
    get_model_info,
    is_parallel,
    occupy_mem,
    synchronize,
)

from yolox.core import Trainer as _Trainer
from .wandb_logger import WandbLogger as WandbLogger


def save_exp(exp, exp_file, run_dir):
    marker = "\n\n# Resolved training configuration\n"
    source = Path(exp_file).read_text(encoding="utf-8").split(marker, 1)[0]
    settings = {}
    for key, value in vars(exp).items():
        if key.startswith("_"):
            continue
        try:
            ast.literal_eval(repr(value))
        except (ValueError, SyntaxError):
            continue
        settings[key] = value
    exp_class = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == "Exp")
    init = next(node for node in exp_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    snapshot = "    def __init__(self):\n        super().__init__()\n"
    snapshot += "".join(indent(f"self.{key} = {pformat(value, sort_dicts=False)}\n", "        ")
                        for key, value in settings.items())
    lines = source.splitlines(keepends=True)
    lines[init.lineno - 1:init.end_lineno] = [snapshot]
    (Path(run_dir) / "exp.py").write_text("".join(lines), encoding="utf-8")


class Trainer(_Trainer):
    def __init__(self, exp, args):
        warnings.warn("[Modded] Trainer")
        super().__init__(exp, args)
        self.cls_feat_scheduler = None

    def before_train(self):
        # >>> MOD
        if self.rank == 0 and self.args.exp_file:
            save_exp(self.exp, self.args.exp_file, self.file_name)
        # <<< MOD
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                # >>> MOD
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
                # <<< MOD
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))
        # >>> MOD
        # or is lr_scheduler default
        self.cls_feat_proj_head_lr_scheduler = self.exp.get_lr_scheduler(
            getattr(self.exp, "cls_feat_proj_head_lr", None) or self.exp.basic_lr_per_img * self.args.batch_size,
            self.max_iter
        )
        if hasattr(self.exp, 'get_cls_feat_scheduler'):
            self.cls_feat_scheduler = self.exp.get_cls_feat_scheduler(self.exp.cls_feat)
        # <<< MOD

    def before_epoch(self):
        # only update cls_feat_gain per epoch
        if self.cls_feat_scheduler is not None:
            self.model.head.cls_feat = self.cls_feat_scheduler.update_cls_feat(self.epoch)
        super().before_epoch()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        # >>> MOD
        cls_feat_proj_head_lr = self.cls_feat_proj_head_lr_scheduler.update_lr(self.progress_in_iter + 1)
        self.optimizer.param_groups[-1]["lr"] = cls_feat_proj_head_lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            cls_feat_proj_head_lr=cls_feat_proj_head_lr,
            cls_feat=self.model.head.cls_feat,
            **outputs,
        )
        # <<< MOD

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (metrics, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )
        ap50 = metrics["eval/mAP50"]
        ap50_95 = metrics["eval/mAP50-95"]
        # metrics = {f"val/{k}": v for k, v in metrics.items()}

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                for k, v in metrics.items():
                    self.tblogger.add_scalar(k, v, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({**metrics, "train/epoch": self.epoch + 1})
                # self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {**metrics, "val/best_ap": round(self.best_ap, 3), "train/epoch": self.epoch + 1}
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        if self.args.logger == "mlflow":
            metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
            self.mlflow_logger.save_checkpoints(self.args, self.exp, self.file_name, self.epoch,
                                                metadata, update_best_ckpt)

    def after_iter(self):
        super().after_iter()
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar(
                    "train/cls_feat_proj_head_lr", self.meter["cls_feat_proj_head_lr"].latest, self.progress_in_iter)
                self.tblogger.add_scalar(
                    "train/cls_feat", self.meter["cls_feat"].latest, self.progress_in_iter)
            if self.args.logger == "wandb":
                metrics = {
                    "train/cls_feat_proj_head_lr": self.meter["cls_feat_proj_head_lr"].latest,
                    "train/cls_feat": self.meter["cls_feat"].latest,
                }
                self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
            if self.args.logger == 'mlflow':
                logs = {
                    "train/cls_feat_proj_head_lr": self.meter["cls_feat_proj_head_lr"].latest,
                    "train/cls_feat": self.meter["cls_feat"].latest,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)
