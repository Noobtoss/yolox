import time
from loguru import logger
import warnings
import torch
from yolox.utils import adjust_status, is_parallel, synchronize

from yolox.core import Trainer as _Trainer


# THS, Copied from yolox.core.trainer


class Trainer(_Trainer):
    def __init__(self, exp, args):
        warnings.warn("[Modded] Trainer")
        super().__init__(exp, args)
        self.cls_feat_scheduler = None

    def before_train(self):
        super().before_train()
        # or is lr_scheduler default
        self.cls_feat_proj_head_lr_scheduler = self.exp.get_lr_scheduler(
            getattr(self.exp, "cls_feat_proj_head_lr", None) or self.exp.basic_lr_per_img * self.args.batch_size,
            self.max_iter
        )
        if hasattr(self.exp, 'get_cls_feat_scheduler'):
            self.cls_feat_scheduler = self.exp.get_cls_feat_scheduler(self.exp.cls_feat)

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
        ap50 = metrics["metrics/mAP50"]
        ap50_95 = metrics["metrics/mAP50-95"]
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
