import warnings
import torch
import torch.nn as nn

from .exp import Exp as _Exp
from .cls_feat_loss import ClsFeatLoss
from .cls_feat_proj_head import ClsFeatProjHeadFactory


# THS, Copied from yolox.exp.yolox_base.py


class Exp(_Exp):
    def __init__(self):
        super().__init__()
        self.save_history_ckpt      = False  # True
        self.train_subset_pct       = None
        self.train_min_cls_pct      = None
        self.seed                   = 2024
        self.cls_feat               = None   # 1
        self.cls_feat_dim           = 320  # hard encoding dangerous
        self.cls_feat_loss          = None   # SupervisedContrastiveLoss()
        self.cls_feat_temperature   = 0.07
        self.cls_feat_mask          = None
        self.cls_feat_mask_pct      = 0.4
        self.cls_feat_min_per_class = 4
        self.cls_feat_weight        = None
        self.cls_feat_proj_head     = None
        self.cls_feat_proj_head_lr  = None
        self.cls_feat_scheduler     = None

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import TrainTransform
        from .coco_dataset import COCODataset

        return COCODataset(
            name="Images",  # self.train_ann.split("annotation_")[-1].removesuffix(".json"),
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            train_subset_pct=self.train_subset_pct,
            train_min_cls_pct=self.train_min_cls_pct,
            seed=int(self.seed) if self.seed is not None else None,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from .coco_evaluator import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

    def get_optimizer(self, batch_size):
        optimizer = super().get_optimizer(batch_size)

        cls_feat_proj_head_params = [
            p for k, p in self.model.named_parameters()
            if p.requires_grad and "cls_feat_proj_head" in k
        ]

        if cls_feat_proj_head_params:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                # or is parent default
                lr = getattr(self, "cls_feat_proj_head_lr", None) or self.basic_lr_per_img * batch_size

            ids = {id(p) for p in cls_feat_proj_head_params}
            for group in self.optimizer.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in ids]

            self.optimizer.add_param_group({
                "params": cls_feat_proj_head_params,
                "lr": lr,
                "weight_decay": self.weight_decay,
            })
            warnings.warn(f"[Modded] Moved cls_feat_proj_head to new group lr={lr}")

        return self.optimizer

    def get_trainer(self, args):
        from .trainer import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def get_model(self):
        from yolox.models import YOLOPAFPN  # , YOLOXHead # THS
        from .yolox import YOLOX
        from .yolox_head_accv_2026 import YOLOXHead

        if self.cls_feat_loss is None:
            raise NotImplementedError("cls_feat_loss must be set before calling get_model().")
        else:
            print(f"cls_feat_loss: {self.cls_feat_loss}")

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)

            kwargs = {k.removeprefix("cls_feat_"): v for k, v in vars(self).items() if k.startswith("cls_feat_")}
            cls_feat_loss = getattr(self, "_cls_feat_loss", None) or ClsFeatLoss(**kwargs)
            cls_feat_proj_head = getattr(self, "_cls_feat_proj_head", None) or ClsFeatProjHeadFactory.get(**kwargs)

            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                             cls_feat=float(self.cls_feat) if self.cls_feat is not None else None,
                             cls_feat_loss=cls_feat_loss,
                             cls_feat_proj_head=cls_feat_proj_head)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_cls_feat_scheduler(self, cls_feat):
        from .cls_feat_scheduler import ClsFeatScheduler
        kwargs = {k.removeprefix("cls_feat_"): v for k, v in vars(self).items() if k.startswith("cls_feat_")}
        scheduler = ClsFeatScheduler(
            getattr(self, "cls_feat_scheduler", None) or "constant",
            cls_feat,
            self.max_epoch,
            **kwargs
        )
        return scheduler
