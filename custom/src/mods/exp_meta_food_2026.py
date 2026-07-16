import torch.nn as nn

from .exp import Exp as _Exp
from .yolox_head_meta_food_2026 import YOLOXHead


# THS, Copied from yolox.exp.yolox_base.py


class Exp(_Exp):
    def __init__(self):
        super().__init__()
        self.save_history_ckpt   = False  # True
        self.cls_feat            = None   # 1
        self.cls_feat_loss       = None   # SupervisedContrastiveLoss()
        self.save_history_ckpt   = False  # True

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN  # , YOLOXHead # THS

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
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                             cls_feat=float(self.cls_feat) if self.cls_feat is not None else None,
                             cls_feat_loss=self.cls_feat_loss)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
