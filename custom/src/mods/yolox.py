import warnings
from yolox.models.yolox import YOLOX


class YOLOX(YOLOX):
    def __init__(self, *args, **kwargs):
        warnings.warn("[Modded] YOLOX")
        super().__init__(*args, **kwargs)

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, cls_feat_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "cls_feat_loss": cls_feat_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
