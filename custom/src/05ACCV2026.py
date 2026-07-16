import os
import warnings

from mods import ClsFeatLoss, ClsFeatProjHeadFactory, ExpACCV2026

EXP_FLOAT_VALUES = {"cls_feat", "train_subset_pct", "train_min_cls_pct"}


def check_exp_value(exp):
    for value in EXP_FLOAT_VALUES:
        if getattr(exp, value, None) is not None:
            setattr(exp, value, float(getattr(exp, value)))


class Exp(ExpACCV2026):
    def __init__(self):
        super().__init__()
        self.cls_feat_dim = 320  # hard encoding dangerous
        self.cls_feat = 0
        self.cls_feat_loss = "sup_con_loss"  # ClsFeatLossFactory.get("sup_con_loss", temperature=self.cls_feat_temperature)
        self.cls_feat_temperature = 0.07
        # self.cls_feat_mask = "conf"
        # self.cls_feat_mask_pct = 0.4
        # self.cls_feat_min_per_class = 4
        # self.cls_feat_weight = "conf"
        # self.cls_feat_proj_head = "s"
        # self.cls_feat_proj_head_lr = 0.01
        # kwargs = {k.removeprefix("cls_feat_"): v for k, v in vars(self).items() if k.startswith("cls_feat_")}
        # self._cls_feat_loss = ClsFeatLoss(**kwargs)
        # self._cls_feat_proj_head = ClsFeatProjHeadFactory.get(**kwargs)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = f"{self.exp_name}_baseline"

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/05ACCV2026"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        self.train_subset_pct = None
        self.train_min_cls_pct = None

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        # ---------------- semmel config ---------------- #

        self.num_classes = 37
        self.names = {
            1: "Backware",
            2: "Bauernbrot",
            3: "Floesserbrot",
            4: "Salzstange",
            5: "Sonnenblumensemmel",
            6: "Kuerbiskernsemmel",
            7: "Roggensemmel",
            8: "Dinkelsemmel",
            9: "LaugenstangeSchinkenKaese",
            10: "Pfefferlaugenbrezel",
            11: "KernigeStange",
            12: "Schokocroissant",
            13: "Apfeltasche",
            14: "Quarktasche",
            15: "Mohnschnecke",
            16: "Nussschnecke",
            17: "Vanillehoernchen",
            18: "Osterei",
            19: "Osterbrezel",
            20: "Kirschtasche",
            21: "Fruechteschiffchen",
            22: "Anisbrezel",
            23: "Doppelsemmel",
            24: "Fruestuecksemmel",
            25: "Kaisersemmel",
            26: "Kornknacker",
            27: "Landbrot",
            28: "Laugenbrezel",
            29: "Laugenstange",
            30: "Laugenzopf",
            31: "Mohnsemmel",
            32: "Mohnstange",
            33: "Partybrot",
            34: "Sandwichbroetchen",
            35: "Sesamsemmel",
            36: "Sesamstange",
            37: "Vollgutsemmel"
        }
        self.img_size = (1280, 1280)  # (640, 640)  # (height, width)

        # ---------------- model config ---------------- #

        scale = "yolox_x"  # "yolox_m" # "yolox_l" # "yolox_x"

        if scale == "yolox_s":
            self.depth = 0.33
            self.width = 0.50
        if scale == "yolox_m":
            self.depth = 0.67
            self.width = 0.75
        if scale == "yolox_l":
            self.depth = 1.0
            self.width = 1.0
        if scale == "yolox_x":
            self.depth = 1.33
            self.width = 1.25

        # self.ckpt = f"checkpoints/{scale}.pth"

    def merge(self, *args, **kwargs):
        super().merge(*args, **kwargs)
        check_exp_value(self)
