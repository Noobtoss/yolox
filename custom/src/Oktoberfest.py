import os

from mods import ExpACCV2026

EXP_FLOAT_VALUES = {"cls_feat", "train_subset_pct", "train_min_cls_pct"}


def check_exp_value(exp):
    for value in EXP_FLOAT_VALUES:
        if getattr(exp, value, None) is not None:
            setattr(exp, value, float(getattr(exp, value)))


class Exp(ExpACCV2026):
    def __init__(self):
        super().__init__()
        self.cls_feat_dim = 320
        self.cls_feat = 0
        self.cls_feat_loss = "sup_con_loss"
        self.cls_feat_temperature = 0.07

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = f"{self.exp_name}_baseline"

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/Oktoberfest"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        self.train_subset_pct = None
        self.train_min_cls_pct = None

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        # ---------------- semmel config ---------------- #

        self.num_classes = 15
        self.names = {
            1: 'Bier',
            2: 'Biermass',
            3: 'Weissbier',
            4: 'Cola',
            5: 'Wasser',
            6: 'Currywurst',
            7: 'Weisswein',
            8: 'Apfelschorle',
            9: 'Jaegermeister',
            10: 'Pommes',
            11: 'Burger',
            12: 'Williamsbirne',
            13: 'Almbrezel',
            14: 'Brotzeitkorb',
            15: 'Kaesespaetzle',
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
