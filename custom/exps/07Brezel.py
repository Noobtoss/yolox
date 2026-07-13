import os

from mods import Exp as _Exp


class Exp(_Exp):
    def __init__(self):
        super().__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = f"{self.exp_name}"

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/07Brezel"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 8

        # ---------------- semmel config ---------------- #

        self.num_classes = 1
        self.names = {
            1: "Brezel",
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
