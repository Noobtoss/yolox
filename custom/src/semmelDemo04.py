import os

from mods import Exp as _Exp


class Exp(_Exp):
    def __init__(self):
        super().__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/semmelDemo04"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 8

        # ---------------- semmel config ---------------- #

        self.num_classes = 21
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
            21: "Fruechteschiffchen"
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
