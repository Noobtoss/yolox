import os
from pathlib import Path
from yolox.exp import Exp as _Exp


# THS, Copied from yolox.exp.yolox_base.py


def tag_exp(exp, cfg_list):
    exp.dataset_name = Path(getattr(exp, "data_dir")).stem
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if k == "ckpt":
            exp.model_name = Path(v).stem


class Exp(_Exp):
    def __init__(self):
        super().__init__()

        self.output_dir = "./runs"

        # ---------------- semmel config ---------------- #

        self.num_classes = 17
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
            17: "Vanillehoernchen"
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

        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = self.img_size or (1280, 1280)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = None
        # name of annotation file for training
        self.train_ann = "annotation_train.json"
        # name of annotation file for evaluation
        self.val_ann = "annotation_val.json"
        # name of annotation file for testing
        self.test_ann = "annotation_test.json"

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False  # True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = self.img_size or (1280, 1280)  # (height, width)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform

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
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        def read_useful_info(coco):
            coco.dataset["info"] = None
            return coco

        coco_dataset = COCODataset(
            name="Images",
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        read_useful_info(coco_dataset.coco)
        return coco_dataset

    def merge(self, cfg_list):
        super().merge(cfg_list)
        tag_exp(self, cfg_list)
