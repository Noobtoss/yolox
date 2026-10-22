import os
import cv2
import warnings
from pycocotools.coco import COCO
from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.coco import remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset
from yolox.data.datasets.coco import COCODataset as _COCODataset


class COCODataset(_COCODataset):
    def __init__(
            self,
            data_dir=None,
            json_file="instances_train2017.json",
            name="train2017",
            img_size=(416, 416),
            preproc=None,
            cache=False,
            cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        # >>> MOD
        self.coco = COCO(os.path.join(self.data_dir, self.json_file))
        # <<< MOD
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

        # >>> MOD
        path_filename = [anno[3] for anno in self.annotations]
        # <<< MOD
        # >>> MOD
        CacheDataset.__init__(
            self,
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
        # <<< MOD

    def load_image(self, index):
        file_name = self.annotations[index][3]

        # >>> MOD
        img_file = os.path.join(self.data_dir, file_name)
        # <<< MOD

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img
