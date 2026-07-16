import contextlib
import io
import warnings
import json
import tempfile
from loguru import logger
from yolox.utils import is_main_process
from yolox.evaluators.coco_evaluator import per_class_AP_table, per_class_AR_table
from yolox.evaluators import COCOEvaluator as _COCOEvaluator


class COCOEvaluator(_COCOEvaluator):

    def __init__(self, *args, **kwargs):
        warnings.warn("[Modded] COCOEvaluator")
        super().__init__(*args, **kwargs)

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            # >>> MOD
            return {"metrics/mAP50-95": 0, "metrics/mAP50": 0}, None
            # <<< MOD
        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")
            # >>> MOD
            with contextlib.redirect_stdout(io.StringIO()):
                # class-agnostic AP
                cocoEval_agnostic = COCOeval(cocoGt, cocoDt, annType[1])
                cocoEval_agnostic.params.useCats = 0
                cocoEval_agnostic.evaluate()
                cocoEval_agnostic.accumulate()
                cocoEval_agnostic.summarize()
                # AP ignoring class 0
                cocoEval_no0 = COCOeval(cocoGt, cocoDt, annType[1])
                cocoEval_no0.params.catIds = [c for c in cocoGt.getCatIds() if c != 0]
                cocoEval_no0.evaluate()
                cocoEval_no0.accumulate()
                cocoEval_no0.summarize()
                # AP ignoring classes below 22
                cocoEval_new = COCOeval(cocoGt, cocoDt, annType[1])
                cocoEval_new.params.catIds = [c for c in cocoGt.getCatIds() if c >= 22]
                cocoEval_new.evaluate()
                cocoEval_new.accumulate()
                cocoEval_new.summarize()
            # <<< MOD
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            # >>> MOD
            metrics = {
                "metrics/mAP50-95": cocoEval.stats[0],
                "metrics/mAP50": cocoEval.stats[1],
                "metrics/mAP50-95_agnostic": cocoEval_agnostic.stats[0],
                "metrics/mAP50-95_No0": cocoEval_no0.stats[0],
                "metrics/mAP50-95_New": cocoEval_new.stats[0],
            }
            return metrics, info
        else:
            return {"metrics/mAP50-95": 0, "metrics/mAP50": 0}, info
        # <<< MOD
