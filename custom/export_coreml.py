import argparse
import os
from loguru import logger
import torch
import torch.nn as nn
from yolox.exp import get_exp
import coremltools as ct

DEVICE = torch.device("cpu")

# Example: python -m custom.export_coreml --output-name yolox.mlpackage --ckpt runs/06Base_baseline/best_ckpt.pth --exp_file custom/src/06Base.py

def make_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="YOLOX Core ML Deployment")
    parser.add_argument("--output-name", type=str, default="yolox.mlpackage", help="Output name of models")
    parser.add_argument("--input", type=str, default="images", help="Input node name of ONNX model")
    parser.add_argument("--output", type=str, default="output", help="Output node name of ONNX model")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, choices=[1], help="Batch size")
    parser.add_argument("--dynamic", action="store_true", help="Whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", help="Use onnxsim or not")
    parser.add_argument("--exp_file", type=str, default=None, help="Experiment description file")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--name", type=str, default=None, help="Model name")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    parser.add_argument("--decode_in_inference", action="store_true", help="Decode in inference or not")
    parser.add_argument("--class_name", type=str, default=None, help="Class name for the object")

    return parser


class YoloxDetectModel(nn.Module):
    """Wrap a YOLOX model for Apple iOS CoreML export."""

    def __init__(self, model, im, num_of_class, device='cpu'):
        """Initialize the YOLOXDetectModel class with a YOLO model and example image."""
        super().__init__()
        _, _, h, w = im.shape
        self.model = model
        self.nc = num_of_class  # number of classes
        self.register_buffer("normalize", torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h], device=device))

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        x = x[:, [2, 1, 0], :, :]  # RGB to BGR
        out_pred = self.model(x)
        xywh = out_pred[:, :, :4][0]
        objectness = out_pred[0][:, 4:5]
        class_conf = out_pred[0][:, 5:5 + self.nc]
        class_scores = objectness * class_conf
        return class_scores, xywh * self.normalize


@logger.catch(reraise=True)
def main():
    """Main function to run the script."""
    args = make_parser().parse_args()
    logger.info(f"Arguments: {args}")

    exp = get_exp(args.exp_file, args.name)
    if args.ckpt:
        exp.ckpt = args.ckpt
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    ckpt_file = args.ckpt if args.ckpt else os.path.join(exp.output_dir, args.experiment_name, "best_ckpt.pth")

    ckpt = torch.load(ckpt_file, map_location=DEVICE)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    model.head.decode_in_inference = True
    logger.info("Checkpoint loaded.")

    names = [name for _, name in sorted(exp.names.items())]
    nc = exp.num_classes
    if len(names) != nc:
        raise ValueError("Experiment class names do not match num_classes")

    im = torch.zeros(args.batch_size, 3, exp.test_size[0], exp.test_size[1]).to(DEVICE)
    model = YoloxDetectModel(model, im, nc, DEVICE)
    model.eval().to(DEVICE)

    with torch.no_grad():
        y = model(im)
        traced_model = torch.jit.trace(model, im, strict=False)

    bias = [0.0, 0.0, 0.0]
    ct_model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        inputs=[ct.ImageType("image", shape=im.shape, bias=bias)]
    )
    _, _, h, w = list(im.shape)
    spec = ct_model.get_spec()
    out0, out1 = iter(spec.description.output)

    out0_shape = tuple(y[0].size())
    out1_shape = tuple(y[1].size())

    out0.type.multiArrayType.shape[:] = out0_shape
    out1.type.multiArrayType.shape[:] = out1_shape

    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = "confidence"
    nms_spec.description.output[1].name = "coordinates"

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name
    nms.coordinatesInputFeatureName = out1.name
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = 0.5
    nms.confidenceThreshold = 0.4
    nms.pickTop.perClass = False  # True  # THS: class_agnostic nms
    nms.stringClassLabels.vector.extend(names)
    nms_model = ct.models.MLModel(nms_spec)

    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, h, w)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=["confidence", "coordinates"],
    )

    # Model from spec
    _model = ct.models.MLModel(spec)

    pipeline.add_model(_model)
    pipeline.add_model(nms_model)

    pipeline.spec.description.input[0].ParseFromString(_model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.userDefined.update(
        {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
    )

    ct_model = ct.models.MLModel(pipeline.spec)
    ct_model.input_description["image"] = "Input image"
    ct_model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
    ct_model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
    )
    ct_model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    ct_model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"

    metadata = {
        "description": "yolox",
        "author": "Noobtoss"
    }
    ct_model.short_description = metadata.pop("description")
    ct_model.author = metadata.pop("author")
    ct_model.user_defined_metadata.update({k: str(v) for k, v in metadata.items()})

    ct_model.save(args.output_name)


if __name__ == "__main__":
    main()
