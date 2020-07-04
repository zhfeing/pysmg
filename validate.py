import yaml
import argparse
import timeit

import torch
from torch.utils import data

from ptsemseg.model import get_model
from ptsemseg.data import get_dataset
from ptsemseg.metrics import RunningScore
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True


@torch.no_grads()
def validate(cfg, args):
    # Setup Dataloader
    data_loader = get_dataset(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"])
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(
        loader,
        batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["validation"]["n_workers"],
        shuffle=False,
        pin_memory=True
    )
    running_metrics = RunningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes)
    # data parrall model state
    state = torch.load(args.model_path, map_location="cpu")["model_state"]
    state = convert_state_dict(state)
    model.load_state_dict(state)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = images.cuda()
        if args.eval_flip:
            flipped_images = images.flip(-1)

            outputs = model(images)
            outputs_flipped = model(flipped_images)
            outputs = (outputs + outputs_flipped.flip(-1)) / 2.0
        else:
            outputs = model(images)
        pred = outputs.argmax(-1)

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(labels, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )
    parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, yaml.SafeLoader)

    validate(cfg, args)
