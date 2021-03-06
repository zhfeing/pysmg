import yaml
import argparse
import os
import tqdm

import torch

from pysmg.model import get_model
from pysmg.dataloader import get_dataloader
from model_generator import get_config_iter

import kamal


def export_models(global_config: str, log_dir: str, file_name_cfg: str, export_path: str):
    """
    Args:
        global_config: global config yaml filepath
    """
    with open(global_config) as fp:
        global_config = yaml.load(fp, Loader=yaml.SafeLoader)

    for cfg in get_config_iter(global_config):
        cfg["training"] = global_config["training"]
        cfg["validation"] = dict()
        if "encoder_name" not in cfg["model"].keys():
            cfg["model"]["encoder_name"] = None
            cfg["model"]["encoder_weights"] = None
        # dataset configs useless
        cfg["training"]["batch_size"] = global_config["running_args"]["batch_size"]
        cfg["training"]["seed"] = global_config["running_args"]["seed"]
        cfg["training"]["n_workers"] = 0
        cfg["validation"]["batch_size"] = global_config["running_args"]["batch_size"]
        cfg["validation"]["n_workers"] = 0

        train_loader, val_loader, n_classes = get_dataloader(cfg)

        formatter = (
            cfg["model"]["arch"],
            cfg["model"]["encoder_name"],
            cfg["model"]["encoder_weights"],
            cfg["data"]["dataset"]
        )
        model_name = file_name_cfg.format(*formatter)
        ckpt_path = os.path.join(log_dir, "ckpt", model_name)
        print("exporting model: ", model_name)

        for name in tqdm.tqdm(os.listdir(ckpt_path)):
            ckpt_fp = os.path.join(ckpt_path, name)
            if ckpt_fp[-4:] == ".pth":
                ckpt = torch.load(ckpt_fp, map_location="cpu")
                # dict_keys(['iter', 'model_state', 'optimizer_state', 'scheduler_state', 'iou'])
                parallel_model_state = ckpt["model_state"]
                model_state = dict((name[7:], val) for name, val in parallel_model_state.items())

                model = get_model(cfg, n_classes)
                model.load_state_dict(model_state)

                export_name = "{}-entry-{}".format(model_name, name)
                kamal.hub.save(
                    model=model,
                    save_path=export_path,
                    entry_name=export_name,
                    spec_name=model_name,
                    code_path='.',
                    metadata=kamal.hub.meta.Metadata(
                        name=export_name,
                        dataset=cfg["data"]["dataset"],
                        task=kamal.hub.meta.TASK.SEGMENTATION,
                        url="https://github.com/zhfeing/pysmg",
                        input=kamal.hub.meta.ImageInput(
                            size=[cfg["data"]["img_rows"], cfg["data"]["img_cols"]],
                            range=[0, 1],
                            space='rgb',
                            normalize=dict(
                                mean=cfg["data"]["normalize_mean"],
                                std=cfg["data"]["normalize_std"]
                            ),
                        ),
                        other_metadata=dict(
                            num_classes=n_classes
                        ),
                        entry_args=dict(
                            num_classes=n_classes,
                            pretrained_backbone=False
                        )
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--file-name-cfg", type=str)
    parser.add_argument("--export-path", type=str)
    args = parser.parse_args()

    print("starting to export models")
    export_models(args.global_config, args.log_dir, args.file_name_cfg, args.export_path)
