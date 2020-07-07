import copy


def get_dataset_iter(datasets_cfg: dict):
    for cfg in datasets_cfg.values():
        new_dict = dict(data=cfg)
        yield new_dict


def get_encoder_iter(encoders_cfg: dict):
    for arch, cfg in encoders_cfg.items():
        if isinstance(cfg, str):
            cfg = [cfg]
        for weights in cfg:
            new_dict = dict(encoder_name=arch, encoder_weights=weights)
            yield new_dict


def get_model_iter(models_cfg: dict):
    for cfg in models_cfg.values():
        new_dict = dict(model=cfg)
        yield new_dict


def get_config_iter(global_config: dict):
    datasets = global_config["datasets"]
    encoders = global_config["encoders"]
    models = global_config["models"]
    for data_cfg in get_dataset_iter(datasets):
        for model_cfg in get_model_iter(models):
            if model_cfg["model"]["customizable_encoder"]:
                for encoder_cfg in get_encoder_iter(encoders):
                    config_dict = dict()
                    config_dict.update(copy.deepcopy(data_cfg))
                    config_dict.update(copy.deepcopy(model_cfg))
                    config_dict["model"].update(copy.deepcopy(encoder_cfg))
                    config_dict["model"].pop("customizable_encoder")
                    yield config_dict
            else:
                config_dict = dict()
                config_dict.update(copy.deepcopy(data_cfg))
                config_dict.update(copy.deepcopy(model_cfg))
                config_dict["model"].pop("customizable_encoder")
                yield config_dict
