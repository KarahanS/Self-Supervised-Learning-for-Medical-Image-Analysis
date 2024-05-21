import argparse

from src.downstream.eval.train import train as eval_train
from src.ssl.simclr.train import train as simclr_train
from src.utils.config.config import Config
from src.utils.enums import DatasetEnum, SSLMethod, DownstreamMethod
import src.utils.setup as setup


parser = argparse.ArgumentParser(
    description="PyTorch SSL Training for Medical Image Analysis"
)

parser.add_argument(
    "--cfg-path", default=None, help="Path to the configuration file."
)


def main():
    args = parser.parse_args()
    cfg = Config(args.cfg_path)

    setup.configure_paths(cfg)
    setup.setup_device(cfg)  # GPU setup if available

    if cfg.Dataset.name == DatasetEnum.MIMETA:
        cfg.Dataset.params.image_size = 224  # there is no other option for MIMETA dataset

    # main can be used either for self-supervised pretraining or downstream task evaluation
    if "Pretrain" in cfg.Training:
        assert (
            cfg.Training.Pretrain.params.n_views == 2
        ), "Only two view training is supported. Please use --n-views 2."

        # Dataset should be read in the train.py of related SSL method
        if cfg.Training.Pretrain.ssl_method == SSLMethod.SIMCLR:
            simclr_train(cfg)
        else:
            raise ValueError("Other SSL methods are not supported yet.")
    else:  # Downstream
        if cfg.Training.Downstream.eval_method in [DownstreamMethod.LINEAR, DownstreamMethod.NONLINEAR]:
            eval_train(cfg)  # logistic regression or mLP
        else:
            raise ValueError("Other evaluation methods are not supported yet.")


if __name__ == "__main__":
    main()
