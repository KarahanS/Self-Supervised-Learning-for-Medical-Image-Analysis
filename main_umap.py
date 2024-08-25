# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.args.umap import parse_args_umap
from src.data.classification_dataloader import prepare_data
from src.ssl.methods import METHODS
from src.utils.auto_umap import OfflineUMAP



def main():
    args = parse_args_umap()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")]
    # get the ckpth that has best_val in its basename
    ckpt_path = [ckpt for ckpt in ckpt_path if "best_val" in ckpt.name][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
    cfg = OmegaConf.create(method_args)

    # build the model
    print(f"Loading model from {ckpt_path}")
    model = (
        METHODS[method_args["method"]]
        .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
        .backbone
    )
    # prepare data
    train_loader, val_loader = prepare_data(
        args.dataset,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        auto_augment=False,
    )

    umap = OfflineUMAP()

    # move model to the gpu
    device = "cuda:0"
    model = model.to(device)

    # Create a folder with the same name method
    os.makedirs(ckpt_dir / "umap", exist_ok=True)
    
    save_dir = ckpt_dir / "umap"
    # Parse checkpoint and get names (dataset-method-backbone)
    print(os.path.basename(ckpt_path))
    method_dataset, method, backbone = os.path.basename(ckpt_path).split("-")[:3]
    is_pretrained = "True" in os.path.basename(ckpt_path)

    umap.plot(device, model, train_loader, save_dir / f"{args.dataset}-{method}-{backbone}-{method_dataset}-{is_pretrained}-train_umap.pdf")
    umap.plot(device, model, val_loader, save_dir / f"{args.dataset}-{method}-{backbone}-{method_dataset}-{is_pretrained}-val_umap.pdf")


if __name__ == "__main__":
    main()