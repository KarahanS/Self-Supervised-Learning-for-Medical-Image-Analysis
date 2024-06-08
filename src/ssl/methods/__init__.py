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

from src.ssl.methods.barlow_twins import BarlowTwins
from src.ssl.methods.base import BaseMethod
from src.ssl.methods.byol import BYOL
from src.ssl.methods.deepclusterv2 import DeepClusterV2
from src.ssl.methods.dino import DINO
from src.ssl.methods.linear import LinearModel
from src.ssl.methods.mae import MAE
from src.ssl.methods.mocov2plus import MoCoV2Plus
from src.ssl.methods.mocov3 import MoCoV3
from src.ssl.methods.nnbyol import NNBYOL
from src.ssl.methods.nnclr import NNCLR
from src.ssl.methods.nnsiam import NNSiam
from src.ssl.methods.ressl import ReSSL
from src.ssl.methods.simclr import SimCLR
from src.ssl.methods.simsiam import SimSiam
from src.ssl.methods.supcon import SupCon
from src.ssl.methods.swav import SwAV
from src.ssl.methods.vibcreg import VIbCReg
from src.ssl.methods.vicreg import VICReg
from src.ssl.methods.wmse import WMSE
from src.ssl.methods.all4one import All4One


METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mae": MAE,
    "mocov2plus": MoCoV2Plus,
    "mocov3": MoCoV3,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
    "all4one": All4One,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "MAE",
    "LinearModel",
    "MoCoV2Plus",
    "MoCoV3",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
    "All4One",
]
