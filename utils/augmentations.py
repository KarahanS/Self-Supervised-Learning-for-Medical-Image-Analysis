from enum import Enum
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# https://github.com/j-freddy/simclr-medical-imaging


## TODO: Some of the transformations are applied on 28x28 images. Modify them to have a more general approach.


class RandomAdjustSharpness:
    def __init__(self, factor_low, factor_high):
        """
        Adjust image sharpness with factor randomly chosen between factor_low
        and factor_high. A factor of 0 gives blurred image and a factor of 1
        gives the original image.

        Args:
            factor_low (float): The lower bound of the sharpness factor.
            factor_high (float): The upper bound of the sharpness factor.
        """
        self.factor_low = factor_low
        self.factor_high = factor_high

    def __call__(self, img):
        factor = np.random.uniform(self.factor_low, self.factor_high)
        return ImageEnhance.Sharpness(img).enhance(factor)


class RandomEqualize:
    def __init__(self, p=0.5):
        """
        Equalise the image histogram. Can be applied to colour images by
        converting to YCbCr format first (which separates raw intensity from
        other channels), applying equalisation, then converting back to RGB.

        Args:
            p (float, optional): The probability of applying equalization.
                Defaults to 0.5.
        """
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = img.convert("YCbCr")
            y, cb, cr = img.split()
            y_eq = ImageOps.equalize(y)
            img = Image.merge("YCbCr", (y_eq, cb, cr))
            img = img.convert("RGB")
            return img
        return img


class AugmentationSequenceType(Enum):
    """
    Augmentation sequence for SimCLR pretraining

    Natural:
        random horizontal flip > crop-and-resize > colour distortion > random
        greyscale > Gaussian blur
    Default:
        crop-and-resize > colour distortion > Gaussian blur   # https://arxiv.org/abs/2101.05224
    Novel:
        random horizontal flip > crop-and-resize > colour distortion > random
        greyscale > Gaussian blur > random histogram equalisation > random
        sharpness
    Greyscale:
        random horizontal flip > crop-and-resize > random greyscale > Gaussian
        blur > random histogram equalisation > random sharpness
    """

    NATURAL = "natural"
    DEFAULT = "default"
    NOVEL = "novel"
    NOVEL_GREYSCALE = "greyscale"


def to_rgb(img):
    return img.convert("RGB")


augmentation_sequence_map = {
    AugmentationSequenceType.NATURAL.value: transforms.Compose(
        [
            # Normalise to 3 channels
            transforms.Lambda(to_rgb),
            # Transformation 1: random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Transformation 2: crop-and-resize
            transforms.RandomResizedCrop(size=28),
            # Transformation 3: colour distortion
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.8,
            ),
            # Transformation 4: random greyscale
            transforms.RandomGrayscale(p=0.2),
            # Transformation 5: Gaussian blur
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    AugmentationSequenceType.DEFAULT.value: transforms.Compose(
        [
            # Normalise to 3 channels
            transforms.Lambda(to_rgb),
            # Transformation 1: crop-and-resize
            transforms.RandomResizedCrop(size=28),
            # Transformation 2: colour distortion
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.8,
            ),
            # Transformation 3: Gaussian blur
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    AugmentationSequenceType.NOVEL.value: transforms.Compose(
        [
            # Normalise to 3 channels
            transforms.Lambda(to_rgb),
            # Transformation 1: random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Transformation 2: crop-and-resize
            transforms.RandomResizedCrop(size=28),
            # Transformation 3: smaller colour distortion
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.04,
                    )
                ],
                p=0.8,
            ),
            # Transformation 4: random greyscale
            transforms.RandomGrayscale(p=0.2),
            # Transformation 5: Gaussian blur
            transforms.GaussianBlur(kernel_size=9),
            # Transformation 6: Histogram equalisation and sharpness to tackle low
            # contrast
            RandomEqualize(0.5),
            RandomAdjustSharpness(factor_low=1, factor_high=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    AugmentationSequenceType.NOVEL_GREYSCALE.value: transforms.Compose(
        [
            # Normalise to 3 channels
            transforms.Lambda(to_rgb),
            # Transformation 1: random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Transformation 2: crop-and-resize
            transforms.RandomResizedCrop(size=28),
            # Transformation 3: Gaussian blur
            transforms.GaussianBlur(kernel_size=9),
            # Transformation 4: Histogram equalisation and sharpness to tackle low
            # contrast
            RandomEqualize(0.5),
            RandomAdjustSharpness(factor_low=1, factor_high=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}
