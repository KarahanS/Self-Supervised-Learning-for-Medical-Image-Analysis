from enum import Enum


class DatasetEnum(Enum):
    MEDMNIST = "medmnist"
    MIMETA = "mimeta"

    def __str__(self):
        return self.value


class MedMNISTCategory(Enum):
    """
    MedMNIST v2 modalities - https://medmnist.com/
    """

    PATH = "pathmnist"
    CHEST = "chestmnist"
    DERMA = "dermamnist"
    OCT = "octmnist"
    PNEUMONIA = "pneumoniamnist"
    RETINA = "retinamnist"
    BREAST = "breastmnist"
    BLOOD = "bloodmnist"
    TISSUE = "tissuemnist"
    ORGANA = "organamnist"
    ORGANC = "organcmnist"
    ORGANS = "organsmnist"

    def __str__(self):
        return self.value
