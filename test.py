from loader.medmnist_loader import MedMNISTLoader
import torchvision.models as models


def test_medmnist_loader():
    loader = MedMNISTLoader("pathmnist", download=True, batch_size=256)
    loader.show_info()
    loader.display_data()

    print("resnet50" in models.list_models())


test_medmnist_loader()
