## Load an npz file and save it as a correct format for ImageFolder Pytorch
import os
import numpy as np
from src.utils.constants import DATASETS_DIR
from PIL import Image
from typing import Optional, Literal
import tqdm
import argparse
SAVE_FORMAT = '.jpg'

def load_npz(dataset_path):
    """
    Load a dataset in npz format and save it in jpg format.
    """
    npz_path = os.path.join(DATASETS_DIR,dataset_path)
    return np.load(npz_path)
    

def medmnist_npz_to_image_folders(
        dataset : np.array,
        save_path : str,
        split : Literal['val','train','test'],
        ):

    images = dataset[f'{split}_images']
    labels = dataset[f'{split}_labels']

    # Create the folder 
    for label in np.unique(labels):
        os.makedirs(os.path.join(DATASETS_DIR, save_path, split, str(label)), exist_ok=True)

    tqdm_bar = tqdm.tqdm(total=len(images), desc=f'Converting {split} images to jpg')
    for i, (image, label) in enumerate(zip(images, labels)):
        tqdm_bar.update(1)
        image = Image.fromarray(image)
        image.save(os.path.join(DATASETS_DIR, save_path, split, str(label[0]), f'{i}{SAVE_FORMAT}'))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert npz to image folders')
    parser.add_argument('--dataset', type=str, help='Path to the npz file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = load_npz(args.dataset)
    dataset_name = args.dataset.split('/')[-1].split('.')[0]
    medmnist_npz_to_image_folders(dataset, dataset_name, 'train')
    medmnist_npz_to_image_folders(dataset, dataset_name, 'val')
    medmnist_npz_to_image_folders(dataset, dataset_name, 'test')