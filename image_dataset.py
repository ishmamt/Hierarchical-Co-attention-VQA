import numpy as np
import os
import errno
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from PIL import Image
from tqdm import tqdm
from image_encoder import load_resnet


def load_image(image_path):
    ''' Loads a single image from the given path.
    Parameters:
        image_path: string; Path to the image.
    Returns:
        image: PIL image object; Image converted to RGB format. 
    '''

    if not os.path.exists(image_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image_path)

    with open(image_path, "rb") as file:
        image = Image.open(file)
        return image.convert("RGB")


def get_image_ids(image_names, image_prefix, type):
    ''' Fetches the image ids from the file names.
    Parameters:
        image_names: list; List of image names.
        image_prefix: string; Prefix of image names i.e. "COCO_train2014_".
        type: string; Data type of how we want the image ids i.e "dict" or "list".
    Returns:
        image_ids: list/dict/None; Collection of image ids.
    '''

    print("Fetching Image IDs...")
    image_ids = None

    if type == "dict":
        image_ids = dict()
        for idx, image_name in enumerate(image_names):
            id = image_name.split(".")[0].rpartition(image_prefix)[-1]  # image name: COCO_train2014_000000000123.jpg
            image_ids[int(id)] = idx
    elif type == "list":
        image_ids = list()
        for idx, image_name in enumerate(image_names):
            id = image_name.split(".")[0].rpartition(image_prefix)[-1]  # image name: COCO_train2014_000000000123.jpg
            image_ids.append(int(id))
    return image_ids


def get_image_names(image_dir):
    ''' Returns names of all the images.
    Parameters:
        image_dir: string; Image directory.
    Returns:
        image_names: list; Names of all the images.
    '''

    return [file for file in os.listdir(image_dir)]


class VQAImageDataset(Dataset):
    ''' Class to handle the images from the VQA dataset.
    '''

    def __init__(self, image_dir, image_prefix, name, metadata_dir, type="dict"):
        ''' Constructor for VQAImageDataset Class.
        Parameters:
            image_dir: string; Image directory.
            image_prefix: string; Prefix of image names i.e. "COCO_train2014_".
            name: string; "train"/"test"/"val"
            metadata_dir: string; Directory for saving metadata about images.
            type: string; Data type of how we want the image ids i.e "dict" or "list".
        '''

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image_dir)

        self.image_dir = image_dir
        self.image_names = get_image_names(self.image_dir)
        self.transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
        self.image_ids = get_image_ids(self.image_names, image_prefix, type)

        if not os.path.isdir(metadata_dir):
            print(f"Making directory: {metadata_dir}")
            os.mkdir(metadata_dir)

        with open(os.path.join(metadata_dir, f"{name}_enc_idx.npy"), "wb") as file:
            print(f"Saving image ids in {metadata_dir}{name}_enc_idx.npy")
            pickle.dump(self.image_ids, file)


    def __len__(self):
        ''' Overwritten method for returning size of the dataset.
        Returns:
            len(self.image_names): int; size of the dataset.
        '''

        return len(self.image_names)


    def __getitem__(self, idx):
        ''' Overwritten method for getting a specific image from the dataset.
        Parameters:
            idx: int; The index of the image we want to fetch.
        Returns:
            image; float; Image from the dataset.
        '''

        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = self.transform(load_image(image_path))
        return image.float()


def save_image_encodings(image_dir, image_prefix, name, metadata_dir, encoded_image_dir, batch_size=8, num_of_workers=4):
    ''' Passes images through the image encoder and saves the encoded images for later usage.
    Parameters:
        image_dir: string; Image directory.
        image_prefix: string; Prefix of image names i.e. "COCO_train2014_".
        name: string; "train"/"test"/"val"
        metadata_dir: string; Directory for saving metadata about images.
        encoded_image_dir: string; Directory for saving encoded images.
        batch_size: int; Batch size for dataloader.
        num_of_workers: int; Number of CPU workers.
    '''

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = load_resnet(device)

    image_dataset = VQAImageDataset(image_dir, image_prefix, name, metadata_dir)
    image_dataset_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_of_workers)

    if not os.path.isdir(encoded_image_dir):
        print(f"Making directory: {encoded_image_dir}")
        os.mkdir(encoded_image_dir)
    
    print(f"Dumping {name} image encodings at {encoded_image_dir}...")
    progress_bar = tqdm(total=len(image_dataset_loader))

    for idx, image in enumerate(image_dataset_loader):
        progress_bar.update(1)
        image = image.to(device)
        output = model(image)
        output = output.view(output.size(0), output.size(1), -1)  # -1 is inferred from other dimensions
        output = output.cpu().numpy()
        
        np.savez_compressed(os.path.join(encoded_image_dir, f"{idx}.npz"), image=output)


if __name__ == "__main__":
    train_image_dir = "Data/train/images/train10K/"
    train_metadata_dir = "Data/train/images/encoded_images_index/"
    train_encoded_image_dir = "Data/train/images/encoded_images/"
    train_image_prefix = "COCO_train2014_"
    name = "train"

    # train = VQAImageDataset(train_image_dir, train_image_prefix, name, train_metadata_dir)
    save_image_encodings(train_image_dir, train_image_prefix, name, train_metadata_dir, train_encoded_image_dir)

    val_image_dir = "Data/val/images/val3K/"
    val_metadata_dir = "Data/val/images/encoded_images_index/"
    val_encoded_image_dir = "Data/val/images/encoded_images/"
    val_image_prefix = "COCO_val2014_"
    name = "val"

    # val = VQAImageDataset(val_image_dir, val_image_prefix, name, val_metadata_dir)
    save_image_encodings(val_image_dir, val_image_prefix, name, val_metadata_dir, val_encoded_image_dir)
