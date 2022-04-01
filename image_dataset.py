import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from PIL import Image
from tqdm import tqdm
from image_encoder import load_resnet


def laod_image(image_path):
    with open(image_path, "rb") as file:
        image = Image.open(file)
        return image.convert("RGB")


def get_image_ids(image_names, image_prefix, type="dict"):
    if type == "dict":
        image_ids = dict()
        for idx, image_name in tqdm(enumerate(image_names)):
            id = image_name.split(".")[0].rpartition(image_prefix)[-1]
            image_ids[int(id)] = idx
    else:
        image_ids = list()
        for idx, image_name in tqdm(enumerate(image_names)):
            id = image_name.split(".")[0].rpartition(image_prefix)[-1]
            image_ids.append(int(id))
    return image_ids


class VQAImageDataset(Dataset):
    """
    Class to handle the images from the VQA dataset.
    Parameters:
        image_dir = image directory
        image_prefix = prefix of image names
        name = train/test/val
        metadata_dir = directory to save metadata about images
    """

    def __init__(self, image_dir, image_prefix, name, metadata_dir):
        self.image_dir = image_dir
        self.image_names = [file for file in os.listdir(self.image_dir)]
        self.transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
        self.image_ids = get_image_ids(self.image_names, image_prefix)

        if not os.path.isdir(metadata_dir):
            print(f"Making directory: {metadata_dir}")
            os.mkdir(metadata_dir)

        with open(os.path.join(metadata_dir, f"{name}_enc_idx.npy"), "wb") as file:
            print(f"Saving image ids in {metadata_dir}{name}_enc_idx.npy")
            pickle.dump(self.image_ids, file)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = self.transform(laod_image(image_path))
        return image.float()


def save_image_encodings(image_dir, image_prefix, name, metadata_dir,
                         encoded_image_dir, device="cpu", batch_size=8, num_of_workers=4):
    """
    Passes images through the image encoder and saves the encoded images for later usage.
    """
    model = load_resnet(device)

    image_dataset = VQAImageDataset(image_dir, image_prefix, name, metadata_dir)
    image_dataset_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_of_workers)

    print(f"Dumping {name} image encodings at {encoded_image_dir}")
    for idx, image in tqdm(enumerate(image_dataset_loader)):
        image = image.to(device)
        output = model(image)
        output = output.view(output.size(0), output.size(1), -1)  # -1 is inferred from other dimensions
        output = output.cpu().numpy()

        path = os.path.join(encoded_image_dir, f"{idx}.npz")
        if not os.path.isdir(encoded_image_dir):
            print(f"Making directory: {encoded_image_dir}")
            os.mkdir(encoded_image_dir)
        np.savez_compressed(path, image=output)
        print(path)


if __name__ == "__main__":
    train_image_dir = "Data/VQA/train/images/train10K/"
    train_metadata_dir = "Data/VQA/train/images/encoded_images_index/"
    train_encoded_image_dir = "Data/VQA/train/images/encoded_images/"
    train_image_prefix = "COCO_train2014_"
    name = "train"

    # train = VQAImageDataset(train_image_dir, train_image_prefix, name, train_metadata_dir)
    save_image_encodings(train_image_dir, train_image_prefix, name, train_metadata_dir, train_encoded_image_dir)

    val_image_dir = "Data/VQA/val/images/3K/"
    val_metadata_dir = "Data/VQA/val/images/encoded_images_index/"
    val_encoded_image_dir = "Data/VQA/val/images/encoded_images/"
    val_image_prefix = "COCO_val2014_"
    name = "val"

    # val = VQAImageDataset(val_image_dir, val_image_prefix, name, val_metadata_dir)
    # save_image_encodings(val_image_dir, val_image_prefix, name, val_metadata_dir, val_encoded_image_dir)
