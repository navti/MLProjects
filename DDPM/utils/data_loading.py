import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import pickle
from torchvision.transforms import transforms
import os

__all__ = ["make_cifar_set"]


def unpickle(file):
    with open(file, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return {"data": data_dict[b"data"], "labels": data_dict[b"labels"]}


def merge_dicts(dicts: list) -> dict:
    # keys: data, labels
    n = len(dicts)
    data, labels = dicts[0]["data"], dicts[0]["labels"]
    for i in range(1, n):
        data = np.concatenate([data, dicts[i]["data"]], axis=0)
        labels = labels + dicts[i]["labels"]
    return {"data": data, "labels": labels}


def get_cifar_data(data_dir, test=False):
    data_dir = os.path.abspath(data_dir)
    if test:
        batch_files = glob.glob(f"{data_dir}/**/test_batch", recursive=True)
    else:
        batch_files = glob.glob(f"{data_dir}/**/*_batch_*", recursive=True)
    data_dicts = list(map(unpickle, batch_files))
    data_dict = merge_dicts(data_dicts)
    imgs = data_dict["data"]
    images = list(map(preprocess_image, imgs))
    labels = data_dict["labels"]
    return images, labels


def preprocess_image(img):
    torch_img = torch.from_numpy(img) / 255
    torch_img = (torch_img - 0.5) * 2
    torch_img = torch_img.view(size=(3, 32, 32))
    return torch_img


class CIFARDataset(Dataset):
    """
    Make CIFAR dataset
    """

    def __init__(self, data_dir, diffuser=None, image_transforms=None, test=False):
        super(CIFARDataset, self).__init__()
        self.images, self.labels = get_cifar_data(data_dir, test)
        self.image_transforms = image_transforms
        self.diffuser = diffuser

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.diffuser:
            x0 = image.unsqueeze(0)
            xt, eps, t_emb, ts = self.diffuser(x0)
            return xt.squeeze(), eps.squeeze(), t_emb.squeeze(), ts.squeeze(), label
        return image, label


def make_cifar_set(data_dir="../data/", diffuser=None):
    image_transforms = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize()
        ]
    )
    train_set = CIFARDataset(data_dir, diffuser, image_transforms)
    test_set = CIFARDataset(data_dir, None, image_transforms, test=True)
    return train_set, test_set
