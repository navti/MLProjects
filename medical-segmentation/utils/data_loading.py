import torch
from torch.utils.data import Dataset
import cv2
import json
import glob


__all__ = ['CVCDataset', 'ThresholdTransform', 'parse_config']

"""
Make custom CVC dataset
"""
class CVCDataset(Dataset):
    def __init__(self, images, image_transforms=None, mask_transforms=None):
        super(CVCDataset, self).__init__()
        self.images = images
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_split = img_path.split('/')
        img_name = img_split[-1]
        img_dir = '/'.join(img_split[:-2])
        mask_path = img_dir+"/Ground Truth/"+img_name
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        # inv_mask = 1 - mask
        # mask = torch.concat([inv_mask, mask], dim=0)
        return image, mask

"""
Custom transform to threshold the mask
"""
class ThresholdTransform(object):
  def __init__(self, thr=0.5):
    self.thr = thr

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


def parse_config(file="./config.json"):
    with open("./config.json","r") as f:
        config = json.load(f)
    image_path = config['data_path']+"Original/"
    mask_path  = config['data_path']+"Ground Truth/"
    images = glob.glob(image_path+"*"+config['file_extn'])
    masks  = glob.glob(mask_path+"*"+config['file_extn'])
    return images, masks