# dataset.py
import os
from PIL import Image
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from torch.utils.data import Dataset


class HuggingFaceATLAS(Dataset):
    def __init__(
        self,
        split="train",
        prompts_file="atlas_prompt_50k.txt",
        cache_dir="~/huggingface/hf_datasets",
        resize_dim=512,
    ):
        self.dataset = load_dataset("ggxxii/ATLAS", split=split, cache_dir=cache_dir)
        self.prompts = self._load_prompts(prompts_file)
        assert len(self.dataset) == len(
            self.prompts
        ), "Mismatch between number of images and prompts."

        self.transform = transforms.Compose(
            [transforms.Resize((resize_dim, resize_dim)), transforms.ToTensor()]
        )

    def _load_prompts(self, prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        image = self.transform(image)
        prompt = self.prompts[idx]
        return {"prompt": prompt, "uv_image": image}


class ATLASLarge(Dataset):
    def __init__(
        self,
        task="t2uv",
        cache_dir="/mnt/ssdp3/datasets/atlas_large",
        resize_tex=512,
        resize_img=(512, 384),
        normalize=True,
    ):
        self.task = task
        self.dataset = load_from_disk(cache_dir)
        tex_transforms = [
            transforms.Resize((resize_tex, resize_tex)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ]
        if normalize:
            tex_transforms.append(transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
        self.transform_tex = transforms.Compose(tex_transforms)

        if task == "i2uv":
            img_transforms = [transforms.Resize(resize_img), transforms.ToTensor()]
            if normalize:
                img_transforms.append(
                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
                )
            self.transform_img = transforms.Compose(img_transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        texture = self.transform_tex(self.dataset[idx]["texture"])
        prompt = self.dataset[idx]["prompt"]
        items = {"prompt": prompt, "texture": texture}
        if self.task == "i2uv":
            image = self.transform_img(self.dataset[idx]["image"])
            items["image"] = image
        return items
