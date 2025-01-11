# dataset.py
import os
from PIL import Image
from datasets import load_dataset
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


class AtlasI2UVDataset(Dataset):
    def __init__(self, split="train", prompts_file=None, cache_dir=None):
        self.dataset = load_dataset("ggxxii/ATLAS", split=split, cache_dir=cache_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),  # Converts to [0,1]
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        uv_image = self.dataset[idx]["uv_texture"]  # This may vary based on key name
        if isinstance(uv_image, Image.Image):
            uv_image = uv_image.convert("RGB")

        image_tensor = self.transform(image)
        uv_tensor = self.transform(uv_image)

        return {"image": image_tensor, "uv_image": uv_tensor}
