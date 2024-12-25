# dataset.py

from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset


class HuggingFaceATLAS(Dataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("ggxxii/ATLAS", split=split)
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item["prompt"]
        image = self.transform(item["image"]).float()
        return {"prompt": prompt, "uv_image": image}
