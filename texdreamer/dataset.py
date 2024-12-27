# dataset.py

from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset


class HuggingFaceATLAS(Dataset):
    def __init__(
        self,
        split="train",
        prompts_file="atlas_prompt_50k.txt",
        cache_dir="~/huggingface/hf_datasets",
    ):
        self.dataset = load_dataset("ggxxii/ATLAS", split=split, cache_dir=cache_dir)
        self.prompts = self._load_prompts(prompts_file)
        assert len(self.dataset) == len(
            self.prompts
        ), "Mismatch between number of images and prompts."

        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )

    def _load_prompts(self, prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx]["image"]).float()
        prompt = self.prompts[idx]
        return {"prompt": prompt, "uv_image": image}
