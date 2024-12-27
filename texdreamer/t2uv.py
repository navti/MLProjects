# t2uv.py

import yaml
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os

from dataset import HuggingFaceATLAS
from lora_patch import apply_lora_to_model


with open("config/train_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
model_cache_path = os.path.expanduser(cfg["model"]["cache_dir"])
data_cache_path = os.path.expanduser(cfg["data"]["cache_dir"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Load SD pipeline & tokenizer
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["pretrained_model"],
        cache_dir=model_cache_path,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg["model"]["text_encoder"], cache_dir=model_cache_path
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg["model"]["text_encoder"], cache_dir=model_cache_path
    ).to(device)

    # Apply LoRA
    unet, text_encoder = apply_lora_to_model(
        pipe.unet,
        text_encoder,
        r=cfg["model"]["lora"]["r"],
        alpha=cfg["model"]["lora"]["alpha"],
    )

    # Dataset and dataloader
    dataset = HuggingFaceATLAS(
        split=cfg["data"]["split"],
        prompts_file=cfg["data"]["prompts_file"],
        cache_dir=data_cache_path,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
    )

    optimizer = AdamW(
        list(unet.parameters()) + list(text_encoder.parameters()),
        lr=cfg["training"]["learning_rate"],
    )
    scheduler = DDPMScheduler.from_pretrained(
        cfg["model"]["pretrained_model"], cache_dir=model_cache_path
    )

    for epoch in range(cfg["training"]["num_epochs"]):
        for step, batch in enumerate(tqdm(dataloader)):
            text_inputs = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            input_ids = text_inputs.input_ids.to(device)
            text_embeds = text_encoder(input_ids).last_hidden_state

            images = batch["uv_image"].to(device)
            images = images * 2 - 1
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeds
            ).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg["output"]["log_every"] == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
