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

from atlas_dataset.dataset import HuggingFaceATLAS
from lora_utils import *


with open("config/train_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
model_cache_path = os.path.expanduser(cfg["model"]["cache_dir"])
data_cache_path = os.path.expanduser(cfg["data"]["cache_dir"])


def set_requires_grad(module, requires_grad: bool = True):
    """
    Freezes or unfreezes the parameters of a given module.

    Args:
        module (torch.nn.Module): The module to modify.
        requires_grad (bool): Whether the parameters should require gradients.
                              Set to False to freeze, True to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load SD pipeline & tokenizer
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["pretrained_model"],
        cache_dir=model_cache_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.enable_gradient_checkpointing()

    tokenizer = CLIPTokenizer.from_pretrained(
        cfg["model"]["text_encoder"], cache_dir=model_cache_path
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg["model"]["text_encoder"], cache_dir=model_cache_path
    ).to(device)

    # Apply LoRA
    unet = apply_lora_to_model(
        pipe.unet,
        r=cfg["model"]["lora"]["r"],
        alpha=cfg["model"]["lora"]["alpha"],
    )

    set_requires_grad(text_encoder, requires_grad=False)
    freeze_all_but_lora(unet)

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
        unet.parameters(),
        lr=cfg["training"]["learning_rate"],
    )
    scheduler = pipe.scheduler

    enable_amp = False
    grad_accum_steps = 16
    accumulated_loss = 0
    grad_scaler = torch.amp.GradScaler(enabled=enable_amp)
    for epoch in range(cfg["training"]["num_epochs"]):
        step = 0
        total_steps = len(dataloader) // grad_accum_steps
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast(device.type, dtype=torch.float16, enabled=enable_amp):
                text_inputs = tokenizer(
                    batch["prompt"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                )
                input_ids = text_inputs.input_ids.to(device)
                text_embeds = text_encoder(input_ids=input_ids).last_hidden_state
                text_embeds = text_embeds.half()

                images = batch["uv_image"].to(device).half()
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
            loss = loss / grad_accum_steps
            grad_scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % grad_accum_steps == 0:
                step += 1
                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                optimizer.zero_grad()
                if step % cfg["output"]["log_every"] == 0:
                    print(
                        f"Epoch {epoch}, Step {step}/{total_steps}, Loss: {accumulated_loss:.4f}"
                    )
                # print(f"Step {step}, Loss: {accumulated_loss:.4f}")
                accumulated_loss = 0
    # Save LoRA adapter weights
    save_path = cfg["output"].get(
        "lora_save_path", "checkpoints/t2uv/lora_adapter_t2uv.pth"
    )
    save_path = os.path.expanduser(save_path)
    save_lora_adapters(unet, save_path)


if __name__ == "__main__":
    main()
