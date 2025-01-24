import yaml, os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from lora_utils import *
from atlas_dataset.dataset import ATLASLarge
from i2uv import ImageAligner  # reuse aligner class
from torch.utils.tensorboard import SummaryWriter
import datetime

with open("config/train_config.yaml") as f:
    cfg = yaml.safe_load(f)
model_cache = os.path.expanduser(cfg["model"]["cache_dir"])
data_cache = os.path.expanduser(cfg["data"]["cache_dir"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    log_dir = os.path.expanduser("logs")
    writer = SummaryWriter(log_dir=log_dir)

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["pretrained_model"],
        cache_dir=model_cache,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.enable_gradient_checkpointing()

    # Tokenizer and Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg["model"]["text_encoder"], cache_dir=model_cache
    )
    text_encoder = (
        CLIPTextModel.from_pretrained(
            cfg["model"]["text_encoder"], cache_dir=model_cache
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    # Vision Encoder and Aligner
    vision_model = (
        CLIPVisionModel.from_pretrained(
            cfg["model"]["text_encoder"], cache_dir=model_cache
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    vision_proc = CLIPImageProcessor.from_pretrained(
        cfg["model"]["text_encoder"], cache_dir=model_cache
    )
    aligner = ImageAligner(use_all_tokens=True, pooling="attention").to(device)

    # LoRA and UNet
    unet = apply_lora_to_model(
        pipe.unet, r=cfg["model"]["lora"]["r"], alpha=cfg["model"]["lora"]["alpha"]
    )
    pipe.unet = unet
    set_requires_grad(pipe.vae, False)
    freeze_all_but_lora(unet)

    # Dataset
    dataset = ATLASLarge(
        task="joint", cache_dir=data_cache, resize_img=(512, 384), resize_tex=512
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["joint"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(
        list(aligner.parameters()) + list(unet.parameters()),
        lr=cfg["training"]["joint"]["learning_rate"],
    )
    scheduler = pipe.scheduler
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    # Training
    alpha = cfg["training"]["joint"].get("alpha", 1.0)  # T2UV loss weight
    beta = cfg["training"]["joint"].get("beta", 1.0)  # I2UV loss weight

    enable_amp = False
    batch_size = cfg["training"]["joint"]["batch_size"]
    effective_batch_size = cfg["training"]["joint"]["effective_batch_size"]
    grad_scaler = GradScaler(enabled=enable_amp)
    total_steps = 0
    for epoch in range(cfg["training"]["joint"]["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            with torch.autocast(device.type, dtype=dtype):
                # T2UV
                text_inputs = tokenizer(
                    batch["prompt"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                )
                input_ids = text_inputs.input_ids.to(device)
                text_embeds = text_encoder(input_ids=input_ids).last_hidden_state.half()

                uv_imgs = batch["texture"].to(device, dtype)  # [B,3,512,512]
                uv_latents = pipe.vae.encode(uv_imgs).latent_dist.sample() * 0.18215
                noise = torch.randn_like(uv_latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (uv_latents.shape[0],),
                    device=device,
                ).long()
                noisy_uv_latents = scheduler.add_noise(uv_latents, noise, timesteps)

                noise_pred_text = unet(
                    noisy_uv_latents, timesteps, encoder_hidden_states=text_embeds
                ).sample
                loss_t2uv = F.mse_loss(noise_pred_text, noise)

                # I2UV
                raw_img = (batch["image"] * 0.5 + 0.5).clamp(0, 1)
                proc = vision_proc(
                    images=[img.cpu().permute(1, 2, 0).numpy() for img in raw_img],
                    return_tensors="pt",
                    do_rescale=False,
                ).to(device, dtype)

                with torch.no_grad():
                    clip_hidden = vision_model(
                        **proc
                    ).last_hidden_state  # [B, 257, 1024]
                cond_embeds = aligner(clip_hidden).to(dtype)

                noisy_img_latents = scheduler.add_noise(uv_latents, noise, timesteps)
                noise_pred_image = unet(
                    noisy_img_latents, timesteps, encoder_hidden_states=cond_embeds
                ).sample
                loss_i2uv = F.mse_loss(noise_pred_image, noise)

                # Combine and Backprop
                total_loss = alpha * loss_t2uv + beta * loss_i2uv
                total_loss = total_loss / (effective_batch_size / batch_size)

            grad_scaler.scale(total_loss).backward()

            if (step + 1) % (effective_batch_size / batch_size) == 0:
                total_steps += 1
                grad_scaler.unscale_(optimizer)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()

            if (step + 1) % cfg["output"]["log_every"] == 0:
                writer.add_scalar("Loss/t2uv", loss_t2uv.item(), total_steps)
                writer.add_scalar("Loss/i2uv", loss_i2uv.item(), total_steps)
                writer.add_scalar("Loss/combined", total_loss.item(), total_steps)
                pbar.set_postfix(
                    {
                        "loss_t2uv": loss_t2uv.item(),
                        "loss_i2uv": loss_i2uv.item(),
                        "total": total_loss.item(),
                    }
                )
            # break
        # Save checkpoint
        save_lora_adapters(unet, f"checkpoints/joint/epoch_{epoch}")
        torch.save(
            aligner.state_dict(),
            f"checkpoints/joint/epoch_{epoch}/aligner.pth",
        )
        # break
    writer.close()


if __name__ == "__main__":
    main()
