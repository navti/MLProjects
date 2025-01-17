# i2uv.py
import yaml, os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm
from lora_utils import *
from atlas_dataset.dataset import ATLASLarge  # your dataset class

# read config
with open("config/train_config.yaml") as f:
    cfg = yaml.safe_load(f)
model_cache = os.path.expanduser(cfg["model"]["cache_dir"])
data_cache = os.path.expanduser(cfg["data"]["cache_dir"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# image encoder  (frozen)
vision_model = (
    CLIPVisionModel.from_pretrained(
        cfg["model"]["text_encoder"],
        cache_dir=model_cache,
    )
    .to(device)
    .eval()
    .requires_grad_(False)
)
vision_proc = CLIPImageProcessor.from_pretrained(
    cfg["model"]["text_encoder"], cache_dir=model_cache
)


# projection / aligner  (trainable)
class ImageAligner(nn.Module):
    """
    Maps CLIP-ViT (1024-D) CLS token → 768-D and repeats to 77 tokens if use_all_tokens is false.
    """

    def __init__(
        self,
        input_dim=1024,  # CLIP ViT-L/14 token dim
        output_dim=768,  # Text latent dim from T2UV training
        use_all_tokens=False,  # Set to True to use all 257 tokens
        pooling="mean",  # Options: "mean", "attention"
    ):
        super().__init__()
        self.use_all_tokens = use_all_tokens
        self.pooling = pooling

        if use_all_tokens:
            if pooling == "attention":
                self.attn_pool = nn.MultiheadAttention(
                    embed_dim=input_dim, num_heads=8, batch_first=True
                )
                self.cls_token = nn.Parameter(
                    torch.randn(1, 1, input_dim)
                )  # learnable query
            # else pooling == "mean", no extra layers needed

        self.projector = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.GELU(), nn.Linear(1024, output_dim)
        )

    def forward(self, clip_output):
        """
        clip_output: shape [B, 257, 1024] from CLIP vision encoder
        returns: [B, 77, output_dim]
        """
        if not self.use_all_tokens:
            # Use only CLS token
            x = clip_output[:, 0]  # [B, 1024]
            x = self.projector(x)  # [B, 768]
            x = x.unsqueeze(1).repeat(1, 77, 1)  # Repeat to 77 tokens
        else:
            if self.pooling == "mean":
                x = clip_output.mean(dim=1)  # [B, 1024]
            elif self.pooling == "attention":
                B = clip_output.size(0)
                cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 1024]
                x, _ = self.attn_pool(
                    cls_token, clip_output, clip_output
                )  # [B, 1, 1024]
                x = x.squeeze(1)  # [B, 1024]
            x = self.projector(x)  # [B, 768]
            x = x.unsqueeze(1).repeat(1, 77, 1)  # Still repeat to 77 for SD UNet

        return x  # [B, 77, 768]


aligner = ImageAligner(use_all_tokens=True, pooling="attention").to(device)

# SD pipeline parts
pipe = StableDiffusionPipeline.from_pretrained(
    cfg["model"]["pretrained_model"],
    cache_dir=model_cache,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

pipe.enable_xformers_memory_efficient_attention()
pipe.unet.enable_gradient_checkpointing()

# Apply LoRA structure to UNet first
unet = apply_lora_to_model(
    pipe.unet,
    r=cfg["model"]["lora"]["r"],
    alpha=cfg["model"]["lora"]["alpha"],
)

# Then load pre-trained adapter weights
lora_path = cfg["model"].get("lora_load_path", "checkpoints/t2uv/lora_adapter_t2uv.pth")
lora_path = os.path.expanduser(lora_path)
load_lora_adapters(unet, lora_path)

set_requires_grad(pipe.vae, False)
freeze_all_but_lora(unet)

vae = pipe.vae  # frozen
scheduler = pipe.scheduler  # DDPM config

# dataset / dataloader
dataset = ATLASLarge(
    task="i2uv",
    cache_dir=data_cache,
    normalize=False,
    resize_tex=512,
    resize_img=(512, 384),
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# optimizer
optim = AdamW(
    list(aligner.parameters()) + list(unet.parameters()),
    lr=cfg["training"]["learning_rate"],
)

scaler = torch.cuda.amp.GradScaler(enabled=True)
grad_accum_steps = 16
scheduler.set_timesteps(scheduler.config.num_train_timesteps)

# training loop
for epoch in range(cfg["training"]["num_epochs"]):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optim.zero_grad()
    for step, batch in enumerate(pbar):
        # encode image
        raw_img = (batch["image"] * 0.5 + 0.5).clamp(0, 1)  # back to [0,1]
        proc = vision_proc(
            images=[img.cpu().permute(1, 2, 0).numpy() for img in raw_img],
            return_tensors="pt",
        ).to(device, dtype)
        with torch.no_grad():
            clip_hidden = vision_model(**proc).last_hidden_state  # [B,257,1024]

        cond_embeds = aligner(clip_hidden).to(dtype)  # [B,77,768]

        # encode UV target
        target_uv = batch["texture"].to(device, dtype)  # [B,3,512,512]
        latents = vae.encode(target_uv).latent_dist.sample() * 0.18215

        # diffusion noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
        ).long()
        noisy_lat = scheduler.add_noise(latents, noise, timesteps)

        # UNet forward
        with torch.autocast(device.type, dtype=dtype):
            noise_pred = unet(
                noisy_lat, timesteps, encoder_hidden_states=cond_embeds
            ).sample
            loss = F.mse_loss(noise_pred, noise) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optim)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        if (step + 1) % (grad_accum_steps * cfg["output"]["log_every"]) == 0:
            pbar.set_postfix(loss=loss.item() * grad_accum_steps)

    # optional: save checkpoint each epoch
    torch.save(
        {
            "unet": unet.state_dict(),
            "aligner": aligner.state_dict(),
        },
        f"checkpoints/i2uv/epoch_{epoch}.pth",
    )
