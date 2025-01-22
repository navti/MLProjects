# infer_uv.py
import torch.nn.functional as F
import argparse, os, torch, yaml
from diffusers import StableDiffusionPipeline
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from PIL import Image
from torchvision import transforms
from lora_utils import load_lora_adapters
from joint import ImageAligner  # reuse the aligner class you trained


# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Generate UV texture from text and/or image"
)
parser.add_argument("--prompt", type=str, default="", help="Text prompt (optional)")
parser.add_argument(
    "--image", type=str, default="", help="Path to conditioning image (optional)"
)
parser.add_argument("--lora", type=str, required=True, help="LoRA adapter path")
parser.add_argument("--aligner", type=str, required=True, help="Aligner weights (.pth)")
parser.add_argument("--steps", type=int, default=50, help="Diffusion sampling steps")
parser.add_argument(
    "--img-weight",
    type=float,
    default=0.5,
    help="Weight for image conditioning when both prompt & image present (0-1)",
)
args = parser.parse_args()

# sanity check
if not args.prompt and not args.image:
    raise ValueError("Provide at least --prompt or --image")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# Config
with open("config/train_config.yaml") as f:
    cfg = yaml.safe_load(f)
model_cache = os.path.expanduser(cfg["model"]["cache_dir"])

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    cfg["model"]["pretrained_model"],
    cache_dir=model_cache,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)
pipe.enable_xformers_memory_efficient_attention()
pipe.unet.enable_gradient_checkpointing()

# Apply LoRA
pipe.unet = load_lora_adapters(pipe.unet, args.lora)

# Load Aligner
aligner = ImageAligner(use_all_tokens=True, pooling="attention").to(device).eval()
aligner.load_state_dict(torch.load(args.aligner, map_location="cpu"))

# Encoders
tokenizer = CLIPTokenizer.from_pretrained(
    cfg["model"]["text_encoder"], cache_dir=model_cache
)
text_encoder = (
    CLIPTextModel.from_pretrained(cfg["model"]["text_encoder"], cache_dir=model_cache)
    .to(device)
    .eval()
)
vision_encoder = (
    CLIPVisionModel.from_pretrained(cfg["model"]["text_encoder"], cache_dir=model_cache)
    .to(device)
    .eval()
)
vision_proc = CLIPImageProcessor.from_pretrained(
    cfg["model"]["text_encoder"], cache_dir=model_cache
)

# freeze everything
for m in (text_encoder, vision_encoder, pipe.vae):  # vae is frozen anyway
    m.requires_grad_(False)

# Build conditioning
cond_embeds = None

if args.prompt:
    tokens = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).to(device)
    cond_embeds = text_encoder(**tokens).last_hidden_state.half()  # (1,77,768)

if args.image:
    img = (
        Image.open(args.image).convert("RGB").resize((384, 512))  # match training size
    )
    proc = vision_proc(images=img, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        vision_hidden = vision_encoder(**proc).last_hidden_state  # (1,257,1024)
    image_embeds = aligner(vision_hidden).to(dtype)  # (1,77,768)
    if cond_embeds is None:
        cond_embeds = image_embeds
    else:
        # blend text & image embeddings
        w = args.img_weight
        cond_embeds = (1 - w) * cond_embeds + w * image_embeds

if cond_embeds is None:
    raise RuntimeError("Conditioning embeddings could not be computed.")

# Diffusion sampling
latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)  # 512×512 resolution
pipe.scheduler.set_timesteps(args.steps)

for t in pipe.scheduler.timesteps:
    with torch.no_grad():
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=cond_embeds).sample
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# Decode UV texture
with torch.no_grad():
    uv = pipe.vae.decode(latents / 0.18215).sample
uv = (uv / 2 + 0.5).clamp(0, 1)  # back to [0,1]

# Resize to 1024x1024
resized_uv = F.interpolate(uv, size=(1024, 1024), mode="bilinear", align_corners=False)

# Save
out_path = "generated_uv.png"
transforms.ToPILImage()(resized_uv[0].cpu()).save(out_path)
print(f"Saved UV texture to {out_path}")
