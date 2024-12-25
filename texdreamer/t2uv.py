# t2uv.py

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch
import torch.nn.functional as F

from dataset import HuggingFaceATLAS
from lora_patch import apply_lora_to_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SD pipeline & tokenizer
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14-336"
    ).to(device)

    # Apply LoRA
    unet, text_encoder = apply_lora_to_model(pipe.unet, text_encoder)

    # Dataset and dataloader
    dataset = HuggingFaceATLAS(split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(
        list(unet.parameters()) + list(text_encoder.parameters()), lr=1e-4
    )
    scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base")

    for epoch in range(10):
        for batch in tqdm(dataloader):
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

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
