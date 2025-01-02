# lora_patch.py

from peft import get_peft_model, LoraConfig, TaskType
from diffusers.models.attention_processor import LoRAAttnProcessor


def apply_lora_to_model(unet, text_encoder, r=16, alpha=32):
    # Apply LoRA to U-Net using diffusers-native API
    for name, module in unet.named_modules():
        if hasattr(module, "set_attn_processor") and hasattr(module, "to_q"):
            hidden_size = module.to_q.in_features
            module.set_attn_processor(
                LoRAAttnProcessor(hidden_size=hidden_size, rank=r)
            )

    return unet, text_encoder
