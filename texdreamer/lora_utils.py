# lora_utils.py

import torch
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


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


def freeze_all_but_lora(module):
    for name, param in module.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_lora_to_model(unet, r=16, alpha=None):
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha or r,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        # task_type="FEATURE_EXTRACTION",
    )

    # Apply LoRA to the UNet
    unet = get_peft_model(unet, lora_config)
    return unet


def save_lora_adapters(unet, save_path):
    """
    Saves only the LoRA adapter weights from the PEFT-wrapped UNet.
    """
    if hasattr(unet, "save_pretrained"):
        unet.save_pretrained(save_path)
        print(f"LoRA adapters saved to: {save_path}")
    else:
        raise ValueError("UNet is not a PEFT model with LoRA adapters applied.")


def load_lora_adapters(unet, lora_path):
    """
    Loads LoRA adapters from `lora_path` and injects them into the provided UNet model.
    Returns the LoRA-wrapped UNet.
    """
    # Load the config to verify it's a PEFT model
    unet = PeftModel.from_pretrained(unet, lora_path)
    print(f"LoRA adapters loaded from: {lora_path}")
    return unet
