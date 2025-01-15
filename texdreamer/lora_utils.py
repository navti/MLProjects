# lora_utils.py

import torch
from diffusers.models.attention_processor import LoRAAttnProcessor


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
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_lora_to_model(unet, r=16, alpha=None):
    for name, module in unet.named_modules():
        if hasattr(module, "set_attn_processor"):
            hidden_size = module.to_q.in_features
            module.set_attn_processor(
                LoRAAttnProcessor(hidden_size=hidden_size, rank=r, network_alpha=alpha)
            )
    return unet


def save_lora_adapters(unet, save_path):
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "processor"):
            processor = module.processor
            if hasattr(processor, "lora_linear_layer"):
                # For LoRAAttnProcessor (diffusers < v0.24)
                for pname, param in processor.lora_linear_layer.named_parameters():
                    lora_state_dict[f"{name}.processor.lora_linear_layer.{pname}"] = (
                        param
                    )
            elif hasattr(processor, "to_q_lora"):
                # For diffusers >= v0.24 which uses separate A/B for each of Q/K/V
                for pname, param in processor.named_parameters():
                    lora_state_dict[f"{name}.processor.{pname}"] = param

    torch.save(lora_state_dict, save_path)
    print(f"LoRA adapters saved to: {save_path}")


def load_lora_adapters(unet, lora_path):
    """
    Loads LoRA weights from `lora_path` and injects them into the corresponding attention processors in UNet.
    """
    lora_state_dict = torch.load(lora_path, map_location="cpu")

    unet_keys = dict(unet.named_modules())

    for full_key, param in lora_state_dict.items():
        # full_key example: 'mid_block.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight'
        parts = full_key.split(".")
        module_key = ".".join(parts[:-4])  # get up to the processor (e.g., '...attn1')
        processor_attr = parts[-4]  # e.g., 'to_q_lora'
        weight_type = parts[-3] + "." + parts[-2]  # e.g., 'down.weight' or 'up.weight'

        # Locate processor module
        if module_key in unet_keys:
            processor = getattr(unet_keys[module_key], "processor", None)
            if processor and hasattr(processor, processor_attr):
                target_module = getattr(processor, processor_attr)
                weight_name = parts[-2]  # 'weight' or 'bias'
                linear_layer = getattr(target_module, parts[-3])  # 'down' or 'up'
                setattr(linear_layer, weight_name, param)
            else:
                print(f"Warning: Processor or attribute not found for {full_key}")
        else:
            print(f"Warning: UNet module not found for {module_key}")

    print(f"LoRA adapters loaded from: {lora_path}")
