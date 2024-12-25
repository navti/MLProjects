# lora_patch.py

from peft import get_peft_model, LoraConfig, TaskType


def apply_lora_to_model(unet, text_encoder, r=16, alpha=32):
    config_unet = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out"],
        task_type=TaskType.UNET,
    )
    unet = get_peft_model(unet, config_unet)

    config_text = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    text_encoder = get_peft_model(text_encoder, config_text)

    return unet, text_encoder
