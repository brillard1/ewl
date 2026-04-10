"""
Vision Transformer (ViT) model setup with LoRA
"""

import torch
import timm
from peft import LoraConfig, get_peft_model


def get_vit_lora_config(config):
    """
    Create LoRA configuration for Vision Transformers

    Args:
        config: Configuration dictionary

    Returns:
        LoRA configuration object
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        modules_to_save=lora_cfg.get("modules_to_save", ["head"]),
    )

    return lora_config


def setup_vit_model(config):
    """
    Setup Vision Transformer with LoRA for fine-tuning

    Args:
        config: Configuration dictionary

    Returns:
        model: LoRA-adapted ViT model
    """
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    # Determine number of classes
    dataset_num_classes = {
        "cub200": 200,
        "aircraft": 100,
        "stanford_dogs": 120,
        "food101": 101,
    }
    num_classes = dataset_num_classes.get(
        dataset_cfg["name"], config["dataset"].get("num_classes", 1000)
    )

    # Load pretrained ViT
    print(
        f"[MODEL] Loading {model_cfg['model_id']} (pretrained={model_cfg.get('pretrained', True)})..."
    )
    model = timm.create_model(
        model_cfg["model_id"],
        pretrained=model_cfg.get("pretrained", True),
        num_classes=num_classes,
        drop_rate=model_cfg.get("drop_rate", 0.0),
    )

    # Count base parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(
        f"[MODEL] Base model: {total_params:,} params ({trainable_params_before:,} trainable)"
    )

    # Apply LoRA
    print("[MODEL] Applying LoRA...")
    lora_config = get_vit_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Count LoRA parameters
    trainable_params_after = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(
        f"[MODEL] LoRA applied: {trainable_params_after:,} trainable params "
        + f"({trainable_params_after/total_params*100:.2f}% of base)"
    )

    return model
