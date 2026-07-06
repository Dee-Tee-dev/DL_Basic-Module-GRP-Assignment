from torchvision import transforms
import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from diffusers import StableDiffusionPipeline
from peft import LoraConfig


MODEL_NAME = "sd-legacy/stable-diffusion-v1-5"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion 1.5")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--instance_token", type=str, default="<sks>")
    parser.add_argument("--output_dir", type=str, default="lora_out")

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


class GhibliDataset(Dataset):
    def __init__(self, data_dir, instance_token):
        self.data_dir = data_dir
        self.instance_token = instance_token

        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        prompt = f"{self.instance_token} style"

        return {
            "pixel_values": image,
            "prompt": prompt,
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(instance_token):
    print("Loading Stable Diffusion 1.5...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    num_added_tokens = tokenizer.add_tokens(instance_token)

    if num_added_tokens == 0:
        print(f"Token {instance_token} already exists in tokenizer.")
    else:
        print(f"Added new token: {instance_token}")

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(instance_token)
    print(f"Token ID for {instance_token}: {token_id}")

    return pipe


def freeze_base_models(pipe):
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)


def add_lora_adapters(pipe, rank):
    print("Adding LoRA adapters to UNet and text encoder...")

    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    text_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    pipe.unet.add_adapter(unet_lora_config)
    pipe.text_encoder.add_adapter(text_encoder_lora_config)

    trainable_params = []

    for name, param in pipe.unet.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    for name, param in pipe.text_encoder.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    print(f"Trainable parameter tensors: {len(trainable_params)}")

    return trainable_params


if __name__ == "__main__":
    args = parse_args()

    dataset = GhibliDataset(args.data_dir, args.instance_token)
    print(f"Images found: {len(dataset)}")

    device = get_device()
    print(f"Using device: {device}")

    pipe = load_pipeline(args.instance_token)
    freeze_base_models(pipe)
    trainable_params = add_lora_adapters(pipe, args.rank)

    print("Dual-adapter LoRA setup complete.")