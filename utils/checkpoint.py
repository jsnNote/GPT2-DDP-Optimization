import os
import torch

parent_path = os.path.dirname(os.path.dirname(__file__))
checkpoint_path = os.path.join(parent_path, "model", "checkpoint")
latest_path = os.path.join(checkpoint_path, "latest.pt")


def save_checkpoint(checkpoint):
    if os.path.exists(latest_path):
        latest_ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)
        new_name = f"{latest_ckpt['sampler_config']['name']}.pt"
        os.rename(
            latest_path,
            os.path.join(checkpoint_path, new_name),
        )

    torch.save(checkpoint, os.path.join(checkpoint_path, "latest.pt"))


def load_checkpoint():
    if os.path.exists(latest_path):
        return torch.load(latest_path, map_location="cpu", weights_only=True)
    else:
        return None
