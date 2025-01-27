import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms

from src.gmae import GaussianMAE
from src.CUBDataset import CUBDataset

def train_gmae():
    # 1) Initialize W&B
    wandb.init(
        project="GaussianMAE-CUB"
    )

    # 2) Device and hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4
    epochs = 400
    batch_size = 1

    # 3) Create your dataset & dataloader
    train_dataset = CUBDataset(
        root_dir="datasets/CUB_200_2011/images",
        img_size=256
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    # 4) Define your model
    gmae = GaussianMAE(
        image_size = 256,
        channels = 3,
        patch_size = 4,
        masking_ratio = 0.75,

        # Encoder configs
        encoder_dim = 512,
        encoder_depth = 6,
        encoder_heads = 6,
        encoder_dim_head = 64,

        # Decoder configs
        decoder_dim = 512,
        decoder_depth = 6,
        decoder_heads = 6,
        decoder_dim_head = 64
    ).to(device)

    # 5) Define optimizer and scheduler (AdamW + Cosine Decay)
    optimizer = torch.optim.AdamW(gmae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # 6) Training loop
    global_step = 0
    for epoch in range(epochs):
        gmae.train()
        total_loss = 0.0

        # Go through one epoch
        for step, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            imgs = imgs.to(device, dtype=torch.float32)

            # Forward pass through the GaussianMAE
            rgb_image, alpha, metadata, recon_loss = gmae(imgs, c=0.1, focal_length=175)

            # Backprop & update
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            total_loss += recon_loss.item()
            global_step += 1

            # Log metrics every N steps
            if step % 10 == 0:
                wandb.log({
                    "train_loss": recon_loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                    "global_step": global_step
                })

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # Average loss for this epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        # Log epoch-level metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch": epoch + 1
        })

        # Optionally, log example images (reconstruction) at end of epoch
        # (Here we'll just log the last batch's first image)
        with torch.no_grad():
            # rgb_image range is [0, 1], so it is already suitable for logging
            example_recon = rgb_image[0].clamp(0, 1).cpu()
            # Log original image side-by-side if you want
            original_img = imgs[0].permute(1, 2, 0).cpu() * 0.5 + 0.5  # if your dataset transforms are [-1,1]
            # Note: Adjust the above "0.5 + 0.5" if your transforms differ

            wandb.log({
                "Reconstruction": [
                    wandb.Image(
                        example_recon,
                        caption=f"Reconstructed (Epoch {epoch+1})"
                    )
                ],
                "Original": [
                    wandb.Image(
                        original_img,
                        caption=f"Original (Epoch {epoch+1})"
                    )
                ]
            })

    # 7) Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    train_gmae()