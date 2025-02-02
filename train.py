import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.gmae import GaussianMAE
from src.CUBDataset import CUBDataset

from torchvision.utils import make_grid

from dotenv import load_dotenv
load_dotenv()

def train_gmae():
    # 1) Initialize W&B
    wandb.init(
        project="GaussianMAE-CUB",
    )

    # 2) Device and hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    lr = 1e-3
    epochs = 400
    batch_size = 3
    num_workers = 3
    log_interval = 10

    # 3) Create your dataset & dataloader
    train_dataset = CUBDataset(
        root_dir="datasets/CUB_200_2011/images",
        img_size=256
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 4) Define your model
    gmae = GaussianMAE(
        image_size = 256,
        channels = 3,
        patch_size = 4,
        masking_ratio = 0.75,

        # Encoder configs
        encoder_dim = 384,
        encoder_depth = 6,
        encoder_heads = 6,
        encoder_dim_head = 64,

        # Decoder configs
        decoder_dim = 384,
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

    global_step = 0
    for epoch in range(epochs):
        gmae.train()

        for step, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            
            optimizer.zero_grad()
            
            imgs = imgs.to(device, dtype)

            # Forward pass through the GaussianMAE
            gaussian_params_shared = gmae(imgs)
            gaussian_params = gaussian_params_shared.detach()
            gaussian_params.requires_grad = True

            images = []
            total_loss = 0.0
            avg_scales = torch.zeros(3)
            avg_means = torch.zeros(3)
            avg_opacities = torch.zeros(1)
            avg_colors = torch.zeros(3)
            for gaussian_param, img in zip(gaussian_params, imgs):
                (
                    means,
                    quats,
                    scales,
                    opacities,
                    colors,
                ) = gmae.decode_gaussian_params(gaussian_param, c=0.1)

                rgb_image, alpha, metadata = gmae.rasterize(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors,
                    focal_length=130.0
                )
                rgb_image = rgb_image[0].permute(2, 0, 1)

                loss = F.mse_loss(rgb_image, img)
                images.append(img.clone().detach().cpu() * 0.5 + 0.5)
                images.append(rgb_image.clone().detach().cpu())

                avg_scales += scales.clone().detach().cpu().mean(dim=0)
                avg_means += means.clone().detach().cpu().mean(dim=0)
                avg_opacities += opacities.clone().detach().cpu().mean(dim=0)
                avg_colors += colors.clone().detach().cpu().mean(dim=0)
                total_loss += loss.item()

                loss.backward()

            gaussian_params_shared.backward(gradient=gaussian_params.grad)

            optimizer.step()

            global_step += 1
            if step % log_interval == 0:
                wandb.log({
                    "train_loss": total_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "scale x": avg_scales[0],
                    "scale y": avg_scales[1],
                    "scale z": avg_scales[2],
                    "mean x": avg_means[0],
                    "mean y": avg_means[1],
                    "mean z": avg_means[2],
                    "opacity": avg_opacities[0],
                    "color r": avg_colors[0],
                    "color g": avg_colors[1],
                    "color b": avg_colors[2],
                    "images": [
                        wandb.Image(
                            make_grid(images, nrow=4, padding=2, normalize=True)
                        )
                    ]
                })

        scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    train_gmae()
