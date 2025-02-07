#!/usr/bin/env python
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange, repeat
from PIL import Image
import wandb
import os
import hydra
from omegaconf import DictConfig, OmegaConf, listconfig

from gsplat import rasterization

from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# GMAE Implementation
# ------------------------------

def pair(t):
    return t if isinstance(t, (tuple, list, listconfig.ListConfig)) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class GMAE(nn.Module):
    def __init__(
        self,
        # encoder configs
        image_size,         # e.g. (176,208)
        patch_size,         # e.g. 16
        encoder_dim,
        encoder_depth,
        encoder_heads,
        pool = 'cls',
        channels = 3,
        encoder_dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,

        # decoder configs
        decoder_dim = 512,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,

        num_gaussians = 512,
        add_gauss_pos = False,
    ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking_ratio must be between 0 and 1'
        self.masking_ratio = masking_ratio

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.channels = channels

        image_height, image_width = self.image_size
        patch_height, patch_width = self.patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        # patchify
        self.to_patch = lambda img: rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        # Encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, encoder_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.encoder = Transformer(
            dim = encoder_dim, 
            depth = encoder_depth,
            heads = encoder_heads, 
            dim_head = encoder_dim_head,
            mlp_dim = encoder_dim * 4,
            dropout = dropout
        )

        # Decoder
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.gaussian_tokens = nn.Parameter(torch.randn(num_gaussians, decoder_dim))
        self.decoder = Transformer(
            dim = decoder_dim,
            depth = decoder_depth,
            heads = decoder_heads,
            dim_head = decoder_dim_head,
            mlp_dim = decoder_dim * 4
        )
        # using an embedding for position (for the decoder) on a per–patch basis
        self.feat_pos_emb = nn.Embedding(int((1 - self.masking_ratio) * num_patches), decoder_dim)
        self.add_gauss_pos = add_gauss_pos
        if self.add_gauss_pos:
            self.gaussian_pos_emb = nn.Embedding(num_gaussians, decoder_dim)
        self.to_gaussians = nn.Linear(decoder_dim, 14)

        self.num_gaussians = num_gaussians

    def forward(self, img, focal_length, scale_factor, return_imgs=False):
        device = img.device
        dtype = img.dtype

        # 1. Patchify
        patches = self.to_patch(img)  
        tokens = self.patch_to_emb(patches)
        batch, num_patches, _ = tokens.shape

        # 2. Add positional embeddings
        if self.pool == "cls":
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
            tokens = tokens + self.pos_embedding[:, :(num_patches + 1)]
        else:
            tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)]
        tokens = self.dropout(tokens)

        # 3. Masking
        num_to_mask = int(self.masking_ratio * num_patches)
        total_tokens_for_masking = tokens.shape[1] - 1 if self.pool == "cls" else tokens.shape[1]
        rand_indices = torch.rand(batch, total_tokens_for_masking, device=device).argsort(dim=-1)
        masked_indices = rand_indices[:, :num_to_mask]
        unmasked_indices = rand_indices[:, num_to_mask:]
        if self.pool == "cls":
            patch_tokens = tokens[:, 1:]
        else:
            patch_tokens = tokens
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = patch_tokens[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]

        if self.pool == "cls":
            cls_tok = tokens[:, 0].unsqueeze(1)
            tokens_for_encoder = torch.cat((cls_tok, unmasked_tokens), dim=1)
        else:
            tokens_for_encoder = unmasked_tokens

        # 4. Encoder
        encoded_tokens = self.encoder(tokens_for_encoder)
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # 5. Prepare tokens for decoder
        all_decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        if self.pool == "cls":
            unmasked_decoder_tokens = decoder_tokens[:, 1:]
        else:
            unmasked_decoder_tokens = decoder_tokens

        unmasked_decoder_tokens = unmasked_decoder_tokens + self.feat_pos_emb.weight
        gaussian_tokens = self.gaussian_tokens.to(device)
        if self.add_gauss_pos:
            gaussian_tokens += self.gaussian_pos_emb.weight
        gaussian_tokens = repeat(gaussian_tokens, 'n d -> b n d', b=batch).to(device, dtype)
        all_decoder_tokens = torch.cat((
            unmasked_decoder_tokens,
            gaussian_tokens,
        ), dim=1)

        # 6. Decoder
        decoded_tokens = self.decoder(all_decoder_tokens)

        # 7. Predict pixel values for masked patches only
        gaussian_params = self.to_gaussians(decoded_tokens[:, unmasked_decoder_tokens.shape[1]:])

        loss = 0
        pred_imgs = []
        pos = []
        cols = []
        for gaussian_param, gt_img in zip(gaussian_params, img):
            (
                means,
                quats,
                scales,
                opacities,
                colors,
            ) = self.decode_gaussian_params(gaussian_param, scale_factor)

            rgb_image, alpha, metadata = self.rasterize(
                means,
                quats,
                scales,
                opacities,
                colors,
                focal_length
            )
            rgb_image = rgb_image[0].permute(2, 1, 0)

            if return_imgs:
                pred_imgs.append(rgb_image)
                pos.append(means)
                cols.append(colors)

            loss += F.mse_loss(rgb_image, gt_img)

        if return_imgs:
            return loss, torch.stack(pred_imgs, dim=0), torch.stack(pos, dim=0), torch.stack(cols, dim=0)
        else:
            return loss
    
    def decode_gaussian_params(self, gaussian_params, scale_factor: float = 1.0):
        means, quats, scales, opacities, colors = torch.split(
            gaussian_params,
            [3, 4, 3, 1, 3], 
            dim=-1
        )

        means = F.tanh(means)
        quats = quats / (torch.norm(quats, dim=-1, keepdim=True) + 1e-8)
        scales = scale_factor * F.sigmoid(scales)
        opacities = F.sigmoid(opacities).squeeze(-1)
        colors = F.sigmoid(colors)

        return means, quats, scales, opacities, colors
    
    def rasterize(self, means, quats, scales, opacities, colors, focal_length: float = 175.0):
        device = means.device
        dtype = means.dtype

        image_width, image_height = self.image_size

        viewmats = torch.eye(4, device=device, dtype=dtype)[None]
        viewmats[:, 2, 3] = 5
        Ks = torch.tensor(
            [[[focal_length, 0.0, image_width // 2],
            [0.0, focal_length, image_height // 2],
            [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        # https://github.com/nerfstudio-project/gsplat/blob/0880d2b471e6650d458aa09fe2b2834531f6e93b/gsplat/rendering.py#L28-L54
        rgb_image, alpha, metadata = rasterization(
            means, quats, scales, opacities, colors,
            viewmats, Ks, image_width, image_height,
            camera_model="ortho", rasterize_mode="classic",
            backgrounds=torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype)
        )

        return rgb_image, alpha, metadata

# ------------------------------
# Lightning Module
# ------------------------------
class GMAELightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.gmae = GMAE(
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            encoder_dim=self.hparams.encoder_dim,
            encoder_depth=self.hparams.encoder_depth,
            encoder_heads=self.hparams.encoder_heads,
            encoder_dim_head=self.hparams.encoder_dim_head,
            num_gaussians=self.hparams.num_gaussians,
            dropout=self.hparams.dropout,
            emb_dropout=self.hparams.emb_dropout,
            decoder_dim=self.hparams.decoder_dim,
            masking_ratio=self.hparams.masking_ratio,
            decoder_depth=self.hparams.decoder_depth,
            decoder_heads=self.hparams.decoder_heads,
            decoder_dim_head=self.hparams.decoder_dim_head,
            channels=3,
            add_gauss_pos=self.hparams.add_gauss_pos
        )

    def forward(self, x):
        return self.mae(x)

    def training_step(self, batch, batch_idx):
        imgs = batch
        loss = self.gmae(imgs, self.hparams.focal_length, self.hparams.scale_factor)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch
        loss, pred_imgs, means, colors = self.gmae(imgs, self.hparams.focal_length, self.hparams.scale_factor, return_imgs=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        # For the first batch only, compute reconstructions for logging.
        if batch_idx == 0:
            # Make a grid (clamp to [0,1] for visualization)
            recon_grid = make_grid(pred_imgs.clamp(0,1), nrow=8)
            points3d = torch.cat((means, colors * 255), dim=-1).detach().cpu().numpy()
            self.logger.experiment.log({
                "reconstructed_images": [wandb.Image(recon_grid, caption="Epoch {}".format(self.current_epoch))],
                "point_clouds": [wandb.Object3D(pts) for pts in points3d]
            })
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# ------------------------------
# DataModule for CelebA
# ------------------------------
class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=4, crop_size=(176,208)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # We apply a center crop so that the resulting image size is divisible by patch size
        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        image_paths = [
            os.path.join(self.data_dir, img_path)
            for img_path in os.listdir(self.data_dir)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.dataset = CelebAImages(image_paths, transform=self.transform)
        n = len(self.dataset)
        train_size = int(0.8 * n)
        val_size = n - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

class CelebAImages(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        img = self.transform(img)
        return img

# ------------------------------
# Main function
# ------------------------------
@hydra.main(config_path="configs", config_name="gmae")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    wandb_logger = WandbLogger(project=cfg.project, name=cfg.run_name) if cfg.log else None

    dm = CelebADataModule(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        crop_size=cfg.crop_size
    )

    hparams = {
        "image_size": cfg.crop_size,
        "patch_size": cfg.patch_size,
        "encoder_dim": cfg.encoder_dim,
        "encoder_depth": cfg.encoder_depth,
        "encoder_heads": cfg.encoder_heads,
        "encoder_dim_head": cfg.encoder_dim_head,
        "dropout": cfg.dropout,
        "emb_dropout": cfg.emb_dropout,
        "decoder_dim": cfg.decoder_dim,
        "masking_ratio": cfg.masking_ratio,
        "decoder_depth": cfg.decoder_depth,
        "decoder_heads": cfg.decoder_heads,
        "decoder_dim_head": cfg.decoder_dim_head,
        "learning_rate": cfg.learning_rate,
        "num_gaussians": cfg.num_gaussians,
        "focal_length": cfg.focal_length,
        "scale_factor": cfg.scale_factor,
        "add_gauss_pos": cfg.add_gauss_pos,
    }

    model = GMAELightningModule(argparse.Namespace(**hparams))

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        devices=cfg.devices,
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
        strategy=cfg.strategy,
        precision=cfg.precision,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=cfg.log_every_n_steps,
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    main()