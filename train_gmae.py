#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from PIL import Image
import wandb
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.gmae import GMAE

from dotenv import load_dotenv
load_dotenv()

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