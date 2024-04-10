#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/4/9 15:16
# @Author : ZhangKuo
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
from tqdm import tqdm

from Model import GeneratorResNet, Discriminator
from util import get_transforms

# Define writer
writer = SummaryWriter("logs")

# Define Loss
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Initialize Generator and Discriminator
G_AB = GeneratorResNet(3, num_residual_blocks=9)
D_B = Discriminator()

G_BA = GeneratorResNet(3, num_residual_blocks=9)
D_A = Discriminator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
G_AB.to(device)
D_B.to(device)
G_BA.to(device)
D_A.to(device)
criterion_GAN.to(device)
criterion_cycle.to(device)
criterion_identity.to(device)

# Initialize Optimizer
lr = 0.0002
b1 = 0.5
b2 = 0.999
optimizer_G = torch.optim.AdamW(
    list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_A = torch.optim.AdamW(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.AdamW(D_B.parameters(), lr=lr, betas=(b1, b2))

# Learning Rate Scheduler Setting
n_epochs = 200
delay = 20
lambda_func = lambda epoch: 1.0 - max(0, epoch - delay) / (n_epochs - delay)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_func)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=lambda_func
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=lambda_func
)


# Define Dataset
class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img) if self.transform else img
        return img


photo_dir = "data/photo"
monet_dir = "data/monet"
photo_dataset = Dataset(photo_dir, get_transforms())
monet_dataset = Dataset(monet_dir, get_transforms())
BATCH_SIZE = 8
monet_dataloader = DataLoader(monet_dataset, batch_size=8, shuffle=True)
photo_dataloader = DataLoader(photo_dataset, batch_size=8, shuffle=True)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def sample_images(real_A, real_B, epoch, figside=4):
    assert (
        real_A.size() == real_B.size()
    ), "The image size for two domains must be the same"

    G_AB.eval()
    G_BA.eval()

    real_A = real_A.type(Tensor)
    fake_B = G_AB(real_A).detach()
    real_B = real_B.type(Tensor)
    fake_A = G_BA(real_B).detach()

    nrows = real_A.size(0)
    real_A = make_grid(real_A, nrow=nrows, normalize=True)
    fake_B = make_grid(fake_B, nrow=nrows, normalize=True)
    real_B = make_grid(real_B, nrow=nrows, normalize=True)
    fake_A = make_grid(fake_A, nrow=nrows, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1).cpu().permute(1, 2, 0)

    plt.figure(figsize=(figside * nrows, figside * 4))
    plt.imshow(image_grid)
    plt.axis("off")
    plt.savefig(f"images/{epoch}.png")


# Training
for epoch in range(n_epochs):
    assert len(photo_dataloader) == len(
        monet_dataloader
    ), "The length of two dataloaders must be the same"
    process_bar = tqdm(
        enumerate(zip(photo_dataloader, monet_dataloader)),
        total=len(photo_dataloader),
        desc=f"Epoch {epoch}",
    )
    for i, (photo, monet) in process_bar:
        photo = photo.to(device)
        monet = monet.to(device)
        fid = FrechetInceptionDistance(feature=64, normalize=True)
        fid.to(device)

        # Adversarial ground truths
        valid = torch.ones((photo.size(0), 1)).to(device)
        fake = torch.zeros((photo.size(0), 1)).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(photo), photo)
        loss_id_B = criterion_identity(G_AB(monet), monet)

        # GAN loss
        fake_B = G_AB(photo)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

        fake_A = G_BA(monet)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        # Cycle loss
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, photo)

        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, monet)

        # Total loss
        loss_G = (
            loss_id_A
            + loss_id_B
            + loss_GAN_AB
            + loss_GAN_BA
            + loss_cycle_A
            + loss_cycle_B
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(photo), valid)

        # Fake loss
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)

        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(monet), valid)

        # Fake loss
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        writer.add_scalar("loss_G", loss_G, epoch)
        writer.add_scalar("loss_D_A", loss_D_A, epoch)
        writer.add_scalar("loss_D_B", loss_D_B, epoch)
        writer.add_scalar("loss_id_A", loss_id_A, epoch)
        writer.add_scalar("loss_id_B", loss_id_B, epoch)
        writer.add_scalar("loss_GAN_AB", loss_GAN_AB, epoch)
        writer.add_scalar("loss_GAN_BA", loss_GAN_BA, epoch)
        writer.add_scalar("loss_cycle_A", loss_cycle_A, epoch)
        writer.add_scalar("loss_cycle_B", loss_cycle_B, epoch)

        # fake_B = fake_B.to(torch.uint8)
        # monet = monet.to(torch.uint8)
        fid.update(fake_B, real=False)
        fid.update(monet, real=True)
        fid_value = fid.compute()
        writer.add_scalar("FID", fid_value, epoch)

        process_bar.set_postfix(
            loss_G=loss_G.item(),
            loss_D_A=loss_D_A.item(),
            loss_D_B=loss_D_B.item(),
            loss_id_A=loss_id_A.item(),
            loss_id_B=loss_id_B.item(),
            loss_GAN_AB=loss_GAN_AB.item(),
            loss_GAN_BA=loss_GAN_BA.item(),
            loss_cycle_A=loss_cycle_A.item(),
            loss_cycle_B=loss_cycle_B.item(),
            FID=fid_value,
        )

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if epoch % 10 == 0:
        torch.save(G_AB.state_dict(), f"checkpoints/G_AB_{epoch}.pth")
        torch.save(G_BA.state_dict(), f"checkpoints/G_BA_{epoch}.pth")
        torch.save(D_A.state_dict(), f"checkpoints/D_A_{epoch}.pth")
        torch.save(D_B.state_dict(), f"checkpoints/D_B_{epoch}.pth")
        read_A = next(iter(photo_dataloader))
        read_B = next(iter(monet_dataloader))
        sample_images(read_A, read_B, epoch)
writer.close()