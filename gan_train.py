"""
gan_train.py — Training loop and synthetic image generation functions for the Conditional WGAN-GP.
"""

import os
import time
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from config import Config
from gan_model import Generator, Critic
from data_preprocessing import preprocess_image, load_dataset


# ═══════════════════════════════════════════════════════════════════════════
#  GAN DATASET (Image-only, no graphs)
# ═══════════════════════════════════════════════════════════════════════════

class GANImageDataset(Dataset):
    """
    Standard dataset that loads RGB imagery to [-1, 1] necessary for Tanh.
    """
    def __init__(self, samples, cfg: Config):
        self.samples = samples
        self.size = cfg.image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # read + resize + normalize [0, 1]
        img = preprocess_image(path, self.size)
        
        # scale to [-1, 1] for GANs
        img = (img * 2.0) - 1.0
        
        # to tensor (C, H, W)
        tensor_img = torch.tensor(img).permute(2, 0, 1).float()
        return tensor_img, label


# ═══════════════════════════════════════════════════════════════════════════
#  WGAN-GP Gradient Penalty
# ═══════════════════════════════════════════════════════════════════════════

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates, labels)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ═══════════════════════════════════════════════════════════════════════════
#  GAN TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_gan(cfg: Config, num_classes: int, device: torch.device):
    """
    Trains the WGAN-GP on the existing training images.
    """
    print("\n" + "="*60)
    print("  TRAINING CONDITIONAL WGAN-GP")
    print("="*60)
    
    # We load the entire dataset. For the GAN, we can just train on everything 
    # or just the training set. Let's use the full pool of images available.
    samples, label_map = load_dataset(cfg.data_dir)
    dataset = GANImageDataset(samples, cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.gan_batch_size, shuffle=True, drop_last=True, num_workers=2)

    gen = Generator(num_classes, cfg.gan_latent_dim, cfg.gan_hidden_dim).to(device)
    critic = Critic(num_classes, cfg.gan_hidden_dim).to(device)
    
    # WGAN-GP typically uses Adam with beta1=0.0, beta2=0.9
    opt_gen = torch.optim.Adam(gen.parameters(), lr=cfg.gan_lr, betas=(0.0, 0.9))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.gan_lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(16, cfg.gan_latent_dim, 1, 1).to(device)
    fixed_labels = torch.randint(0, num_classes, (16,)).to(device)
    
    gan_ckpt_dir = cfg.checkpoint_dir / "gan"
    gan_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, cfg.gan_epochs + 1):
        for batch_idx, (real_imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.gan_epochs}")):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            current_batch_size = real_imgs.size(0)

            # ── Train Critic ──────────────────────────────────────────────────
            for _ in range(cfg.gan_n_critic):
                opt_critic.zero_grad()
                
                # Real images Score
                critic_real = critic(real_imgs, labels).reshape(-1)
                
                # Fake images Score
                z = torch.randn(current_batch_size, cfg.gan_latent_dim, 1, 1).to(device)
                fake_imgs = gen(z, labels).detach()
                critic_fake = critic(fake_imgs, labels).reshape(-1)
                
                # Gradient Penalty
                gp = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)
                
                # Maximize Expected[Critic(Real)] - Expected[Critic(Fake)] -> minimize inverted
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + cfg.gan_lambda_gp * gp
                loss_critic.backward()
                opt_critic.step()

            # ── Train Generator ───────────────────────────────────────────────
            opt_gen.zero_grad()
            z = torch.randn(current_batch_size, cfg.gan_latent_dim, 1, 1).to(device)
            generated = gen(z, labels)
            
            # Generator wants Critic to output HIGH scores for fakes
            gen_fake = critic(generated, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            
            loss_gen.backward()
            opt_gen.step()
        
        # Save sample and checkpoint
        print(f"  Epoch [{epoch}/{cfg.gan_epochs}] Loss C: {loss_critic:.4f}, Loss G: {loss_gen:.4f}")
        
        if epoch % 5 == 0 or epoch == cfg.gan_epochs:
            torch.save({
                'gen': gen.state_dict(),
                'critic': critic.state_dict(),
            }, gan_ckpt_dir / f"gan_ckpt_epoch_{epoch}.pt")


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC IMAGE GENERATION 
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_synthetic_data(cfg: Config, num_classes: int, target_images_per_class: int, device: torch.device):
    """
    Loads custom Generator and generates samples until every class folder 
    in dataset/images/ has at least `target_images_per_class` total images.
    """
    ckpt_path = cfg.checkpoint_dir / "gan" / f"gan_ckpt_epoch_{cfg.gan_epochs}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[error] Cannot find GAN checkpoint: {ckpt_path}. Train the GAN first.")
    
    print("\n▸ SYNTHESIZING MISSING DATA with WGAN-GP")
    gen = Generator(num_classes, cfg.gan_latent_dim, cfg.gan_hidden_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['gen'])
    gen.eval()
    
    samples, label_map = load_dataset(cfg.data_dir)
    
    # Count current images per class
    counts = {name: 0 for name in label_map.values()}
    for _, l in samples:
        counts[label_map[l]] += 1
        
    for class_idx in range(num_classes):
        cls_name = label_map[class_idx]
        current_count = counts[cls_name]
        
        if current_count < target_images_per_class:
            missing = target_images_per_class - current_count
            print(f"  Synthesizing {missing:4d} images for {cls_name} ...")
            
            cls_out_dir = cfg.data_dir / cls_name
            
            # Generate in batches of 32
            generated_so_far = 0
            while generated_so_far < missing:
                bs = min(32, missing - generated_so_far)
                z = torch.randn(bs, cfg.gan_latent_dim, 1, 1).to(device)
                labels = torch.full((bs,), class_idx, dtype=torch.long).to(device)
                
                fakes = gen(z, labels) # shape (bs, 3, 256, 256), range [-1, 1]
                
                # Denormalize to [0, 255] BGR to write with CV2
                fakes = (fakes + 1) / 2.0
                fakes = fakes.permute(0, 2, 3, 1).cpu().numpy() * 255.0
                fakes = np.clip(fakes, 0, 255).astype(np.uint8)
                
                for i in range(bs):
                    img_rgb = fakes[i]
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    fname = f"synth_{generated_so_far + i:05d}.png"
                    cv2.imwrite(str(cls_out_dir / fname), img_bgr)
                
                generated_so_far += bs

    print("\n✓ Synthetic data integration complete. You can now run preprocessing to include them in the splits!")
