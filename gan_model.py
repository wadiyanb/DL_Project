"""
gan_model.py — Conditional WGAN-GP Architecture for synthesizing 256x256 images.
"""

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
#  GENERATOR (Conditional)
# ═══════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Takes latent noise (z) and class label embedding and generates 256x256 RGB image.
    Uses Transposed Convolutions.
    """
    def __init__(self, num_classes: int, latent_dim: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        
        # We embed the class label into a 50-dimensional vector
        # Then concatenate it with latent z
        self.label_emb = nn.Embedding(num_classes, 50)
        
        in_dim = latent_dim + 50
        
        # Input to first layer goes from (in_dim, 1, 1) -> (hidden_dim*8, 4, 4)
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, hidden_dim * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 16),
            nn.ReLU(True)
        )
        # Dimensions: (hidden_dim*16) x 4 x 4

        self.upsample = nn.Sequential(
            # -> (hidden_dim*8) x 8 x 8
            self._block(hidden_dim * 16, hidden_dim * 8, 4, 2, 1),
            # -> (hidden_dim*4) x 16 x 16
            self._block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            # -> (hidden_dim*2) x 32 x 32
            self._block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            # -> (hidden_dim) x 64 x 64
            self._block(hidden_dim * 2, hidden_dim, 4, 2, 1),
            # -> (hidden_dim//2) x 128 x 128
            self._block(hidden_dim, hidden_dim // 2, 4, 2, 1),
            # -> 3 x 256 x 256
            nn.ConvTranspose2d(hidden_dim // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, z, labels):
        # z: (batch, latent_dim, 1, 1)
        # labels: (batch,)
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3) # (batch, 50, 1, 1)
        x = torch.cat([z, c], dim=1)                         # (batch, latent_dim+50, 1, 1)
        x = self.init_block(x)                               # (batch, ... , 4, 4)
        return self.upsample(x)                              # (batch, 3, 256, 256)


# ═══════════════════════════════════════════════════════════════════════════
#  CRITIC (Discriminator) (Conditional)
# ═══════════════════════════════════════════════════════════════════════════

class Critic(nn.Module):
    """
    Takes 256x256 RGB image and class label and outputs real/fake score.
    WGAN-GP uses a Critic rather than a Discriminator (no sigmoid at the end).
    """
    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        
        # We embed the label into a full 256x256 channel explicitly
        self.label_emb = nn.Embedding(num_classes, 256 * 256)
        
        # Input is 3 image channels + 1 label channel = 4 channels
        # Note: No BatchNorm in Critic for WGAN-GP; InstanceNorm is allowed but plain layers are standard.
        self.model = nn.Sequential(
            # Input: 4 x 256 x 256 -> hidden_dim/2 x 128 x 128
            nn.Conv2d(4, hidden_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> hidden_dim x 64 x 64
            self._block(hidden_dim // 2, hidden_dim, 4, 2, 1),
            # -> (hidden_dim*2) x 32 x 32
            self._block(hidden_dim, hidden_dim * 2, 4, 2, 1),
            # -> (hidden_dim*4) x 16 x 16
            self._block(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            # -> (hidden_dim*8) x 8 x 8
            self._block(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            # -> (hidden_dim*16) x 4 x 4
            self._block(hidden_dim * 8, hidden_dim * 16, 4, 2, 1),
            
            # Output: 1 scalar score per image
            nn.Conv2d(hidden_dim * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_c, affine=True), # Use InstanceNorm to keep independent batch gradients for WGAN-GP
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, images, labels):
        # images: (batch, 3, 256, 256)
        # labels: (batch,)
        batch_size = images.size(0)
        
        c = self.label_emb(labels).view(batch_size, 1, 256, 256) # (batch, 1, 256, 256)
        x = torch.cat([images, c], dim=1)                        # (batch, 4, 256, 256)
        
        out = self.model(x)                                      # (batch, 1, 1, 1)
        return out.view(batch_size, -1)                          # (batch, 1)
