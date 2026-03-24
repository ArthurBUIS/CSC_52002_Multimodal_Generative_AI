from torch.utils.data import Dataset, DataLoader
from datasets import ChairV2Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

# ─────────────────────────────────────────────
#  SketchMapper  (Es)
# ─────────────────────────────────────────────

class SketchMapper(nn.Module):
    """
    Autoregressive Sketch-to-Latent Mapper (Es) – Section 5.1 of the paper.

    Differences vs PhotoMapper (Er / teacher):
      • Input is a sketch (grayscale → replicated to 3-ch if needed).
      • Only `num_predicted` (≤ num_latents) latent vectors are predicted;
        the remaining (num_latents_total - num_predicted) are filled with
        random Gaussian vectors to enable multi-modal generation.
      • Trained with 4 losses: Lrec + LLPIPS + Ldisc + LKD.

    Architecture is kept intentionally close to PhotoMapper so that the
    distillation loss (LKD) is well-defined in the shared latent space.
    """

    def __init__(
        self,
        latent_dim: int = 256,       # d  – StyleGAN latent dimension
        num_latents_total: int = 12,  # total W+ vectors expected by G
        num_predicted: int = 10,      # max vectors predicted (paper: 10/12)
        feature_dim: int = 2048,      # ResNet-50 last feature map channels
        hidden_dim: int = 256,        # GRU hidden size
    ):
        super().__init__()
        self.latent_dim        = latent_dim
        self.num_latents_total = num_latents_total
        self.num_predicted     = num_predicted

        # ── Backbone (shared with PhotoMapper architecture) ──────────────
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.gap               = nn.AdaptiveAvgPool2d(1)

        # ── h0 initialisation from global average-pooled feature ─────────
        self.fc_init = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )

        # ── GRU sequential decoder ────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # ── η: Hadamard(fs, w_{k-1}) → d-dim summary ─────────────────────
        self.feature_proj  = nn.Conv2d(feature_dim, latent_dim, kernel_size=1)
        self.eta_network   = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),          # → (B, latent_dim)
        )

        # ── Output projection: hidden_dim → latent_dim ───────────────────
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        sketch: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sketch    : (B, 1 or 3, H, W)
            num_steps : how many latent vectors to *predict* (1..num_predicted).
                        Defaults to self.num_predicted.
                        Remaining vectors are sampled from N(0,1).

        Returns:
            w_plus        : (B, num_latents_total, latent_dim)  – full W+ code
            predicted_only: (B, num_steps, latent_dim)          – predicted part
                            (used for LKD supervision)
        """
        if num_steps is None:
            num_steps = self.num_predicted
        assert 1 <= num_steps <= self.num_predicted, \
            f"num_steps must be in [1, {self.num_predicted}]"

        batch_size = sketch.size(0)
        device     = sketch.device

        # Greyscale sketch → 3 channels
        if sketch.size(1) == 1:
            sketch = sketch.repeat(1, 3, 1, 1)

        # ── Feature extraction ────────────────────────────────────────────
        fs      = self.feature_extractor(sketch)          # (B, C, Hf, Wf)
        fs_proj = self.feature_proj(fs)                   # (B, d, Hf, Wf)

        vh      = self.gap(fs).squeeze(-1).squeeze(-1)    # (B, feature_dim)
        h0      = self.fc_init(vh)                        # (B, hidden_dim)
        hidden  = h0.unsqueeze(0).repeat(2, 1, 1)         # (2, B, hidden_dim)

        # ── Autoregressive unrolling ──────────────────────────────────────
        predicted_latents = []
        w_prev = torch.zeros(batch_size, self.latent_dim, device=device)

        for _ in range(num_steps):
            # η(fs, w_{k-1})  –  Hadamard product in feature-map space
            w_prev_spatial = w_prev.view(batch_size, self.latent_dim, 1, 1)
            fs_interaction = fs_proj * w_prev_spatial     # (B, d, Hf, Wf)
            eta_out        = self.eta_network(fs_interaction)  # (B, d)

            # GRU step
            gru_input      = eta_out.unsqueeze(1)         # (B, 1, d)
            gru_out, hidden = self.gru(gru_input, hidden) # (B, 1, hidden)
            w_current      = self.fc_out(gru_out.squeeze(1))   # (B, d)

            predicted_latents.append(w_current)
            w_prev = w_current

        predicted_only = torch.stack(predicted_latents, dim=1)  # (B, steps, d)

        # ── Fill remaining positions with N(0,1) noise ────────────────────
        num_random = self.num_latents_total - num_steps
        if num_random > 0:
            random_latents = torch.randn(
                batch_size, num_random, self.latent_dim, device=device
            )
            w_plus = torch.cat([predicted_only, random_latents], dim=1)
        else:
            w_plus = predicted_only

        return w_plus, predicted_only
    
    
#============================================================
# =====================================
# 1. ARCHITECTURE FG-SBIR
# =====================================

class FGSBIRModel(nn.Module):
    """
    Fine-Grained Sketch-Based Image Retrieval
    Siamese network avec shared weights pour sketch et photo
    """
    
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Backbone ResNet50
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
        )
    
    def forward_sketch(self, sketch):
        """Encode sketch"""
        # Si grayscale, répéter sur 3 channels
        if sketch.size(1) == 1:
            sketch = sketch.repeat(1, 3, 1, 1)
        
        features = self.encoder(sketch)
        features = features.squeeze(-1).squeeze(-1)
        embedding = self.projector(features)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward_photo(self, photo):
        """Encode photo"""
        features = self.encoder(photo)
        features = features.squeeze(-1).squeeze(-1)
        embedding = self.projector(features)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x):
        """Forward flexible (détecte sketch vs photo)"""
        if x.size(1) == 1:
            return self.forward_sketch(x)
        else:
            return self.forward_photo(x)

print("✅ FGSBIRModel défini")

#============================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Pas de stop
        else:
            self.counter += 1
            print(f"   ⏳ Early stopping: {self.counter}/{self.patience}")
            return self.counter >= self.patience  # Stop si patience dépassée
        
#============================================================

# ============================================================
# PHOTO-TO-PHOTO MAPPER (Teacher)
# ============================================================

class PhotoMapper(nn.Module):
    """
    Photo-to-Photo Mapper (Er) - Identique à SketchMapper mais pour photos
    Utilisé comme Teacher pour la distillation
    """
    
    def __init__(
        self,
        latent_dim=256,
        num_latents=12,
        feature_dim=2048,
        hidden_dim=256
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-2]
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc_init = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.feature_proj = nn.Conv2d(feature_dim, latent_dim, kernel_size=1)
        
        self.eta_network = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.fc_out = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, photo):
        """Identique à SketchMapper mais pour photos"""
        batch_size = photo.size(0)
        device = photo.device
        
        # Feature extraction
        fs = self.feature_extractor(photo)
        fs_proj = self.feature_proj(fs)
        
        vh = self.gap(fs).squeeze(-1).squeeze(-1)
        h0 = self.fc_init(vh)
        hidden = h0.unsqueeze(0).repeat(2, 1, 1)
        
        # Prédire TOUS les 14 latents (pas de random pour teacher)
        predicted_latents = []
        w_prev = torch.zeros(batch_size, self.latent_dim).to(device)
        
        for step in range(self.num_latents):
            w_prev_expanded = w_prev.view(batch_size, self.latent_dim, 1, 1)
            fs_interaction = fs_proj * w_prev_expanded
            eta_out = self.eta_network(fs_interaction)
            
            gru_input = eta_out.unsqueeze(1)
            gru_out, hidden = self.gru(gru_input, hidden)
            
            w_current = self.fc_out(gru_out.squeeze(1))
            predicted_latents.append(w_current)
            w_prev = w_current
        
        w_plus = torch.stack(predicted_latents, dim=1)
        return w_plus
