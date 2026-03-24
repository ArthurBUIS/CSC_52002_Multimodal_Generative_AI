from torch.utils.data import Dataset, DataLoader
from datasets import ChairV2Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

# ============================================================
# SKETCH MAPPER LOSS AVEC CLIP - VERSION CORRIGÉE
# ============================================================

class SketchMapperLoss(nn.Module):
    """
    Ltotal = λ1·Lrec + λ2·LLPIPS + λ3·Ldisc + λ4·LKD

    Paper default weights: λ1=1, λ2=0.8, λ3=0.5, λ4=0.6

    Args:
        lpips_net   : a callable (r, r_hat) → scalar LPIPS loss.
                      Pass e.g. ``lpips.LPIPS(net='vgg').to(device)``.
        fgsbir_model: pre-trained FGSBIRModel with forward_sketch /
                      forward_photo methods.
        lambda_*    : loss weights from the paper.
    """

    def __init__(
        self,
        lpips_net,
        fgsbir_model: nn.Module,
        lambda_rec  : float = 1.0,
        lambda_lpips: float = 0.8,
        lambda_disc : float = 0.5,
        lambda_kd   : float = 0.6,
    ):
        super().__init__()
        self.lpips       = lpips_net
        self.fgsbir      = fgsbir_model
        self.lambda_rec   = lambda_rec
        self.lambda_lpips = lambda_lpips
        self.lambda_disc  = lambda_disc
        self.lambda_kd    = lambda_kd

    def forward(
        self,
        sketch:          torch.Tensor,   # (B, 1|3, H, W)  – input sketch
        photo_real:      torch.Tensor,   # (B, 3,   H, W)  – ground-truth photo
        photo_generated: torch.Tensor,   # (B, 3,   H, W)  – G(Es(s))
        predicted_latents: torch.Tensor, # (B, m,   d)     – Es predicted part
        teacher_latents:   torch.Tensor, # (B, m,   d)     – Er predicted part (frozen)
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with individual losses and 'total'.
        """
        losses = {}

        # ── Lrec : pixel-level L2 ─────────────────────────────────────────
        losses['rec'] = F.mse_loss(photo_generated, photo_real)

        # ── LLPIPS : perceptual loss ──────────────────────────────────────
        # lpips expects images in [-1, 1]; adjust if yours are in [0, 1]
        losses['lpips'] = self.lpips(photo_generated, photo_real).mean()

        # ── Ldisc : fine-grained discriminative loss (Eq. 6) ─────────────
        # Cosine distance between sketch embedding and generated photo emb.
        with torch.no_grad():
            # Sketch embedding (FGSBIRModel handles grayscale internally)
            emb_sketch = self.fgsbir.forward_sketch(sketch)
        emb_generated  = self.fgsbir.forward_photo(photo_generated)
        # 1 - cosine_similarity  (already L2-normalised in FGSBIRModel)
        losses['disc'] = (1.0 - (emb_sketch * emb_generated).sum(dim=1)).mean()

        # ── LKD : distillation from photo-mapper teacher (Eq. 7) ─────────
        # Only on the predicted latents, not the random ones
        losses['kd'] = F.mse_loss(predicted_latents, teacher_latents.detach())

        # ── Ltotal ────────────────────────────────────────────────────────
        losses['total'] = (
            self.lambda_rec   * losses['rec']
          + self.lambda_lpips * losses['lpips']
          + self.lambda_disc  * losses['disc']
          + self.lambda_kd    * losses['kd']
        )

        return losses

    
# ============================================================
# LOSS
# ============================================================

class PhotoMapperLoss(nn.Module):
    """Loss pour Photo Mapper: reconstruction + perceptual"""
    
    def __init__(self):
        super().__init__()
        
        # LPIPS loss (perceptual)
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        
        # Freeze LPIPS
        for param in self.lpips.parameters():
            param.requires_grad = False
    
    def forward(self, photo_input, photo_reconstructed):
        """
        Args:
            photo_input: Photo d'entrée (GT)
            photo_reconstructed: G(Er(photo_input))
        """
        # L2 reconstruction
        loss_rec = torch.nn.functional.mse_loss(photo_reconstructed, photo_input)
        
        # LPIPS perceptual
        loss_lpips = self.lpips(photo_reconstructed, photo_input).mean()
        
        # Combiné
        loss_total = loss_rec + 0.8 * loss_lpips
        
        return loss_total, {
            'rec': loss_rec.item(),
            'lpips': loss_lpips.item(),
        }


# =====================================
# 2. TRIPLET LOSS
# =====================================
class TripletLoss(nn.Module):
    """
    Triplet Loss avec mining
    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Embeddings sketch (batch, dim)
            positive: Embeddings photo paired (batch, dim)
            negative: Embeddings photo unpaired (batch, dim)
        """
        # Distances euclidiennes
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Triplet loss
        loss = F.relu(dist_pos - dist_neg + self.margin)
        
        return loss.mean(), {
            'triplet': loss.mean().item(),
            'dist_pos': dist_pos.mean().item(),
            'dist_neg': dist_neg.mean().item()
        }

print("✅ TripletLoss définie")