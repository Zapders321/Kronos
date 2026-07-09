"""
Contrastive Learning for Kronos — teaches the model what "similar patterns" look like.

The model learns embeddings where:
  - Positive pairs (same window + small noise) are close together
  - Negative pairs (different windows / regimes) are far apart

This creates a latent space that naturally clusters by market regime,
making the prediction head's job much easier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def augment_ohlcv(x, noise_std=0.005, scale_jitter=0.02, time_warp=0):
    """
    Create an augmented view of price data for contrastive learning.
    Positive pairs should be semantically similar — small realistic perturbations.

    Args:
        x: [batch_size, seq_len, n_features] — OHLCV + indicators
        noise_std: Gaussian noise std (normalized data)
        scale_jitter: random scaling factor
        time_warp: number of frames to shift (0 = disabled)

    Returns:
        augmented x of same shape
    """
    aug = x.clone()

    # 1. Gaussian noise (tiny — doesn't change the pattern, just the exact values)
    aug = aug + torch.randn_like(aug) * noise_std

    # 2. Random scaling (changes magnitude but not direction)
    scale = 1.0 + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * scale_jitter
    aug = aug * scale

    # 3. Time warp (shift the sequence by 1-2 frames)
    if time_warp > 0 and torch.rand(1).item() < 0.3:
        shift = torch.randint(-time_warp, time_warp + 1, (1,)).item()
        if shift != 0:
            if shift > 0:
                aug[:, :-shift] = aug[:, shift:]  # shift right
                aug[:, -shift:] = aug[:, -shift:]  # pad with last values
            else:
                aug[:, -shift:] = aug[:, :shift]   # shift left
                aug[:, :shift] = aug[:, :shift]     # pad with first values

    return aug


def nt_xent_loss(embeddings, temperature=0.15):
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy).
    Used in SimCLR — pulls positive pairs together, pushes negatives apart.

    Args:
        embeddings: [batch_size * 2, d_model] — first half original, second half augmented
        temperature: scaling temperature

    Returns:
        Scalar loss
    """
    batch_size = embeddings.shape[0] // 2

    # Normalize embeddings to unit sphere
    z = F.normalize(embeddings, dim=-1)

    # Similarity matrix
    sim = torch.matmul(z, z.T) / temperature

    # Labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size, device=embeddings.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # Remove self-similarity (diagonal)
    mask = torch.eye(batch_size * 2, device=embeddings.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Cross-entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss


def compute_contrastive_loss(model, x, xs, tokenizer, device, temperature=0.15):
    """
    Compute NT-Xent loss on the model's encoder embeddings.

    Args:
        model: Kronos model (needs to expose encoder output)
        x: [batch_size, seq_len, n_features]
        xs: time features
        tokenizer: KronosTokenizer
        device: torch device

    Returns:
        contrastive_loss (scalar)
    """
    with torch.no_grad():
        # Encode original data to token IDs
        t0t_orig, t1t_orig = tokenizer.encode(x, half=True)

    # Create augmented view
    x_aug = augment_ohlcv(x)
    with torch.no_grad():
        t0t_aug, t1t_aug = tokenizer.encode(x_aug, half=True)

    # Concatenate original and augmented for a single forward pass
    t0t = torch.cat([t0t_orig, t0t_aug], dim=0)
    t1t = torch.cat([t1t_orig, t1t_aug], dim=0)
    xs_combined = torch.cat([xs, xs], dim=0)

    # Forward through model
    logits = model(t0t[:, :-1], t1t[:, :-1], xs_combined[:, :-1, :])

    # Extract encoder embeddings
    # The model's first output logit contains the encoded representation
    # We average over the sequence to get a pooled embedding
    enc = logits[0]  # [batch*2, seq_len-1, d_model]

    # Pool: mean over sequence
    pooled = enc.mean(dim=1)  # [batch*2, d_model]

    # NT-Xent loss
    loss = nt_xent_loss(pooled, temperature)

    return loss