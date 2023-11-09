from typing import Tuple

import random

import torch
import torch.nn.functional as F

from einops import rearrange


def get_anchors_positive_and_negative_pairs(
    addresses: torch.Tensor,
    features: torch.Tensor,
    n_anchors_to_sample: int,
    n_positive_pairs: int,
    n_negative_pairs: int,
    top_k: int,
    bottom_m: int,
    use_patches: bool,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # phase: B, C, H, W
    # features: B, C, H, W

    if use_patches:
        # divide phase and magnitude maps into patches of size p x p
        assert addresses.size()[-1] % patch_size == 0, "Patch-size not a factor of output maps size!!!!"
        h_patched = w_patched = addresses.size()[-1] // patch_size 
        addresses = torch.nn.functional.unfold(addresses, kernel_size=patch_size, stride=patch_size)
        addresses = rearrange(addresses, 'b c (h w) -> b c h w', h=h_patched, w=w_patched)
        features = torch.nn.functional.unfold(features, kernel_size=patch_size, stride=patch_size)
        features = rearrange(features, 'b c (h w) -> b c h w', h=h_patched, w=w_patched)

    anchor = addresses.flatten(start_dim=-2)
    anchor = anchor.unsqueeze(-1).unsqueeze(-1)
    n_anchors_total = anchor.shape[-3]

    per_anchor_addresses = torch.unsqueeze(addresses, dim=-3)
    per_anchor_addresses = torch.repeat_interleave(per_anchor_addresses, repeats=n_anchors_total, dim=-3)

    per_anchor_features = torch.unsqueeze(features, dim=-3)
    per_anchor_features = torch.repeat_interleave(per_anchor_features, repeats=n_anchors_total, dim=-3)

    # Sample random anchors
    n_anchors_to_sample = min(n_anchors_to_sample, n_anchors_total)
    indices = random.sample(list(range(n_anchors_total)), k=n_anchors_to_sample)
    anchor = anchor[:, :, indices]
    anchor_features = features.flatten(start_dim=-2)
    anchor_features = anchor_features[:, :, indices]
    per_anchor_addresses = per_anchor_addresses[:, :, indices]
    per_anchor_features = per_anchor_features[:, :, indices]

    # anchor: B, C, Na, 1, 1
    # per_anchor_phase: B, C, Na, H, W
    n_channels = features.shape[1]
    anchor = F.normalize(anchor, dim=1)
    per_anchor_addresses = F.normalize(per_anchor_addresses, dim=1)
    delta_address = torch.sum(anchor * per_anchor_addresses, dim=1, keepdim=True)
    delta_address = 1 - delta_address

    # Average delta_address over channels, but then repeat it (such that the gather below works properly)
    delta_address = torch.repeat_interleave(delta_address, repeats=n_channels, dim=1)

    # Now sort
    delta_address_for_sorting = torch.flatten(delta_address, start_dim=-2)
    delta_address_argsort = torch.argsort(delta_address_for_sorting, dim=-1)

    # Pick 1/top-K and N-1/bottom-M
    # Note: we start at 1 because we don't want to pick the anchor itself
    positives_idcs = random.sample(list(range(1, top_k+1)), k=n_positive_pairs)
    n_phases = delta_address_argsort.shape[-1]
    negatives_idcs = random.sample(range(n_phases - bottom_m, n_phases), k=n_negative_pairs)

    per_anchor_features_flat = torch.flatten(per_anchor_features, start_dim=-2)
    positives = torch.take_along_dim(per_anchor_features_flat, indices=delta_address_argsort[..., positives_idcs], dim=-1)
    negatives = torch.take_along_dim(per_anchor_features_flat, indices=delta_address_argsort[..., negatives_idcs], dim=-1)

    return anchor_features, positives, negatives


def contrastive_soft_target_loss(anchors, positive_pairs, negative_pairs, temperature):
    # anchors: B, C, Na
    # positive_pairs: B, C, Na, Np
    # negative_pairs: B, C, Na, Nn

    device = anchors.device
    batch_size, n_anchors = anchors.shape[0], anchors.shape[-1]
    n_positive_pairs = positive_pairs.shape[-1]
    n_negative_pairs = negative_pairs.shape[-1]

    pairs_concat = torch.cat([positive_pairs, negative_pairs], dim=-1)

    # Normalize the anchors and pairs before the dot product
    anchors = F.normalize(anchors, dim=1)
    pairs_concat = F.normalize(pairs_concat, dim=1)

    logits = torch.einsum("bcA, bcAK -> bAK", anchors, pairs_concat)

    logits /= temperature
    logits = rearrange(logits, "b A K -> (b A) K")

    labels_pos = torch.ones((batch_size * n_anchors, n_positive_pairs), dtype=torch.float32, device=device)
    labels_pos /= n_positive_pairs  # from (1 1 1 1 0 0 0 0 ..) to (.25 .25 .25 .25 0 0 0 0 ..)
    labels_neg = torch.zeros((batch_size * n_anchors, n_negative_pairs), dtype=torch.float32, device=device)
    labels = torch.cat([labels_pos, labels_neg], dim=-1)
    # Reference:
    # https://timm.fast.ai/loss.cross_entropy#SoftTargetCrossEntropy
    # https://github.com/huggingface/pytorch-image-models/blob/9a38416fbdfd0d38e6922eee5d664e8ec7fbc356/timm/loss/cross_entropy.py#L29
    loss = torch.sum(-labels * F.log_softmax(logits, dim=-1), dim=-1)
    loss = loss.mean()

    return loss
