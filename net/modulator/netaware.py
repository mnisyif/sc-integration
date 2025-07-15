import torch
import torch.nn as nn
from typing import Tuple, Union

class NetAwareMod(nn.Module):
    """
    y' (B,N,C)  +  bandwidth, latency  ➜  y_masked, mask, k_per_sample
    """

    def __init__(self, policy, *, auto_unit: bool = True):
        """
        auto_unit=True  → heuristically converts
                          • bit/s ➜ Mb/s  (if B > 1.2×B_max)
                          • seconds ➜ ms  (if L < 0.5)
        """
        super().__init__()
        self.policy     = policy
        self.auto_unit  = auto_unit

    # ------------------------------------------------------------------ #
    @staticmethod
    def _topk_mask(one_latent: torch.Tensor, k: int) -> torch.Tensor:
        """
        one_latent : (N, C)   (single sample)
        returns    : (1, C) bool mask
        """
        importance = one_latent.abs().sum(dim=-2)            # (C,)
        _, idx = importance.topk(k, largest=True)
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask[idx] = True
        return mask.unsqueeze(0)                             # (1, C)

    # ------------------------------------------------------------------ #
    def forward( self, y_prime: torch.Tensor,B, L) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        y_masked : (B, N, C)
        mask     : (B, 1, C)  bool
        k_vec    : (B,)       int tensor
        """
        # ---------- make B, L 1-D CPU tensors --------------------------
        B_vec = torch.as_tensor(B, dtype=torch.float32).flatten().cpu()
        L_vec = torch.as_tensor(L, dtype=torch.float32).flatten().cpu()
        assert len(B_vec) == len(y_prime), "B vector must match batch size"
        assert len(L_vec) == len(y_prime), "L vector must match batch size"

        # ---------- auto unit conversion -------------------------------
        if self.auto_unit:
            if B_vec.max() > self.policy.B_max * 1.2:        # looks like bit/s
                B_vec /= 1e6                                 # → Mb/s
            if L_vec.max() < 1.0:                            # looks like seconds
                L_vec *= 1e3                                 # → ms

        # ---------- compute k per sample -------------------------------
        k_list = [
            self.policy(float(b), float(l))
            for b, l in zip(B_vec.tolist(), L_vec.tolist())
        ]
        k_vec = torch.tensor(k_list, device=y_prime.device, dtype=torch.int)

        # ---------- build masks & apply -------------------------------
        Bsz, N, C = y_prime.shape
        masks = []
        for i in range(Bsz):
            mask_i = self._topk_mask(y_prime[i], int(k_vec[i]))   # (1,C)
            masks.append(mask_i)

        mask = torch.stack(masks, dim=0)          # (B, 1, C)
        y_masked = y_prime * mask                 # broadcast (B,N,C)

        return y_masked, mask, k_vec
