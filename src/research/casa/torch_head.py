"""
casa.torch_head -- a torch ``nn.Module`` wrapper around certified sparsification.

Certification runs in numpy/float64 on detached tensors (a *decision* does not need
gradients), then the output is recomputed in torch over the retained tokens so that
autograd flows through the values that were actually used.  Gradients with respect
to dropped tokens are zero by construction -- that is the honest cost of pruning and
is why ``eps`` should be small when the head is used inside training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .bounds import DEFAULT_SCHEME, certified_pattern
from .certificate import Certificate, digest_array, digest_indices

__all__ = ["CertifiedAttentionHead"]


class CertifiedAttentionHead(nn.Module):
    """Single-head attention that can prove how far its sparse path drifted."""

    def __init__(
        self,
        block_size: int = 16,
        eps: float = 0.05,
        scheme: str = DEFAULT_SCHEME,
        reference: str = "global_mean",
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.eps = float(eps)
        self.scheme = scheme
        self.reference = reference

    def forward(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        return_certificate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Certificate]]:
        if q.dim() == 1:
            q = q.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if q.shape[0] != 1:
            raise ValueError("CertifiedAttentionHead supports one query at a time")

        q_np = q.detach().to(torch.float64).cpu().numpy().ravel()
        K_np = K.detach().to(torch.float64).cpu().numpy()
        V_np = V.detach().to(torch.float64).cpu().numpy()

        pattern = certified_pattern(
            q_np, K_np, V_np, block_size=self.block_size, eps=self.eps, scheme=self.scheme,
            reference=self.reference,
        )
        kept = np.asarray(pattern.kept, dtype=np.int64)
        idx = torch.as_tensor(kept, dtype=torch.long, device=K.device)
        K_sel, V_sel = K[idx], V[idx]

        s = (K_sel.to(torch.float64) @ q.to(torch.float64).ravel()) / (K.shape[1] ** 0.5)
        w = torch.softmax(s, dim=0)
        out = (w.unsqueeze(-1) * V_sel.to(torch.float64)).sum(0).to(V.dtype)

        cert = None
        if return_certificate:
            cert = Certificate(
                eps=self.eps,
                error_bound=float(pattern.error_bound),
                tau_bound=float(pattern.tau_bound),
                n_total=int(K.shape[0]),
                n_retained=int(kept.size),
                n_dropped=int(K.shape[0] - kept.size),
                block_size=self.block_size,
                blocks_kept=int(pattern.blocks_kept),
                scheme=self.scheme,
                reference=self.reference,
                max_kept_score=float(pattern.max_kept_score),
                D_kept=float(pattern.D_kept),
                D_all=float(pattern.D_all),
                dropped_logmass=float(pattern.dropped_logmass),
                retained=[int(i) for i in kept],
                retained_digest=digest_indices(kept),
                q_digest=digest_array(q_np),
                k_digest=digest_array(K_np),
                v_digest=digest_array(V_np),
                feasible=bool(pattern.feasible),
                fallback=not pattern.feasible,
            )
        return (out.unsqueeze(0) if squeeze else out), cert
