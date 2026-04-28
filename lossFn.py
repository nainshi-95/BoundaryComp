import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix C of shape [n, n].

    DCT-II:
        Y[k] = alpha(k) * sum_x X[x] * cos(pi / n * (x + 0.5) * k)

    alpha(0) = sqrt(1/n)
    alpha(k) = sqrt(2/n), k > 0
    """
    x = torch.arange(n, device=device, dtype=dtype).view(1, n)
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)

    mat = torch.cos(math.pi / n * (x + 0.5) * k)

    mat[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= math.sqrt(2.0 / n)

    return mat


class DCT2Loss(nn.Module):
    """
    DCT2-domain loss using pre-registered DCT matrices.

    Supports rectangular blocks, e.g.
        4x4, 4x16, 16x8, ...

    Args:
        sizes:
            Iterable of allowed 1D sizes.
            Example: [4, 8, 16]
            DCT matrices for these sizes are precomputed.

        loss_type:
            "l1", "l2", or "charbonnier"

        reduction:
            "mean", "sum", or "none"

        eps:
            epsilon for Charbonnier loss.

        device:
            Device where DCT matrices are initialized.

        dtype:
            dtype of DCT matrices.
    """

    def __init__(
        self,
        sizes: Iterable[int] = (4, 8, 16),
        loss_type: str = "l1",
        reduction: str = "mean",
        eps: float = 1e-3,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.sizes = tuple(sorted(set(int(s) for s in sizes)))
        self.loss_type = loss_type
        self.reduction = reduction
        self.eps = float(eps)

        assert self.loss_type in ("l1", "l2", "charbonnier")
        assert self.reduction in ("mean", "sum", "none")

        for s in self.sizes:
            assert s > 0 and (s & (s - 1)) == 0, f"size must be power of 2: {s}"

            mat = make_dct_matrix(s, device=device, dtype=dtype)
            self.register_buffer(f"dct_{s}", mat, persistent=False)

    def _get_dct(self, n: int, x: torch.Tensor) -> torch.Tensor:
        name = f"dct_{n}"

        if not hasattr(self, name):
            raise ValueError(
                f"DCT matrix for size {n} is not registered. "
                f"Available sizes: {self.sizes}. "
                f"Initialize with sizes including {n}."
            )

        mat = getattr(self, name)

        # Match dtype/device in case module/buffer was moved or AMP is used.
        if mat.device != x.device or mat.dtype != x.dtype:
            mat = mat.to(device=x.device, dtype=x.dtype)

        return mat

    def dct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B,C,H,W]

        Returns:
            y: [B,C,H,W], DCT2 coefficients.
        """
        assert x.ndim == 4, f"x must be [B,C,H,W], got {x.shape}"

        B, C, H, W = x.shape

        C_h = self._get_dct(H, x)  # [H,H]
        C_w = self._get_dct(W, x)  # [W,W]

        # y = C_h @ x @ C_w.T
        #
        # Height transform:
        #   x:   [B,C,H,W]
        #   C_h: [H,H]
        #   tmp[b,c,k,w] = sum_h C_h[k,h] * x[b,c,h,w]
        tmp = torch.einsum("kh,bchw->bckw", C_h, x)

        # Width transform:
        #   C_w: [W,W]
        #   out[b,c,h,k] = sum_w tmp[b,c,h,w] * C_w[k,w]
        out = torch.einsum("lw,bchw->bchl", C_w, tmp)

        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B,C,H,W]
            target: [B,C,H,W]

        Returns:
            scalar loss unless reduction="none".
        """
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        assert pred.ndim == 4

        pred_dct = self.dct2(pred)
        target_dct = self.dct2(target)

        diff = pred_dct - target_dct

        if self.loss_type == "l1":
            loss = diff.abs()
        elif self.loss_type == "l2":
            loss = diff.pow(2)
        elif self.loss_type == "charbonnier":
            loss = torch.sqrt(diff.pow(2) + self.eps * self.eps)
        else:
            raise RuntimeError("unreachable")

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss












class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")











class CharbonnierPlusDCT2Loss(nn.Module):
    def __init__(
        self,
        sizes: Sequence[int] = (4, 8, 16),
        lambda_dct: float = 0.1,
        charbonnier_eps: float = 1e-3,
        dct_eps: float = 1e-3,
        dct_loss_type: str = "charbonnier",
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.lambda_dct = float(lambda_dct)

        self.pix_loss = CharbonnierLoss(eps=charbonnier_eps, reduction="mean")
        self.dct_loss = DCT2Loss(
            sizes=sizes,
            loss_type=dct_loss_type,
            reduction="mean",
            eps=dct_eps,
            device=device,
            dtype=dtype,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss_pix = self.pix_loss(pred, target)
        loss_dct = self.dct_loss(pred, target)

        loss = loss_pix + self.lambda_dct * loss_dct

        return loss, {
            "loss_pix": loss_pix.detach(),
            "loss_dct": loss_dct.detach(),
        }


























