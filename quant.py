import math
from typing import Iterable, Optional, Tuple, Dict

import torch
import torch.nn as nn


def ste_replace(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return x + (x_hat - x).detach()


def floor_log2_int(x: int) -> int:
    assert x > 0
    return int(math.floor(math.log2(x)))


def clip3(x: torch.Tensor, min_val: int, max_val: int) -> torch.Tensor:
    return torch.clamp(x, min=float(min_val), max=float(max_val))


def make_orthonormal_dct2_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix.

    Forward:
        coeff = C @ x

    Inverse:
        x = C.T @ coeff
    """
    assert n > 0 and (n & (n - 1)) == 0

    x = torch.arange(n, device=device, dtype=dtype).view(1, n)
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)

    mat = torch.cos(math.pi / n * (x + 0.5) * k)

    mat[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= math.sqrt(2.0 / n)

    return mat


class VVCInterDCT2QuantIDCT2STEStable(nn.Module):
    """
    Stable codec-like differentiable DCT2/Q/DQ/IDCT2 module.

    This version intentionally uses orthonormal DCT2 instead of VTM integer-core
    transform shifts, because the previous integer-shift approximation can make
    qcoeff and dequant scale inconsistent, causing all dequantized coefficients
    to become zero.

    Scope:
        - luma/inter residual
        - DCT2 only
        - no transform skip
        - no MTS/LFNST/sub-block transform
        - no scaling list
        - no RDOQ/sign hiding
        - normalized input domain
    """

    QUANT_SCALES = {
        0: [26214, 23302, 20560, 18396, 16384, 14564],
        1: [18396, 16384, 14564, 13107, 11651, 10280],
    }

    INV_QUANT_SCALES = {
        0: [40, 45, 51, 57, 64, 72],
        1: [57, 64, 72, 80, 90, 102],
    }

    COEFF_SHIFT_ARRAY = [
        0, 63, 31, 21, 15, 12, 10, 9, 7, 7, 6, 5, 5, 4, 4, 4,
        3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]

    def __init__(
        self,
        sizes: Iterable[int] = (4, 8, 16, 32),
        qp: int = 32,
        bit_depth: int = 10,
        use_sqrt_adjustment: bool = True,
        use_dead_zone_quant: bool = True,
        use_qabs64_correction: bool = True,
        use_ste: bool = True,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.sizes = tuple(sorted(set(int(s) for s in sizes)))
        self.qp = int(qp)
        self.bit_depth = int(bit_depth)
        self.max_val = float((1 << bit_depth) - 1)

        self.use_sqrt_adjustment = bool(use_sqrt_adjustment)
        self.use_dead_zone_quant = bool(use_dead_zone_quant)
        self.use_qabs64_correction = bool(use_qabs64_correction)
        self.use_ste = bool(use_ste)

        for s in self.sizes:
            mat = make_orthonormal_dct2_matrix(s, device=device, dtype=dtype)
            self.register_buffer(f"dct2_{s}", mat, persistent=False)

        coeff_shift = torch.tensor(self.COEFF_SHIFT_ARRAY, device=device, dtype=dtype)
        self.register_buffer("coeff_shift_array", coeff_shift, persistent=False)

    def _get_mat(self, n: int, x: torch.Tensor) -> torch.Tensor:
        name = f"dct2_{n}"
        if not hasattr(self, name):
            raise ValueError(f"DCT2 matrix for size {n} is not registered. Available: {self.sizes}")
        return getattr(self, name).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def needs_sqrt_adjustment(h: int, w: int) -> bool:
        area = h * w
        is_power_of_two = (area & (area - 1)) == 0
        log2_area = floor_log2_int(area)
        is_power_of_four = is_power_of_two and ((log2_area % 2) == 0)
        return not is_power_of_four

    def qp_per_rem(self, qp: Optional[int] = None) -> Tuple[int, int]:
        if qp is None:
            qp = self.qp
        qp = int(qp)
        return qp // 6, qp % 6

    def dct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W]
        coeff = C_h @ x @ C_w.T
        """
        B, C, H, W = x.shape
        Ch = self._get_mat(H, x)
        Cw = self._get_mat(W, x)

        tmp = torch.einsum("kh,bchw->bckw", Ch, x)
        coeff = torch.einsum("lw,bchw->bchl", Cw, tmp)
        return coeff

    def idct2(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        coeff: [B,C,H,W]
        x = C_h.T @ coeff @ C_w
        """
        B, C, H, W = coeff.shape
        Ch = self._get_mat(H, coeff)
        Cw = self._get_mat(W, coeff)

        tmp = torch.einsum("hk,bckw->bchw", Ch, coeff)
        x = torch.einsum("wl,bchw->bchl", Cw, tmp)
        return x

    def get_qstep_like_scale(self, H: int, W: int, qp: Optional[int] = None):
        """
        Stable scalar quant scale for orthonormal DCT coefficient.

        Instead of VTM integer iQBits/rightShift coupling, use a consistent
        effective step derived from inv quant scale and qp_per.

        This is not bit-exact, but avoids qcoeff nonzero -> dequant zero collapse.
        """
        qp_per, qp_rem = self.qp_per_rem(qp)

        need_sqrt = self.needs_sqrt_adjustment(H, W) if self.use_sqrt_adjustment else False
        scale_idx = 1 if need_sqrt else 0

        inv_scale = float(self.INV_QUANT_SCALES[scale_idx][qp_rem])

        # Effective coefficient step.
        # VVC invQuant scale grows with 2^qp_per.
        qstep = inv_scale * float(1 << qp_per) / 64.0

        return qstep, scale_idx, qp_per, qp_rem

    def quantize(self, coeff: torch.Tensor, qp: Optional[int] = None) -> torch.Tensor:
        """
        Quantize orthonormal DCT coefficient using a stable codec-like scalar step.

        q_abs = floor(abs(coeff) / qstep + offset)

        For inter dead-zone, use offset around 85/512.
        For round-to-nearest, use 0.5.
        """
        _, _, H, W = coeff.shape

        qstep, _, _, _ = self.get_qstep_like_scale(H, W, qp)
        qstep_t = coeff.new_tensor(qstep)

        if self.use_dead_zone_quant:
            offset = 85.0 / 512.0
        else:
            offset = 0.5

        q_abs_real = coeff.abs() / qstep_t + offset
        q_abs_hat = torch.floor(q_abs_real)

        q_hat = q_abs_hat * coeff.sign()

        if self.use_ste:
            q_real_signed = q_abs_real * coeff.sign()
            q = ste_replace(q_real_signed, q_hat)
        else:
            q = q_hat

        return q

    def dequantize(self, qcoeff: torch.Tensor, qp: Optional[int] = None) -> torch.Tensor:
        """
        Dequantize qcoeff back to orthonormal DCT coefficient domain.

        Includes qAbs < 64 interpolation correction in a scale-consistent way.
        """
        _, _, H, W = qcoeff.shape

        qstep, _, _, _ = self.get_qstep_like_scale(H, W, qp)
        qstep_t = qcoeff.new_tensor(qstep)

        q_abs = qcoeff.abs()
        q_sgn = torch.where(qcoeff < 0, -torch.ones_like(qcoeff), torch.ones_like(qcoeff))

        # Base dequant.
        coeff_abs = q_abs * qstep_t

        if self.use_qabs64_correction:
            small = q_abs < 64

            q_abs2 = q_abs + 1.0
            coeff_abs2 = q_abs2 * qstep_t

            q_abs_idx = torch.clamp(q_abs.detach().long(), 0, 63)
            coef_shift = self.coeff_shift_array.to(device=qcoeff.device, dtype=qcoeff.dtype)[q_abs_idx]

            coeff_abs_corr = ((1024.0 - coef_shift) * coeff_abs + coef_shift * coeff_abs2) / 1024.0
            coeff_abs_hat = torch.where(small, coeff_abs_corr, coeff_abs)
        else:
            coeff_abs_hat = coeff_abs

        coeff_hat = coeff_abs_hat * q_sgn

        if self.use_ste:
            coeff_real = coeff_abs * q_sgn
            coeff = ste_replace(coeff_real, coeff_hat)
        else:
            coeff = coeff_hat

        return coeff

    def forward(
        self,
        residual_norm: torch.Tensor,
        qp: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        assert residual_norm.ndim == 4
        _, _, H, W = residual_norm.shape

        if H not in self.sizes or W not in self.sizes:
            raise ValueError(f"Unsupported size {H}x{W}. Registered sizes: {self.sizes}")

        residual_int = residual_norm * self.max_val

        coeff = self.dct2(residual_int)
        qcoeff = self.quantize(coeff, qp=qp)
        coeff_hat = self.dequantize(qcoeff, qp=qp)
        rec_residual_int = self.idct2(coeff_hat)

        rec_residual_norm = rec_residual_int / self.max_val

        aux = {
            "residual_int": residual_int,
            "coeff": coeff,
            "qcoeff": qcoeff,
            "coeff_hat": coeff_hat,
            "rec_residual_int": rec_residual_int,
            "q_nonzero_ratio": (qcoeff.abs() > 0).float().mean().detach(),
            "q_abs_mean": qcoeff.abs().mean().detach(),
        }

        return rec_residual_norm, aux
