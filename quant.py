import math
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn


# ============================================================
# STE helper
# ============================================================

def ste_replace(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Forward:
        x_hat
    Backward:
        identity gradient w.r.t. x

    Standard STE:
        x + (x_hat - x).detach()
    """
    return x + (x_hat - x).detach()


def floor_log2_int(x: int) -> int:
    assert x > 0
    return int(math.floor(math.log2(x)))


# ============================================================
# Orthonormal DCT-II matrix
# ============================================================

def make_orthonormal_dct2_matrix(
    n: int,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix C, shape [n,n].

    Forward:
        coeff = C @ x

    Inverse:
        x = C.T @ coeff

    This avoids VTM integer-core shift/scale instability.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of two, got {n}"

    x = torch.arange(n, device=device, dtype=dtype).view(1, n)
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)

    mat = torch.cos(math.pi / n * (x + 0.5) * k)

    mat[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= math.sqrt(2.0 / n)

    return mat


# ============================================================
# Stable codec-like DCT2 / Quant / Dequant / IDCT2 proxy
# ============================================================

class StableDCT2QuantIDCT2STE(nn.Module):
    """
    Stable differentiable DCT2 -> Quant -> Dequant -> IDCT2 module.

    This is NOT bit-exact VTM.
    This is a stable training proxy.

    Scope:
        - luma/inter residual proxy
        - DCT-II only
        - no transform skip
        - no MTS/LFNST/sub-block transform
        - no scaling list
        - no RDOQ/sign hiding
        - normalized input/output

    Input:
        residual_norm:
            [B,C,H,W], normalized residual, usually pred - target.

    Internal:
        residual_int = residual_norm * ((1 << bit_depth) - 1)

    Transform:
        Orthonormal DCT2.

    Quant:
        q = floor(abs(coeff) / qstep + offset) * sign(coeff)
        with STE.

    Dequant:
        coeff_hat = q * qstep
        optional qAbs<64-like interpolation correction.

    Output:
        rec_residual_norm:
            IDCT2(coeff_hat) / max_val
    """

    # VVC inverse quant scale table, scaling-list-off style.
    # Used only to derive a monotonic QP-dependent qstep.
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
        use_dead_zone_quant: bool = True,
        dead_zone_offset: float = 85.0 / 512.0,
        use_sqrt_adjustment: bool = True,
        use_qabs64_correction: bool = False,
        use_ste: bool = True,
        qstep_scale: float = 1.0,
        min_qstep: float = 1e-6,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.sizes = tuple(sorted(set(int(s) for s in sizes)))
        self.qp = int(qp)
        self.bit_depth = int(bit_depth)
        self.max_val = float((1 << self.bit_depth) - 1)

        self.use_dead_zone_quant = bool(use_dead_zone_quant)
        self.dead_zone_offset = float(dead_zone_offset)
        self.use_sqrt_adjustment = bool(use_sqrt_adjustment)
        self.use_qabs64_correction = bool(use_qabs64_correction)
        self.use_ste = bool(use_ste)
        self.qstep_scale = float(qstep_scale)
        self.min_qstep = float(min_qstep)

        for s in self.sizes:
            assert s > 0 and (s & (s - 1)) == 0, f"size must be power of two: {s}"
            mat = make_orthonormal_dct2_matrix(s, device=device, dtype=dtype)
            self.register_buffer(f"dct2_{s}", mat, persistent=False)

        coeff_shift = torch.tensor(self.COEFF_SHIFT_ARRAY, device=device, dtype=dtype)
        self.register_buffer("coeff_shift_array", coeff_shift, persistent=False)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _get_mat(self, n: int, x: torch.Tensor) -> torch.Tensor:
        name = f"dct2_{n}"
        if not hasattr(self, name):
            raise ValueError(
                f"DCT2 matrix for size {n} is not registered. "
                f"Available sizes: {self.sizes}"
            )
        return getattr(self, name).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def needs_sqrt_adjustment(h: int, w: int) -> bool:
        """
        Approximation of VVC transform scale adjustment.

        For this stable training proxy, this only chooses between
        normal inv quant scale table and sqrt-adjusted table.

        You can disable it with use_sqrt_adjustment=False.
        """
        area = h * w
        is_power_of_two = (area & (area - 1)) == 0
        log2_area = floor_log2_int(area)
        is_power_of_four = is_power_of_two and ((log2_area % 2) == 0)
        return not is_power_of_four

    def _prepare_qp(
        self,
        qp: Optional[Union[int, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supports:
            qp=None
            qp=int
            qp=scalar tensor
            qp=[B] tensor

        Returns:
            qp_tensor: [B,1,1,1], float tensor
        """
        B = x.shape[0]

        if qp is None:
            qp_t = torch.full((B,), self.qp, device=x.device, dtype=torch.float32)
        elif torch.is_tensor(qp):
            qp_t = qp.to(device=x.device, dtype=torch.float32)
            if qp_t.ndim == 0:
                qp_t = qp_t.view(1).expand(B)
            elif qp_t.ndim == 1:
                assert qp_t.shape[0] == B, f"qp shape {qp_t.shape}, B={B}"
            else:
                raise ValueError(f"qp must be scalar or [B], got {qp_t.shape}")
        else:
            qp_t = torch.full((B,), int(qp), device=x.device, dtype=torch.float32)

        return qp_t.view(B, 1, 1, 1)

    def get_qstep(
        self,
        h: int,
        w: int,
        x: torch.Tensor,
        qp: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Stable effective quant step in orthonormal DCT coefficient domain.

        qstep is derived from VVC inverse quant scale:

            invScale[qp_rem] * 2^qp_per / 64

        This is NOT VTM bit-exact, but it is scale-consistent:

            q = quant(coeff, qstep)
            coeff_hat = dequant(q, qstep)

        Returns:
            qstep: [B,1,1,1]
        """
        qp_t = self._prepare_qp(qp, x)  # [B,1,1,1]

        qp_floor = torch.floor(qp_t)
        qp_per = torch.div(qp_floor, 6, rounding_mode="floor")
        qp_rem = torch.remainder(qp_floor, 6).long().view(-1)

        need_sqrt = self.needs_sqrt_adjustment(h, w) if self.use_sqrt_adjustment else False
        scale_idx = 1 if need_sqrt else 0

        inv_table = x.new_tensor(self.INV_QUANT_SCALES[scale_idx])  # [6]
        inv_scale = inv_table[qp_rem].view(x.shape[0], 1, 1, 1)

        qstep = inv_scale * torch.pow(x.new_tensor(2.0), qp_per) / 64.0
        qstep = qstep * self.qstep_scale
        qstep = torch.clamp(qstep, min=self.min_qstep)

        return qstep

    # ------------------------------------------------------------
    # DCT / IDCT
    # ------------------------------------------------------------

    def dct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            [B,C,H,W]

        coeff:
            [B,C,H,W]
        """
        assert x.ndim == 4, f"Expected [B,C,H,W], got {x.shape}"
        _, _, H, W = x.shape

        if H not in self.sizes or W not in self.sizes:
            raise ValueError(
                f"Unsupported DCT size {H}x{W}. Registered sizes: {self.sizes}"
            )

        Ch = self._get_mat(H, x)
        Cw = self._get_mat(W, x)

        # coeff = Ch @ x @ Cw.T
        tmp = torch.einsum("kh,bchw->bckw", Ch, x)
        coeff = torch.einsum("lw,bchw->bchl", Cw, tmp)

        return coeff

    def idct2(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        coeff:
            [B,C,H,W]

        x:
            [B,C,H,W]
        """
        assert coeff.ndim == 4, f"Expected [B,C,H,W], got {coeff.shape}"
        _, _, H, W = coeff.shape

        if H not in self.sizes or W not in self.sizes:
            raise ValueError(
                f"Unsupported IDCT size {H}x{W}. Registered sizes: {self.sizes}"
            )

        Ch = self._get_mat(H, coeff)
        Cw = self._get_mat(W, coeff)

        # x = Ch.T @ coeff @ Cw
        tmp = torch.einsum("hk,bckw->bchw", Ch, coeff)
        x = torch.einsum("wl,bchw->bchl", Cw, tmp)

        return x

    # ------------------------------------------------------------
    # Quant / dequant
    # ------------------------------------------------------------

    def quantize(
        self,
        coeff: torch.Tensor,
        qstep: torch.Tensor,
    ) -> torch.Tensor:
        """
        coeff:
            [B,C,H,W]

        qstep:
            [B,1,1,1]

        Quant:
            q_abs = floor(abs(coeff) / qstep + offset)

        Offset:
            dead-zone inter-like offset: 85/512
            or round-to-nearest offset: 0.5
        """
        if self.use_dead_zone_quant:
            offset = self.dead_zone_offset
        else:
            offset = 0.5

        q_abs_real = coeff.abs() / qstep + offset
        q_abs_hat = torch.floor(q_abs_real)

        q_hat = q_abs_hat * coeff.sign()

        if self.use_ste:
            q_real_signed = q_abs_real * coeff.sign()
            q = ste_replace(q_real_signed, q_hat)
        else:
            q = q_hat

        return q

    def dequantize(
        self,
        qcoeff: torch.Tensor,
        qstep: torch.Tensor,
    ) -> torch.Tensor:
        """
        qcoeff:
            [B,C,H,W]

        qstep:
            [B,1,1,1]

        Base:
            coeff_hat = qcoeff * qstep

        Optional qAbs<64 correction:
            qAbs<64 interpolation between q and q+1.
            This is NOT bit-exact VTM in orthonormal domain.
            Default is off for stability.
        """
        q_abs = qcoeff.abs()
        q_sgn = torch.where(qcoeff < 0, -torch.ones_like(qcoeff), torch.ones_like(qcoeff))

        coeff_abs_base = q_abs * qstep

        if self.use_qabs64_correction:
            small = q_abs < 64.0

            q_abs2 = q_abs + 1.0
            coeff_abs2 = q_abs2 * qstep

            q_abs_idx = torch.clamp(q_abs.detach().long(), 0, 63)
            coef_shift = self.coeff_shift_array.to(
                device=qcoeff.device,
                dtype=qcoeff.dtype,
            )[q_abs_idx]

            coeff_abs_corr = (
                (1024.0 - coef_shift) * coeff_abs_base
                + coef_shift * coeff_abs2
            ) / 1024.0

            coeff_abs_hat = torch.where(small, coeff_abs_corr, coeff_abs_base)
        else:
            coeff_abs_hat = coeff_abs_base

        coeff_hat = coeff_abs_hat * q_sgn

        if self.use_ste:
            coeff_real = coeff_abs_base * q_sgn
            coeff = ste_replace(coeff_real, coeff_hat)
        else:
            coeff = coeff_hat

        return coeff

    # ------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------

    def forward(
        self,
        residual_norm: torch.Tensor,
        qp: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        residual_norm:
            [B,C,H,W], normalized residual.

        qp:
            None, int, scalar tensor, or [B] tensor.

        Returns:
            rec_residual_norm:
                [B,C,H,W]

            aux:
                intermediate tensors.
        """
        assert residual_norm.ndim == 4, f"Expected [B,C,H,W], got {residual_norm.shape}"

        _, _, H, W = residual_norm.shape

        if H not in self.sizes or W not in self.sizes:
            raise ValueError(
                f"Unsupported transform size {H}x{W}. "
                f"Registered sizes: {self.sizes}"
            )

        # normalized residual -> codec-like integer residual scale
        residual_int = residual_norm * self.max_val

        # orthonormal DCT
        coeff = self.dct2(residual_int)

        # QP-dependent stable qstep
        qstep = self.get_qstep(H, W, coeff, qp=qp)

        # STE quant/dequant
        qcoeff = self.quantize(coeff, qstep)
        coeff_hat = self.dequantize(qcoeff, qstep)

        # orthonormal IDCT
        rec_residual_int = self.idct2(coeff_hat)
        rec_residual_norm = rec_residual_int / self.max_val

        aux = {
            "residual_int": residual_int,
            "coeff": coeff,
            "qstep": qstep,
            "qcoeff": qcoeff,
            "coeff_hat": coeff_hat,
            "rec_residual_int": rec_residual_int,
            "q_nonzero_ratio": (qcoeff.abs() > 0).float().mean().detach(),
            "q_abs_mean": qcoeff.abs().mean().detach(),
            "coeff_abs_mean": coeff.abs().mean().detach(),
            "coeff_hat_abs_mean": coeff_hat.abs().mean().detach(),
        }

        return rec_residual_norm, aux


# ============================================================
# Loss wrapper
# ============================================================

class StableCodecDCT2Loss(nn.Module):
    """
    Loss using StableDCT2QuantIDCT2STE.

    residual = pred - target
    rec_residual = codec_proxy(residual)

    loss_recon = Charbonnier(rec_residual - residual)
    loss_rate  = mean(log2(1 + abs(qcoeff)))

    total = loss_recon + lambda_rate * loss_rate

    You can additionally use pixel-domain loss outside this class:
        total = pix_loss + lambda_codec * codec_loss
    """

    def __init__(
        self,
        codec_proxy: StableDCT2QuantIDCT2STE,
        lambda_rate: float = 0.01,
        eps: float = 1e-3,
        exclude_dc_from_rate: bool = False,
    ):
        super().__init__()

        self.codec_proxy = codec_proxy
        self.lambda_rate = float(lambda_rate)
        self.eps = float(eps)
        self.exclude_dc_from_rate = bool(exclude_dc_from_rate)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        qp: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"

        residual = pred - target

        rec_residual, aux = self.codec_proxy(residual, qp=qp)

        recon_err = rec_residual - residual
        loss_recon = torch.sqrt(recon_err * recon_err + self.eps * self.eps).mean()

        q_abs = aux["qcoeff"].abs()

        if self.exclude_dc_from_rate:
            q_abs = q_abs.clone()
            q_abs[:, :, 0, 0] = 0.0

        loss_rate = torch.log2(1.0 + q_abs).mean()

        loss = loss_recon + self.lambda_rate * loss_rate

        logs = {
            "loss_codec_recon": loss_recon.detach(),
            "loss_codec_rate": loss_rate.detach(),
            "q_nonzero_ratio": aux["q_nonzero_ratio"],
            "q_abs_mean": aux["q_abs_mean"],
            "coeff_abs_mean": aux["coeff_abs_mean"],
            "coeff_hat_abs_mean": aux["coeff_hat_abs_mean"],
        }

        return loss, logs


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 4, 1, 16, 8

    pred = torch.rand(B, C, H, W, device=device)
    target = torch.rand(B, C, H, W, device=device)

    # Batch-wise QP example.
    # You can also pass qp=32 as int.
    qp = torch.tensor([22, 27, 32, 37], device=device)

    codec_proxy = StableDCT2QuantIDCT2STE(
        sizes=(4, 8, 16, 32),
        qp=32,
        bit_depth=10,

        # Inter-like dead-zone quantization.
        use_dead_zone_quant=True,
        dead_zone_offset=85.0 / 512.0,

        # Rectangular block sqrt-table adjustment.
        # If this makes matching worse, set False.
        use_sqrt_adjustment=True,

        # Keep off for stable training proxy.
        # Turn on only for experiments.
        use_qabs64_correction=False,

        use_ste=True,

        # Tune this if qcoeff is too dense or too sparse.
        # Larger -> stronger quantization -> fewer nonzero coefficients.
        qstep_scale=1.0,

        device=device,
    ).to(device)

    residual = pred - target
    rec_residual, aux = codec_proxy(residual, qp=qp)

    print("rec_residual:", rec_residual.shape)
    print("qstep:", aux["qstep"].view(-1))
    print("q_nonzero_ratio:", float(aux["q_nonzero_ratio"].cpu()))
    print("q_abs_mean:", float(aux["q_abs_mean"].cpu()))
    print("coeff_abs_mean:", float(aux["coeff_abs_mean"].cpu()))
    print("coeff_hat_abs_mean:", float(aux["coeff_hat_abs_mean"].cpu()))

    loss_fn = StableCodecDCT2Loss(
        codec_proxy=codec_proxy,
        lambda_rate=0.01,
        eps=1e-3,
        exclude_dc_from_rate=False,
    ).to(device)

    loss, logs = loss_fn(pred, target, qp=qp)
    loss.backward()

    print("loss:", float(loss.detach().cpu()))
    print({k: float(v.detach().cpu()) for k, v in logs.items()})
