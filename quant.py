import math
from typing import Iterable, Optional, Tuple, Dict

import torch
import torch.nn as nn


# ============================================================
# STE helpers
# ============================================================

def ste_replace(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Forward:
        x_hat
    Backward:
        gradient as if identity w.r.t. x

    This is the standard STE form:
        x + (x_hat - x).detach()
    """
    return x + (x_hat - x).detach()


def ste_floor(x: torch.Tensor) -> torch.Tensor:
    """
    Forward:
        floor(x)
    Backward:
        identity
    """
    return x + (torch.floor(x) - x).detach()


def ste_round_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Codec-like rounded right shift with STE.

    Approximate:
        (x + (1 << (shift - 1))) >> shift

    For float tensor:
        floor((x + 2^(shift-1)) / 2^shift)

    Backward:
        identity-like through x / 2^shift.
    """
    if shift == 0:
        return x

    if shift < 0:
        return x * float(1 << (-shift))

    denom = float(1 << shift)
    x_real = x / denom
    x_hat = torch.floor((x + 0.5 * denom) / denom)
    return ste_replace(x_real, x_hat)


def floor_log2_int(x: int) -> int:
    assert x > 0
    return int(math.floor(math.log2(x)))


def clip3(x: torch.Tensor, min_val: int, max_val: int) -> torch.Tensor:
    return torch.clamp(x, min=float(min_val), max=float(max_val))


# ============================================================
# VVC-like integer DCT-II matrix
# ============================================================

def make_vvc_dct2_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Approximate VVC/VTM DCT-II transform core.

    VVC transform matrices are integer cores with transform_matrix_shift = 6,
    so the basis is approximately:

        round(64 * orthonormal_DCT2)

    Shape:
        [n, n]

    Forward 1D:
        Y[k] = sum_x C[k, x] * X[x]
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of two, got {n}"

    x = torch.arange(n, device=device, dtype=dtype).view(1, n)
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)

    mat = torch.cos(math.pi / n * (x + 0.5) * k)

    mat[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= math.sqrt(2.0 / n)

    mat = torch.round(mat * 64.0)
    return mat


# ============================================================
# Main module
# ============================================================

class VVCInterDCT2QuantIDCT2STE(nn.Module):
    """
    Narrow-scope VVC-like DCT2 -> Quant -> Dequant -> IDCT2 module.

    This is intended for differentiable training, not strict bit-exact VTM replacement.

    Scope:
        - inter residual
        - luma only
        - DCT-II only
        - no transform skip
        - no MTS
        - no LFNST
        - no sub-block transform
        - no scaling list
        - no RDOQ
        - no sign hiding

    Input:
        residual_norm:
            [B, C, H, W]
            normalized residual, usually pred - target.
            If bit_depth=10, internal integer-like residual is residual_norm * 1023.

    Output:
        rec_residual_norm:
            [B, C, H, W]
            reconstructed residual after DCT2/Q/DQ/IDCT2, normalized back.

        aux:
            dictionary containing coeff, qcoeff, dequant coeff, etc.

    Notes:
        The forward path uses codec-like quant/dequant integer formulas.
        The backward path uses STE.
    """

    # VVC/VTM quant scale tables for scaling-list-off path.
    # Index 0: normal
    # Index 1: sqrt(2) adjustment case
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
        max_log2_tr_dynamic_range: int = 15,
        quant_shift: int = 14,
        iquant_shift: int = 20,
        transform_matrix_shift: int = 6,
        com16_trans_prec: int = 0,
        use_ste: bool = True,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.sizes = tuple(sorted(set(int(s) for s in sizes)))
        self.qp = int(qp)
        self.bit_depth = int(bit_depth)
        self.max_val = float((1 << self.bit_depth) - 1)

        self.max_log2_tr_dynamic_range = int(max_log2_tr_dynamic_range)
        self.quant_shift = int(quant_shift)
        self.iquant_shift = int(iquant_shift)
        self.transform_matrix_shift = int(transform_matrix_shift)
        self.com16_trans_prec = int(com16_trans_prec)
        self.use_ste = bool(use_ste)

        for s in self.sizes:
            assert s > 0 and (s & (s - 1)) == 0, f"size must be power of two: {s}"
            mat = make_vvc_dct2_matrix(s, device=device, dtype=dtype)
            self.register_buffer(f"dct2_{s}", mat, persistent=False)

        coeff_shift = torch.tensor(self.COEFF_SHIFT_ARRAY, device=device, dtype=dtype)
        self.register_buffer("coeff_shift_array", coeff_shift, persistent=False)

    # ------------------------------------------------------------
    # Basic parameter helpers
    # ------------------------------------------------------------

    def _get_mat(self, n: int, x: torch.Tensor) -> torch.Tensor:
        name = f"dct2_{n}"
        if not hasattr(self, name):
            raise ValueError(
                f"DCT2 matrix for size {n} is not registered. "
                f"Available sizes: {self.sizes}. "
                f"Initialize with sizes including {n}."
            )

        mat = getattr(self, name)
        return mat.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def needs_sqrt_adjustment(h: int, w: int) -> bool:
        """
        Approximation of TU::needsBlockSizeTrafoScale.

        For the narrow case here, use sqrt adjustment when transform area
        is not a power of 4.

        Examples:
            4x4:   area=16=4^2   -> False
            8x8:   area=64=4^3   -> False
            4x16:  area=64=4^3   -> False
            16x8:  area=128      -> True
            4x8:   area=32       -> True
        """
        area = h * w
        is_power_of_two = (area & (area - 1)) == 0
        log2_area = floor_log2_int(area)
        is_power_of_four = is_power_of_two and ((log2_area % 2) == 0)
        return not is_power_of_four

    def qp_per_rem(self, qp: Optional[int] = None) -> Tuple[int, int]:
        if qp is None:
            qp = self.qp

        qp = int(qp)
        qp_per = qp // 6
        qp_rem = qp % 6
        return qp_per, qp_rem

    def get_transform_shift(self, h: int, w: int) -> int:
        """
        Approximation of getTransformShift(channelBitDepth, rect.size(), maxLog2TrDynamicRange).

        For square blocks, this matches the common form:
            maxLog2TrDynamicRange - bitDepth - log2TrSize

        For rectangular blocks, use:
            log2TrSize = floor(log2(area) / 2)

        This is the part most likely to require adjustment if exact VTM matching
        differs for rectangular blocks in your build.
        """
        log2_area = floor_log2_int(h * w)
        log2_tr_size = log2_area // 2
        return self.max_log2_tr_dynamic_range - self.bit_depth - log2_tr_size

    # ------------------------------------------------------------
    # Forward / inverse DCT2
    # ------------------------------------------------------------

    def forward_transform(self, residual_int: torch.Tensor) -> torch.Tensor:
        """
        Approximate VVC forward 2D DCT-II.

        residual_int:
            [B, C, H, W], float tensor in codec residual scale.

        returns:
            coeff:
                [B, C, H, W]
        """
        assert residual_int.ndim == 4
        _, _, H, W = residual_int.shape

        Cw = self._get_mat(W, residual_int)  # [W, W]
        Ch = self._get_mat(H, residual_int)  # [H, H]

        # VTM-like xTrMxN shift approximation.
        shift1 = (
            floor_log2_int(W)
            + self.bit_depth
            + self.transform_matrix_shift
            - self.max_log2_tr_dynamic_range
            + self.com16_trans_prec
        )
        shift2 = (
            floor_log2_int(H)
            + self.transform_matrix_shift
            + self.com16_trans_prec
        )

        # Horizontal transform:
        # tmp[b,c,y,k] = sum_x residual[b,c,y,x] * Cw[k,x]
        tmp = torch.einsum("kx,bcyx->bcyk", Cw, residual_int)
        tmp = ste_round_shift(tmp, shift1) if self.use_ste else self._round_shift_no_ste(tmp, shift1)

        # Vertical transform:
        # coeff[b,c,k,x] = sum_y Ch[k,y] * tmp[b,c,y,x]
        coeff = torch.einsum("ky,bcyx->bckx", Ch, tmp)
        coeff = ste_round_shift(coeff, shift2) if self.use_ste else self._round_shift_no_ste(coeff, shift2)

        return coeff

    def inverse_transform(self, coeff: torch.Tensor) -> torch.Tensor:
        """
        Approximate VVC inverse 2D DCT-II.

        coeff:
            [B, C, H, W]

        returns:
            rec_residual_int:
                [B, C, H, W]
        """
        assert coeff.ndim == 4
        _, _, H, W = coeff.shape

        Cw = self._get_mat(W, coeff)
        Ch = self._get_mat(H, coeff)

        # VTM-like xITrMxN shift approximation.
        shift1 = self.transform_matrix_shift + 1 + self.com16_trans_prec
        shift2 = (
            self.transform_matrix_shift
            + self.max_log2_tr_dynamic_range
            - 1
            - self.bit_depth
            + self.com16_trans_prec
        )

        # Inverse vertical:
        # tmp[b,c,y,x] = sum_k Ch[k,y] * coeff[b,c,k,x]
        tmp = torch.einsum("ky,bckx->bcyx", Ch, coeff)
        tmp = ste_round_shift(tmp, shift1) if self.use_ste else self._round_shift_no_ste(tmp, shift1)

        # Inverse horizontal:
        # rec[b,c,y,x] = sum_k Cw[k,x] * tmp[b,c,y,k]
        rec = torch.einsum("kx,bcyk->bcyx", Cw, tmp)
        rec = ste_round_shift(rec, shift2) if self.use_ste else self._round_shift_no_ste(rec, shift2)

        return rec

    @staticmethod
    def _round_shift_no_ste(x: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return x
        if shift < 0:
            return x * float(1 << (-shift))

        denom = float(1 << shift)
        return torch.floor((x + 0.5 * denom) / denom)

    # ------------------------------------------------------------
    # Quant / dequant
    # ------------------------------------------------------------

    def quantize(self, coeff: torch.Tensor, qp: Optional[int] = None) -> torch.Tensor:
        """
        VTM-like scalar quantization without scaling list/RDOQ.

        Inter dead-zone style:
            qAbs = floor((abs(coeff) * quantScale + add) / 2^qbits)

        where for ordinary inter:
            add = 85 << (qbits - 9)

        coeff:
            [B, C, H, W]

        returns:
            qcoeff:
                forward value is integer-like quant coefficient.
                backward uses STE through real-valued quant proxy.
        """
        assert coeff.ndim == 4
        _, _, H, W = coeff.shape

        qp_per, qp_rem = self.qp_per_rem(qp)

        need_sqrt = self.needs_sqrt_adjustment(H, W)
        scale_idx = 1 if need_sqrt else 0

        quant_scale = float(self.QUANT_SCALES[scale_idx][qp_rem])

        i_transform_shift = self.get_transform_shift(H, W)
        if need_sqrt:
            i_transform_shift -= 1

        i_qbits = self.quant_shift + qp_per + i_transform_shift

        if i_qbits >= 9:
            add = float(85 << (i_qbits - 9))
        else:
            add = 85.0 / float(1 << (9 - i_qbits))

        abs_coeff = coeff.abs()

        # Real-valued quant proxy.
        denom = float(1 << i_qbits)
        q_abs_real = (abs_coeff * quant_scale + add) / denom

        # Codec-like integer forward.
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
        VTM-like dequantization without scaling list.

        Includes exact qAbs < 64 interpolation correction from VTM-style Quant::dequant:

            nom(q)  = ((qAbs    * scale + add) >> shift)
            nom(q+1)= (((qAbs+1)* scale + add) >> shift)

            if qAbs < 64:
                coefShift = coeffShiftArray[qAbs]
                nom = ((1024 - coefShift) * nom(q)
                       + coefShift * nom(q+1)) >> 10

        qcoeff:
            [B, C, H, W]

        returns:
            coeff_hat:
                dequantized transform coefficient.
        """
        assert qcoeff.ndim == 4
        _, _, H, W = qcoeff.shape

        qp_per, qp_rem = self.qp_per_rem(qp)

        need_sqrt = self.needs_sqrt_adjustment(H, W)
        scale_idx = 1 if need_sqrt else 0

        i_transform_shift = self.get_transform_shift(H, W)
        if need_sqrt:
            i_transform_shift -= 1

        right_shift = self.iquant_shift - (i_transform_shift + qp_per)

        # VTM scaling-list-off branch:
        # if rightShift > 0:
        #     scale = invQuantScale
        #     shift = rightShift
        # else:
        #     scale = invQuantScale << (-rightShift)
        #     shift = 0
        if right_shift > 0:
            scale = float(self.INV_QUANT_SCALES[scale_idx][qp_rem])
            shift = right_shift
        else:
            scale = float(self.INV_QUANT_SCALES[scale_idx][qp_rem]) * float(1 << (-right_shift))
            shift = 0

        add = float((1 << shift) >> 1)

        transform_minimum = -(1 << self.max_log2_tr_dynamic_range)
        transform_maximum = (1 << self.max_log2_tr_dynamic_range) - 1

        # Approximate VTM input clipping range.
        scale_bits = self.iquant_shift + 1
        intermediate_bits = 64
        target_input_bit_depth = min(
            self.max_log2_tr_dynamic_range + 1,
            intermediate_bits + right_shift - scale_bits,
        )
        input_minimum = -(1 << (target_input_bit_depth - 1))
        input_maximum = (1 << (target_input_bit_depth - 1)) - 1

        clip_q = clip3(qcoeff, input_minimum, input_maximum)

        q_abs = clip_q.abs()
        q_sgn = torch.where(clip_q < 0, -torch.ones_like(clip_q), torch.ones_like(clip_q))

        # nomTCoeff(qAbs)
        if shift > 0:
            denom = float(1 << shift)
            nom = torch.floor((q_abs * scale + add) / denom)
        else:
            nom = q_abs * scale

        # nomTCoeff(qAbs + 1)
        q_abs2 = q_abs + 1.0
        if shift > 0:
            denom = float(1 << shift)
            nom2 = torch.floor((q_abs2 * scale + add) / denom)
        else:
            nom2 = q_abs2 * scale

        # Exact small-qAbs correction.
        small = q_abs < 64

        q_abs_idx = torch.clamp(q_abs.detach().long(), 0, 63)
        coef_shift = self.coeff_shift_array.to(device=qcoeff.device, dtype=qcoeff.dtype)[q_abs_idx]

        nom_corr = torch.floor(
            ((1024.0 - coef_shift) * nom + coef_shift * nom2) / 1024.0
        )

        nom_final = torch.where(small, nom_corr, nom)

        coeff_hat = clip3(nom_final * q_sgn, transform_minimum, transform_maximum)

        if self.use_ste:
            # Differentiable proxy through unrounded dequant value.
            if shift > 0:
                coeff_real_abs = (q_abs * scale) / float(1 << shift)
            else:
                coeff_real_abs = q_abs * scale

            coeff_real = coeff_real_abs * q_sgn
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
        qp: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        residual_norm:
            [B, C, H, W]
            normalized residual.

        qp:
            optional integer QP override.

        returns:
            rec_residual_norm:
                [B, C, H, W]

            aux:
                dict of intermediate tensors.
        """
        assert residual_norm.ndim == 4, f"Expected [B,C,H,W], got {residual_norm.shape}"

        _, _, H, W = residual_norm.shape
        if H not in self.sizes or W not in self.sizes:
            raise ValueError(
                f"Unsupported transform size {H}x{W}. "
                f"Registered sizes: {self.sizes}"
            )

        residual_int = residual_norm * self.max_val

        coeff = self.forward_transform(residual_int)
        qcoeff = self.quantize(coeff, qp=qp)
        coeff_hat = self.dequantize(qcoeff, qp=qp)
        rec_residual_int = self.inverse_transform(coeff_hat)

        rec_residual_norm = rec_residual_int / self.max_val

        aux = {
            "residual_int": residual_int,
            "coeff": coeff,
            "qcoeff": qcoeff,
            "coeff_hat": coeff_hat,
            "rec_residual_int": rec_residual_int,
            "need_sqrt_adjustment": torch.tensor(
                float(self.needs_sqrt_adjustment(H, W)),
                device=residual_norm.device,
                dtype=residual_norm.dtype,
            ),
        }

        return rec_residual_norm, aux


# ============================================================
# Optional training loss using this codec-like module
# ============================================================

class CodecLikeDCT2Loss(nn.Module):
    """
    Training loss:

        residual = pred - target
        rec_residual = codec_tq(residual)

        loss_recon = Charbonnier(rec_residual - residual)
        loss_rate  = mean(log2(1 + abs(qcoeff)))

        total = loss_recon + lambda_rate * loss_rate

    Usually combine this with pixel-domain pred-target loss if needed.
    """

    def __init__(
        self,
        codec_tq: VVCInterDCT2QuantIDCT2STE,
        lambda_rate: float = 0.01,
        eps: float = 1e-3,
        exclude_dc_from_rate: bool = False,
    ):
        super().__init__()

        self.codec_tq = codec_tq
        self.lambda_rate = float(lambda_rate)
        self.eps = float(eps)
        self.exclude_dc_from_rate = bool(exclude_dc_from_rate)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        qp: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        pred, target:
            [B,C,H,W], normalized domain.
        """
        assert pred.shape == target.shape

        residual = pred - target

        rec_residual, aux = self.codec_tq(residual, qp=qp)

        recon_err = rec_residual - residual
        loss_recon = torch.sqrt(recon_err * recon_err + self.eps * self.eps).mean()

        q = aux["qcoeff"]
        q_abs = q.abs()

        if self.exclude_dc_from_rate:
            q_abs = q_abs.clone()
            q_abs[:, :, 0, 0] = 0.0

        loss_rate = torch.log2(1.0 + q_abs).mean()

        loss = loss_recon + self.lambda_rate * loss_rate

        logs = {
            "loss_codec_recon": loss_recon.detach(),
            "loss_codec_rate": loss_rate.detach(),
            "q_nonzero_ratio": (q.abs() > 0).float().mean().detach(),
            "q_abs_mean": q.abs().mean().detach(),
            "coeff_abs_mean": aux["coeff"].abs().mean().detach(),
        }

        return loss, logs


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, H, W = 2, 1, 16, 8

    pred = torch.rand(B, C, H, W, device=device)
    target = torch.rand(B, C, H, W, device=device)

    codec_tq = VVCInterDCT2QuantIDCT2STE(
        sizes=(4, 8, 16, 32),
        qp=32,
        bit_depth=10,
        use_ste=True,
        device=device,
        dtype=torch.float32,
    ).to(device)

    residual = pred - target
    rec_residual, aux = codec_tq(residual)

    print("rec_residual:", rec_residual.shape)
    print("qcoeff:", aux["qcoeff"].shape)
    print("q nonzero ratio:", (aux["qcoeff"].abs() > 0).float().mean().item())

    codec_loss = CodecLikeDCT2Loss(
        codec_tq=codec_tq,
        lambda_rate=0.01,
        eps=1e-3,
        exclude_dc_from_rate=False,
    ).to(device)

    loss, logs = codec_loss(pred, target)
    loss.backward()

    print("loss:", float(loss.detach().cpu()))
    print({k: float(v.detach().cpu()) for k, v in logs.items()})
