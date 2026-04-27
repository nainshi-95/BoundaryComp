import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):
    """
    1x1 pointwise -> 3x3 depthwise -> 1x1 pointwise
    """
    def __init__(self, in_ch, out_ch, hidden_ch=None):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = out_ch

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResDWBlock(nn.Module):
    def __init__(self, ch, bottleneck=None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(ch // 2, 8)

        self.body = nn.Sequential(
            nn.Conv2d(ch, bottleneck, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(bottleneck, ch, 1),
        )

    def forward(self, x):
        return x + self.body(x)


def crop_scaled_center(feat, margin, out_h, out_w, scale):
    """
    Crop the feature region that corresponds to the final center CU.

    Original full-res layout:
        input: [H + 2M, W + 2M]
        output CU: [M : M+H, M : M+W]

    At scale s:
        feature crop:
            [M/s : M/s + H/s, M/s : M/s + W/s]

    Args:
        feat:   [B,C,Hs,Ws]
        margin: full-resolution margin M
        out_h:  final CU height H
        out_w:  final CU width W
        scale:  1, 2, or 4

    Returns:
        cropped feature.
    """
    assert margin % scale == 0
    assert out_h % scale == 0
    assert out_w % scale == 0

    m = margin // scale
    h = out_h // scale
    w = out_w // scale

    return feat[:, :, m:m+h, m:m+w]


class ResidualUNetQuarterCenterDecode(nn.Module):
    """
    Efficient residual U-Net.

    Input:
        inp: [B,3,H+2M,W+2M]
        ref: [B,1,H+2M,W+2M]

    Encoder:
        full scale: base_ch
        1/2 scale: 32
        1/4 scale: 48

    Decoder:
        crop 1/4 center first
        upsample to 1/2 center
        concat cropped 1/2 encoder feature
        upsample to 1x center
        concat cropped 1x encoder feature
        output 1-channel residual

    Final:
        pred = ref_center + residual
    """

    def __init__(
        self,
        base_ch=24,
        ch_half=32,
        ch_quarter=48,
        residual_scale=0.5,
        margin=12,
    ):
        super().__init__()

        self.base_ch = base_ch
        self.ch_half = ch_half
        self.ch_quarter = ch_quarter
        self.residual_scale = float(residual_scale)
        self.margin = int(margin)

        # --------------------------------------------------------
        # Encoder: full -> 1/2 -> 1/4
        # --------------------------------------------------------
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, base_ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(base_ch, base_ch, hidden_ch=base_ch),
            ResDWBlock(base_ch),
        )

        self.enc1 = nn.Sequential(
            nn.AvgPool2d(2),
            DWConvBlock(base_ch, ch_half, hidden_ch=ch_half),
            ResDWBlock(ch_half),
        )

        self.enc2 = nn.Sequential(
            nn.AvgPool2d(2),
            DWConvBlock(ch_half, ch_quarter, hidden_ch=ch_quarter),
            ResDWBlock(ch_quarter),
        )

        # 1/4 bottleneck, applied only after encoder has seen full canvas
        self.bottleneck = nn.Sequential(
            DWConvBlock(ch_quarter, ch_quarter, hidden_ch=ch_quarter),
            ResDWBlock(ch_quarter),
        )

        # --------------------------------------------------------
        # Decoder center path
        # --------------------------------------------------------

        # 1/4 center -> up to 1/2 center, concat 1/2 skip
        self.dec_half = nn.Sequential(
            nn.Conv2d(ch_quarter + ch_half, ch_half, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch_half, ch_half, hidden_ch=ch_half),
            ResDWBlock(ch_half),
        )

        # 1/2 center -> up to 1x center, concat full-res skip
        self.dec_full = nn.Sequential(
            nn.Conv2d(ch_half + base_ch, base_ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(base_ch, base_ch, hidden_ch=base_ch),
            ResDWBlock(base_ch),
        )

        # final pixel residual domain
        self.out_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, inp, ref):
        """
        inp: [B,3,H+2M,W+2M]
        ref: [B,1,H+2M,W+2M]
        """
        assert inp.ndim == 4 and inp.shape[1] == 3
        assert ref.ndim == 4 and ref.shape[1] == 1
        assert inp.shape[0] == ref.shape[0]
        assert inp.shape[-2:] == ref.shape[-2:]

        B, _, H_full, W_full = inp.shape
        M = self.margin

        H = H_full - 2 * M
        W = W_full - 2 * M

        assert H > 0 and W > 0
        assert M % 4 == 0, "margin should be divisible by 4. For M=12, OK."
        assert H % 4 == 0 and W % 4 == 0, "H and W should be divisible by 4."

        # --------------------------------------------------------
        # 1. Encoder on full canvas
        # --------------------------------------------------------
        e0 = self.enc0(inp)   # [B,base_ch,H+2M,W+2M]
        e1 = self.enc1(e0)    # [B,32,(H+2M)/2,(W+2M)/2]
        e2 = self.enc2(e1)    # [B,48,(H+2M)/4,(W+2M)/4]

        e2 = self.bottleneck(e2)

        # --------------------------------------------------------
        # 2. Crop 1/4 region first
        #
        # This is the exact 1/4 region which becomes H,W
        # after total 4x upsampling.
        # --------------------------------------------------------
        e2c = crop_scaled_center(e2, M, H, W, scale=4)  # [B,48,H/4,W/4]

        # 1/4 center -> 1/2 center
        up_half = F.interpolate(
            e2c,
            size=(H // 2, W // 2),
            mode="bilinear",
            align_corners=False,
        )  # [B,48,H/2,W/2]

        # Crop 1/2 encoder feature corresponding to final CU
        e1c = crop_scaled_center(e1, M, H, W, scale=2)  # [B,32,H/2,W/2]

        d_half = self.dec_half(torch.cat([up_half, e1c], dim=1))  # [B,32,H/2,W/2]

        # --------------------------------------------------------
        # 3. 1/2 center -> full center
        # --------------------------------------------------------
        up_full = F.interpolate(
            d_half,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # [B,32,H,W]

        # Crop full-res encoder feature corresponding to final CU
        e0c = crop_scaled_center(e0, M, H, W, scale=1)  # [B,base_ch,H,W]

        d_full = self.dec_full(torch.cat([up_full, e0c], dim=1))  # [B,base_ch,H,W]

        # --------------------------------------------------------
        # 4. Residual output and final prediction
        # --------------------------------------------------------
        raw_residual = self.out_head(d_full)
        residual = self.residual_scale * torch.tanh(raw_residual)

        base = ref[:, :, M:M+H, M:M+W]
        pred = base + residual

        aux = {
            "base": base,
            "residual": residual,
            "raw_residual": raw_residual,
            "e0c": e0c,
            "e1c": e1c,
            "e2c": e2c,
            "d_half": d_half,
            "d_full": d_full,
        }

        return pred, aux
