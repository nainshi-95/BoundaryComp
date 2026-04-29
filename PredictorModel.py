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














































import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):
    """
    Lightweight block:
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


class FlowWarper(nn.Module):
    """
    Bilinear warping using pixel-unit flow.

    src:
        [B,C,H,W]

    flow:
        [B,H,W,2], xy order, pixel unit

    output:
        [B,C,H,W]

    Convention:
        output[y,x] = src[y + flow_y, x + flow_x]
    """

    def __init__(
        self,
        h: int,
        w: int,
        align_corners: bool = False,
        padding_mode: str = "border",
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.h = int(h)
        self.w = int(w)
        self.align_corners = bool(align_corners)
        self.padding_mode = padding_mode

        assert padding_mode in ("zeros", "border", "reflection")

        base_grid = self._make_base_grid(
            self.h,
            self.w,
            align_corners=self.align_corners,
            device=device,
            dtype=dtype,
        )

        self.register_buffer("base_grid", base_grid, persistent=False)

    @staticmethod
    def _make_base_grid(h, w, align_corners=False, device=None, dtype=torch.float32):
        if align_corners:
            if w == 1:
                xs = torch.zeros(1, device=device, dtype=dtype)
            else:
                xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)

            if h == 1:
                ys = torch.zeros(1, device=device, dtype=dtype)
            else:
                ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        else:
            xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * 2.0 / w - 1.0
            ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * 2.0 / h - 1.0

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
        return grid.unsqueeze(0)              # [1,H,W,2]

    def flow_to_norm(self, flow):
        """
        flow: [B,H,W,2], xy order, pixel unit
        """
        if self.align_corners:
            if self.w > 1:
                dx = flow[..., 0] * (2.0 / (self.w - 1))
            else:
                dx = torch.zeros_like(flow[..., 0])

            if self.h > 1:
                dy = flow[..., 1] * (2.0 / (self.h - 1))
            else:
                dy = torch.zeros_like(flow[..., 1])
        else:
            dx = flow[..., 0] * (2.0 / self.w)
            dy = flow[..., 1] * (2.0 / self.h)

        return torch.stack([dx, dy], dim=-1)

    def forward(self, src, flow):
        """
        src:
            [B,C,H,W]

        flow:
            [B,H,W,2]
        """
        assert src.ndim == 4
        assert flow.ndim == 4 and flow.shape[-1] == 2

        B, C, H, W = src.shape
        assert H == self.h and W == self.w
        assert flow.shape == (B, H, W, 2)

        base_grid = self.base_grid.to(device=src.device, dtype=src.dtype)
        flow = flow.to(device=src.device, dtype=src.dtype)

        grid = base_grid + self.flow_to_norm(flow)

        return F.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class FeatureWarpResidualNet(nn.Module):
    """
    Flow + reference frame/crop -> warped predictor.

    Input:
        ref:
            [B,1,H,W]

        flow:
            [B,H,W,2], xy order, pixel unit

    Output:
        pred:
            [B,1,H,W]

    Structure:
        1. pixel-domain warp:
            base = warp(ref, flow)

        2. feature encoder:
            ref -> feature pyramid at 1x and 1/2x

        3. feature-domain warp:
            warp features using flow and downsampled flow

        4. residual decoder:
            residual = f(warped features, base, flow)

        5. final:
            pred = base + residual
    """

    def __init__(
        self,
        h: int,
        w: int,
        ch: int = 32,
        ch_half: int = 48,
        residual_scale: float = 0.25,
        align_corners: bool = False,
        padding_mode: str = "border",
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.h = int(h)
        self.w = int(w)
        self.ch = int(ch)
        self.ch_half = int(ch_half)
        self.residual_scale = float(residual_scale)
        self.align_corners = bool(align_corners)

        self.pixel_warper = FlowWarper(
            h=h,
            w=w,
            align_corners=align_corners,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.feature_warper_1x = FlowWarper(
            h=h,
            w=w,
            align_corners=align_corners,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        assert h % 2 == 0 and w % 2 == 0, "H and W should be divisible by 2 for half-scale branch."

        self.feature_warper_half = FlowWarper(
            h=h // 2,
            w=w // 2,
            align_corners=align_corners,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # 1x reference feature
        self.ref_enc_1x = nn.Sequential(
            nn.Conv2d(1, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch, ch, hidden_ch=ch),
            ResDWBlock(ch),
        )

        # 1/2 reference feature
        self.ref_enc_half = nn.Sequential(
            nn.AvgPool2d(2),
            DWConvBlock(ch, ch_half, hidden_ch=ch_half),
            ResDWBlock(ch_half),
        )

        # flow feature embedding at 1x
        # flow is 2-channel, xy pixel unit
        self.flow_enc = nn.Sequential(
            nn.Conv2d(2, ch // 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch // 2, ch // 2, hidden_ch=ch // 2),
        )

        # decoder:
        # warped 1x feature: ch
        # upsampled warped half feature: ch_half
        # base warped pixel: 1
        # flow feature: ch//2
        dec_in = ch + ch_half + 1 + (ch // 2)

        self.res_decoder = nn.Sequential(
            nn.Conv2d(dec_in, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch, ch, hidden_ch=ch),
            ResDWBlock(ch),
            nn.Conv2d(ch, 1, 1),
        )

        # 안정적으로 identity에서 시작하고 싶으면 마지막 conv zero init
        nn.init.zeros_(self.res_decoder[-1].weight)
        nn.init.zeros_(self.res_decoder[-1].bias)

    @staticmethod
    def downsample_flow(flow, target_h, target_w, scale: float):
        """
        flow:
            [B,H,W,2], original pixel unit

        Return:
            [B,target_h,target_w,2], pixel unit at downsampled resolution

        If resolution is half, flow magnitude should be divided by 2.
        """
        flow_chw = flow.permute(0, 3, 1, 2).contiguous()  # [B,2,H,W]

        flow_ds = F.interpolate(
            flow_chw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        flow_ds = flow_ds / scale

        return flow_ds.permute(0, 2, 3, 1).contiguous()

    def forward(self, ref, flow):
        """
        ref:
            [B,1,H,W]

        flow:
            [B,H,W,2], xy order, pixel unit

        Returns:
            pred:
                [B,1,H,W]

            aux:
                dict
        """
        assert ref.ndim == 4 and ref.shape[1] == 1
        B, _, H, W = ref.shape
        assert H == self.h and W == self.w
        assert flow.shape == (B, H, W, 2)

        # --------------------------------------------------------
        # 1. Pixel-domain warping: base predictor
        # --------------------------------------------------------
        base = self.pixel_warper(ref, flow)  # [B,1,H,W]

        # --------------------------------------------------------
        # 2. Feature extraction
        # --------------------------------------------------------
        feat_1x = self.ref_enc_1x(ref)        # [B,ch,H,W]
        feat_half = self.ref_enc_half(feat_1x)  # [B,ch_half,H/2,W/2]

        # --------------------------------------------------------
        # 3. Feature-domain warping
        # --------------------------------------------------------
        warped_feat_1x = self.feature_warper_1x(feat_1x, flow)

        flow_half = self.downsample_flow(
            flow,
            target_h=H // 2,
            target_w=W // 2,
            scale=2.0,
        )

        warped_feat_half = self.feature_warper_half(feat_half, flow_half)
        warped_feat_half_up = F.interpolate(
            warped_feat_half,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # --------------------------------------------------------
        # 4. Flow feature
        # --------------------------------------------------------
        flow_chw = flow.permute(0, 3, 1, 2).contiguous()
        flow_feat = self.flow_enc(flow_chw)

        # --------------------------------------------------------
        # 5. Pixel residual
        # --------------------------------------------------------
        dec_in = torch.cat(
            [
                warped_feat_1x,
                warped_feat_half_up,
                base,
                flow_feat,
            ],
            dim=1,
        )

        raw_residual = self.res_decoder(dec_in)
        residual = self.residual_scale * torch.tanh(raw_residual)

        pred = base + residual

        aux = {
            "base": base,
            "residual": residual,
            "raw_residual": raw_residual,
            "warped_feat_1x": warped_feat_1x,
            "warped_feat_half": warped_feat_half,
            "flow_half": flow_half,
        }

        return pred, aux






