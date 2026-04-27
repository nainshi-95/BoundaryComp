import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):
    """
    Lightweight spatial block:
      1x1 pointwise
      3x3 depthwise
      1x1 pointwise
    """
    def __init__(self, in_ch, out_ch, hidden_ch=None, k=3):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = out_ch

        pad = k // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, k, padding=pad, groups=hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyCenterFlowResidualNet(nn.Module):
    """
    Efficient version.

    Input:
        inp: [B,3,H+2M,W+2M]
             3-channel input canvas.
        ref: [B,1,H+2M,W+2M]
             reference crop.

    Output:
        pred: [B,1,H,W]

    Design:
        - Encode full inp/ref canvas once.
        - Predict flow only on center HxW.
        - Warp ref feature only for center HxW output.
        - Predict residual only for center HxW.
        - pred = ref_center + residual.
    """

    def __init__(
        self,
        ch=8,
        max_flow=4.0,
        residual_scale=0.25,
        margin=12,
    ):
        super().__init__()

        self.ch = int(ch)
        self.max_flow = float(max_flow)
        self.residual_scale = float(residual_scale)
        self.margin = int(margin)

        # Input/context encoder over full canvas.
        self.inp_encoder = nn.Sequential(
            nn.Conv2d(3, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch, ch),
        )

        # Reference encoder over full reference crop.
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(1, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch, ch),
        )

        # Flow is predicted only from center features.
        # Input:
        #   inp_center_feat [ch]
        #   ref_center_feat [ch]
        self.flow_head = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch, ch),
            nn.Conv2d(ch, 2, 1),
        )

        # Residual prediction only on center.
        # Input:
        #   warped_ref_center_feat [ch]
        #   inp_center_feat        [ch]
        #   ref_center_pixel       [1]
        self.residual_head = nn.Sequential(
            nn.Conv2d(ch * 2 + 1, ch, 1),
            nn.LeakyReLU(0.1, inplace=True),
            DWConvBlock(ch, ch),
            nn.Conv2d(ch, 1, 1),
        )

    @staticmethod
    def _flow_to_center_grid(flow, H_src, W_src, origin_y, origin_x):
        """
        Make grid_sample grid for center-only warping.

        Args:
            flow:
                [B,2,H,W], xy order, pixel unit.
            H_src, W_src:
                source feature map size.
            origin_y, origin_x:
                center block top-left coordinate inside source feature.

        Returns:
            grid:
                [B,H,W,2], normalized coordinates.
        """
        B, _, H, W = flow.shape
        device = flow.device
        dtype = flow.dtype

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )

        xx = xx.unsqueeze(0).expand(B, H, W)
        yy = yy.unsqueeze(0).expand(B, H, W)

        sample_x = xx + origin_x + flow[:, 0]
        sample_y = yy + origin_y + flow[:, 1]

        # align_corners=False convention
        grid_x = 2.0 * (sample_x + 0.5) / W_src - 1.0
        grid_y = 2.0 * (sample_y + 0.5) / H_src - 1.0

        return torch.stack([grid_x, grid_y], dim=-1)

    @staticmethod
    def _warp_center(src_feat, flow, origin_y, origin_x):
        """
        Warp only center HxW output from full source feature.

        Args:
            src_feat:
                [B,C,H_src,W_src]
            flow:
                [B,2,H,W]
            origin_y, origin_x:
                top-left of center block inside src_feat.

        Returns:
            warped:
                [B,C,H,W]
        """
        _, _, H_src, W_src = src_feat.shape

        grid = TinyCenterFlowResidualNet._flow_to_center_grid(
            flow=flow,
            H_src=H_src,
            W_src=W_src,
            origin_y=origin_y,
            origin_x=origin_x,
        )

        return F.grid_sample(
            src_feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def forward(self, inp, ref):
        """
        Args:
            inp:
                [B,3,H+2M,W+2M]
            ref:
                [B,1,H+2M,W+2M]

        Returns:
            pred:
                [B,1,H,W]
            aux:
                dict
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

        # --------------------------------------------------------
        # 1. Encode full canvas once
        # --------------------------------------------------------
        inp_feat = self.inp_encoder(inp)   # [B,ch,H+2M,W+2M]
        ref_feat = self.ref_encoder(ref)   # [B,ch,H+2M,W+2M]

        # --------------------------------------------------------
        # 2. Center crop features and base reference pixel
        # --------------------------------------------------------
        inp_c = inp_feat[:, :, M:M + H, M:M + W]
        ref_c_feat = ref_feat[:, :, M:M + H, M:M + W]
        ref_c_pixel = ref[:, :, M:M + H, M:M + W]

        # --------------------------------------------------------
        # 3. Predict flow only for center HxW
        # --------------------------------------------------------
        flow_in = torch.cat([inp_c, ref_c_feat], dim=1)
        raw_flow = self.flow_head(flow_in)          # [B,2,H,W]
        flow = self.max_flow * torch.tanh(raw_flow) # [B,2,H,W]

        # --------------------------------------------------------
        # 4. Warp reference feature only for center HxW
        # --------------------------------------------------------
        warped_ref_c = self._warp_center(
            src_feat=ref_feat,
            flow=flow,
            origin_y=M,
            origin_x=M,
        )  # [B,ch,H,W]

        # --------------------------------------------------------
        # 5. Predict residual only for center HxW
        # --------------------------------------------------------
        res_in = torch.cat([warped_ref_c, inp_c, ref_c_pixel], dim=1)
        raw_residual = self.residual_head(res_in)

        residual = self.residual_scale * torch.tanh(raw_residual)

        # --------------------------------------------------------
        # 6. Final predictor
        # --------------------------------------------------------
        pred = ref_c_pixel + residual

        aux = {
            "flow": flow,                  # [B,2,H,W]
            "residual": residual,          # [B,1,H,W]
            "base": ref_c_pixel,           # [B,1,H,W]
            "warped_ref_feat": warped_ref_c,
        }

        return pred, aux
















B, H, W = 4, 32, 32
M = 12

inp = torch.randn(B, 3, H + 2*M, W + 2*M).cuda()
ref = torch.randn(B, 1, H + 2*M, W + 2*M).cuda()
gt  = torch.randn(B, 1, H, W).cuda()

model = TinyCenterFlowResidualNet(
    ch=8,
    max_flow=4.0,
    residual_scale=0.25,
    margin=M,
).cuda()

pred, aux = model(inp, ref)

loss = F.mse_loss(pred, gt)
loss.backward()

print(pred.shape)          # [B,1,H,W]
print(aux["flow"].shape)   # [B,2,H,W]
print(aux["residual"].shape)







