import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class MaskedOuterpolationNet(nn.Module):
    """
    Input:
        x: [B,1,H,W]

    Internally:
        canvas: [B,1,H+4,W+4]
            canvas[:,:,4:,4:] = x
            unknown top/left/corner = 0

        mask: [B,1,H+4,W+4]
            known area = 1
            unknown area = 0

        net input: concat(canvas, mask) -> [B,2,H+4,W+4]

    Output:
        out: [B,1,H+4,W+4]
    """

    def __init__(self, base_ch=32, residual_scale=0.25):
        super().__init__()
        self.residual_scale = residual_scale

        self.net = nn.Sequential(
            nn.Conv2d(2, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),

            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1

        canvas = x.new_zeros(B, 1, H + 4, W + 4)
        mask = x.new_zeros(B, 1, H + 4, W + 4)

        canvas[:, :, 4:, 4:] = x
        mask[:, :, 4:, 4:] = 1.0

        inp = torch.cat([canvas, mask], dim=1)
        pred = self.net(inp)

        # 안정성을 위해 unknown 영역은 border replication + residual 형태로 제한
        base = canvas.clone()
        base[:, :, :4, 4:] = x[:, :, :1, :].expand(B, 1, 4, W)
        base[:, :, 4:, :4] = x[:, :, :, :1].expand(B, 1, H, 4)
        base[:, :, :4, :4] = x[:, :, :1, :1].expand(B, 1, 4, 4)

        unknown = 1.0 - mask
        completed = base + unknown * self.residual_scale * torch.tanh(pred)

        # known 영역은 원본 x를 강제로 유지
        out = mask * canvas + unknown * completed

        return out
