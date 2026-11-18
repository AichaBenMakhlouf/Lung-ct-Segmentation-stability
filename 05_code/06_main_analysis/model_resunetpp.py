"""
06_main_analysis / model_resunetpp.py
-------------------------------------------------------------------------------
Faithful PyTorch port of the ResUNet++ implemented in the released notebook
(Copie_de_Final3D.ipynb, cell [12]). Structure preserved exactly:

  stem_block(16)                      : Conv3-BN-ReLU-Conv3  +  Conv1 shortcut
  resnet_block(f, s): SE(in, r=8) -> [BN-ReLU-Conv3(stride s)] -> [BN-ReLU-Conv3]
                       +  Conv1(stride s) shortcut on the SE'd input
  aspp_block(f)     : parallel Conv3 dilations {6,12,18} + Conv3(1);  Add;  Conv1
  attention_block(x1_skip, x2_dec):
        x1c = Conv3(BN-ReLU(x1)) ; x1p = MaxPool2(x1c)
        x2c = Conv3(BN-ReLU(x2))
        x   = Conv3(BN-ReLU(x1p + x2c)) ;  return x * x2         (at x2 resolution)

  encoder : c1=stem(16) -> c2=res(32,s2) -> c3=res(64,s2) -> c4=res(128,s2)
  bridge  : b1=aspp(256)
  decoder : d = attn(c_i, prev); d = Upsample2(d); d = cat(d, c_i); d = res(f)
            (f = 128, 64, 32 for the three stages)
  head    : aspp(16) -> Conv1(1) -> sigmoid

Released code uses TF/Keras defaults: Conv2D has bias, BN eps 1e-3 momentum 0.99,
'same' padding. We mirror those (bias=True, eps=1e-3). Sigmoid is applied inside
`forward` so BCE (not BCEWithLogits) matches the released `loss="binary_crossentropy"`.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-3
BN_MOM = 0.01          # torch momentum = 1 - keras_momentum(0.99)


def bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, eps=BN_EPS, momentum=BN_MOM)


class SE(nn.Module):
    """Squeeze-and-excitation on the *input* (matches released resnet_block)."""
    def __init__(self, c: int, ratio: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(c, c // ratio, bias=False)
        self.fc2 = nn.Linear(c // ratio, c, bias=False)

    def forward(self, x):
        s = x.mean(dim=(2, 3))                 # global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class StemBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = bn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.sc = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        y = self.c1(x)
        y = F.relu(self.b1(y))
        y = self.c2(y)
        return y + self.sc(x)


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.se = SE(cin)
        self.b1 = bn(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.b2 = bn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1)
        self.sc = nn.Conv2d(cin, cout, 1, stride=stride)

    def forward(self, x):
        x = self.se(x)
        y = self.c1(F.relu(self.b1(x)))
        y = self.c2(F.relu(self.b2(y)))
        return y + self.sc(x)


class ASPP(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=6, dilation=6), bn(cout))
        self.d2 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=12, dilation=12), bn(cout))
        self.d3 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=18, dilation=18), bn(cout))
        self.d4 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), bn(cout))
        self.proj = nn.Conv2d(cout, cout, 1)

    def forward(self, x):
        return self.proj(self.d1(x) + self.d2(x) + self.d3(x) + self.d4(x))


class AttentionBlock(nn.Module):
    """x1 = encoder skip (higher res), x2 = decoder feature (lower res)."""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        f = c2
        self.b1 = bn(c1); self.c1 = nn.Conv2d(c1, f, 3, padding=1)
        self.b2 = bn(c2); self.c2 = nn.Conv2d(c2, f, 3, padding=1)
        self.b3 = bn(f);  self.c3 = nn.Conv2d(f, f, 3, padding=1)

    def forward(self, x1, x2):
        x1c = self.c1(F.relu(self.b1(x1)))
        x1p = F.max_pool2d(x1c, 2)
        x2c = self.c2(F.relu(self.b2(x2)))
        x = self.c3(F.relu(self.b3(x1p + x2c)))
        return x * x2


class ResUNetPP(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.c1 = StemBlock(in_ch, 16)
        self.c2 = ResnetBlock(16, 32, stride=2)
        self.c3 = ResnetBlock(32, 64, stride=2)
        self.c4 = ResnetBlock(64, 128, stride=2)
        self.b1 = ASPP(128, 256)

        self.a1 = AttentionBlock(64, 256)          # attn(c3, b1)
        self.r1 = ResnetBlock(256 + 64, 128)       # cat(up(d1), c3)
        self.a2 = AttentionBlock(32, 128)          # attn(c2, d1)
        self.r2 = ResnetBlock(128 + 32, 64)
        self.a3 = AttentionBlock(16, 64)           # attn(c1, d2)
        self.r3 = ResnetBlock(64 + 16, 32)

        self.head_aspp = ASPP(32, 16)
        self.head_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        b1 = self.b1(c4)

        d1 = self.a1(c3, b1)
        d1 = F.interpolate(d1, scale_factor=2, mode="nearest")
        d1 = self.r1(torch.cat([d1, c3], dim=1))

        d2 = self.a2(c2, d1)
        d2 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d2 = self.r2(torch.cat([d2, c2], dim=1))

        d3 = self.a3(c1, d2)
        d3 = F.interpolate(d3, scale_factor=2, mode="nearest")
        d3 = self.r3(torch.cat([d3, c1], dim=1))

        out = self.head_conv(self.head_aspp(d3))
        return torch.sigmoid(out)


if __name__ == "__main__":
    m = ResUNetPP()
    n = sum(p.numel() for p in m.parameters())
    print(f"ResUNetPP parameters: {n:,}  (Keras released summary: 4,092,337)")
    y = m(torch.randn(2, 1, 256, 256))
    print("output:", tuple(y.shape), "range", float(y.min()), float(y.max()))
