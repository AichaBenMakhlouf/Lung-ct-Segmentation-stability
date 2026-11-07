"""Segmentation networks compared in the article (Table 1).

All networks take a 1-channel 256 x 256 slice and return a sigmoid probability map.

| name        | architecture                                   | parameters |
|-------------|------------------------------------------------|-----------:|
| unet        | U-Net (Ronneberger et al., 2015)               |  7,765,409 |
| attn_unet   | Attention U-Net (Oktay et al., 2018)           |  7,853,173 |
| resunet     | ResU-Net (Zhang et al., 2018)                  |  8,115,073 |
| resunetpp   | ResUNet++ (Jha et al., 2019)                   |  4,085,873 |
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS, BN_MOMENTUM = 1e-3, 0.01


def _bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, eps=BN_EPS, momentum=BN_MOMENTUM)


# ============================================================================ ResUNet++
class SqueezeExcite(nn.Module):
    def __init__(self, c: int, ratio: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(c, c // ratio, bias=False)
        self.fc2 = nn.Linear(c // ratio, c, bias=False)

    def forward(self, x):
        s = F.relu(self.fc1(x.mean((2, 3))))
        return x * torch.sigmoid(self.fc2(s))[:, :, None, None]


class StemBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, cout, 3, padding=1), _bn(cout)
        self.conv2, self.shortcut = nn.Conv2d(cout, cout, 3, padding=1), nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.bn1(self.conv1(x)))) + self.shortcut(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.se = SqueezeExcite(cin)
        self.bn1, self.conv1 = _bn(cin), nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.bn2, self.conv2 = _bn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride=stride)

    def forward(self, x):
        x = self.se(x)
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        return y + self.shortcut(x)


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling: dilations 6, 12, 18 and 1, summed, then 1x1 conv."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=d, dilation=d), _bn(cout)) for d in (6, 12, 18))
        self.local = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), _bn(cout))
        self.project = nn.Conv2d(cout, cout, 1)

    def forward(self, x):
        return self.project(sum(b(x) for b in self.branches) + self.local(x))


class AttentionBlock(nn.Module):
    """Attention between an encoder skip (higher resolution) and a decoder feature."""

    def __init__(self, c_skip: int, c_dec: int):
        super().__init__()
        self.bn1, self.conv1 = _bn(c_skip), nn.Conv2d(c_skip, c_dec, 3, padding=1)
        self.bn2, self.conv2 = _bn(c_dec), nn.Conv2d(c_dec, c_dec, 3, padding=1)
        self.bn3, self.conv3 = _bn(c_dec), nn.Conv2d(c_dec, c_dec, 3, padding=1)

    def forward(self, skip, dec):
        a = F.max_pool2d(self.conv1(F.relu(self.bn1(skip))), 2)
        b = self.conv2(F.relu(self.bn2(dec)))
        return self.conv3(F.relu(self.bn3(a + b))) * dec


class ResUNetPP(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.enc1 = StemBlock(in_ch, 16)
        self.enc2, self.enc3, self.enc4 = (ResidualSEBlock(16, 32, 2), ResidualSEBlock(32, 64, 2),
                                           ResidualSEBlock(64, 128, 2))
        self.bridge = ASPP(128, 256)
        self.att1, self.dec1 = AttentionBlock(64, 256), ResidualSEBlock(320, 128)
        self.att2, self.dec2 = AttentionBlock(32, 128), ResidualSEBlock(160, 64)
        self.att3, self.dec3 = AttentionBlock(16, 64), ResidualSEBlock(80, 32)
        self.head_aspp, self.head = ASPP(32, 16), nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        d = self.bridge(e4)
        d = self.dec1(torch.cat([F.interpolate(self.att1(e3, d), scale_factor=2), e3], 1))
        d = self.dec2(torch.cat([F.interpolate(self.att2(e2, d), scale_factor=2), e2], 1))
        d = self.dec3(torch.cat([F.interpolate(self.att3(e1, d), scale_factor=2), e1], 1))
        return torch.sigmoid(self.head(self.head_aspp(d)))


# ============================================================================ U-Net family
def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), _bn(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), _bn(cout), nn.ReLU(inplace=True))


class _ResidualDoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = _double_conv(cin, cout)
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    def __init__(self, c_gate: int, c_skip: int, c_inter: int):
        super().__init__()
        self.w_gate, self.w_skip = nn.Conv2d(c_gate, c_inter, 1), nn.Conv2d(c_skip, c_inter, 1)
        self.psi = nn.Conv2d(c_inter, 1, 1)

    def forward(self, g, x):
        return x * torch.sigmoid(self.psi(F.relu(self.w_gate(g) + self.w_skip(x))))


class UNet(nn.Module):
    """U-Net with four down-sampling stages (base width 32).

    residual=True gives ResU-Net (residual double-conv blocks); attention=True gives
    Attention U-Net (attention gates on the skip connections).
    """

    def __init__(self, in_ch: int = 1, base: int = 32, residual: bool = False, attention: bool = False):
        super().__init__()
        block = _ResidualDoubleConv if residual else _double_conv
        b = base
        self.enc = nn.ModuleList([block(in_ch, b), block(b, 2 * b), block(2 * b, 4 * b), block(4 * b, 8 * b)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(8 * b, 16 * b)
        chans = [8 * b, 4 * b, 2 * b, b]
        self.up = nn.ModuleList()
        self.gates = nn.ModuleList() if attention else None
        self.dec = nn.ModuleList()
        for c in chans:
            self.up.append(nn.ConvTranspose2d(2 * c, c, 2, 2))
            if attention:
                self.gates.append(AttentionGate(c, c, max(c // 2, 1)))
            self.dec.append(block(2 * c, c))
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        x = self.bottleneck(self.pool(x))
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            u = up(x)
            s = skips[-1 - i]
            if self.gates is not None:
                s = self.gates[i](u, s)
            x = dec(torch.cat([u, s], 1))
        return torch.sigmoid(self.out(x))


ARCHITECTURES = {
    "unet": lambda: UNet(),
    "attn_unet": lambda: UNet(attention=True),
    "resunet": lambda: UNet(residual=True),
    "resunetpp": lambda: ResUNetPP(),
}


def build_model(name: str) -> nn.Module:
    return ARCHITECTURES[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
