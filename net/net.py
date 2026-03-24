import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out)

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            ChannelAttention(channels)
        )

    def forward(self, x):
        return x + self.body(x)

class L_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_feat=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_feat, 3, padding=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock(num_feat, dilation=1),
            ResidualBlock(num_feat, dilation=2),
            ResidualBlock(num_feat, dilation=4)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_feat, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv_in(x)
        feat = self.res_blocks(feat)
        return self.conv_out(feat)

class R_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feat=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_feat, 3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_feat) for _ in range(6)])
        self.conv_out = nn.Sequential(
            nn.Conv2d(num_feat, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.res_blocks:
            h = block(h)
        return self.conv_out(h)

class enhance_net(nn.Module):
    def __init__(self, num_feat=32):
        super().__init__()
        self.l_net = L_net(num_feat=num_feat)
        self.r_net = R_net(num_feat=num_feat)

    def forward(self, x):
        # x: [B, 3, H, W]
        l = self.l_net(x)
        r = self.r_net(x)
        return l, r