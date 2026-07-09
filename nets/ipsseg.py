import torch
import torch.nn as nn
import torch.nn.functional as F



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=3, act_type="mish"):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act_type = act_type.lower()
        if self.act_type == "leakyrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif self.act_type == "tanh":
            self.act = nn.Tanh()
        elif self.act_type == "mish":
            # Mish: x * tanh(softplus(x))
            self.act = None  # sẽ xử lý trong forward
        elif self.act_type == "swish":
            # Swish: x * sigmoid(x)
            self.act = None  # sẽ xử lý trong forward
        else:
            self.act = nn.ReLU(inplace=True)  # fallback

    def forward(self, x):
        x = self.conv(x)
        if self.act_type == "mish":
            # Mish: x * tanh(softplus(x))
            return x * torch.tanh(F.softplus(x))
        elif self.act_type == "swish":
            # Swish: x * sigmoid(x)
            return x * torch.sigmoid(x)
        elif self.act is not None:
            return self.act(x)
        else:
            return x


class IdentityTransformerBlock(nn.Module):
    def __init__(self, emb_dim, mlp_ratio):
        super(IdentityTransformerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, emb_dim)  # Use GroupNorm instead of LayerNorm for 4D tensors
        self.identity = nn.Identity()
        self.norm2 = nn.GroupNorm(1, emb_dim)  # Use GroupNorm instead of LayerNorm for 4D tensors
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(emb_dim * mlp_ratio, emb_dim),
        )

    def forward(self, x):
        # Get original shape
        B, C, H, W = x.shape
        
        # Normalize before token mixing
        x_norm = self.norm1(x)
        # Token mixing with identity (no transformation)
        mixed = self.identity(x_norm)
        # Add residual connection
        x = x + mixed

        # Normalize before feed-forward network
        x_norm = self.norm2(x)
        
        # Reshape for linear layers: (B, C, H, W) -> (B*H*W, C)
        x_norm = x_norm.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Feed-forward network
        ffn_out = self.ffn(x_norm)
        
        # Reshape back: (B*H*W, C) -> (B, C, H, W)
        ffn_out = ffn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Add residual connection
        out = x + ffn_out

        return out


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ASPP, self).__init__()

        branch_ch = in_ch

        # ASPP branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d( in_ch, branch_ch, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU()
        )

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch*4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.project(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv2(self.conv1(x))
        return self.act(x + residual)


class IPSSeg(nn.Module):
    def __init__(self, n_channels, n_classes, num_filters_1st=24):
        super(IPSSeg, self).__init__()

        self.conv1_1 = ConvBlock(n_channels, num_filters_1st, 3, act_type="swish")
        self.ite1_1 = IdentityTransformerBlock(num_filters_1st, 4)
        self.ite1_2 = IdentityTransformerBlock(num_filters_1st, 4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBlock(num_filters_1st, int(2*num_filters_1st), 3, act_type="swish")
        self.ite2_1 = IdentityTransformerBlock(int(2*num_filters_1st), 4)
        self.ite2_2 = IdentityTransformerBlock(int(2*num_filters_1st), 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBlock(int(2*num_filters_1st), int(4*num_filters_1st), 3, act_type="swish")
        self.ite3_1 = IdentityTransformerBlock(int(4*num_filters_1st), 4)
        self.ite3_2 = IdentityTransformerBlock(int(4*num_filters_1st), 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBlock(int(4*num_filters_1st), int(8*num_filters_1st), 3, act_type="swish")
        self.ite4_1 = IdentityTransformerBlock(int(8*num_filters_1st), 4)
        self.ite4_2 = IdentityTransformerBlock(int(8*num_filters_1st), 4)
        self.spp_conv = ConvBlock(int(8*num_filters_1st), int(4*num_filters_1st), 1, act_type="swish")

        # ASPP
        self.aspp = ASPP(int(4*num_filters_1st), int(8*num_filters_1st))
        self.drop5 = nn.Dropout(0.5)

        # Decoder
        f = num_filters_1st
        # Decoder Stage 7
        self.up7 = nn.Sequential(
            nn.Conv2d(int(8*f), int(4*f) * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2), nn.BatchNorm2d(int(4*f)), nn.SiLU()
        )
        self.fuse7 = ConvBlock(int(4*f) * 2, int(4*f), 1, act_type="swish")
        self.res7_1 = ResBlock(int(4*f))
        self.res7_2 = ResBlock(int(4*f))

        # Decoder Stage 8
        self.up8 = nn.Sequential(
            nn.Conv2d(int(4*f), int(2*f) * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2), nn.BatchNorm2d(int(2*f)), nn.SiLU()
        )
        self.fuse8 = ConvBlock(int(2*f) * 2, int(2*f), 1, act_type="swish")
        self.res8_1 = ResBlock(int(2*f))
        self.res8_2 = ResBlock(int(2*f))

        # Decoder Stage 9
        self.up9 = nn.Sequential(
            nn.Conv2d(int(2*f), f * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2), nn.BatchNorm2d(f), nn.SiLU()
        )
        self.fuse9 = ConvBlock(f * 2, f, 1, act_type="swish")
        self.res9_1 = ResBlock(f)
        self.res9_2 = ResBlock(f)

        self.outc = OutConv(num_filters_1st, n_classes)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1_1(x)
        conv1 = self.ite1_1(conv1)
        conv1 = self.ite1_2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.ite2_1(conv2)
        conv2 = self.ite2_2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.ite3_1(conv3)
        conv3 = self.ite3_2(conv3)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = self.ite4_1(conv4)
        conv4 = self.ite4_2(conv4)
        conv5 = self.spp_conv(conv4)

        # ASPP
        aspp = self.aspp(conv5)
        drop5 = self.drop5(aspp)

        # Decoder
        # Decoder Stage 7
        up7 = self.up7(drop5)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.fuse7(merge7)
        conv7 = self.res7_1(conv7)
        conv7 = self.res7_2(conv7)

        # Decoder Stage 8
        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.fuse8(merge8)
        conv8 = self.res8_1(conv8)
        conv8 = self.res8_2(conv8)

        # Decoder Stage 9
        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.fuse9(merge9)
        conv9 = self.res9_1(conv9)
        conv9 = self.res9_2(conv9)

        logits = self.outc(conv9)

        return logits