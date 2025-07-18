
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ConvBlock, ELA, MambaBlock, BAM, AdaIN, DenSoA
from utils.color import rgb_to_lab
from models.deconv import DEConv

class UWnet(nn.Module):
    def __init__(self, base_ch=64):
        super(UWnet, self).__init__()
        self.base_ch = base_ch

        # 输入层
        self.rgb_in = nn.Conv2d(3, base_ch, 3, padding=1, bias=False)
        self.lab_in = nn.Conv2d(3, base_ch, 3, padding=1, bias=False)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.mamba_block = MambaBlock(dim=base_ch)
        self.bam = BAM(base_ch)
        self.adain = AdaIN()
        self.densoa = DenSoA(base_ch)
        self.conv_post_adain = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)

        # 浅层特征融合
        self.shallow_conv = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.shallow_norm = nn.InstanceNorm2d(base_ch)

        # 中层增强
        self.blocks = nn.Sequential(*[ConvBlock() for _ in range(2)])
        self.ela = ELA(base_ch)
        self.deconv = DEConv(in_channels=base_ch, out_channels=base_ch)

        self.mid_fuse = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.mid_norm = nn.InstanceNorm2d(base_ch)

        # 深层增强
        self.deep_fuse = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.deep_norm = nn.InstanceNorm2d(base_ch)

        # 输出层
        self.final_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def encode_branch(self, x, conv):
        f = self.relu(conv(x))
        p = self.pool(f)
        B, C, Hp, Wp = p.shape
        seq = p.view(B, C, -1).permute(0, 2, 1)
        seq = self.mamba_block(seq)
        p = seq.permute(0, 2, 1).view(B, C, Hp, Wp)
        u = F.interpolate(p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return self.relu(u), f

    def forward(self, x):
        lab = rgb_to_lab(x)

        # 浅层处理
        o0, _ = self.encode_branch(x, self.rgb_in)
        oI0, _ = self.encode_branch(lab, self.lab_in)

        # 注意力与风格对齐
        fused = self.bam(o0 + oI0)
        stylized = self.densoa(self.adain(fused, oI0))
        stylized = self.relu(self.conv_post_adain(stylized))

        # 浅层融合
        sh = self.shallow_conv(torch.cat([stylized, oI0], dim=1))
        sh = self.shallow_norm(sh)

        # 中层增强
        om = self.ela(self.deconv(self.blocks(sh)))
        olm = self.ela(self.deconv(self.blocks(oI0)))
        fused_mid = self.densoa(self.adain(om, olm))
        fused_mid = self.relu(self.conv_post_adain(fused_mid))
        mid = self.mid_fuse(torch.cat([fused_mid, olm], dim=1))
        mid = self.mid_norm(mid)

        # 深层增强
        od = self.ela(self.deconv(self.blocks(mid)))
        old = self.ela(self.deconv(self.blocks(olm)))
        fused_deep = self.densoa(self.adain(od, old))
        fused_deep = self.relu(self.conv_post_adain(fused_deep))
        dp = self.deep_fuse(torch.cat([fused_deep, old], dim=1))
        dp = self.deep_norm(dp)

        return self.final_conv(dp)
