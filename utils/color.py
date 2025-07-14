import torch
import torch.nn.functional as F
import math

"""
utils2.py
基于 PyTorch 实现的水下图像质量评估（UIQM）指标计算
包含：
 - 色彩保真度（UICM）
 - 清晰度（UISM）
 - 对比度（UIConM）
 - 综合指标 UIQM
"""

# -------- 色彩保真度指标 UICM --------
def mu_a(x: torch.Tensor, alpha_L: float = 0.1, alpha_R: float = 0.1) -> torch.Tensor:
    """
    非对称 alpha 截断均值
    x: 1D 张量
    """
    x_sorted, _ = torch.sort(x)
    K = x_sorted.numel()
    T_L = math.ceil(alpha_L * K)
    T_R = math.floor(alpha_R * K)
    trimmed = x_sorted[T_L:K - T_R]
    return trimmed.mean()


def s_a(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    计算样本方差
    """
    return torch.mean((x - mu)**2)


def _uicm(img: torch.Tensor) -> torch.Tensor:
    """
    Underwater Image Colorfulness Measure
    img: [H,W,3] 或 [3,H,W]
    返回 UICM 标量
    """
    # 假设输入为 [3,H,W]
    if img.ndim == 3:
        R, G, B = img[0], img[1], img[2]
    else:
        R, G, B = img[...,0], img[...,1], img[...,2]
    RG = (R - G).flatten()
    YB = ((R + G) * 0.5 - B).flatten()
    mu_rg = mu_a(RG)
    mu_yb = mu_a(YB)
    sa_rg = s_a(RG, mu_rg)
    sa_yb = s_a(YB, mu_yb)
    l = torch.sqrt(mu_rg**2 + mu_yb**2)
    r = torch.sqrt(sa_rg + sa_yb)
    return -0.0268 * l + 0.1586 * r

# -------- 清晰度指标 UISM --------
def sobel(x: torch.Tensor) -> torch.Tensor:
    """
    Sobel 边缘检测
    x: 单通道 [H,W]
    """
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1,-2)
    gx = F.conv2d(x.unsqueeze(0).unsqueeze(0), kernel_x, padding=1)[0,0]
    gy = F.conv2d(x.unsqueeze(0).unsqueeze(0), kernel_y, padding=1)[0,0]
    mag = torch.hypot(gx, gy)
    return mag


def eme(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Enhancement Measure Estimation
    x: 单通道 [H,W]
    """
    H, W = x.shape
    n_h = H // window
    n_w = W // window
    val = 0.0
    for i in range(n_h):
        for j in range(n_w):
            block = x[i*window:(i+1)*window, j*window:(j+1)*window]
            max_v = block.max()
            min_v = block.min()
            if min_v > 0 and max_v > 0:
                val += math.log(max_v / min_v)
    return (2.0 / (n_h * n_w)) * val


def _uism(img: torch.Tensor) -> torch.Tensor:
    """
    Underwater Image Sharpness Measure
    img: [3,H,W]
    """
    R, G, B = img[0], img[1], img[2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    r_eme = eme(Rs * R, 8)
    g_eme = eme(Gs * G, 8)
    b_eme = eme(Bs * B, 8)
    return 0.299*r_eme + 0.587*g_eme + 0.114*b_eme

# -------- 对比度指标 UIConM --------
# PLIP 相关算子
def plip_g(x: torch.Tensor, mu: float = 1026.0) -> torch.Tensor:
    return mu - x

def plip_phi(x: torch.Tensor, mu: float = 1026.0, beta: float = 1.0) -> torch.Tensor:
    return -mu * torch.log(1 - x/mu)

def plip_phi_inv(x: torch.Tensor, mu: float = 1026.0, beta: float = 1.0) -> torch.Tensor:
    return mu * (1 - torch.exp(-x/mu))


def _uiconm(img: torch.Tensor, window: int) -> torch.Tensor:
    """
    Underwater Image Contrast Measure
    img: [3,H,W]
    """
    C1 = img.shape[1] // window
    C2 = img.shape[2] // window
    val = 0.0
    for i in range(C2):
        for j in range(C1):
            block = img[:, i*window:(i+1)*window, j*window:(j+1)*window]
            max_v = block.max()
            min_v = block.min()
            top = max_v - min_v
            bot = max_v + min_v
            if bot>0 and top>0:
                val += (top/bot)**1 * math.log(top/bot)
    return -val/(C1*C2)

# -------- 综合指标 UIQM --------
def getUIQM(img: torch.Tensor) -> torch.Tensor:
    """
    计算 UIQM 指标
    """
    uicm = _uicm(img)
    uism = _uism(img)
    uiconm = _uiconm(img, 8)
    # 原论文权重
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1*uicm + c2*uism + c3*uiconm
'''

# 写入 utils2.py
(utils2_path := project_root / "utils" / "utils2.py").write_text(blocks_py_content)

# 打包项目
zip_path = "/mnt/data/UWnet_integrated_full.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for path in project_root.rglob("*"):
        zipf.write(path, arcname=path.relative_to(project_root))

zip_path
