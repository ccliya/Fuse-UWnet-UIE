import torch
import torch.nn.functional as F
import math
import numpy as np
import zipfile
from pathlib import Path


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
def mu_a(x, alpha_L: float = 0.1, alpha_R: float = 0.1) -> torch.Tensor:
    """
    非对称 alpha 截断均值
    x: 1D 张量或 ndarray
    """
    # 如果是 numpy.ndarray，则转换为 torch.Tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # 如果是 torch.Tensor 且非 1D，也可以加判断处理
    if x.dim() != 1:
        x = x.view(-1)

    x_sorted, _ = torch.sort(x)
    K = x_sorted.numel()
    T_L = math.ceil(alpha_L * K)
    T_R = math.floor(alpha_R * K)
    trimmed = x_sorted[T_L:K - T_R]
    return trimmed.float().mean()



def s_a(x, mu):
    """
    计算样本方差（支持 x 为 numpy 或 Tensor，mu 为标量或 Tensor）
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(mu, np.ndarray):
        mu = torch.from_numpy(mu).float()
    elif isinstance(mu, (int, float)):
        mu = torch.tensor(mu).float()
    elif isinstance(mu, torch.Tensor) and mu.numel() == 1:
        mu = mu.float()
    return torch.mean((x - mu) ** 2)


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
def sobel(x) -> torch.Tensor:
    """
    Sobel 边缘检测
    x: 输入图像 [H,W]，可为 numpy.ndarray 或 torch.Tensor，要求单通道
    返回: 梯度幅值图像，类型为 float32 Tensor
    """
    # 转换为 float32 Tensor，确保在同一设备上
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        if not x.is_floating_point():
            x = x.float()
    else:
        raise TypeError("Input must be a numpy array or torch Tensor")

    device = x.device if x.is_cuda else torch.device("cpu")

    # 构建 Sobel 卷积核
    kernel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)

    # 添加 batch 和 channel 维度: [1,1,H,W]
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    gx = F.conv2d(x, kernel_x, padding=1)[0, 0]
    gy = F.conv2d(x, kernel_y, padding=1)[0, 0]

    mag = torch.hypot(gx, gy)  # √(gx² + gy²)
    return mag


def eme(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Enhancement Measure Estimation
    x: 单通道图像，维度 [H, W]
    window: 分块窗口大小
    """
    H, W = x.shape
    n_h = H // window
    n_w = W // window

    if n_h == 0 or n_w == 0:
        # 图像太小，无法分块，返回 0
        return torch.tensor(0.0, dtype=torch.float32)

    val = 0.0
    for i in range(n_h):
        for j in range(n_w):
            block = x[i*window:(i+1)*window, j*window:(j+1)*window]
            max_v = block.max().item()
            min_v = block.min().item()
            if min_v > 0 and max_v > 0:
                val += math.log(max_v / min_v + 1e-8)  # 避免 log(0)

    result = (2.0 / (n_h * n_w)) * val
    return torch.tensor(result, dtype=torch.float32)


def _uism(img: torch.Tensor) -> torch.Tensor:
    """
    Underwater Image Sharpness Measure (UISM)
    img: torch.Tensor of shape [3, H, W]
    Returns: scalar tensor
    """
    R, G, B = img[0], img[1], img[2]

    # Apply Sobel edge detection
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # Multiply with original channels to enhance edge + intensity contrast
    try:
        r_eme = eme(Rs * R, 8)
        g_eme = eme(Gs * G, 8)
        b_eme = eme(Bs * B, 8)
    except ZeroDivisionError:
        print("[Warning] ZeroDivisionError in eme calculation, returning 0")
        return torch.tensor(0.0)

    # Weighted sum of the EME values per channel
    return 0.299 * r_eme + 0.587 * g_eme + 0.114 * b_eme


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
    Underwater Image Contrast Measure (UIConM)
    img: [3, H, W]
    window: block size
    """
    H, W = img.shape[1], img.shape[2]
    C1 = H // window
    C2 = W // window
    eps = 1e-6  # 防止除零

    if C1 == 0 or C2 == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    val = 0.0
    valid_blocks = 0

    for i in range(C2):
        for j in range(C1):
            block = img[:, i * window:(i + 1) * window, j * window:(j + 1) * window]
            max_v = block.max().item()
            min_v = block.min().item()
            top = max_v - min_v
            bot = max_v + min_v
            if bot > eps and top > eps:
                val += plip_phi(top / bot)
                valid_blocks += 1
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
'''