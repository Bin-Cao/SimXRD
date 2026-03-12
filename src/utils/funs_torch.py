################################################################
# funs_torch.py - GPU加速工具函数
# 支持 CUDA / Ascend NPU (torch_npu) / CPU
# 使用前调用 init_precision() 设置全局精度
################################################################
import numpy as np
import torch
import math

try:
    import torch_npu
except ImportError:
    torch_npu = None

from .funs import atomics, get_float


# ======================== 精度控制 ========================

DTYPE: torch.dtype = torch.float32
NP_DTYPE = np.float32


def get_device():
    """自动检测设备: NPU > CUDA > CPU"""
    if (torch_npu is not None
            and hasattr(torch_npu, 'npu')
            and torch_npu.npu.is_available()):
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _upcast_dtype():
    """需要 >= float32 精度的运算(FFT, linalg, arcsin等)使用的类型"""
    if DTYPE in (torch.float16, torch.bfloat16):
        return torch.float32
    return DTYPE


def init_precision(dtype=None):
    """初始化全局计算精度

    Parameters
    ----------
    dtype : torch.dtype | str | None
        None   → 自动 (NPU/CUDA: float32, CPU: float64)
        'fp16' / torch.float16
        'fp32' / torch.float32 (推荐)
        'fp64' / torch.float64
        'bf16' / torch.bfloat16
    """
    global DTYPE, NP_DTYPE

    if dtype is None:
        dev = get_device()
        if dev.type in ('npu', 'cuda'):
            DTYPE = torch.float32
        else:
            DTYPE = torch.float64
    elif isinstance(dtype, str):
        _m = {
            'fp16': torch.float16, 'float16': torch.float16, 'half': torch.float16,
            'fp32': torch.float32, 'float32': torch.float32, 'float': torch.float32,
            'fp64': torch.float64, 'float64': torch.float64, 'double': torch.float64,
            'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
        }
        DTYPE = _m.get(dtype.lower(), torch.float32)
    else:
        DTYPE = dtype

    _np = {
        torch.float16: np.float16, torch.float32: np.float32,
        torch.float64: np.float64, torch.bfloat16: np.float32,
    }
    NP_DTYPE = _np.get(DTYPE, np.float32)
    return DTYPE


# ======================== 度量张量 ========================

def cosd_t(x):
    return torch.cos(torch.deg2rad(x))


def fillgmat_t(cell, device=None):
    """实空间度量张量 [3,3]，输入 cell = [a,b,c,α,β,γ]"""
    if device is None:
        device = cell.device
    a, b, c, alp, bet, gam = cell
    g = torch.zeros((3, 3), dtype=DTYPE, device=device)
    g[0, 0] = a * a
    g[1, 1] = b * b
    g[2, 2] = c * c
    g[0, 1] = g[1, 0] = a * b * cosd_t(gam)
    g[0, 2] = g[2, 0] = a * c * cosd_t(bet)
    g[1, 2] = g[2, 1] = b * c * cosd_t(alp)
    return g


def cell2A_t(cell, device=None):
    """倒空间度量张量 A-vector，内部以 upcast 精度计算"""
    if device is None:
        device = cell.device
    up = _upcast_dtype()
    g = fillgmat_t(cell, device).to(up)
    G = torch.linalg.inv(g)
    return torch.stack([G[0, 0], G[1, 1], G[2, 2],
                        2 * G[0, 1], 2 * G[0, 2], 2 * G[1, 2]])


# ======================== 衍射指标 ========================

def Diffraction_index_torch(latt, wavelength, two_theta_range):
    """向量化衍射指标与面间距计算

    使用与CPU版 grid_atom() 相同的 HKL 网格排序，保证一致的 HKL 代表。

    Returns: (res_HKL [P,3], res_d [P])
    """
    device = get_device()
    up = _upcast_dtype()

    # 复现 CPU grid_atom() 排序
    hh, kk, ll = np.mgrid[-11:13:1, -11:13:1, -11:13:1]
    grid_np = np.c_[hh.ravel(), kk.ravel(), ll.ravel()]
    sorted_indices = np.argsort(np.linalg.norm(grid_np, axis=1))
    grid_np = grid_np[sorted_indices]
    idx_origin = np.where(
        (grid_np[:, 0] == 0) & (grid_np[:, 1] == 0) & (grid_np[:, 2] == 0)
    )[0][0]
    grid_np[[0, idx_origin]] = grid_np[[idx_origin, 0]]
    grid_np = grid_np[1:]  # 跳过原点

    # GPU 计算 d-spacing (upcast 精度)
    latt_t = torch.as_tensor(latt, dtype=up, device=device)
    A = cell2A_t(latt_t, device)

    grid_t = torch.from_numpy(grid_np.astype(np.float32)).to(device).to(up)
    h = grid_t[:, 0]
    k = grid_t[:, 1]
    l = grid_t[:, 2]

    rdsq = (h ** 2 * A[0] + k ** 2 * A[1] + l ** 2 * A[2] +
            h * k * A[3] + h * l * A[4] + k * l * A[5])

    valid = rdsq > 1e-12
    h_v, k_v, l_v = h[valid], k[valid], l[valid]
    d_array = 1.0 / torch.sqrt(rdsq[valid])

    bragg_valid = (wavelength / (2 * d_array)) <= 1.0
    h_v, k_v, l_v = h_v[bragg_valid], k_v[bragg_valid], l_v[bragg_valid]
    d_array = d_array[bragg_valid]

    sin_arg = (wavelength / (2 * d_array)).clamp(-1.0, 1.0)
    two_theta = 2 * torch.rad2deg(torch.arcsin(sin_arg))
    range_mask = (two_theta >= two_theta_range[0]) & (two_theta <= two_theta_range[1])
    h_v, k_v, l_v = h_v[range_mask], k_v[range_mask], l_v[range_mask]
    d_array = d_array[range_mask]

    # 去重 (保留首次出现，等价于 CPU de_redundant)
    d_trunc = torch.floor(d_array * 10000) / 10000.0
    unique_d, inverse = torch.unique(d_trunc, sorted=True, return_inverse=True)
    n_all = d_trunc.shape[0]
    n_unique = unique_d.shape[0]
    reverse_order = torch.arange(n_all - 1, -1, -1, device=device)
    first_idx = torch.empty(n_unique, dtype=torch.long, device=device)
    first_idx.scatter_(0, inverse[reverse_order], reverse_order)

    res_HKL = torch.stack([h_v[first_idx], k_v[first_idx], l_v[first_idx]], dim=1).to(DTYPE)
    res_d = d_trunc[first_idx].to(DTYPE)

    return res_HKL, res_d


# ======================== 原子散射因子 ========================

def cal_atoms_torch(ion, two_theta, wavelength, device=None):
    """向量化原子散射因子插值 (bucketize + lerp)"""
    import re
    if device is None:
        device = two_theta.device
    up = _upcast_dtype()

    dict_data = atomics()
    if ion not in dict_data:
        bare = re.sub(r'[^A-Za-z]+', '', ion)
        ion = bare if bare in dict_data else ion
    if ion not in dict_data:
        return torch.zeros(two_theta.shape[0], dtype=DTYPE, device=device)

    ion_data = dict_data[ion]
    raw_keys = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1','1.1']
    x_pts_np = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    y_pts_np = np.array([ion_data[k] for k in raw_keys])
    x_pts = torch.as_tensor(x_pts_np, dtype=up, device=device)
    y_pts = torch.as_tensor(y_pts_np, dtype=up, device=device)

    loc = torch.sin(torch.deg2rad(two_theta.to(up) / 2)) / wavelength
    loc = torch.clamp(loc, x_pts[0], x_pts[-1])

    idx = torch.bucketize(loc, x_pts).clamp(1, len(x_pts) - 1)
    x_lo, x_hi = x_pts[idx - 1], x_pts[idx]
    y_lo, y_hi = y_pts[idx - 1], y_pts[idx]
    t = (loc - x_lo) / (x_hi - x_lo + 1e-30)
    return (y_lo + t * (y_hi - y_lo)).to(DTYPE)


# ======================== 消光判据 ========================

def lattice_extinction_mask(lattice_type, hkl_t, crystal_system):
    """格子消光，返回 bool [P]"""
    h = hkl_t[:, 0].long()
    k = hkl_t[:, 1].long()
    l = hkl_t[:, 2].long()
    P = hkl_t.shape[0]
    extinct = torch.zeros(P, dtype=torch.bool, device=hkl_t.device)

    if lattice_type in ('P', 'R'):
        pass
    elif lattice_type == 'I':
        extinct = torch.abs(h + k + l) % 2 == 1
    elif lattice_type == 'C':
        if crystal_system in (1, 5):
            extinct = ((torch.abs(h + k) % 2 == 1) |
                       (torch.abs(k + l) % 2 == 1) |
                       (torch.abs(h + l) % 2 == 1))
        else:
            extinct = torch.abs(h + k) % 2 == 1
    elif lattice_type == 'F':
        all_even = (torch.abs(h) % 2 == 0) & (torch.abs(k) % 2 == 0) & (torch.abs(l) % 2 == 0)
        all_odd  = (torch.abs(h) % 2 == 1) & (torch.abs(k) % 2 == 1) & (torch.abs(l) % 2 == 1)
        extinct = ~(all_even | all_odd)
    return extinct


def structure_extinction_mask(AtomCoordinates, hkl_t, two_theta_t, wavelength, device=None):
    """结构因子消光，返回 bool [P]"""
    if device is None:
        device = hkl_t.device
    up = _upcast_dtype()
    P = hkl_t.shape[0]
    N = len(AtomCoordinates)

    symbols = [atom[0] for atom in AtomCoordinates]
    positions = torch.tensor([[atom[1], atom[2], atom[3]] for atom in AtomCoordinates],
                             dtype=up, device=device)

    fi = torch.zeros((N, P), dtype=up, device=device)
    for n in range(N):
        fi[n] = cal_atoms_torch(symbols[n], two_theta_t, wavelength, device).to(up)

    phase = 2 * math.pi * (hkl_t.to(up) @ positions.T)
    F_real = torch.sum(fi.T * torch.cos(phase), dim=1)
    F_imag = torch.sum(fi.T * torch.sin(phase), dim=1)
    F_sq = F_real ** 2 + F_imag ** 2

    return F_sq <= 1e-5


def cal_extinction_torch(Point_group, hkl_t, d_t, crystal_system,
                         AtomCoordinates, wavelength):
    """向量化消光计算，返回 (res_HKL, ex_HKL, d_res, d_ex)"""
    device = hkl_t.device
    up = _upcast_dtype()

    two_theta_t = (2 * torch.rad2deg(
        torch.arcsin((wavelength / (2 * d_t.to(up))).clamp(-1, 1)))).to(DTYPE)

    lat_extinct = lattice_extinction_mask(Point_group, hkl_t, crystal_system)

    remaining_idx = torch.where(~lat_extinct)[0]
    struct_extinct_full = torch.zeros(hkl_t.shape[0], dtype=torch.bool, device=device)
    if remaining_idx.numel() > 0:
        struct_extinct = structure_extinction_mask(
            AtomCoordinates, hkl_t[remaining_idx],
            two_theta_t[remaining_idx], wavelength, device)
        struct_extinct_full[remaining_idx] = struct_extinct

    all_extinct = lat_extinct | struct_extinct_full
    diff_mask = ~all_extinct
    return hkl_t[diff_mask], hkl_t[all_extinct], d_t[diff_mask], d_t[all_extinct]


# ======================== 多重性因子 ========================

def mult_rule_torch(H, K, L, crystal_system, device=None):
    """向量化多重性因子，与 CPU mult_rule 逻辑完全一致"""
    if device is None:
        device = H.device
    P = H.shape[0]

    H0 = (H == 0); K0 = (K == 0); L0 = (L == 0)
    HK = (H == K); HL = (H == L); KL = (K == L)
    HKL = HK & KL
    nH = ~H0; nK = ~K0; nL = ~L0
    all_diff = ~HK & ~KL & ~HL & nH & nK & nL
    two0 = (HK & H0 & nL) | (HL & H0 & nK) | (KL & K0 & nH)
    two_eq_one0 = (HK & nH & L0) | (HL & nH & K0) | (KL & nK & H0)
    hkl_eq_nz = HKL & nH

    mult = torch.ones(P, dtype=DTYPE, device=device)

    if crystal_system == 1:  # Cubic
        mult = torch.where(two0, 6.0, mult)
        mult = torch.where(two_eq_one0, 12.0, mult)
        mult = torch.where(all_diff, 48.0, mult)
        mult = torch.where(hkl_eq_nz, 8.0, mult)
        mult = torch.where(~(two0 | two_eq_one0 | all_diff | hkl_eq_nz), 24.0, mult)

    elif crystal_system == 2:  # Hexagonal
        c6a = (HL & H0 & nK) | (KL & K0 & nH)
        c2  = HK & H0 & nL
        c6b = HK & nH & L0
        mult = torch.where(c6a, 6.0, mult)
        mult = torch.where(c2, 2.0, mult)
        mult = torch.where(c6b, 6.0, mult)
        mult = torch.where(all_diff, 24.0, mult)
        mult = torch.where(hkl_eq_nz, 1.0, mult)
        mult = torch.where(~(c6a | c2 | c6b | all_diff | hkl_eq_nz), 12.0, mult)

    elif crystal_system == 5:  # Trigonal
        mult = torch.where(two0, 6.0, mult)
        mult = torch.where(two_eq_one0, 6.0, mult)
        mult = torch.where(all_diff, 24.0, mult)
        mult = torch.where(hkl_eq_nz, 1.0, mult)
        mult = torch.where(~(two0 | two_eq_one0 | all_diff | hkl_eq_nz), 12.0, mult)

    elif crystal_system == 3:  # Tetragonal
        c4a = (HL & H0 & nK) | (KL & K0 & nH)
        c2  = HK & H0 & nL
        c4b = HK & nH & L0
        mult = torch.where(c4a, 4.0, mult)
        mult = torch.where(c2, 2.0, mult)
        mult = torch.where(c4b, 4.0, mult)
        mult = torch.where(all_diff, 16.0, mult)
        mult = torch.where(hkl_eq_nz, 1.0, mult)
        mult = torch.where(~(c4a | c2 | c4b | all_diff | hkl_eq_nz), 8.0, mult)

    elif crystal_system == 4:  # Orthorhombic
        c1 = hkl_eq_nz | (HK & nH & L0) | (HK & ~KL & nH & nL)
        mult = torch.where(two0, 2.0, mult)
        mult = torch.where(all_diff, 8.0, mult)
        mult = torch.where(c1, 1.0, mult)
        mult = torch.where(~(two0 | all_diff | c1), 4.0, mult)

    elif crystal_system == 6:  # Monoclinic
        c1  = hkl_eq_nz | (HK & nH & L0) | (HK & ~KL & nH & nL)
        c2a = two0
        c2b = ~HL & nH & nL & K0
        mult = torch.where(c2a, 2.0, mult)
        mult = torch.where(all_diff, 4.0, mult)
        mult = torch.where(c1, 1.0, mult)
        mult = torch.where(c2b, 2.0, mult)
        mult = torch.where(~(c2a | all_diff | c1 | c2b), 4.0, mult)

    elif crystal_system == 7:  # Triclinic
        c1 = hkl_eq_nz | (HK & nH & L0) | (HK & ~KL & nH & nL)
        mult = torch.where(c1, 1.0, mult)
        mult = torch.where(~c1, 2.0, mult)

    return mult


# ======================== 批量峰形函数 ========================

def batch_fft_convolve_same(a, b):
    """批量一维 FFT 卷积，输出保持输入精度"""
    out_dtype = a.dtype
    up = _upcast_dtype()
    N = a.shape[-1]
    n_fft = N * 2
    a_up = a.to(up)
    b_up = b.to(up)
    if b_up.shape != a_up.shape:
        b_up = b_up.expand_as(a_up).contiguous()
    A = torch.fft.rfft(a_up, n=n_fft, dim=-1)
    B = torch.fft.rfft(b_up, n=n_fft, dim=-1)
    c = torch.fft.irfft(A * B, n=n_fft, dim=-1)
    start = (N - 1) // 2
    return c[..., start:start + N].to(out_dtype)


def voigt_batch_torch(x_local, gammas, sigma2s, device=None):
    """批量 Voigt 峰形 (Gaussian ⊗ Lorentzian via FFT)"""
    if device is None:
        device = x_local.device
    x   = x_local
    gam = gammas.unsqueeze(1)
    sig = torch.sqrt(sigma2s).unsqueeze(1)

    gauss   = (torch.exp(-x.unsqueeze(0) ** 2 / (2 * sig ** 2))
               / (sig * math.sqrt(2 * math.pi)))
    lorentz = gam / (math.pi * (x.unsqueeze(0) ** 2 + gam ** 2))

    return batch_fft_convolve_same(gauss, lorentz)


def axial_div_batch_torch(x_local, mus, L, H, S, device=None):
    """批量轴向发散函数 (Van Laar & Yelon 1984)"""
    if device is None:
        device = x_local.device
    up = _upcast_dtype()

    x  = x_local.to(up)
    mu = mus.to(up)

    x_abs = x.unsqueeze(0) + mu.unsqueeze(1)
    left_mask = (x_local <= 0).unsqueeze(0).to(up)

    cos_x  = torch.cos(torch.deg2rad(x_abs))
    cos_mu = torch.cos(torch.deg2rad(mu)).unsqueeze(1)

    cos_ratio_sq = torch.clamp((cos_x / cos_mu) ** 2 - 1, min=0)
    h = L * torch.sqrt(cos_ratio_sq)

    W = torch.clamp(H + S - h, min=0) * (h >= H - S).to(up)

    h_safe   = torch.clamp(h, min=1e-10)
    cos_safe = torch.clamp(torch.abs(cos_x), min=1e-10)
    axial_raw = (L / (2 * H * S)) * W / (h_safe * cos_safe) * left_mask

    axial_max  = axial_raw.max(dim=1, keepdim=True)[0].clamp(min=1e-10)
    axial_norm = axial_raw / axial_max

    n_left = int((x_local < 0).sum().item())
    cdf = torch.zeros_like(axial_norm)
    if n_left > 0:
        cdf[:, :n_left] = torch.cumsum(axial_norm[:, :n_left], dim=1)
    return cdf.to(DTYPE)


# ======================== 组合峰形 ========================

def _get_l_gap(mu_val):
    """根据 2θ 位置返回局部窗口宽度 (与 CPU combined_peak 一致)"""
    if mu_val <= 10:
        return 7.8
    elif mu_val <= 15:
        return 10.0
    elif mu_val <= 20:
        return 15.0
    elif mu_val <= 30:
        return 20.0
    else:
        return 30.0


def combined_peak_batch_torch(twotheta_t, Weights_t, mus_t, gammas_t, sigma2s_t,
                              L, H, S, step=0.02):
    """批量峰形生成，按 l_gap 分组处理"""
    device = twotheta_t.device
    up = _upcast_dtype()
    P = mus_t.shape[0]
    M = twotheta_t.shape[0]

    y_sim = torch.zeros(M, dtype=up, device=device)

    mus_np = mus_t.cpu().float().numpy()
    l_gaps = np.array([_get_l_gap(float(m)) for m in mus_np])

    for l_gap in sorted(set(l_gaps)):
        mask = (l_gaps == l_gap)
        if not mask.any():
            continue

        idx = np.where(mask)[0]
        idx_t = torch.from_numpy(idx).long().to(device)
        P_sub = len(idx)

        local_M = int(round(2 * l_gap / step))
        x_local = torch.linspace(-l_gap, l_gap - step, local_M,
                                 dtype=up, device=device)

        # Voigt
        voigt = voigt_batch_torch(
            x_local.to(DTYPE), gammas_t[idx_t], sigma2s_t[idx_t], device
        ).to(up)
        # 轴向发散
        axial = axial_div_batch_torch(
            x_local.to(DTYPE), mus_t[idx_t], L, H, S, device
        ).to(up)
        # 狭缝函数
        slit_w = 0.1
        slit = torch.where(
            (x_local >= -slit_w / 2) & (x_local <= slit_w / 2),
            torch.tensor(1.0 / slit_w, dtype=up, device=device),
            torch.tensor(0.0, dtype=up, device=device),
        ).unsqueeze(0)
        # 晶格畸变
        sigma_d = math.sqrt(0.001)
        distor = (torch.exp(-x_local ** 2 / (2 * 0.001)) /
                  (sigma_d * math.sqrt(2 * math.pi))).unsqueeze(0)

        # 逐步 FFT 卷积
        combined = batch_fft_convolve_same(voigt, axial)
        combined = batch_fft_convolve_same(combined, slit)
        combined = batch_fft_convolve_same(combined, distor)

        # 归一化
        peak_sum = combined.sum(dim=1, keepdim=True) * step
        peak_sum = torch.clamp(peak_sum, min=1e-20)
        combined = combined / peak_sum
        combined = combined * Weights_t[idx_t].unsqueeze(1).to(up)

        # 散布到全局网格
        offsets = torch.round(
            (mus_t[idx_t].to(up) - twotheta_t[0].to(up)) / step
        ).long()
        local_center = local_M // 2
        base_indices = offsets - local_center

        m_indices = torch.arange(local_M, device=device).unsqueeze(0)
        global_indices = base_indices.unsqueeze(1) + m_indices
        valid = (global_indices >= 0) & (global_indices < M)

        flat_idx = global_indices.clamp(0, M - 1).reshape(-1)
        flat_val = (combined * valid.to(up)).reshape(-1)

        y_sim.scatter_add_(0, flat_idx, flat_val)

    return y_sim.to(DTYPE)


# ======================== 辅助函数 ========================

def theta_intensity_area_torch(theta_data, intensity):
    """梯形积分"""
    h  = (intensity[:-1] + intensity[1:]) / 2
    dx = theta_data[1:] - theta_data[:-1]
    return (h * dx).sum()


def scale_list_torch(lst):
    """缩放到 [0, 100]"""
    mx = lst.max()
    if mx == 0:
        return torch.zeros_like(lst)
    return lst * (100.0 / mx)


def generate_random_polynomial_torch(degree, x, device=None):
    """随机多项式 (Horner 法)"""
    if device is None:
        device = x.device
    up = _upcast_dtype()
    coeffs = torch.randn(degree + 1, dtype=up, device=device).to(DTYPE)
    y = coeffs[0]
    for i in range(1, len(coeffs)):
        y = y * x + coeffs[i]
    return y
