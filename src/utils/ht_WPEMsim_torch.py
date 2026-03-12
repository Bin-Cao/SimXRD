################################################################
# ht_WPEMsim_torch.py - 高通量 GPU 向量化 XRD 模拟
#
# 对同一晶体结构批量生成 T 条不同噪声配置的衍射谱图。
# deformation=False: 结构计算一次，T 条噪声 GPU 并行 (最快)
# deformation=True:  循环调用 pxrdsim_torch (每次仍有 ~40× 加速)
################################################################
import numpy as np
import torch
import math
import random

from ase.spacegroup import get_spacegroup

from .funs import prim2conv
from .WPEMsim import c_atom_covert2WPEMformat, space_group_to_crystal_system
from .WPEMsim_torch import pxrdsim_torch, _adjust_length_torch
from .funs_torch import (
    DTYPE, NP_DTYPE, get_device, _upcast_dtype,
    Diffraction_index_torch, cal_extinction_torch,
    cal_atoms_torch, mult_rule_torch,
    batch_fft_convolve_same, voigt_batch_torch, axial_div_batch_torch,
    _get_l_gap,
)


# ======================== 高通量批量峰形生成 ========================

def _combined_peak_ht_torch(twotheta_t, Weights_t, mus_t, gammas_t, sigma2s_t,
                            L, H, S, step=0.02):
    """高通量批量峰形生成

    与 combined_peak_batch_torch 的区别: 支持 T 条谱图并行。
    mus_t 在 T 条谱图间共享 (同一晶体结构)，gamma/sigma2/Weights 各不相同。

    Parameters
    ----------
    twotheta_t : [M]          全局 2θ 网格
    Weights_t  : [T, P]       各峰强度
    mus_t      : [P]          共享 2θ 位置
    gammas_t   : [T, P]       Lorentzian 半宽
    sigma2s_t  : [T, P]       Gaussian 方差
    L, H, S    : float        仪器参数

    Returns
    -------
    y_sim : [T, M]
    """
    device = twotheta_t.device
    up = _upcast_dtype()
    T, P = Weights_t.shape
    M = twotheta_t.shape[0]

    y_flat = torch.zeros(T * M, dtype=up, device=device)

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

        # Voigt: gamma/sigma2 随 T 变化 → [T*P_sub, local_M]
        voigt = voigt_batch_torch(
            x_local.to(DTYPE),
            gammas_t[:, idx_t].reshape(-1),
            sigma2s_t[:, idx_t].reshape(-1),
            device
        ).to(up)

        # 轴向发散: mus 共享 → 计算一次 [P_sub] 再复制 T 份
        axial = axial_div_batch_torch(
            x_local.to(DTYPE), mus_t[idx_t], L, H, S, device
        ).to(up).repeat(T, 1)      # [T*P_sub, local_M]

        # 狭缝 + 晶格畸变 (所有峰共享)
        slit_w = 0.1
        slit = torch.where(
            (x_local >= -slit_w / 2) & (x_local <= slit_w / 2),
            torch.tensor(1.0 / slit_w, dtype=up, device=device),
            torch.tensor(0.0, dtype=up, device=device),
        ).unsqueeze(0)
        sigma_d = math.sqrt(0.001)
        distor = (torch.exp(-x_local ** 2 / (2 * 0.001)) /
                  (sigma_d * math.sqrt(2 * math.pi))).unsqueeze(0)

        # 逐步 FFT 卷积
        combined = batch_fft_convolve_same(voigt, axial)
        combined = batch_fft_convolve_same(combined, slit)
        combined = batch_fft_convolve_same(combined, distor)

        # 归一化 + 加权
        peak_sum = combined.sum(dim=1, keepdim=True) * step
        combined = combined / peak_sum.clamp(min=1e-20)
        combined = combined * Weights_t[:, idx_t].reshape(-1, 1).to(up)

        # 散布到 [T*M] 展平网格
        offsets = torch.round(
            (mus_t[idx_t].to(up) - twotheta_t[0].to(up)) / step
        ).long()                                         # [P_sub]
        base_idx = offsets - local_M // 2                # [P_sub]

        m_idx = torch.arange(local_M, device=device).unsqueeze(0)
        global_idx = base_idx.unsqueeze(1) + m_idx       # [P_sub, local_M]
        valid = (global_idx >= 0) & (global_idx < M)

        # 扩展到 T 个批次: 每个批次偏移 t*M
        global_idx_exp = global_idx.repeat(T, 1)         # [T*P_sub, local_M]
        valid_exp = valid.repeat(T, 1)
        batch_off = torch.arange(T, device=device).repeat_interleave(P_sub) * M
        flat_idx = (global_idx_exp + batch_off.unsqueeze(1)).clamp(0, T * M - 1).reshape(-1)
        flat_val = (combined * valid_exp.to(up)).reshape(-1)

        y_flat.scatter_add_(0, flat_idx, flat_val)

    return y_flat.reshape(T, M).to(DTYPE)


# ======================== 批量模拟 (deformation=False) ========================

def ht_pxrdsim_batch(Volume, Point_group, AtomCoordinates, latt, crystal_system,
                     T, grainsize_arr, orientation_arr, thermo_vib_arr, zero_shift_arr,
                     L, H, S, background_order=6, background_ratio=0.05,
                     mixture_noise_ratio=0.02, xrd='reciprocal'):
    """批量生成 T 条 WPEM 谱图 (结构仅计算一次)

    Parameters
    ----------
    T               : int       谱图数量
    grainsize_arr   : [T]       晶粒尺寸
    orientation_arr : [T]       择优取向标准差
    thermo_vib_arr  : [T]       热振动幅度
    zero_shift_arr  : [T]       零点偏移
    其余参数与 pxrdsim_torch 一致

    Returns
    -------
    x : numpy [M_out]          共享 x 轴
    Y : numpy [T, M_out]       T 条谱图
    """
    device = get_device()
    up = _upcast_dtype()
    wavelength = 1.54184
    two_theta_range = (8, 82.02, 0.02)
    step = two_theta_range[2]

    # ---- 结构计算 (仅一次) ----
    grid_t, d_t = Diffraction_index_torch(latt, wavelength, two_theta_range)
    res_HKL_t, _, d_res_t, _ = cal_extinction_torch(
        Point_group, grid_t, d_t, crystal_system, AtomCoordinates, wavelength)

    if res_HKL_t.shape[0] == 0:
        x_out = np.arange(10, 80, step, dtype=NP_DTYPE)
        return x_out, np.zeros((T, len(x_out)), dtype=NP_DTYPE)

    P = res_HKL_t.shape[0]
    N = len(AtomCoordinates)

    # 2θ 位置
    mu_up = 2 * torch.rad2deg(
        torch.arcsin((wavelength / (2 * d_res_t.to(up))).clamp(-1, 1)))
    mu_t = mu_up.to(DTYPE)                     # [P]

    # |F(hkl)|²
    symbols = [atom[0] for atom in AtomCoordinates]
    pos_up = torch.tensor(
        [[a[1], a[2], a[3]] for a in AtomCoordinates], dtype=up, device=device)
    fi_up = torch.zeros((N, P), dtype=up, device=device)
    for n in range(N):
        fi_up[n] = cal_atoms_torch(symbols[n], mu_t, wavelength, device).to(up)

    phase = 2 * math.pi * (res_HKL_t.to(up) @ pos_up.T)
    F_real = torch.sum(fi_up.T * torch.cos(phase), dim=1)
    F_imag = torch.sum(fi_up.T * torch.sin(phase), dim=1)
    FHKL_sq = F_real ** 2 + F_imag ** 2        # [P]

    # 多重性 + Lorentz-偏振
    H_i = res_HKL_t[:, 0].long()
    K_i = res_HKL_t[:, 1].long()
    L_i = res_HKL_t[:, 2].long()
    Mult = mult_rule_torch(H_i, K_i, L_i, crystal_system, device)

    mu_rad = torch.deg2rad(mu_up)
    mu_half = torch.deg2rad(mu_up / 2)
    _Ints = (FHKL_sq * Mult.to(up) / Volume ** 2 *
             (1 + torch.cos(mu_rad) ** 2) /
             (torch.sin(mu_half) ** 2 *
              torch.cos(torch.deg2rad(Mult.to(up) / 2))))  # [P]

    # ---- 批量噪声参数 → [T, P] ----
    gs = torch.as_tensor(grainsize_arr, dtype=up, device=device)     # [T]
    ori = torch.as_tensor(orientation_arr, dtype=up, device=device)  # [T]
    tv = torch.as_tensor(thermo_vib_arr, dtype=up, device=device)    # [T]
    zs = torch.as_tensor(zero_shift_arr, dtype=up, device=device)    # [T]

    # Scherrer 展宽: [T, P]
    cos_half = torch.cos(mu_half)                                     # [P]
    Gamma = 0.888 * wavelength / (gs.unsqueeze(1) * cos_half.unsqueeze(0))
    gamma_list = (Gamma / 2 + 1e-10).to(DTYPE)
    sigma2_list = (Gamma ** 2 / (8 * math.sqrt(2)) + 1e-10).to(DTYPE)

    # 择优取向: [T, P]
    Ori_coe = torch.clamp(
        torch.normal(1.0, 0.2, size=(T, P), dtype=up, device=device),
        1 - ori.unsqueeze(1), 1 + ori.unsqueeze(1))

    # Debye-Waller: [T, P]
    sin_half_lam = torch.sin(mu_half) / wavelength                    # [P]
    M_dw = (8 / 3 * math.pi ** 2
            * tv.unsqueeze(1) ** 2 * sin_half_lam.unsqueeze(0) ** 2)
    Deb_coe = torch.exp(-2 * M_dw)

    Ints = (_Ints.unsqueeze(0) * Ori_coe * Deb_coe).to(DTYPE)        # [T, P]

    # ---- 批量峰形生成 ----
    x_sim = torch.arange(two_theta_range[0], two_theta_range[1], step,
                         dtype=DTYPE, device=device)
    M = x_sim.shape[0]

    y_sim = _combined_peak_ht_torch(
        x_sim, Ints, mu_t, gamma_list, sigma2_list,
        L, H, S, step)                                               # [T, M]

    # ---- 归一化: [T, M] ----
    h = (y_sim[:, :-1] + y_sim[:, 1:]) / 2
    dx = (x_sim[1:] - x_sim[:-1]).unsqueeze(0)
    area = (h * dx).sum(dim=1, keepdim=True).clamp(min=1e-30)
    nor_y = y_sim / area

    # ---- 背景 + 噪声: [T, M] ----
    x_shifted = x_sim.unsqueeze(0) + zs.unsqueeze(1).to(DTYPE)       # [T, M]

    coeffs = torch.randn(T, background_order + 1, dtype=DTYPE, device=device)
    _bac = coeffs[:, 0:1]
    for i in range(1, background_order + 1):
        _bac = _bac * x_shifted + coeffs[:, i:i + 1]
    _bac = _bac - _bac.min(dim=1, keepdim=True)[0]
    bac_max = _bac.max(dim=1, keepdim=True)[0].clamp(min=1e-30)
    nor_y_max = nor_y.max(dim=1, keepdim=True)[0]
    _bacI = _bac / bac_max * nor_y_max * background_ratio

    mixture = (torch.rand(T, M, dtype=up, device=device).to(DTYPE)
               * nor_y_max * mixture_noise_ratio)

    nor_y = nor_y + torch.flip(_bacI, dims=[1]) + mixture

    # 缩放到 [0, 100]
    row_max = nor_y.max(dim=1, keepdim=True)[0].clamp(min=1e-30)
    nor_y = nor_y * 100.0 / row_max

    # ---- 截取 [10, 80] (逐行，因 zero_shift 不同) ----
    x_out = torch.arange(10, 80, step, dtype=DTYPE, device=device)
    M_out = x_out.shape[0]
    Y_out = torch.zeros(T, M_out, dtype=DTYPE, device=device)

    for t in range(T):
        xs = x_sim + zs[t].to(DTYPE)
        i10 = torch.argmin(torch.abs(xs - 10)).item()
        i80 = torch.argmin(torch.abs(xs - 80)).item() + 1
        row = nor_y[t, i10:i80]
        Y_out[t] = _adjust_length_torch(row, M_out)

    if xrd == 'real':
        x_out = wavelength / (2 * torch.sin(x_out / 2 * math.pi / 180))

    return x_out.cpu().numpy(), Y_out.cpu().numpy()


# ======================== 高层接口 ========================

def ht_process_entry(atoms, times, deformation=False,
                     grainsize_range=(2, 20),
                     orientation_range=(0.0, 0.4),
                     thermo_vib_range=(0.0, 0.3),
                     zero_shift_range=(-1.5, 1.5),
                     lattice_extinction_ratio=0.01,
                     lattice_torsion_ratio=0.01,
                     dis_detector2sample=500,
                     half_height_slit_detector=5,
                     half_height_sample=2.5,
                     background_order=6,
                     background_ratio=0.05,
                     mixture_noise_ratio=0.02,
                     xrd='reciprocal',
                     grainsize_arr=None,
                     orientation_arr=None,
                     thermo_vib_arr=None,
                     zero_shift_arr=None):
    """对单个晶体生成 times 条 WPEM 衍射谱图

    Parameters
    ----------
    atoms       : ase.Atoms
    times       : int           生成谱图数量
    deformation : bool
        False → 全向量化 (结构计算一次，噪声 GPU 并行)
        True  → 循环调用 pxrdsim_torch (每次仍有 ~40× 加速)
    grainsize_arr / orientation_arr / thermo_vib_arr / zero_shift_arr :
        可选预生成参数数组 (长度 times)。为 None 时自动从对应 range 随机生成。

    Returns
    -------
    x_list : list of numpy arrays
    y_list : list of numpy arrays
    """
    L = dis_detector2sample / 10
    H_slit = half_height_slit_detector
    S_slit = half_height_sample

    # 使用预生成参数或自动随机生成
    gs_arr = np.asarray(grainsize_arr) if grainsize_arr is not None \
        else np.array([random.uniform(*grainsize_range) for _ in range(times)])
    ori_arr = np.asarray(orientation_arr) if orientation_arr is not None \
        else np.array([random.uniform(*orientation_range) for _ in range(times)])
    tv_arr = np.asarray(thermo_vib_arr) if thermo_vib_arr is not None \
        else np.array([random.uniform(*thermo_vib_range) for _ in range(times)])
    zs_arr = np.asarray(zero_shift_arr) if zero_shift_arr is not None \
        else np.array([random.uniform(*zero_shift_range) for _ in range(times)])

    if not deformation:
        # 全向量化
        latt, _, c_atom = prim2conv(atoms, False,
                                    lattice_extinction_ratio,
                                    lattice_torsion_ratio)
        sg = get_spacegroup(c_atom)
        crystal_system = space_group_to_crystal_system(sg.no)
        Point_group = sg.symbol[0]
        AtomCoords = c_atom_covert2WPEMformat(
            c_atom.get_scaled_positions(), c_atom.get_chemical_symbols())
        Volume = c_atom.get_volume()

        x, Y = ht_pxrdsim_batch(
            Volume, Point_group, AtomCoords, latt, crystal_system,
            times, gs_arr, ori_arr, tv_arr, zs_arr,
            L, H_slit, S_slit,
            background_order, background_ratio, mixture_noise_ratio, xrd)

        return [x] * times, [Y[t] for t in range(times)]

    else:
        # 循环模式
        x_list, y_list = [], []
        for t in range(times):
            latt, _, c_atom = prim2conv(atoms, True,
                                        lattice_extinction_ratio,
                                        lattice_torsion_ratio)
            sg = get_spacegroup(c_atom)
            crystal_system = space_group_to_crystal_system(sg.no)
            Point_group = sg.symbol[0]
            AtomCoords = c_atom_covert2WPEMformat(
                c_atom.get_scaled_positions(), c_atom.get_chemical_symbols())
            Volume = c_atom.get_volume()

            x, y = pxrdsim_torch(
                Volume, Point_group, AtomCoords, latt, crystal_system,
                gs_arr[t], [ori_arr[t], ori_arr[t]], tv_arr[t], zs_arr[t],
                L, H_slit, S_slit,
                background_order, background_ratio, mixture_noise_ratio, xrd)
            x_list.append(x)
            y_list.append(y)

        return x_list, y_list


def ht_simulator(db_file, sv_file, times=20, deformation=False, verbose=True):
    """批量模拟数据库中所有晶体的衍射谱图

    Parameters
    ----------
    db_file     : str    输入 .db 文件
    sv_file     : str    输出 .db 文件
    times       : int    每个晶体生成的谱图数
    deformation : bool   是否启用晶格畸变
    """
    from tqdm import tqdm
    from ase.db import connect

    database = connect(db_file)
    total = database.count()
    entries = list(range(1, total + 1))
    databs = connect(sv_file)
    count = 0

    iterator = tqdm(entries, desc='高通量模拟') if verbose else entries

    for _id in iterator:
        try:
            entry = database.get(id=_id)
            label = getattr(entry, 'Label', _id)
            atoms = entry.toatoms()

            x_list, y_list = ht_process_entry(
                atoms, times, deformation=deformation)

            for t in range(times):
                databs.write(
                    atoms=atoms,
                    latt_dis=str(x_list[t]),
                    intensity=str(y_list[t]),
                    Label=label,
                )
                count += 1

        except Exception as e:
            if verbose:
                print(f"  跳过 id={_id}: {e}")

    if verbose:
        print(f"完成: 共写入 {count} 条记录到 {sv_file}")
    return True
