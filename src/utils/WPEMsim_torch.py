################################################################
# WPEMsim_torch.py - GPU加速 WPEM 衍射模拟
# 精度由 funs_torch.DTYPE 控制
# 支持 CUDA / Ascend NPU (torch_npu) / CPU
################################################################
from .funs_torch import (
    DTYPE, NP_DTYPE, get_device, _upcast_dtype, init_precision,
    Diffraction_index_torch, cal_extinction_torch,
    cal_atoms_torch, mult_rule_torch, combined_peak_batch_torch,
    theta_intensity_area_torch, scale_list_torch,
    generate_random_polynomial_torch)
from .WPEMsim import c_atom_covert2WPEMformat, space_group_to_crystal_system
import torch
import math
import numpy as np


def pxrdsim_torch(Volume, Point_group, AtomCoordinates, latt, crystal_system,
                  GrainSize, orientation, thermo_vib, zero_shift,
                  L, H, S, background_order, background_ratio,
                  mixture_noise_ratio, xrd):
    """GPU加速的 WPEM XRD 模拟 (与 pxrdsim 接口兼容)"""
    device = get_device()
    up = _upcast_dtype()
    wavelength = 1.54184
    two_theta_range = (8, 82.02, 0.02)

    # 1. 衍射指标
    grid_t, d_t = Diffraction_index_torch(latt, wavelength, two_theta_range)

    # 2. 消光
    res_HKL_t, _, d_res_t, _ = cal_extinction_torch(
        Point_group, grid_t, d_t, crystal_system, AtomCoordinates, wavelength)

    if res_HKL_t.shape[0] == 0:
        x_sim = np.arange(10, 80, two_theta_range[2], dtype=NP_DTYPE)
        return x_sim, np.zeros(len(x_sim), dtype=NP_DTYPE)

    P = res_HKL_t.shape[0]
    N = len(AtomCoordinates)

    # 2θ 位置
    mu_up = 2 * torch.rad2deg(
        torch.arcsin((wavelength / (2 * d_res_t.to(up))).clamp(-1, 1)))
    mu_t = mu_up.to(DTYPE)

    # 3. |F(hkl)|²
    symbols = [atom[0] for atom in AtomCoordinates]
    positions_up = torch.tensor(
        [[atom[1], atom[2], atom[3]] for atom in AtomCoordinates],
        dtype=up, device=device)

    fi_up = torch.zeros((N, P), dtype=up, device=device)
    for n in range(N):
        fi_up[n] = cal_atoms_torch(symbols[n], mu_t, wavelength, device).to(up)

    phase = 2 * math.pi * (res_HKL_t.to(up) @ positions_up.T)
    F_real = torch.sum(fi_up.T * torch.cos(phase), dim=1)
    F_imag = torch.sum(fi_up.T * torch.sin(phase), dim=1)
    FHKL_square = F_real ** 2 + F_imag ** 2

    # 多重性
    H_idx = res_HKL_t[:, 0].long()
    K_idx = res_HKL_t[:, 1].long()
    L_idx = res_HKL_t[:, 2].long()
    Mult = mult_rule_torch(H_idx, K_idx, L_idx, crystal_system, device)

    # Lorentz-偏振因子
    mu_rad      = torch.deg2rad(mu_up)
    mu_half_rad = torch.deg2rad(mu_up / 2)
    _Ints = (FHKL_square * Mult.to(up) / Volume ** 2 *
             (1 + torch.cos(mu_rad) ** 2) /
             (torch.sin(mu_half_rad) ** 2 *
              torch.cos(torch.deg2rad(Mult.to(up) / 2))))

    # 4. Scherrer 展宽
    Gamma = 0.888 * wavelength / (GrainSize * torch.cos(mu_half_rad))
    gamma_list  = (Gamma / 2 + 1e-10).to(DTYPE)
    sigma2_list = (Gamma ** 2 / (8 * math.sqrt(2)) + 1e-10).to(DTYPE)

    # 5. 择优取向 + Debye-Waller
    Ori_coe = torch.clamp(
        torch.normal(mean=1.0, std=0.2, size=(P,),
                     dtype=up, device=device),
        1 - orientation[0], 1 + orientation[0])

    M_dw = (8 / 3 * math.pi ** 2 * thermo_vib ** 2 *
            (torch.sin(mu_half_rad) / wavelength) ** 2)
    Deb_coe = torch.exp(-2 * M_dw)
    Ints = (_Ints * Ori_coe * Deb_coe).to(DTYPE)

    # 6. 批量峰形生成
    x_sim = torch.arange(two_theta_range[0], two_theta_range[1],
                         two_theta_range[2], dtype=DTYPE, device=device)
    y_sim = combined_peak_batch_torch(
        x_sim, Ints, mu_t, gamma_list, sigma2_list,
        L, H, S, two_theta_range[2])

    # 7. 归一化
    area = theta_intensity_area_torch(x_sim, y_sim)
    nor_y = y_sim if area == 0 else y_sim / area

    # 8. 零点偏移 + 背景 + 混合噪声
    x_sim = x_sim + zero_shift
    _bac = generate_random_polynomial_torch(background_order, x_sim, device)
    _bac = _bac - _bac.min()
    bac_max = _bac.max()
    _bacI = (_bac / bac_max * nor_y.max() * background_ratio
             if bac_max > 0 else torch.zeros_like(_bac))

    mixture = torch.empty(x_sim.shape[0], dtype=up, device=device)
    mixture.uniform_(0, nor_y.max().to(up).item() * mixture_noise_ratio)
    mixture = mixture.to(DTYPE)

    nor_y = nor_y + torch.flip(_bacI, dims=[0]) + mixture
    nor_y = scale_list_torch(nor_y)

    # 9. 截取 [10, 80]
    index_10 = torch.argmin(torch.abs(x_sim - 10)).item()
    index_80 = torch.argmin(torch.abs(x_sim - 80)).item() + 1
    nor_y = nor_y[index_10:index_80]

    x_sim_final = torch.arange(10, 80, two_theta_range[2],
                               dtype=DTYPE, device=device)

    if xrd == 'real':
        x_sim_final = wavelength / (2 * torch.sin(x_sim_final / 2 * math.pi / 180))

    nor_y = _adjust_length_torch(nor_y, x_sim_final.shape[0])

    return x_sim_final.cpu().numpy(), nor_y.cpu().numpy()


def _adjust_length_torch(t, desired_length):
    """填充或截断张量至目标长度"""
    n = t.shape[0]
    if n < desired_length:
        return torch.cat([t, t[-1].expand(desired_length - n)])
    elif n > desired_length:
        return t[:desired_length]
    return t
