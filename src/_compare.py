#!/usr/bin/env python
"""CPU vs Torch 各步骤中间结果逐步对比"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch, math
from ase.db import connect
from ase.spacegroup import get_spacegroup

from Pysimxrd.utils.funs import (
    prim2conv, cal_atoms, mult_rule,
    combined_peak, theta_intensity_area, scale_list,
    grid_atom, de_redundant, get_float, atomics
)
from Pysimxrd.utils.WPEMsim import (
    c_atom_covert2WPEMformat, space_group_to_crystal_system,
    Diffraction_index, cal_extinction
)

from Pysimxrd.utils.funs_torch import (
    init_precision, get_device, _upcast_dtype, NP_DTYPE,
    Diffraction_index_torch, cal_extinction_torch,
    cal_atoms_torch, mult_rule_torch, combined_peak_batch_torch,
    theta_intensity_area_torch, scale_list_torch
)
import Pysimxrd.utils.funs_torch as ft

init_precision('fp32')
DTYPE = ft.DTYPE
device = get_device()
up = ft._upcast_dtype()
print(f"Device: {device}, DTYPE: {DTYPE}")

# 加载样本
database = connect('/home/pushuchen/XQueryer/demo_data/XRDbenchmark_train.db')
row = database.get(id=8)
atoms = row.toatoms()
G_latt_consts, _, c_atom = prim2conv(atoms, False, 0.01, 0.01)
G_spacegroup = get_spacegroup(c_atom).no
N_symbols = c_atom.get_chemical_symbols()
positions = c_atom.get_scaled_positions()
Volume = c_atom.get_volume()
spacegroup_obj = get_spacegroup(c_atom)
crystal_system = space_group_to_crystal_system(G_spacegroup)
AtomCoordinates = c_atom_covert2WPEMformat(positions, N_symbols)
Point_group = spacegroup_obj.symbol[0]
latt = G_latt_consts

print(f"Formula: {c_atom.get_chemical_formula()}")
print(f"SpaceGroup: {G_spacegroup}, CrystalSystem: {crystal_system}, Lattice: {Point_group}")
print(f"Latt: {latt}, Volume: {Volume:.2f}")

wavelength = 1.54184
two_theta_range = (8, 82.02, 0.02)

print("\n" + "="*70)
print("STEP 1: Diffraction Index (HKL + d-spacing)")
print("="*70)

# CPU
cpu_grid, cpu_d = Diffraction_index(crystal_system, latt, wavelength, two_theta_range)
cpu_grid = np.array(cpu_grid, dtype=np.float64)
cpu_d = np.array(cpu_d, dtype=np.float64)
cpu_mu = 2 * np.arcsin(wavelength / 2 / cpu_d) * 180 / np.pi

# Torch
torch_grid, torch_d = Diffraction_index_torch(latt, wavelength, two_theta_range)
torch_grid_np = torch_grid.cpu().float().numpy().astype(np.float64)
torch_d_np = torch_d.cpu().float().numpy().astype(np.float64)
torch_mu_np = 2 * np.arcsin(wavelength / 2 / torch_d_np) * 180 / np.pi

print(f"  CPU:   {len(cpu_d)} unique HKL, d range [{cpu_d.min():.6f}, {cpu_d.max():.6f}]")
print(f"  Torch: {len(torch_d_np)} unique HKL, d range [{torch_d_np.min():.6f}, {torch_d_np.max():.6f}]")
print(f"  CPU  mu range: [{cpu_mu.min():.4f}, {cpu_mu.max():.4f}]")
print(f"  Torch mu range: [{torch_mu_np.min():.4f}, {torch_mu_np.max():.4f}]")

cpu_d_set = set(np.round(cpu_d.astype(np.float64), 4))
torch_d_set = set(np.round(torch_d_np.astype(np.float64), 4))
only_cpu = cpu_d_set - torch_d_set
only_torch = torch_d_set - cpu_d_set
common = cpu_d_set & torch_d_set
print(f"\n  d-spacing overlap: {len(common)} common, {len(only_cpu)} CPU-only, {len(only_torch)} torch-only")
if only_cpu:
    only_list = sorted(only_cpu)[:10]
    print(f"  CPU-only d (first 10):  {[float(x) for x in only_list]}")
if only_torch:
    only_list = sorted(only_torch)[:10]
    print(f"  Torch-only d (first 10): {[float(x) for x in only_list]}")

cpu_sort = np.argsort(-cpu_d)
torch_sort = np.argsort(-torch_d_np)

print(f"\n  First 10 HKL sorted by d-descending (CPU vs Torch):")
print(f"    {'CPU_HKL':>14s} {'d_CPU':>10s} {'2θ_CPU':>8s} |  {'T_HKL':>14s} {'d_T':>10s} {'2θ_T':>8s}")
for i in range(min(10, len(cpu_sort))):
    ci = cpu_sort[i]
    ti = torch_sort[i]
    hc = f"[{cpu_grid[ci][0]:3.0f} {cpu_grid[ci][1]:3.0f} {cpu_grid[ci][2]:3.0f}]"
    ht = f"[{torch_grid_np[ti][0]:3.0f} {torch_grid_np[ti][1]:3.0f} {torch_grid_np[ti][2]:3.0f}]"
    print(f"    {hc} {cpu_d[ci]:10.6f} {cpu_mu[ci]:8.4f} |  {ht} {torch_d_np[ti]:10.6f} {torch_mu_np[ti]:8.4f}")

print("\n" + "="*70)
print("STEP 2: Extinction")
print("="*70)

cpu_res_HKL, _, cpu_d_res, _ = cal_extinction(
    Point_group, cpu_grid.tolist(), cpu_d.tolist(),
    crystal_system, AtomCoordinates, wavelength)
cpu_res_HKL = np.array(cpu_res_HKL, dtype=np.float64)
cpu_d_res = np.array(cpu_d_res, dtype=np.float64)

torch_res_HKL, _, torch_d_res, _ = cal_extinction_torch(
    Point_group, torch_grid, torch_d, crystal_system, AtomCoordinates, wavelength)
torch_res_HKL_np = torch_res_HKL.cpu().float().numpy().astype(np.float64)
torch_d_res_np = torch_d_res.cpu().float().numpy().astype(np.float64)

print(f"  CPU:   {len(cpu_d_res)} surviving peaks")
print(f"  Torch: {len(torch_d_res_np)} surviving peaks")

cpu_d_res_set = set(np.round(cpu_d_res, 4))
torch_d_res_set = set(np.round(torch_d_res_np, 4))
only_cpu_ext = cpu_d_res_set - torch_d_res_set
only_torch_ext = torch_d_res_set - cpu_d_res_set
common_ext = cpu_d_res_set & torch_d_res_set
print(f"  After extinction: {len(common_ext)} common, {len(only_cpu_ext)} CPU-only, {len(only_torch_ext)} torch-only")

print("\n" + "="*70)
print("STEP 3: F(hkl)², Multiplicity, Intensity (common peaks)")
print("="*70)

cpu_mu_res = 2 * np.arcsin(wavelength / 2 / cpu_d_res) * 180 / np.pi
torch_mu_res = 2 * np.arcsin(wavelength / 2 / torch_d_res_np) * 180 / np.pi

cpu_d4 = np.round(cpu_d_res, 4)
torch_d4 = np.round(torch_d_res_np, 4)

cpu_idx_map = {float(d): i for i, d in enumerate(cpu_d4)}
torch_idx_map = {float(d): i for i, d in enumerate(torch_d4)}

matched = []
for d_val in sorted(common_ext):
    ci = cpu_idx_map.get(float(d_val))
    ti = torch_idx_map.get(float(d_val))
    if ci is not None and ti is not None:
        matched.append((ci, ti, float(d_val)))

print(f"  Matched {len(matched)} peak pairs by d-value")

mu_t = (2 * torch.rad2deg(
    torch.arcsin((wavelength / (2 * torch_d_res.to(up))).clamp(-1, 1))
)).to(DTYPE)

N = len(AtomCoordinates)
syms = [atom[0] for atom in AtomCoordinates]
pos_t = torch.tensor([[a[1], a[2], a[3]] for a in AtomCoordinates], dtype=DTYPE, device=device)

fi_torch = torch.zeros((N, torch_res_HKL.shape[0]), dtype=DTYPE, device=device)
for n in range(N):
    fi_torch[n] = cal_atoms_torch(syms[n], mu_t, wavelength, device)

phase_t = 2 * math.pi * (torch_res_HKL @ pos_t.T)
Freal_t = torch.sum(fi_torch.T * torch.cos(phase_t), dim=1)
Fimag_t = torch.sum(fi_torch.T * torch.sin(phase_t), dim=1)
Fsq_t = (Freal_t**2 + Fimag_t**2).cpu().float().numpy()

H_t = torch_res_HKL[:, 0].long()
K_t = torch_res_HKL[:, 1].long()
L_t = torch_res_HKL[:, 2].long()
Mult_t = mult_rule_torch(H_t, K_t, L_t, crystal_system, device).cpu().float().numpy()

mu_rad_t = np.deg2rad(torch_mu_res)
mu_half_rad_t = np.deg2rad(torch_mu_res / 2)
_Ints_t = (Fsq_t * Mult_t / Volume**2 *
           (1 + np.cos(mu_rad_t)**2) /
           (np.sin(mu_half_rad_t)**2 * np.cos(np.deg2rad(Mult_t/2))))

print("\n  Comparing first 20 matched peaks:")
print(f"  {'d':>8s} | {'HKL_CPU':>14s} {'HKL_T':>14s} | {'Fsq_CPU':>10s} {'Fsq_T':>10s} | {'M_CPU':>5s} {'M_T':>5s} | {'I_CPU':>10s} {'I_T':>10s}")
print("  " + "-"*110)

max_diff_fhkl = 0
max_diff_mult = 0
max_diff_ints = 0
n_hkl_differ = 0

for idx, (ci, ti, d_val) in enumerate(matched[:20]):
    angle_cpu = cpu_mu_res[ci]
    hkl_cpu = cpu_res_HKL[ci]
    Fleft = 0; Fright = 0
    for atom in AtomCoordinates:
        fi = cal_atoms(atom[0], angle_cpu, wavelength)
        phase = 2*np.pi*(atom[1]*hkl_cpu[0] + atom[2]*hkl_cpu[1] + atom[3]*hkl_cpu[2])
        Fleft += fi * np.cos(phase)
        Fright += fi * np.sin(phase)
    Fsq_cpu = Fleft**2 + Fright**2
    M_cpu = mult_rule(int(hkl_cpu[0]), int(hkl_cpu[1]), int(hkl_cpu[2]), crystal_system)
    I_cpu = float(Fsq_cpu * M_cpu / Volume**2 *
                  (1+np.cos(np.radians(angle_cpu))**2) /
                  (np.sin(np.radians(angle_cpu/2))**2 * np.cos(np.radians(M_cpu/2))))

    hkl_cpu_s = f"[{hkl_cpu[0]:3.0f} {hkl_cpu[1]:3.0f} {hkl_cpu[2]:3.0f}]"
    hkl_t = torch_res_HKL_np[ti]
    hkl_t_s = f"[{hkl_t[0]:3.0f} {hkl_t[1]:3.0f} {hkl_t[2]:3.0f}]"
    if not np.array_equal(np.abs(hkl_cpu), np.abs(hkl_t)):
        n_hkl_differ += 1
    
    mark = " *" if abs(Fsq_cpu - Fsq_t[ti]) > 1.0 else ""
    print(f"  {d_val:8.4f} | {hkl_cpu_s:>14s} {hkl_t_s:>14s} | "
          f"{Fsq_cpu:10.4f} {Fsq_t[ti]:10.4f} | "
          f"{M_cpu:5.0f} {Mult_t[ti]:5.0f} | "
          f"{I_cpu:10.4f} {_Ints_t[ti]:10.4f}{mark}")

    max_diff_fhkl = max(max_diff_fhkl, abs(Fsq_cpu - Fsq_t[ti]))
    max_diff_mult = max(max_diff_mult, abs(M_cpu - Mult_t[ti]))
    max_diff_ints = max(max_diff_ints, abs(I_cpu - _Ints_t[ti]))

all_fsq_diffs = []
all_mult_diffs = []
all_ints_diffs = []
for ci, ti, d_val in matched:
    angle_cpu = cpu_mu_res[ci]
    hkl_cpu = cpu_res_HKL[ci]
    Fl = 0; Fr = 0
    for atom in AtomCoordinates:
        fi = cal_atoms(atom[0], angle_cpu, wavelength)
        ph = 2*np.pi*(atom[1]*hkl_cpu[0] + atom[2]*hkl_cpu[1] + atom[3]*hkl_cpu[2])
        Fl += fi * np.cos(ph)
        Fr += fi * np.sin(ph)
    Fsq = Fl**2 + Fr**2
    M = mult_rule(int(hkl_cpu[0]), int(hkl_cpu[1]), int(hkl_cpu[2]), crystal_system)
    I = float(Fsq * M / Volume**2 *
              (1+np.cos(np.radians(angle_cpu))**2) /
              (np.sin(np.radians(angle_cpu/2))**2 * np.cos(np.radians(M/2))))
    all_fsq_diffs.append(abs(Fsq - Fsq_t[ti]))
    all_mult_diffs.append(abs(M - Mult_t[ti]))
    all_ints_diffs.append(abs(I - _Ints_t[ti]))

all_fsq_diffs = np.array(all_fsq_diffs)
all_ints_diffs = np.array(all_ints_diffs)
print(f"\n  Over all {len(matched)} matched peaks:")
print(f"  Max |ΔF²| = {all_fsq_diffs.max():.6f}, Mean = {all_fsq_diffs.mean():.6f}")
print(f"  Max |ΔMult| = {max(all_mult_diffs):.1f}")
print(f"  Max |Δ_Ints| = {all_ints_diffs.max():.6f}, Mean = {all_ints_diffs.mean():.6f}")
print(f"  HKL representatives differ (|h|,|k|,|l|): {n_hkl_differ}/{min(20,len(matched))}")
print(f"  Peaks with |ΔF²| > 1.0: {(all_fsq_diffs > 1.0).sum()}")
print(f"  Peaks with |Δ_Ints| > 0.1: {(all_ints_diffs > 0.1).sum()}")

print("\n" + "="*70)
print("STEP 4: Full y_sim (deterministic, no orientation/background noise)")
print("="*70)

GrainSize = 17.0
cpu_Ints_all = []
for ci in range(len(cpu_res_HKL)):
    angle_cpu = cpu_mu_res[ci]
    hkl_cpu = cpu_res_HKL[ci]
    Fl = 0; Fr = 0
    for atom in AtomCoordinates:
        fi = cal_atoms(atom[0], angle_cpu, wavelength)
        ph = 2*np.pi*(atom[1]*hkl_cpu[0] + atom[2]*hkl_cpu[1] + atom[3]*hkl_cpu[2])
        Fl += fi * np.cos(ph)
        Fr += fi * np.sin(ph)
    Fsq = Fl**2 + Fr**2
    M = mult_rule(int(hkl_cpu[0]), int(hkl_cpu[1]), int(hkl_cpu[2]), crystal_system)
    I = float(Fsq * M / Volume**2 *
              (1+np.cos(np.radians(angle_cpu))**2) /
              (np.sin(np.radians(angle_cpu/2))**2 * np.cos(np.radians(M/2))))
    cpu_Ints_all.append(I)

cpu_Gamma = 0.888*wavelength/(GrainSize*np.cos(np.radians(np.array(cpu_mu_res)/2)))
cpu_gamma_list = cpu_Gamma / 2 + 1e-10
cpu_sigma2_list = cpu_Gamma**2 / (8*np.sqrt(2)) + 1e-10

x_sim_cpu = np.arange(two_theta_range[0], two_theta_range[1], two_theta_range[2])
y_cpu = np.zeros(len(x_sim_cpu))
for num in range(len(cpu_Ints_all)):
    y_cpu += combined_peak(x_sim_cpu, cpu_Ints_all[num], cpu_mu_res[num],
                           cpu_gamma_list[num], cpu_sigma2_list[num], 500, 50, 25, 0.02)

Gamma_t = 0.888 * wavelength / (GrainSize * torch.cos(torch.deg2rad(mu_t / 2)))
gamma_t_all = Gamma_t / 2 + 1e-10
sigma2_t_all = Gamma_t**2 / (8 * math.sqrt(2)) + 1e-10
Ints_t_all = torch.tensor(_Ints_t, dtype=DTYPE, device=device)
x_t = torch.arange(two_theta_range[0], two_theta_range[1], two_theta_range[2], dtype=DTYPE, device=device)
y_torch = combined_peak_batch_torch(
    x_t, Ints_t_all, mu_t, gamma_t_all, sigma2_t_all,
    500, 50, 25, 0.02).cpu().float().numpy()

print(f"  y_cpu nan: {np.isnan(y_cpu).sum()}, y_torch nan: {np.isnan(y_torch).sum()}")
print(f"  y_cpu range: [{y_cpu.min():.6f}, {y_cpu.max():.6f}]")
print(f"  y_torch range: [{y_torch.min():.6f}, {y_torch.max():.6f}]")

area_cpu = np.sum((y_cpu[:-1] + y_cpu[1:]) / 2 * np.diff(x_sim_cpu))
area_torch = np.sum((y_torch[:-1] + y_torch[1:]) / 2 * np.diff(x_sim_cpu))
print(f"  Area: CPU={area_cpu:.6f}, Torch={area_torch:.6f}")
y_cpu_n = y_cpu / (area_cpu + 1e-30) if area_cpu > 0 else y_cpu
y_torch_n = y_torch / (area_torch + 1e-30) if area_torch > 0 else y_torch

idx10 = np.abs(x_sim_cpu - 10).argmin()
idx80 = np.abs(x_sim_cpu - 80).argmin() + 1
y_cpu_trim = np.array(scale_list(y_cpu_n[idx10:idx80]), dtype=np.float64)
y_torch_trim = y_torch_n[idx10:idx80]
if y_torch_trim.max() > 0:
    y_torch_trim = y_torch_trim * 100.0 / y_torch_trim.max()

x_final = np.arange(10, 80, 0.02)
y_cpu_trim = y_cpu_trim[:len(x_final)]
y_torch_trim = y_torch_trim[:len(x_final)]

corr_full = np.corrcoef(y_cpu_trim, y_torch_trim)[0, 1]
rmse = np.sqrt(np.mean((y_cpu_trim - y_torch_trim)**2))
peak_pos_cpu = x_final[np.argmax(y_cpu_trim)]
peak_pos_torch = x_final[np.argmax(y_torch_trim)]

print(f"\n  CPU  main peak at 2θ = {peak_pos_cpu:.4f}°, max = {y_cpu_trim.max():.4f}")
print(f"  Torch main peak at 2θ = {peak_pos_torch:.4f}°, max = {y_torch_trim.max():.4f}")
print(f"  Correlation: {corr_full:.6f}")
print(f"  RMSE: {rmse:.4f}")

try:
    from scipy.signal import find_peaks
except ImportError:
    def find_peaks(x, height=None, distance=None):
        idx = []
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                if height is None or x[i] >= height:
                    if not idx or (i - idx[-1]) >= (distance or 1):
                        idx.append(i)
        return np.array(idx), {}
cpu_peaks_idx, _ = find_peaks(y_cpu_trim, height=5, distance=20)
torch_peaks_idx, _ = find_peaks(y_torch_trim, height=5, distance=20)

cpu_top5 = cpu_peaks_idx[np.argsort(y_cpu_trim[cpu_peaks_idx])[::-1][:5]]
torch_top5 = torch_peaks_idx[np.argsort(y_torch_trim[torch_peaks_idx])[::-1][:5]]

print(f"\n  Top 5 peaks by intensity:")
print(f"    CPU:   {[f'{x_final[i]:.2f}° ({y_cpu_trim[i]:.1f})' for i in sorted(cpu_top5)]}")
print(f"    Torch: {[f'{x_final[i]:.2f}° ({y_torch_trim[i]:.1f})' for i in sorted(torch_top5)]}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

axes[0].plot(x_final, y_cpu_trim, 'b-', linewidth=1.0, alpha=0.8, label='CPU (WPEM)')
axes[0].plot(x_final, y_torch_trim, 'r--', linewidth=1.0, alpha=0.8, label='Torch')
axes[0].set_title(f'Full profile (no noise) — corr={corr_full:.4f}, RMSE={rmse:.2f}',
                   fontsize=13, weight='bold')
axes[0].set_xlabel('2θ (degrees)')
axes[0].set_ylabel('Intensity (scaled)')
axes[0].legend()
axes[0].grid(True, linestyle='--', linewidth=0.3)

diff = y_cpu_trim - y_torch_trim
axes[1].plot(x_final, diff, 'g-', linewidth=0.8)
axes[1].set_title('Difference (CPU − Torch)', fontsize=13, weight='bold')
axes[1].set_xlabel('2θ (degrees)')
axes[1].set_ylabel('ΔIntensity')
axes[1].grid(True, linestyle='--', linewidth=0.3)
axes[1].axhline(y=0, color='k', linewidth=0.5)

win = 5
axes[2].plot(x_final, y_cpu_trim, 'b-', linewidth=1.5, alpha=0.8, label='CPU')
axes[2].plot(x_final, y_torch_trim, 'r--', linewidth=1.5, alpha=0.8, label='Torch')
axes[2].set_xlim(peak_pos_cpu - win, peak_pos_cpu + win)
axes[2].set_title(f'Zoom on main peak ({peak_pos_cpu:.1f}°)', fontsize=13, weight='bold')
axes[2].set_xlabel('2θ (degrees)')
axes[2].set_ylabel('Intensity')
axes[2].legend()
axes[2].grid(True, linestyle='--', linewidth=0.3)

fig.tight_layout()
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, 'cpu_vs_torch_step_compare.png'), dpi=200)
plt.close(fig)
print(f"\n  Plot saved to {outdir}/cpu_vs_torch_step_compare.png")

print("\nDone.")
