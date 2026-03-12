#!/usr/bin/env python
"""WPEM (CPU) vs WPEM_torch (GPU/NPU) benchmark
Single-pattern and high-throughput simulation comparison.
"""

import sys
import os
import time
import random as _random

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.db import connect

from Pysimxrd.generator import parser
from Pysimxrd.utils.funs_torch import init_precision, get_device, DTYPE
from Pysimxrd.utils.ht_WPEMsim_torch import ht_process_entry

DB_PATH    = './gpu_test.db'
ENTRY_ID   = 8
N_RUNS     = 5
HT_TIMES   = 10
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'benchmark_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

database = connect(DB_PATH)
atoms = database.get_atoms(id=ENTRY_ID)

resolved = init_precision()
dev = get_device()
from Pysimxrd.utils.funs_torch import DTYPE as RESOLVED_DTYPE

# ================================================================
# 1. Single-pattern: WPEM (CPU)
# ================================================================
print("\n" + "=" * 60)
print("WPEM (CPU) — {} runs".format(N_RUNS))
print("=" * 60)

cpu_times   = []
cpu_results = []

for i in range(N_RUNS):
    t0 = time.perf_counter()
    x, y = parser(database, ENTRY_ID, sim_model='WPEM')
    dt = time.perf_counter() - t0
    cpu_times.append(dt)
    cpu_results.append((x, y))
    print("  Run {}/{}: {:.4f} s".format(i + 1, N_RUNS, dt))

cpu_avg = np.mean(cpu_times)
cpu_std = np.std(cpu_times)
print("  Average: {:.4f} ± {:.4f} s\n".format(cpu_avg, cpu_std))

# ================================================================
# 2. Single-pattern: WPEM_torch (GPU/NPU)
# ================================================================
print("=" * 60)
print("WPEM_torch (torch) — {} runs".format(N_RUNS))
print("=" * 60)
print("  Device : {}".format(dev))
print("  DTYPE  : {}".format(RESOLVED_DTYPE))

gpu_times   = []
gpu_results = []

for i in range(N_RUNS):
    t0 = time.perf_counter()
    x, y = parser(database, ENTRY_ID, sim_model='WPEM_torch')
    dt = time.perf_counter() - t0
    gpu_times.append(dt)
    gpu_results.append((x, y))
    print("  Run {}/{}: {:.4f} s".format(i + 1, N_RUNS, dt))

gpu_avg = np.mean(gpu_times)
gpu_std = np.std(gpu_times)
print("  Average: {:.4f} ± {:.4f} s\n".format(gpu_avg, gpu_std))

speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float('inf')

print("=" * 60)
print("Single-pattern SUMMARY")
print("=" * 60)
print("  WPEM (CPU)         : {:.4f} ± {:.4f} s".format(cpu_avg, cpu_std))
print("  WPEM_torch ({:<5s}) : {:.4f} ± {:.4f} s".format(str(dev), gpu_avg, gpu_std))
print("  Speedup            : {:.2f}x".format(speedup))
print("=" * 60)

# ================================================================
# 3. Pre-generate shared random parameters for HT comparison
#    (matching ht_simxrd.py process_entry defaults)
# ================================================================
_random.seed(42)
np.random.seed(42)

# Same ranges as ht_simxrd.py: grainsize=[2,20], orientation=[0,0.4],
# thermo_vib=[0,0.3], zero_shift=[-1.5,1.5], deformation=True
ht_gs_arr   = [_random.uniform(2, 20)     for _ in range(HT_TIMES)]
ht_ori_arr  = [_random.uniform(0., 0.4)   for _ in range(HT_TIMES)]
ht_ori_arr2 = [_random.uniform(0., 0.4)   for _ in range(HT_TIMES)]
ht_tv_arr   = [_random.uniform(0., 0.3)   for _ in range(HT_TIMES)]
ht_zs_arr   = [_random.uniform(-1.5, 1.5) for _ in range(HT_TIMES)]

# Fixed defaults (same as ht_simxrd.py)
HT_DEFORMATION              = True
HT_LATTICE_EXTINCTION_RATIO = 0.01
HT_LATTICE_TORSION_RATIO    = 0.01
HT_BACKGROUND_ORDER         = 6
HT_BACKGROUND_RATIO         = 0.05
HT_MIXTURE_NOISE_RATIO      = 0.02
HT_DIS_DETECTOR2SAMPLE      = 500
HT_HALF_HEIGHT_SLIT_DET     = 5
HT_HALF_HEIGHT_SAMPLE       = 2.5

print("\n  Shared HT params (first 3, deformation={}):"
      .format(HT_DEFORMATION))
for i in range(min(3, HT_TIMES)):
    print("    [{}] gs={:.2f}  ori=[{:.3f},{:.3f}]  tv={:.3f}  zs={:.3f}"
          .format(i, ht_gs_arr[i], ht_ori_arr[i], ht_ori_arr2[i],
                  ht_tv_arr[i], ht_zs_arr[i]))

# ================================================================
# 4. High-throughput: CPU sequential (deformation=True, like ht_simxrd.py)
# ================================================================
print("\n" + "=" * 60)
print("HT CPU — {} patterns (deformation={}, like ht_simxrd.py)"
      .format(HT_TIMES, HT_DEFORMATION))
print("=" * 60)

ht_cpu_results = []
t0 = time.perf_counter()
for i in range(HT_TIMES):
    x, y = parser(database, ENTRY_ID, sim_model='WPEM',
                  deformation=HT_DEFORMATION,
                  grainsize=ht_gs_arr[i],
                  prefect_orientation=[ht_ori_arr[i], ht_ori_arr2[i]],
                  thermo_vibration=ht_tv_arr[i],
                  zero_shift=ht_zs_arr[i],
                  lattice_extinction_ratio=HT_LATTICE_EXTINCTION_RATIO,
                  lattice_torsion_ratio=HT_LATTICE_TORSION_RATIO,
                  background_order=HT_BACKGROUND_ORDER,
                  background_ratio=HT_BACKGROUND_RATIO,
                  mixture_noise_ratio=HT_MIXTURE_NOISE_RATIO,
                  dis_detector2sample=HT_DIS_DETECTOR2SAMPLE,
                  half_height_slit_detector=HT_HALF_HEIGHT_SLIT_DET,
                  half_height_sample=HT_HALF_HEIGHT_SAMPLE)
    ht_cpu_results.append((x, y))
ht_cpu_time = time.perf_counter() - t0
print("  Total : {:.4f} s  ({:.1f} ms/pattern)".format(
    ht_cpu_time, ht_cpu_time / HT_TIMES * 1000))

# ================================================================
# 5. High-throughput: GPU batch (deformation=False, shared params)
# ================================================================
print("\n" + "=" * 60)
print("HT GPU batch (deformation=False) — {} patterns".format(HT_TIMES))
print("=" * 60)

t0 = time.perf_counter()
ht_batch_x, ht_batch_y = ht_process_entry(
    atoms, HT_TIMES, deformation=False,
    grainsize_arr=ht_gs_arr, orientation_arr=ht_ori_arr,
    thermo_vib_arr=ht_tv_arr, zero_shift_arr=ht_zs_arr,
    lattice_extinction_ratio=HT_LATTICE_EXTINCTION_RATIO,
    lattice_torsion_ratio=HT_LATTICE_TORSION_RATIO,
    background_order=HT_BACKGROUND_ORDER,
    background_ratio=HT_BACKGROUND_RATIO,
    mixture_noise_ratio=HT_MIXTURE_NOISE_RATIO,
    dis_detector2sample=HT_DIS_DETECTOR2SAMPLE,
    half_height_slit_detector=HT_HALF_HEIGHT_SLIT_DET,
    half_height_sample=HT_HALF_HEIGHT_SAMPLE)
ht_batch_time = time.perf_counter() - t0
print("  Total : {:.4f} s  ({:.1f} ms/pattern)".format(
    ht_batch_time, ht_batch_time / HT_TIMES * 1000))

# ================================================================
# 6. High-throughput: GPU loop (deformation=True, shared params)
# ================================================================
print("\n" + "=" * 60)
print("HT GPU loop (deformation=True) — {} patterns".format(HT_TIMES))
print("=" * 60)

t0 = time.perf_counter()
ht_loop_x, ht_loop_y = ht_process_entry(
    atoms, HT_TIMES, deformation=True,
    grainsize_arr=ht_gs_arr, orientation_arr=ht_ori_arr,
    thermo_vib_arr=ht_tv_arr, zero_shift_arr=ht_zs_arr,
    lattice_extinction_ratio=HT_LATTICE_EXTINCTION_RATIO,
    lattice_torsion_ratio=HT_LATTICE_TORSION_RATIO,
    background_order=HT_BACKGROUND_ORDER,
    background_ratio=HT_BACKGROUND_RATIO,
    mixture_noise_ratio=HT_MIXTURE_NOISE_RATIO,
    dis_detector2sample=HT_DIS_DETECTOR2SAMPLE,
    half_height_slit_detector=HT_HALF_HEIGHT_SLIT_DET,
    half_height_sample=HT_HALF_HEIGHT_SAMPLE)
ht_loop_time = time.perf_counter() - t0
print("  Total : {:.4f} s  ({:.1f} ms/pattern)".format(
    ht_loop_time, ht_loop_time / HT_TIMES * 1000))

# ================================================================
# 7. High-throughput summary
# ================================================================
sp_batch = ht_cpu_time / ht_batch_time if ht_batch_time > 0 else float('inf')
sp_loop  = ht_cpu_time / ht_loop_time  if ht_loop_time  > 0 else float('inf')

print("\n" + "=" * 60)
print("High-throughput SUMMARY ({} patterns)".format(HT_TIMES))
print("=" * 60)
print("  CPU sequential     : {:.4f} s  ({:.1f} ms/pattern)".format(
    ht_cpu_time, ht_cpu_time / HT_TIMES * 1000))
print("  GPU batch          : {:.4f} s  ({:.1f} ms/pattern)  {:.1f}x speedup".format(
    ht_batch_time, ht_batch_time / HT_TIMES * 1000, sp_batch))
print("  GPU loop (deform)  : {:.4f} s  ({:.1f} ms/pattern)  {:.1f}x speedup".format(
    ht_loop_time, ht_loop_time / HT_TIMES * 1000, sp_loop))
print("=" * 60)

# ================================================================
# 8. Plots — Single-pattern
# ================================================================

# 8a. CPU 5 runs overlay
fig, ax = plt.subplots(figsize=(10, 5))
for i, (x, y) in enumerate(cpu_results):
    ax.plot(x, y, linewidth=1.0, alpha=0.7,
            label='Run {} ({:.3f}s)'.format(i + 1, cpu_times[i]))
ax.set_xlabel('2θ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.set_title('WPEM CPU — {} Runs'.format(N_RUNS), fontsize=14, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'wpem_cpu_5runs.png'), dpi=200)
plt.close(fig)

# 8b. Torch 5 runs overlay
fig, ax = plt.subplots(figsize=(10, 5))
for i, (x, y) in enumerate(gpu_results):
    ax.plot(x, y, linewidth=1.0, alpha=0.7,
            label='Run {} ({:.3f}s)'.format(i + 1, gpu_times[i]))
ax.set_xlabel('2θ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.set_title('WPEM_torch ({}, {}) — {} Runs'.format(dev, RESOLVED_DTYPE, N_RUNS),
             fontsize=14, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'wpem_torch_5runs.png'), dpi=200)
plt.close(fig)

# 8c. Per-run timing bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(N_RUNS)
width = 0.35
ax.bar(x_pos - width / 2, cpu_times, width, label='WPEM (CPU)', color='steelblue')
ax.bar(x_pos + width / 2, gpu_times, width,
       label='WPEM_torch ({})'.format(dev), color='coral')
ax.set_xlabel('Run', fontsize=13)
ax.set_ylabel('Time (s)', fontsize=13)
ax.set_title('Per-run Timing Comparison', fontsize=14, weight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([str(i + 1) for i in range(N_RUNS)])
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'timing_comparison.png'), dpi=200)
plt.close(fig)

# 8d. CPU vs Torch side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(cpu_results[0][0], cpu_results[0][1],
             color='steelblue', linewidth=1.5)
axes[0].set_title('WPEM CPU (Run 1: {:.3f}s)'.format(cpu_times[0]),
                   fontsize=13, weight='bold')
axes[0].set_xlabel('2θ (degrees)')
axes[0].set_ylabel('Intensity (a.u.)')
axes[0].grid(True, linestyle='--', linewidth=0.4)
axes[1].plot(gpu_results[0][0], gpu_results[0][1],
             color='coral', linewidth=1.5)
axes[1].set_title('WPEM_torch {} (Run 1: {:.3f}s)'.format(dev, gpu_times[0]),
                   fontsize=13, weight='bold')
axes[1].set_xlabel('2θ (degrees)')
axes[1].set_ylabel('Intensity (a.u.)')
axes[1].grid(True, linestyle='--', linewidth=0.4)
fig.suptitle('Speedup: {:.2f}x'.format(speedup),
             fontsize=15, weight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'cpu_vs_torch_compare.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)

# ================================================================
# 9. Plots — High-throughput
# ================================================================

N_SHOW = min(HT_TIMES, 10)

# 9a. HT CPU overlay
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(N_SHOW):
    ax.plot(ht_cpu_results[i][0], ht_cpu_results[i][1],
            linewidth=0.8, alpha=0.6)
ax.set_xlabel('2θ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.set_title('HT CPU Sequential — {} patterns (showing first {})'.format(
    HT_TIMES, N_SHOW), fontsize=14, weight='bold')
ax.grid(True, linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ht_cpu_overlay.png'), dpi=200)
plt.close(fig)

# 9b. HT GPU batch overlay
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(N_SHOW):
    ax.plot(ht_batch_x[i], ht_batch_y[i],
            linewidth=0.8, alpha=0.6)
ax.set_xlabel('2θ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.set_title('HT GPU Batch (deform=False) — {} patterns (showing first {})'.format(
    HT_TIMES, N_SHOW), fontsize=14, weight='bold')
ax.grid(True, linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ht_batch_overlay.png'), dpi=200)
plt.close(fig)

# 9c. HT GPU loop overlay
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(N_SHOW):
    ax.plot(ht_loop_x[i], ht_loop_y[i],
            linewidth=0.8, alpha=0.6)
ax.set_xlabel('2θ (degrees)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.set_title('HT GPU Loop (deform=True) — {} patterns (showing first {})'.format(
    HT_TIMES, N_SHOW), fontsize=14, weight='bold')
ax.grid(True, linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ht_loop_overlay.png'), dpi=200)
plt.close(fig)

# 9d. Three-mode comparison (1st pattern each)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(ht_cpu_results[0][0], ht_cpu_results[0][1],
             color='steelblue', linewidth=1.5)
axes[0].set_title('CPU Sequential\n{:.1f} ms/pattern'.format(
    ht_cpu_time / HT_TIMES * 1000), fontsize=12, weight='bold')
axes[0].set_xlabel('2θ (degrees)')
axes[0].set_ylabel('Intensity (a.u.)')
axes[0].grid(True, linestyle='--', linewidth=0.4)

axes[1].plot(ht_batch_x[0], ht_batch_y[0],
             color='coral', linewidth=1.5)
axes[1].set_title('GPU Batch (deform=False)\n{:.1f} ms/pattern'.format(
    ht_batch_time / HT_TIMES * 1000), fontsize=12, weight='bold')
axes[1].set_xlabel('2θ (degrees)')
axes[1].grid(True, linestyle='--', linewidth=0.4)

axes[2].plot(ht_loop_x[0], ht_loop_y[0],
             color='seagreen', linewidth=1.5)
axes[2].set_title('GPU Loop (deform=True)\n{:.1f} ms/pattern'.format(
    ht_loop_time / HT_TIMES * 1000), fontsize=12, weight='bold')
axes[2].set_xlabel('2θ (degrees)')
axes[2].grid(True, linestyle='--', linewidth=0.4)

fig.suptitle('High-throughput Comparison — {} patterns'.format(HT_TIMES),
             fontsize=15, weight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ht_three_mode_compare.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)

# 9e. HT timing bar chart
fig, ax = plt.subplots(figsize=(8, 5))
modes = ['CPU Sequential', 'GPU Batch\n(deform=False)', 'GPU Loop\n(deform=True)']
times_ht = [ht_cpu_time, ht_batch_time, ht_loop_time]
colors = ['steelblue', 'coral', 'seagreen']
bars = ax.bar(modes, times_ht, color=colors, width=0.5)
for bar, t in zip(bars, times_ht):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            '{:.2f}s'.format(t), ha='center', va='bottom', fontsize=11, weight='bold')
ax.set_ylabel('Time (s)', fontsize=13)
ax.set_title('High-throughput Timing — {} patterns'.format(HT_TIMES),
             fontsize=14, weight='bold')
ax.grid(axis='y', linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ht_timing_comparison.png'), dpi=200)
plt.close(fig)

# 9f. Overall speedup summary
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Single Torch\nvs CPU', 'HT Batch\nvs CPU', 'HT Loop\nvs CPU']
speedups = [speedup, sp_batch, sp_loop]
colors = ['coral', '#e8735a', 'seagreen']
bars = ax.bar(labels, speedups, color=colors, width=0.5)
for bar, s in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            '{:.1f}x'.format(s), ha='center', va='bottom', fontsize=12, weight='bold')
ax.set_ylabel('Speedup', fontsize=13)
ax.set_title('GPU Acceleration Speedup Summary', fontsize=14, weight='bold')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8)
ax.grid(axis='y', linestyle='--', linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'speedup_summary.png'), dpi=200)
plt.close(fig)

# ================================================================
# 10. Output summary
# ================================================================
print("\nPlots saved to: {}/".format(OUTPUT_DIR))
print("  Single-pattern:")
print("    - wpem_cpu_5runs.png")
print("    - wpem_torch_5runs.png")
print("    - timing_comparison.png")
print("    - cpu_vs_torch_compare.png")
print("  High-throughput:")
print("    - ht_cpu_overlay.png")
print("    - ht_batch_overlay.png")
print("    - ht_loop_overlay.png")
print("    - ht_three_mode_compare.png")
print("    - ht_timing_comparison.png")
print("    - speedup_summary.png")
print("\nDone.")
