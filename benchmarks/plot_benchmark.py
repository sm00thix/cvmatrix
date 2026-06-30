"""
This script reads the benchmark results from 'benchmark_results.csv' and plots
the results. The first plot compares the CVMatrix and NaiveCVMatrix models
across different preprocessing combinations. The second plot shows the CVMatrix
model across different preprocessing combinations.

The plots are saved as 'benchmark_cvmatrix_vs_naive.png' and 'benchmark_cvmatrix.png'
respectively.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.text import Text


def plot_cvmatrix_vs_naive(df, combination_to_color_map, weights):
    # Set text and label sizes to 20
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 15, 'axes.titlesize': 20, 'legend.fontsize': 15})
    fig, ax = plt.subplots(figsize=(10, 10))
    preprocessing_combinations = []
    preprocessing_combinations.append(
        {'center_X': False, 'center_Y': False, 'scale_X': False, 'scale_Y': False, 'weights': weights}
    )
    preprocessing_combinations.append(
        {'center_X': True, 'center_Y': True, 'scale_X': False, 'scale_Y': False, 'weights': weights}
    )
    preprocessing_combinations.append(
        {'center_X': True, 'center_Y': True, 'scale_X': True, 'scale_Y': True, 'weights': weights}
    )
    N = df['N'].unique()[0]
    K = df['K'].unique()[0]
    M = df['M'].unique()[0]
    for combination in preprocessing_combinations:
        # Do not use the 'weights' key in combination when getting color
        combination_no_weights = {k: v for k, v in combination.items() if k != 'weights'}
        color = combination_to_color_map[str(combination_no_weights)]
        
        fast_times = []
        naive_times = []
        Ps = []
        for P in df['P'].unique():
            fast_time = df[
                (df['model'] == 'CVMatrix') &
                (df['P'] == P) &
                (df['center_X'] == combination['center_X']) &
                (df['center_Y'] == combination['center_Y']) &
                (df['scale_X'] == combination['scale_X']) &
                (df['scale_Y'] == combination['scale_Y']) &
                (df['weights'] == combination['weights'])
            ]['time'].values[0]
            naive_time = df[
                (df['model'] == 'NaiveCVMatrix') &
                (df['P'] == P) &
                (df['center_X'] == combination['center_X']) &
                (df['center_Y'] == combination['center_Y']) &
                (df['scale_X'] == combination['scale_X']) &
                (df['scale_Y'] == combination['scale_Y']) &
                (df['weights'] == combination['weights'])
            ]['time'].values[0]
            fast_times.append(fast_time)
            naive_times.append(naive_time)
            Ps.append(P)
        # label = "Baseline, " + ', '.join(
        #         [f"{k}={v}" for k, v in combination.items() if k != 'weights']
        #     )
        if combination['scale_X']:
            label = "Algorithm 6"
        elif combination['center_X']:
            label = "Algorithm 4"
        else:
            label = "Algorithm 2"
        ax.plot(
            Ps,
            naive_times,
            marker='s',
            color=color,
            linestyle='dotted',
            label=label
        )
        # label = "Fast (CVMatrix), " + ', '.join([f"{k}={v}" for k, v in combination.items() if k != 'weights'])
        if combination['scale_X']:
            label = "Algorithm 7"
        elif combination['center_X']:
            label = "Algorithm 5"
        else:
            label = "Algorithm 3"
        ax.plot(
            Ps,
            fast_times,
            marker='D',
            color=color,
            linestyle='dashed',
            label=label
        )
    lines = [1, 60, 3600, 86400]
    line_names = ['Second', 'Minute', 'Hour', 'Day']
    for j, line in enumerate(lines):
        ax.axhline(y=line, color='k', linestyle='--', linewidth=1)
        ax.text(
            310000, line, f"1 {line_names[j]}", fontsize=12, ha='center', va='center'
        )
    cvmatrix_version = df['version'].unique()[0]
    version_text = f'cvmatrix version: {cvmatrix_version}'
    ax.text(1, 330000, version_text, fontsize=10, ha='center', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    current_x_ticks, current_x_labels = plt.xticks()
    extra_x_ticks = np.array([0, 3, 5])
    extra_x_labels = [Text(0 , 0, '0'), Text(3 , 0, '3'), Text(5, 0, '5')]
    new_x_ticks = np.concatenate((current_x_ticks, extra_x_ticks))
    new_x_labels = current_x_labels + extra_x_labels
    sort_idxs = np.argsort(new_x_ticks)
    new_x_ticks = new_x_ticks[sort_idxs]
    new_x_labels = (np.array(new_x_labels)[sort_idxs]).tolist()
    start_idx = np.where(new_x_ticks == 3)[0][0]
    stop_idx = np.where(new_x_ticks == 1e5)[0][0] + 1
    new_x_ticks = new_x_ticks[start_idx:stop_idx]
    new_x_labels = new_x_labels[start_idx:stop_idx]
    ax.set_xticks(new_x_ticks)
    ax.set_xticklabels(new_x_labels)
    ax.set_title(f'Fast (cvmatrix) vs. Baseline Weighted Cross-Validation\n(N={N:,}, K={K}, M={M})', fontsize=20)
    ax.set_xlabel('P (cross-validation folds)')
    ax.set_ylabel('Time (s)')
    # fig.set_size_inches(8.27, 0.37 * 11.7)
    ax.legend(loc='upper left')#, fontsize='xx-small')
    # plt.tight_layout()
    if weights:
        plt.savefig('benchmark_cvmatrix_vs_naive_weights.png')
        plt.savefig('benchmark_cvmatrix_vs_naive_weights.eps')
    else:
        plt.savefig('benchmark_cvmatrix_vs_naive_no_weights.png')
    plt.clf()

def plot_cvmatrix(df, combination_to_color_map, weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    preprocessing_combinations = []
    center_Xs = [False, True]
    center_Ys = [False, True]
    scale_Xs = [False, True]
    scale_Ys = [False, True]
    for center_X, center_Y, scale_X, scale_Y \
        in product(center_Xs, center_Ys, scale_Xs, scale_Ys):
        preprocessing_combinations.append(
            {
                'center_X': center_X,
                'center_Y': center_Y,
                'scale_X': scale_X,
                'scale_Y': scale_Y,
                'weights': weights
            }
        )
    N = df['N'].unique()[0]
    K = df['K'].unique()[0]
    M = df['M'].unique()[0]
    for combination in preprocessing_combinations:
        combination_no_weights = {k: v for k, v in combination.items() if k != 'weights'}
        color = combination_to_color_map[str(combination_no_weights)]
        fast_times = []
        Ps = []
        for P in df['P'].unique():
            fast_time = df[
                (df['model'] == 'CVMatrix') &
                (df['P'] == P) &
                (df['center_X'] == combination['center_X']) &
                (df['center_Y'] == combination['center_Y']) &
                (df['scale_X'] == combination['scale_X']) &
                (df['scale_Y'] == combination['scale_Y']) &
                (df['weights'] == combination['weights'])
            ]['time'].values[0]
            fast_times.append(fast_time)
            Ps.append(P)
        label = ', '.join([f"{k}={v}" for k, v in combination.items() if k != 'weights'])
        ax.plot(
            Ps,
            fast_times,
            marker='D',
            color=color,
            linestyle='dashed',
            label=label
        )
    lines = [1, 60, 3600, 86400]
    line_names = ['Second', 'Minute', 'Hour', 'Day']
    for j, line in enumerate(lines):
        ax.axhline(y=line, color='k', linestyle='--', linewidth=1)
        ax.text(
            310000,line, f"1 {line_names[j]}", fontsize=20, ha='center', va='center'
        )
    cvmatrix_version = df['version'].unique()[0]
    version_text = f'CVMatrix version: {cvmatrix_version}'
    ax.text(1, 330000, version_text, fontsize=20, ha='center', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    current_x_ticks, current_x_labels = plt.xticks()
    extra_x_ticks = np.array([0, 3, 5])
    extra_x_labels = [Text(0 , 0, '0'), Text(3 , 0, '3'), Text(5, 0, '5')]
    new_x_ticks = np.concatenate((current_x_ticks, extra_x_ticks))
    new_x_labels = current_x_labels + extra_x_labels
    sort_idxs = np.argsort(new_x_ticks)
    new_x_ticks = new_x_ticks[sort_idxs]
    new_x_labels = (np.array(new_x_labels)[sort_idxs]).tolist()
    start_idx = np.where(new_x_ticks == 3)[0][0]
    stop_idx = np.where(new_x_ticks == 1e5)[0][0] + 1
    new_x_ticks = new_x_ticks[start_idx:stop_idx]
    new_x_labels = new_x_labels[start_idx:stop_idx]
    ax.set_xticks(new_x_ticks)
    ax.set_xticklabels(new_x_labels)
    ax.set_title(f'Fast (CVMatrix) Cross-Validation\n(N={N:,}, K={K}, M={M})', fontsize=10)
    ax.set_xlabel('P (cross-validation folds)')
    ax.set_ylabel('Time (s)')
    fig.set_size_inches(8.27, 0.37 * 11.7)
    ax.legend(loc='upper left', fontsize='xx-small')
    plt.tight_layout()
    if weights:
        plt.savefig('benchmark_cvmatrix_weights.png')
    else:
        plt.savefig('benchmark_cvmatrix_no_weights.png')
    plt.clf()

def _add_time_reflines(ax, ymax):
    """Draw "1 Second/Minute/Hour/Day" reference lines up to (a little above) ymax."""
    from matplotlib.text import Text  # noqa: F401  (kept for parity with other plots)
    for y, name in [(1, 'Second'), (60, 'Minute'), (3600, 'Hour'), (86400, 'Day')]:
        if y <= ymax * 2:
            ax.axhline(y=y, color='k', linestyle='--', linewidth=1)
            ax.text(310000, y, f'1 {name}', fontsize=11, ha='center', va='center')


_COMBOS = [
    ((False, False, False, False), 'solid', 'no preprocessing'),
    ((True, True, False, False), 'dashed', 'centering'),
    ((True, True, True, True), 'dotted', 'centering + scaling'),
]


def _combo_dict(key):
    return dict(zip(('center_X', 'center_Y', 'scale_X', 'scale_Y'), key))


def _series_times(df, model, key, Ps_all):
    """Per-P times (weighted) for a given model and (center_X, center_Y, scale_X,
    scale_Y) combination; returns (Ps, times) over the P values that have a row."""
    cX, cY, sX, sY = key
    times, Ps = [], []
    for P in Ps_all:
        sel = df[
            (df['model'] == model) & (df['P'] == P) & (df['weights'] == True)
            & (df['center_X'] == cX) & (df['center_Y'] == cY)
            & (df['scale_X'] == sX) & (df['scale_Y'] == sY)
        ]['time'].values
        if len(sel):
            times.append(sel[0])
            Ps.append(P)
    return Ps, times


def plot_numpy_vs_jax(df, combination_to_color_map):
    """
    benchmark_cvmatrix_numpy_vs_jax.png: total weighted cross-validation time for the
    CVMatrix NumPy backend vs. the JAX backend (warm, JIT-compiled jax.vmap over folds) on
    CPU and GPU, across the three preprocessing combinations. All measured multi-threaded
    (no thread limit) at cvmatrix 3.2.0. Color encodes preprocessing; line style encodes
    backend.
    """
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 15,
                         'axes.titlesize': 18, 'legend.fontsize': 11})
    fig, ax = plt.subplots(figsize=(10, 10))
    # (model, linestyle, marker, legend label)
    series = [
        ('CVMatrix', 'dashed', 'D', 'NumPy'),
        ('CVMatrix-jax-cpu-warmjit', 'dashdot', '^', 'JAX warm (CPU)'),
        ('CVMatrix-jax-gpu-warmjit', 'solid', 'o', 'JAX warm (GPU)'),
    ]
    N = df['N'].unique()[0]
    K = df['K'].unique()[0]
    M = df['M'].unique()[0]
    Ps_all = sorted(df['P'].unique())
    ymax = 0.0
    for key, _ls, combo_lab in _COMBOS:
        color = combination_to_color_map[str(_combo_dict(key))]
        for model, linestyle, marker, _lab in series:
            Ps, times = _series_times(df, model, key, Ps_all)
            if times:
                ax.plot(Ps, times, marker=marker, color=color, linestyle=linestyle)
                ymax = max(ymax, max(times))
    _add_time_reflines(ax, ymax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('CVMatrix NumPy vs. JAX (warm) Weighted Cross-Validation\n'
                 f'(N={N:,}, K={K}, M={M})', fontsize=18)
    ax.set_xlabel('P (cross-validation folds)')
    ax.set_ylabel('Time (s)')
    backend_handles = [Line2D([0], [0], color='gray', linestyle=ls, marker=mk, label=lab)
                       for _m, ls, mk, lab in series]
    combo_handles = [Line2D([0], [0], color=combination_to_color_map[str(_combo_dict(k))],
                            lw=3, label=lab) for k, _ls, lab in _COMBOS]
    leg1 = ax.legend(handles=backend_handles, loc='upper left', title='Backend')
    ax.add_artist(leg1)
    ax.legend(handles=combo_handles, loc='lower right', title='Preprocessing')
    fig.text(0.5, 0.01, 'cvmatrix 3.2.0; multi-threaded CPU; GPU: RTX 3090 Ti',
             ha='center', va='bottom', fontsize=9, alpha=0.7)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig('benchmark_cvmatrix_numpy_vs_jax.png')
    plt.clf()


def plot_jax_variants(df):
    """
    benchmark_jax_variants.png: total weighted cross-validation time for the CVMatrix JAX
    backend under no-JIT (eager vmap), cold-JIT (compilation + run) and warm-JIT (run
    only), on CPU and GPU, across the three preprocessing combinations. Multi-threaded
    CPU, cvmatrix 3.2.0. Color encodes the JAX mode/device; line style encodes
    preprocessing.
    """
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 15,
                         'axes.titlesize': 18, 'legend.fontsize': 10})
    fig, ax = plt.subplots(figsize=(10, 10))
    variants = [
        ('CVMatrix-jax-gpu-nojit', 'GPU, no-JIT'),
        ('CVMatrix-jax-gpu-coldjit', 'GPU, cold-JIT'),
        ('CVMatrix-jax-gpu-warmjit', 'GPU, warm-JIT'),
        ('CVMatrix-jax-cpu-nojit', 'CPU, no-JIT'),
        ('CVMatrix-jax-cpu-coldjit', 'CPU, cold-JIT'),
        ('CVMatrix-jax-cpu-warmjit', 'CPU, warm-JIT'),
    ]
    cm = colormaps.get_cmap('tab10')
    variant_colors = {m: cm(i) for i, (m, _lab) in enumerate(variants)}
    N = df['N'].unique()[0]
    K = df['K'].unique()[0]
    M = df['M'].unique()[0]
    Ps_all = sorted(df['P'].unique())
    ymax = 0.0
    for key, linestyle, _combo_lab in _COMBOS:
        for model, _lab in variants:
            Ps, times = _series_times(df, model, key, Ps_all)
            if times:
                ax.plot(Ps, times, color=variant_colors[model], linestyle=linestyle,
                        linewidth=1.8)
                ymax = max(ymax, max(times))
    _add_time_reflines(ax, ymax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('CVMatrix JAX backend: execution-mode comparison\n'
                 f'(weighted, N={N:,}, K={K}, M={M})', fontsize=18)
    ax.set_xlabel('P (cross-validation folds)')
    ax.set_ylabel('Time (s)')
    variant_handles = [Line2D([0], [0], color=variant_colors[m], lw=2.5, label=lab)
                       for m, lab in variants]
    combo_handles = [Line2D([0], [0], color='gray', linestyle=ls, label=lab)
                     for _k, ls, lab in _COMBOS]
    leg1 = ax.legend(handles=variant_handles, loc='upper left', title='JAX mode / device')
    ax.add_artist(leg1)
    ax.legend(handles=combo_handles, loc='lower right', title='Preprocessing')
    fig.text(0.5, 0.01, 'cvmatrix 3.2.0; multi-threaded CPU; GPU: RTX 3090 Ti',
             ha='center', va='bottom', fontsize=9, alpha=0.7)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig('benchmark_jax_variants.png')
    plt.clf()


def get_combination_to_color_map():
    preprocessing_combinations = []
    center_Xs = [False, True]
    center_Ys = [False, True]
    scale_Xs = [False, True]
    scale_Ys = [False, True]
    for center_X, center_Y, scale_X, scale_Y \
        in product(center_Xs, center_Ys, scale_Xs, scale_Ys):
        preprocessing_combinations.append(
            {
                'center_X': center_X,
                'center_Y': center_Y,
                'scale_X': scale_X,
                'scale_Y': scale_Y
            }
        )
    # Define a list of 16 colorblind-friendly colors
    cm = colormaps.get_cmap('tab20')
    colors = [cm(i) for i in range(16)]
    combination_to_color_map = {}
    for i, combination in enumerate(preprocessing_combinations):
        combination_to_color_map[str(combination)] = colors[i]
    return combination_to_color_map

if __name__ == '__main__':
    # Set the font size, legend size, and axis label size
    plt.rcParams.update({'font.size': 12})
    df = pd.read_csv('benchmark_results.csv')
    combination_to_color_map = get_combination_to_color_map()
    for weights in [True, False]:
        plot_cvmatrix_vs_naive(df, combination_to_color_map, weights)
        plot_cvmatrix(df, combination_to_color_map, weights)
    # Multi-threaded CVMatrix NumPy-vs-JAX and JAX execution-mode figures, from the
    # separate results CSV produced by the BENCH_JAX_VARIANTS / multi-threaded runs.
    if os.path.exists('benchmark_results_jax.csv'):
        df_jax = pd.read_csv('benchmark_results_jax.csv')
        plot_numpy_vs_jax(df_jax, combination_to_color_map)
        plot_jax_variants(df_jax)
