"""
Publication-quality spatial contour plots for PRANO-Cav MoE.

Generates for each test case:
  - True vs Predicted vs Error contours for u, v, p, alpha_v
  - Force coefficient time series

Requires cell center coordinates from the OpenFOAM mesh.
Run once:  python -m src.plot_contours --extract-mesh /path/to/any/CN_case
Then:      python -m src.plot_contours
"""

import os
import re
import argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader

from src.dataset import PRANOCavDataset, REGIME_NAMES
from src.model_moe import PRANOCavMoEModel


# =====================================================================
# Publication style
# =====================================================================

def set_pub_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'mathtext.fontset': 'dejavuserif',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
    })


# =====================================================================
# Cell center extraction from OpenFOAM
# =====================================================================

def read_openfoam_field(file_path):
    """Read internalField from an OpenFOAM field file."""
    with open(file_path, 'r') as f:
        content = f.read()

    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//[^\n]*', '', content)

    m = re.search(r'\binternalField\b\s+(\S+)', content)
    if m is None:
        raise ValueError(f"No internalField in {file_path}")

    keyword = m.group(1)
    rest = content[m.end():]

    if keyword == 'uniform':
        vec = re.search(r'\(\s*([-+\d.eE\s]+)\)', rest)
        if vec:
            return np.array([float(v) for v in vec.group(1).split()],
                            dtype=np.float32)
        raise ValueError(f"Cannot parse uniform in {file_path}")

    if keyword == 'nonuniform':
        count_m = re.search(r'(\d+)\s*\(', rest)
        if count_m is None:
            raise ValueError(f"No data block in {file_path}")
        op = count_m.end() - 1
        depth = 0
        cp = -1
        for i in range(op, len(rest)):
            if rest[i] == '(':
                depth += 1
            elif rest[i] == ')':
                depth -= 1
                if depth == 0:
                    cp = i
                    break
        block = rest[op + 1:cp]
        vec_entries = re.findall(r'\(([^()]+)\)', block)
        if vec_entries:
            return np.array([[float(v) for v in e.split()]
                             for e in vec_entries], dtype=np.float32)
        scalars = [float(v) for v in
                   re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block)]
        return np.array(scalars, dtype=np.float32)

    raise ValueError(f"Unknown keyword '{keyword}' in {file_path}")


def extract_cell_centers(case_dir, save_path='data/cell_centers.npy'):
    """
    Extract cell centers from an OpenFOAM case.

    Looks for the C (cell center) file in:
      - <case>/0/C
      - <case>/constant/polyMesh/C
      - <case>/<first_time_dir>/C

    Or computes from U field coordinates if C not available.
    """
    candidates = [
        os.path.join(case_dir, '0', 'C'),
        os.path.join(case_dir, 'constant', 'polyMesh', 'C'),
        os.path.join(case_dir, 'constant', 'C'),
    ]

    # Also check numeric time directories
    if os.path.isdir(case_dir):
        for d in sorted(os.listdir(case_dir)):
            try:
                float(d)
                candidates.append(os.path.join(case_dir, d, 'C'))
                candidates.append(os.path.join(case_dir, d, 'ccx'))
            except ValueError:
                pass

    for c_path in candidates:
        if os.path.exists(c_path):
            print(f"[mesh] Reading cell centers from: {c_path}")
            centers = read_openfoam_field(c_path)
            if centers.ndim == 2 and centers.shape[1] >= 2:
                centers_2d = centers[:, :2]  # take x, y only
            else:
                raise ValueError(f"Unexpected shape {centers.shape}")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, centers_2d)
            print(f"[mesh] Saved {centers_2d.shape[0]} cell centers "
                  f"to {save_path}")
            return centers_2d

    raise FileNotFoundError(
        f"No cell center file found in {case_dir}.\n"
        f"Tried: {candidates}\n"
        f"You can create it by running in OpenFOAM:\n"
        f"  writeCellCentres -case {case_dir}\n"
        f"This creates 0/C with cell center coordinates."
    )


def load_cell_centers(path='data/cell_centers.npy', stride=10):
    """Load saved cell centers and apply the same subsampling as dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cell centers not found at {path}.\n"
            f"Run first:\n"
            f"  python -m src.plot_contours --extract-mesh /path/to/CN_case\n"
            f"where CN_case is any OpenFOAM case directory."
        )
    centers = np.load(path)
    centers_sub = centers[::stride]
    print(f"[mesh] Loaded {centers.shape[0]} centers, "
          f"subsampled to {centers_sub.shape[0]} (stride={stride})")
    return centers_sub


# =====================================================================
# Channel metadata
# =====================================================================

CHANNELS = [
    {'key': 'u',     'symbol': r'$u$',         'unit': r'm s$^{-1}$'},
    {'key': 'v',     'symbol': r'$v$',         'unit': r'm s$^{-1}$'},
    {'key': 'p',     'symbol': r'$p$',         'unit': r'kPa'},
    {'key': 'alpha', 'symbol': r'$\alpha_v$',  'unit': None},
]


def _label(ch):
    return f"{ch['symbol']} ({ch['unit']})" if ch['unit'] else ch['symbol']


# =====================================================================
# Spatial contour plot (1 figure per field channel)
# =====================================================================

def plot_contour_triplet(channel_idx, y_true, y_pred, cell_centers,
                         case_name, output_dir):
    """
    Three-panel contour: True | Predicted | Error
    for a single field channel on the hydrofoil mesh.
    """
    ch = CHANNELS[channel_idx]
    key = ch['key']

    # Scale pressure Pa -> kPa
    scale = 1.0 / 1000 if key == 'p' else 1.0
    true_vals = y_true[:, channel_idx] * scale
    pred_vals = y_pred[:, channel_idx] * scale
    err_vals  = pred_vals - true_vals

    vmin = float(min(true_vals.min(), pred_vals.min()))
    vmax = float(max(true_vals.max(), pred_vals.max()))
    err_lim = float(max(abs(err_vals.min()), abs(err_vals.max())))
    if err_lim < 1e-10:
        err_lim = 1.0

    label_str = _label(ch)
    delta_str = (f"$\\Delta${ch['symbol']} ({ch['unit']})"
                 if ch['unit'] else f"$\\Delta${ch['symbol']}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ['True (CFD)', 'Predicted (MoE)', 'Error']

    plot_data = [
        (true_vals, 'viridis', vmin, vmax, label_str),
        (pred_vals, 'viridis', vmin, vmax, label_str),
        (err_vals,  'RdBu_r', -err_lim, err_lim, delta_str),
    ]

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]

    n_cells = len(x)
    n_vals  = len(true_vals)
    if n_vals != n_cells:
        print(f"  [WARNING] cell centers ({n_cells}) != field values "
              f"({n_vals}). Using min of both.")
        n = min(n_cells, n_vals)
        x = x[:n]
        y = y[:n]
        plot_data = [
            (d[:n], cm, vmn, vmx, lb)
            for d, cm, vmn, vmx, lb in plot_data
        ]

    for col, (ax, (vals, cmap, vmn, vmx, cbar_label), title) in enumerate(
            zip(axes, plot_data, titles)):

        sc = ax.scatter(x, y, c=vals, s=0.5, cmap=cmap,
                        vmin=vmn, vmax=vmx, rasterized=True)
        ax.set_xlabel(r'$x$ (m)')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13)

        if col == 0:
            ax.set_ylabel(r'$y$ (m)')
        else:
            ax.set_ylabel('')

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=18,
                            extend='both', orientation='horizontal',
                            pad=0.20)
        cbar.ax.set_title(cbar_label, fontsize=11, pad=5)
        cbar.ax.tick_params(labelsize=9)

        if cmap == 'RdBu_r':
            cbar.set_ticks([-err_lim, 0.0, err_lim])

    fig.suptitle(f'{ch["symbol"]} field  --  {case_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88,
                        bottom=0.22, wspace=0.35)

    name = f'contour_{key}_{case_name}'
    fig.savefig(os.path.join(output_dir, f'{name}.png'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f'{name}.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved {name}")


# =====================================================================
# Force coefficient time series
# =====================================================================

def plot_force_timeseries(cl_true, cl_pred, cd_true, cd_pred,
                          case_name, output_dir):
    """CL and CD vs window index (proxy for time)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    t = np.arange(len(cl_true))

    ax1.plot(t, cl_true, color='#1F77B4', lw=1.2, label='True (CFD)')
    ax1.plot(t, cl_pred, color='#D62728', lw=1.2, alpha=0.85,
             label='Predicted (MoE)')
    ax1.set_ylabel(r'$C_L$')
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(True, alpha=0.2)

    ax2.plot(t, cd_true, color='#1F77B4', lw=1.2, label='True (CFD)')
    ax2.plot(t, cd_pred, color='#D62728', lw=1.2, alpha=0.85,
             label='Predicted (MoE)')
    ax2.set_ylabel(r'$C_D$')
    ax2.set_xlabel('Window index')
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f'Force Coefficients  --  {case_name}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    name = f'forces_{case_name}'
    fig.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, f'{name}.pdf'), dpi=300)
    plt.close(fig)
    print(f"    saved {name}")


# =====================================================================
# Summary metrics table
# =====================================================================

def print_metrics_table(all_metrics):
    """Print a summary table of RMSE and correlation per case per field."""
    print(f"\n{'='*75}")
    print(f"  FIELD PREDICTION METRICS (normalised space)")
    print(f"{'='*75}")
    header = f"{'Case':<28s}  {'u RMSE':>8s}  {'v RMSE':>8s}  " \
             f"{'p RMSE':>8s}  {'a RMSE':>8s}  {'u R':>6s}  " \
             f"{'v R':>6s}  {'p R':>6s}  {'a R':>6s}"
    print(header)
    print('-' * len(header))
    for case, m in all_metrics.items():
        print(f"{case:<28s}  "
              f"{m['u_rmse']:8.4f}  {m['v_rmse']:8.4f}  "
              f"{m['p_rmse']:8.4f}  {m['a_rmse']:8.4f}  "
              f"{m['u_corr']:6.3f}  {m['v_corr']:6.3f}  "
              f"{m['p_corr']:6.3f}  {m['a_corr']:6.3f}")
    print(f"{'='*75}\n")


# =====================================================================
# Main
# =====================================================================

def generate_contour_plots(config_path='configs/config.yaml',
                           model_path='outputs/prano_cav_moe_model.pt',
                           stats_path='outputs/norm_stats.npz',
                           centers_path='data/cell_centers.npy',
                           output_dir='outputs/contours',
                           time_idx=-1):
    """
    Generate spatial contour plots for all test cases.

    Args:
        time_idx: which time step to plot (-1 = last predicted frame)
    """

    set_pub_style()
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    stats = np.load(stats_path, allow_pickle=True)
    stats = {k: stats[k] for k in stats.files}

    cell_centers = load_cell_centers(centers_path, stride=10)

    # Test cases (same as train/evaluate scripts)
    test_names = [
        'CN_0.8_AOA8_Re7.8e5',
        'CN_0.28_AOA8_Re5e5',
        'CN_0.28_AOA8_Re1.2e6',
        'CN_1.4_AOA8_Re7.8e5',
    ]
    data_dir = config.get('data_dir', 'data/processed')

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PRANOCavMoEModel(
        hidden_size=config['hidden_size'],
        num_regimes=config.get('num_regimes', 4),
        field_channels=config['field_channels'],
        scalar_outputs=config['scalar_outputs'],
        global_cond_size=3,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[contour] Model loaded, device={device}\n")

    all_metrics = {}

    for case_name in test_names:
        fpath = os.path.join(data_dir, f'{case_name}.npz')
        if not os.path.exists(fpath):
            print(f"[WARNING] {fpath} not found, skipping")
            continue

        print(f"[contour] Processing {case_name}...")
        ds = PRANOCavDataset([fpath], config['window_size'], stats=stats)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        y_true_all = []
        y_pred_all = []
        cl_true_all = []
        cl_pred_all = []
        cd_true_all = []
        cd_pred_all = []

        with torch.no_grad():
            for batch in loader:
                x_hist, y_target, global_cond, scalar_target, \
                    scalar_mask, regime_label = batch

                x_hist = x_hist.to(device)
                global_cond = global_cond.to(device)

                y_pred, scalar_pred, _, _ = model(x_hist, global_cond)

                y_true_all.append(y_target.cpu().numpy())
                y_pred_all.append(y_pred.cpu().numpy())
                cl_true_all.append(scalar_target[:, 0].numpy())
                cl_pred_all.append(scalar_pred[:, 0].cpu().numpy())
                cd_true_all.append(scalar_target[:, 1].numpy())
                cd_pred_all.append(scalar_pred[:, 1].cpu().numpy())

        y_true = np.concatenate(y_true_all)  # (W, N, 4)
        y_pred = np.concatenate(y_pred_all)
        cl_true = np.concatenate(cl_true_all)
        cl_pred = np.concatenate(cl_pred_all)
        cd_true = np.concatenate(cd_true_all)
        cd_pred = np.concatenate(cd_pred_all)

        # Pick one time frame for contour plots
        if time_idx == -1:
            t = y_true.shape[0] // 2   # middle frame
        else:
            t = min(time_idx, y_true.shape[0] - 1)

        y_true_frame = y_true[t]   # (N, 4)
        y_pred_frame = y_pred[t]

        # Contour plots for each field channel
        for ch_idx in range(4):
            plot_contour_triplet(ch_idx, y_true_frame, y_pred_frame,
                                cell_centers, case_name, output_dir)

        # Force time series
        plot_force_timeseries(cl_true, cl_pred, cd_true, cd_pred,
                              case_name, output_dir)

        # Metrics
        flat_true = y_true.reshape(-1, 4)
        flat_pred = y_pred.reshape(-1, 4)
        metrics = {}
        for i, name in enumerate(['u', 'v', 'p', 'a']):
            t_vals = flat_true[:, i]
            p_vals = flat_pred[:, i]
            rmse = np.sqrt(np.mean((p_vals - t_vals) ** 2))
            corr = np.corrcoef(t_vals, p_vals)[0, 1]
            metrics[f'{name}_rmse'] = rmse
            metrics[f'{name}_corr'] = corr
        all_metrics[case_name] = metrics

        print(f"  u: RMSE={metrics['u_rmse']:.4f} R={metrics['u_corr']:.3f}")
        print(f"  v: RMSE={metrics['v_rmse']:.4f} R={metrics['v_corr']:.3f}")
        print(f"  p: RMSE={metrics['p_rmse']:.4f} R={metrics['p_corr']:.3f}")
        print(f"  a: RMSE={metrics['a_rmse']:.4f} R={metrics['a_corr']:.3f}")

    print_metrics_table(all_metrics)
    print(f"[contour] All plots saved to {output_dir}/")


# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-mesh', type=str, default=None,
                        help='Path to an OpenFOAM case dir to extract '
                             'cell centers from. Run this once before '
                             'plotting.')
    parser.add_argument('--centers', type=str,
                        default='data/cell_centers.npy',
                        help='Path to saved cell centers .npy file')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/contours')
    parser.add_argument('--time-idx', type=int, default=-1,
                        help='Time step index for contour plots '
                             '(-1 = middle frame)')
    args = parser.parse_args()

    if args.extract_mesh:
        extract_cell_centers(args.extract_mesh, args.centers)
        print("\nCell centers extracted. Now run without --extract-mesh "
              "to generate plots.")
    else:
        generate_contour_plots(
            centers_path=args.centers,
            output_dir=args.output_dir,
            time_idx=args.time_idx,
        )
