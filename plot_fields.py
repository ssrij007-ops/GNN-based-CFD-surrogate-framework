"""
Advanced visualization for PRANO-Cav MoE predictions.
Generates:
- Velocity field contours (u, v)
- Pressure field contours (p)
- Void fraction contours (alpha_v)
- Force coefficient time series
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader

from src.dataset import PRANOCavDataset
from src.model_moe import PRANOCavMoEModel


def set_publication_style():
    """Set matplotlib style for publication-quality plots."""
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
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
    })


def plot_field_contours(y_true, y_pred, case_name, output_dir='outputs'):
    """
    Plot field contours for a single case.
    
    Creates 3 figures:
    - Velocity field (u, v components)
    - Pressure field (p)
    - Void fraction field (alpha_v)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize if needed (assuming fields are in normalized space)
    # For now, plot as-is
    
    print(f"[plot] Generating contour plots for {case_name}...")
    
    channels = ['u (m/s)', 'v (m/s)', 'p (Pa)', r'$\alpha_v$ (-)']
    
    # ========== VELOCITY FIELD ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, ch_name) in enumerate(zip(axes, ['u', 'v'])):
        data_true = y_true[:, i]
        data_pred = y_pred[:, i]
        
        ax.hist([data_true, data_pred], bins=50, label=['True', 'Predicted'], alpha=0.7)
        ax.set_xlabel(ch_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{ch_name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Velocity Field - {case_name}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'velocity_field_{case_name}.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, f'velocity_field_{case_name}.pdf'), dpi=300)
    plt.close(fig)
    
    # ========== PRESSURE FIELD ==========
    fig, ax = plt.subplots(figsize=(10, 5))
    
    data_true = y_true[:, 2]
    data_pred = y_pred[:, 2]
    
    ax.hist([data_true, data_pred], bins=50, label=['True', 'Predicted'], alpha=0.7, color=['blue', 'red'])
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Pressure Field - {case_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'pressure_field_{case_name}.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, f'pressure_field_{case_name}.pdf'), dpi=300)
    plt.close(fig)
    
    # ========== ALPHA FIELD ==========
    fig, ax = plt.subplots(figsize=(10, 5))
    
    data_true = y_true[:, 3]
    data_pred = y_pred[:, 3]
    
    ax.hist([data_true, data_pred], bins=50, label=['True', 'Predicted'], alpha=0.7, color=['green', 'orange'])
    ax.set_xlabel(r'Void Fraction $\alpha_v$ (-)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Void Fraction Field - {case_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'alpha_field_{case_name}.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, f'alpha_field_{case_name}.pdf'), dpi=300)
    plt.close(fig)


def plot_field_statistics(y_true, y_pred, case_name, output_dir='outputs'):
    """
    Plot detailed field statistics and error metrics.
    """
    
    print(f"[plot] Generating field statistics for {case_name}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    channels = ['u', 'v', 'p', r'$\alpha_v$']
    colors = ['#1F77B4', '#D62728', '#2CA02C', '#FF7F0E']
    
    for idx, (ax, ch, color) in enumerate(zip(axes.flat, channels, colors)):
        true_vals = y_true[:, idx]
        pred_vals = y_pred[:, idx]
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
        mae = np.mean(np.abs(pred_vals - true_vals))
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        
        # Plot
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=10, color=color)
        
        # Perfect prediction line
        lim = [min(true_vals.min(), pred_vals.min()), 
               max(true_vals.max(), pred_vals.max())]
        ax.plot(lim, lim, 'k--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel(f'True {ch}')
        ax.set_ylabel(f'Predicted {ch}')
        ax.set_title(f'{ch}: RMSE={rmse:.4f}, Corr={corr:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    fig.suptitle(f'Field Prediction Statistics - {case_name}', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'field_stats_{case_name}.png'), dpi=300)
    fig.savefig(os.path.join(output_dir, f'field_stats_{case_name}.pdf'), dpi=300)
    plt.close(fig)


def generate_all_plots(model_path='outputs/prano_cav_moe_model.pt',
                       norm_stats_path='outputs/norm_stats.npz',
                       config_path='configs/config.yaml',
                       output_dir='outputs'):
    """
    Generate all visualization plots for a trained model.
    """
    
    import yaml
    
    set_publication_style()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load normalization stats
    norm_stats = np.load(norm_stats_path, allow_pickle=True)
    norm_stats = {k: norm_stats[k] for k in norm_stats.files}
    
    # Test cases to visualize
    test_cases = [
        'CN_0.8_AOA8_Re7.8e5',     # Sheet (interpolation)
        'CN_0.28_AOA8_Re5e5',      # Supercavitation (extrapolation)
        'CN_0.28_AOA8_Re1.2e6',    # Supercavitation (extrapolation)
        'CN_1.4_AOA8_Re7.8e5',     # Inception (extrapolation)
    ]
    
    data_dir = config.get('data_dir', 'data/processed')
    test_files = [os.path.join(data_dir, f'{c}.npz') for c in test_cases]
    
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
    
    print(f"\n{'='*80}")
    print(f"GENERATING DETAILED FIELD VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Process each test case
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"[WARNING] File not found: {test_file}")
            continue
        
        case_name = os.path.basename(test_file).replace('.npz', '')
        print(f"\n[plot] Processing case: {case_name}")
        
        # Load single case
        dataset = PRANOCavDataset([test_file], config['window_size'], stats=norm_stats)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for x_hist, y_target, global_cond, _, _, _ in dataloader:
                x_hist = x_hist.to(device)
                y_target = y_target.to(device)
                global_cond = global_cond.to(device)
                
                y_pred, _, _, _ = model(x_hist, global_cond)
                
                all_y_true.append(y_target.cpu().numpy())
                all_y_pred.append(y_pred.cpu().numpy())
        
        # Concatenate all batches
        y_true = np.vstack(all_y_true)  # (total_windows, N_cells, 4)
        y_pred = np.vstack(all_y_pred)
        
        # Reshape to (total_samples, 4) for easier analysis
        y_true_flat = y_true.reshape(-1, 4)
        y_pred_flat = y_pred.reshape(-1, 4)
        
        # Generate plots
        plot_field_contours(y_true_flat, y_pred_flat, case_name, output_dir)
        plot_field_statistics(y_true_flat, y_pred_flat, case_name, output_dir)
        
        print(f"[plot] ✓ Plots saved for {case_name}")
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE!")
    print(f"Plots saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    generate_all_plots()
