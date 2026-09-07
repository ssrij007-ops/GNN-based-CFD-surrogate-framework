"""
PRANO-Cav MoE Evaluation  –  Corrected version
────────────────────────────────────────────────
Uses TRUE regime labels from the dataset for evaluation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml

from src.dataset import PRANOCavDataset, REGIME_NAMES, NUM_REGIMES
from src.model_moe import PRANOCavMoEModel


def find_npz_files(data_dir):
    files = []
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.npz'):
                files.append(os.path.join(data_dir, f))
    return sorted(files)


def evaluate_moe(model_path='outputs/prano_cav_moe_model.pt',
                 norm_stats_path='outputs/norm_stats.npz',
                 config_path='configs/config.yaml'):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    norm_stats = np.load(norm_stats_path, allow_pickle=True)
    norm_stats = {k: norm_stats[k] for k in norm_stats.files}

    # Same test cases as training script
    data_dir  = config.get('data_dir', 'data/processed')
    test_names = [
        'CN_0.8_AOA8_Re7.8e5',     # Sheet (interpolation)
        'CN_0.28_AOA8_Re5e5',      # Supercavitation (extrapolation)
        'CN_0.28_AOA8_Re1.2e6',    # Supercavitation (extrapolation)
        'CN_1.4_AOA8_Re7.8e5',     # Inception (extrapolation)
    ]
    test_files = [os.path.join(data_dir, f'{n}.npz') for n in test_names]

    print(f"[evaluate] Test files ({len(test_files)}):")
    for f in test_files:
        print(f"    {os.path.basename(f)}")

    test_ds = PRANOCavDataset(test_files, config['window_size'],
                               stats=norm_stats)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'],
                             shuffle=False, num_workers=0)

    # ── load model ───────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PRANOCavMoEModel(
        hidden_size    = config['hidden_size'],
        num_regimes    = config.get('num_regimes', NUM_REGIMES),
        field_channels = config['field_channels'],
        scalar_outputs = config['scalar_outputs'],
        global_cond_size = 3,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[evaluate] Model loaded from {model_path}")

    # ── inference ────────────────────────────────────────────────────
    all_regime_true = []
    all_regime_pred = []
    all_regime_conf = []
    all_gates       = []
    all_field_rmse  = []
    all_cl_true     = []
    all_cl_pred     = []
    all_cd_true     = []
    all_cd_pred     = []

    criterion = nn.MSELoss(reduction='none')

    print(f"\n{'='*70}")
    print(f"  REGIME-AWARE MoE EVALUATION")
    print(f"{'='*70}\n")

    with torch.no_grad():
        for batch in test_loader:
            x_hist, y_target, global_cond, scalar_target, \
                scalar_mask, regime_label = batch

            x_hist       = x_hist.to(device)
            y_target     = y_target.to(device)
            global_cond  = global_cond.to(device)
            scalar_target = scalar_target.to(device)
            regime_label = regime_label.to(device)

            y_pred, scalar_pred, regime_logits, gates = \
                model(x_hist, global_cond)

            # regime predictions
            probs = torch.softmax(regime_logits, dim=-1)
            pred  = probs.argmax(dim=-1)
            conf  = probs.max(dim=-1)[0]

            # per-sample field RMSE
            mse = criterion(y_pred, y_target).mean(dim=(1, 2))  # (B,)
            rmse = torch.sqrt(mse)

            all_regime_true.append(regime_label.cpu().numpy())
            all_regime_pred.append(pred.cpu().numpy())
            all_regime_conf.append(conf.cpu().numpy())
            all_gates.append(gates.cpu().numpy())
            all_field_rmse.append(rmse.cpu().numpy())
            all_cl_true.append(scalar_target[:, 0].cpu().numpy())
            all_cl_pred.append(scalar_pred[:, 0].cpu().numpy())
            all_cd_true.append(scalar_target[:, 1].cpu().numpy())
            all_cd_pred.append(scalar_pred[:, 1].cpu().numpy())

    # concatenate
    regime_true = np.concatenate(all_regime_true)
    regime_pred = np.concatenate(all_regime_pred)
    regime_conf = np.concatenate(all_regime_conf)
    gates       = np.concatenate(all_gates)
    field_rmse  = np.concatenate(all_field_rmse)
    cl_true     = np.concatenate(all_cl_true)
    cl_pred     = np.concatenate(all_cl_pred)
    cd_true     = np.concatenate(all_cd_true)
    cd_pred     = np.concatenate(all_cd_pred)

    # ── print summary ────────────────────────────────────────────────
    N = len(regime_true)
    acc = (regime_true == regime_pred).mean()
    print(f"Samples: {N}")
    print(f"\nRegime Classification Accuracy: {acc:.3f}")
    for r in range(NUM_REGIMES):
        mask = regime_true == r
        if mask.sum() > 0:
            r_acc = (regime_pred[mask] == r).mean()
            print(f"  {REGIME_NAMES[r]:18s}: {r_acc:.3f}  ({mask.sum()} samples)")

    print(f"\nRegime Confidence: mean={regime_conf.mean():.3f}  "
          f"std={regime_conf.std():.3f}")

    print(f"\nExpert Utilisation (avg gating weight):")
    g_mean = gates.mean(axis=0)
    for i in range(NUM_REGIMES):
        print(f"  {REGIME_NAMES[i]:18s}: {g_mean[i]:.3f}")

    cl_rmse = np.sqrt(np.mean((cl_pred - cl_true)**2))
    cd_rmse = np.sqrt(np.mean((cd_pred - cd_true)**2))
    print(f"\nForce RMSE:  CL={cl_rmse:.4f}   CD={cd_rmse:.4f}")
    print(f"Field RMSE:  mean={field_rmse.mean():.6f}")

    # ── plots ────────────────────────────────────────────────────────
    out = 'outputs'
    os.makedirs(out, exist_ok=True)

    # 1. Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(regime_true, regime_pred,
                          labels=list(range(NUM_REGIMES)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=REGIME_NAMES, yticklabels=REGIME_NAMES)
    ax.set_xlabel('Predicted');  ax.set_ylabel('True')
    ax.set_title('Regime Classification Confusion Matrix')
    fig.tight_layout()
    fig.savefig(f'{out}/regime_classification.png', dpi=300)
    fig.savefig(f'{out}/regime_classification.pdf', dpi=300)
    plt.close(fig)

    # 2. Gating heatmap
    sort_idx = np.argsort(regime_true)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(gates[sort_idx], aspect='auto', cmap='viridis',
                   vmin=0, vmax=1)
    changes = np.where(np.diff(regime_true[sort_idx]) != 0)[0]
    for b in changes:
        ax.axhline(b + 0.5, color='red', ls='--', lw=1)
    ax.set_xticks(range(NUM_REGIMES))
    ax.set_xticklabels(REGIME_NAMES)
    ax.set_ylabel('Sample (sorted by regime)')
    ax.set_title('Gating Weights per Sample')
    plt.colorbar(im, ax=ax, label='Weight')
    fig.tight_layout()
    fig.savefig(f'{out}/gating_weights.png', dpi=300)
    fig.savefig(f'{out}/gating_weights.pdf', dpi=300)
    plt.close(fig)

    # 3. Force predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(cl_true, cl_pred, alpha=0.5, s=15)
    lim = [min(cl_true.min(), cl_pred.min()),
           max(cl_true.max(), cl_pred.max())]
    ax1.plot(lim, lim, 'r--', lw=2)
    ax1.set_xlabel('True CL');  ax1.set_ylabel('Pred CL')
    ax1.set_title('Lift Coefficient');  ax1.grid(True, alpha=0.3)

    ax2.scatter(cd_true, cd_pred, alpha=0.5, s=15, color='#D62728')
    lim = [min(cd_true.min(), cd_pred.min()),
           max(cd_true.max(), cd_pred.max())]
    ax2.plot(lim, lim, 'r--', lw=2)
    ax2.set_xlabel('True CD');  ax2.set_ylabel('Pred CD')
    ax2.set_title('Drag Coefficient');  ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{out}/force_predictions.png', dpi=300)
    fig.savefig(f'{out}/force_predictions.pdf', dpi=300)
    plt.close(fig)

    # 4. Field RMSE per regime
    fig, ax = plt.subplots(figsize=(8, 5))
    data_box   = []
    labels_box = []
    for r in range(NUM_REGIMES):
        mask = regime_true == r
        if mask.sum() > 0:
            data_box.append(field_rmse[mask])
            labels_box.append(REGIME_NAMES[r])
    if data_box:
        ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True)
        for i, d in enumerate(data_box):
            ax.text(i+1, np.mean(d), f'{np.mean(d):.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Field RMSE');  ax.set_xlabel('Regime')
    ax.set_title('Field Prediction Error per Regime')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{out}/field_rmse_per_regime.png', dpi=300)
    fig.savefig(f'{out}/field_rmse_per_regime.pdf', dpi=300)
    plt.close(fig)

    print(f"\n[evaluate] plots saved → {out}/")
    print(f"[evaluate] done ✓")


if __name__ == '__main__':
    evaluate_moe()
