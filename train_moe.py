"""
PRANO-Cav MoE Training  --  v4 with 5 improvements
────────────────────────────────────────────────────
1. Load-balancing loss to prevent expert collapse
2. Channel-weighted field loss (alpha_v weighted higher)
3. Cosine annealing LR scheduler (200 epochs)
4. Attention-based surface pooling (in model)
5. Alpha bounds penalty (physical constraint)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
torch.set_num_threads(48)

import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from src.dataset import PRANOCavDataset, REGIME_NAMES, NUM_REGIMES
from src.model_moe import PRANOCavMoEModel


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =====================================================================
# Regime-specific physics losses
# =====================================================================

def physics_loss_inception(y_target, y_pred):
    """Inception: minimal cavitation, pressure accuracy."""
    p_t = y_target[..., 2]
    p_p = y_pred[..., 2]
    return torch.mean((p_p - p_t) ** 2)


def physics_loss_sheet(y_target, y_pred):
    """Sheet: attached cavity, pressure + alpha + smoothness."""
    p_t = y_target[..., 2]
    p_p = y_pred[..., 2]
    a_t = y_target[..., 3]
    a_p = y_pred[..., 3]
    loss_p = torch.mean((p_p - p_t) ** 2)
    loss_a = torch.mean((a_p - a_t) ** 2)
    p_diff = torch.diff(p_p, dim=-1)
    loss_smooth = torch.mean(p_diff ** 2)
    return loss_p + loss_a + 0.1 * loss_smooth

def physics_loss_cloud(y_target, y_pred):
    """Cloud: shedding dynamics, void fraction."""
    a_t = y_target[..., 3]
    a_p = y_pred[..., 3]
    loss_a = torch.mean((a_p - a_t) ** 2)
    # Bounds check in correct normalized range [-0.096, 10.35]
    ALPHA_NORM_MIN = -0.096
    ALPHA_NORM_MAX = 10.35
    below = torch.clamp(ALPHA_NORM_MIN - a_p, min=0)
    above = torch.clamp(a_p - ALPHA_NORM_MAX, min=0)
    loss_bound = torch.mean(below ** 2 + above ** 2)
    return loss_a + 0.05 * loss_bound

def physics_loss_supercav(y_target, y_pred):
    """Supercavitation: velocity + large cavity."""
    u_t = y_target[..., 0]
    u_p = y_pred[..., 0]
    v_t = y_target[..., 1]
    v_p = y_pred[..., 1]
    a_t = y_target[..., 3]
    a_p = y_pred[..., 3]
    loss_vel = torch.mean((u_p - u_t)**2) + torch.mean((v_p - v_t)**2)
    loss_a   = torch.mean((a_p - a_t) ** 2)
    return loss_vel + loss_a


PHYSICS_LOSSES = [
    physics_loss_inception,
    physics_loss_sheet,
    physics_loss_cloud,
    physics_loss_supercav,
]


def compute_regime_physics_loss(y_target, y_pred, regime_labels):
    """Apply correct physics loss per sample based on true regime."""
    total = torch.tensor(0.0, device=y_target.device)
    B = y_target.shape[0]
    for r_idx in range(NUM_REGIMES):
        mask = (regime_labels == r_idx)
        if mask.sum() == 0:
            continue
        y_t = y_target[mask]
        y_p = y_pred[mask]
        total = total + PHYSICS_LOSSES[r_idx](y_t, y_p) * mask.sum()
    return total / B


# =====================================================================
# IMPROVEMENT 1: Load-balancing loss
# =====================================================================

def load_balancing_loss(gates):
    """
    Penalise uneven expert usage across the batch.

    For each expert i:
      f_i = fraction of batch where expert i has highest gate
      P_i = mean gate probability for expert i
    L_balance = num_experts * sum(f_i * P_i)

    Minimum when all experts are used equally (f_i = P_i = 1/K).
    """
    num_experts = gates.shape[1]
    # f_i: fraction dispatched to each expert
    expert_assignments = gates.argmax(dim=-1)           # (B,)
    f = torch.zeros(num_experts, device=gates.device)
    for i in range(num_experts):
        f[i] = (expert_assignments == i).float().mean()
    # P_i: mean gate probability
    P = gates.mean(dim=0)                               # (K,)
    return num_experts * (f * P).sum()


def gating_entropy_loss(gates):
    """Encourage peaked gating distributions."""
    entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1)
    return entropy.mean()


# =====================================================================
# IMPROVEMENT 2: Channel-weighted field loss
# =====================================================================

def channel_weighted_mse(y_pred, y_target, channel_weights):
    """
    MSE with per-channel weights.
    channel_weights: tensor of shape (C,), e.g. [1, 1, 1, 3] for
    higher alpha_v weight.
    """
    # (B, N, C)
    sq_err = (y_pred - y_target) ** 2
    # weight each channel
    w = channel_weights.to(y_pred.device)
    weighted = sq_err * w.unsqueeze(0).unsqueeze(0)     # broadcast (1,1,C)
    return weighted.mean()


# =====================================================================
# IMPROVEMENT 5: Alpha bounds penalty
# =====================================================================

def alpha_bounds_penalty(y_pred):
    """
    Penalise alpha.water predictions outside physical bounds [0, 1]
    in NORMALIZED space.

    Normalized alpha.water bounds:
      physical 0.0 -> normalized = (0.0 - 0.009172) / 0.09577 = -0.0958
      physical 1.0 -> normalized = (1.0 - 0.009172) / 0.09577 = 10.348

    So valid normalized range is approximately [-0.096, 10.35].
    """
    ALPHA_NORM_MIN = -0.096   # corresponds to physical alpha.water = 0
    ALPHA_NORM_MAX = 10.35    # corresponds to physical alpha.water = 1

    alpha = y_pred[..., 3]
    below = torch.clamp(ALPHA_NORM_MIN - alpha, min=0)
    above = torch.clamp(alpha - ALPHA_NORM_MAX, min=0)
    return torch.mean(below ** 2 + above ** 2)

# =====================================================================
# Train/test split
# =====================================================================

def get_train_test_files(data_dir):
    train_names = [
        'CN_1.4_AOA8_Re1.2e6',
        'CN_1.4_AOA8_Re5e5',
        'CN_0.28_AOA8_Re7.8e5',
        'CN_0.4_AOA8_Re5e5',
        'CN_0.4_AOA8_Re7.8e5',
        'CN_0.5_AOA8_Re5e5',
        'CN_0.5_AOA8_Re7.8e5',
        'CN_0.8_AOA12_Re7.8e5',
        'CN_0.8_AOA4_Re7.8e5',
        'CN_0.8_AOA6_Re7.8e5',
        'CN_0.8_AOA8_Re1.2e6',
        'CN_0.8_AOA8_Re5e5',
    ]
    test_names = [
        'CN_0.8_AOA8_Re7.8e5',
        'CN_0.28_AOA8_Re5e5',
        'CN_0.28_AOA8_Re1.2e6',
        'CN_1.4_AOA8_Re7.8e5',
    ]
    train_files = [os.path.join(data_dir, f'{n}.npz') for n in train_names]
    test_files  = [os.path.join(data_dir, f'{n}.npz') for n in test_names]
    for p in train_files + test_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data file not found: {p}")
    return train_files, test_files


# =====================================================================
# Training loop
# =====================================================================

def train_moe(config):

    data_dir = config.get('data_dir', 'data/processed')
    train_files, test_files = get_train_test_files(data_dir)

    print(f"[train] TRAINING ({len(train_files)} cases):")
    for f in train_files:
        print(f"    {os.path.basename(f)}")
    print(f"[train] TEST ({len(test_files)} cases):")
    for f in test_files:
        print(f"    {os.path.basename(f)}")

    # -- Datasets --
    train_ds = PRANOCavDataset(train_files, config['window_size'])
    test_ds  = PRANOCavDataset(test_files,  config['window_size'],
                                stats=train_ds.stats)

    os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/norm_stats.npz', **train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    # -- Model --
    model = PRANOCavMoEModel(
        hidden_size    = config['hidden_size'],
        num_regimes    = config.get('num_regimes', NUM_REGIMES),
        field_channels = config['field_channels'],
        scalar_outputs = config['scalar_outputs'],
        global_cond_size = 3,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[train] device = {device}")
    print(f"[train] parameters = {sum(p.numel() for p in model.parameters()):,}")

    criterion_regime = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # IMPROVEMENT 3: Cosine annealing LR scheduler
    epochs = config.get('epochs', 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss weights
    W_SCALAR  = config.get('lambda_scalar',  0.1)
    W_REGIME  = config.get('lambda_regime',  0.05)
    W_PHYSICS = config.get('lambda_physics', 0.10)
    W_GATING  = config.get('lambda_gating',  0.01)
    W_BALANCE = config.get('lambda_balance', 0.05)   # NEW: load balancing
    W_BOUNDS  = config.get('lambda_bounds',  0.02)   # NEW: alpha bounds

    # IMPROVEMENT 2: Channel weights [u, v, p, alpha_v]
    alpha_weight = config.get('alpha_weight', 3.0)
    channel_weights = torch.tensor(
        [1.0, 1.0, 1.0, alpha_weight], dtype=torch.float32)

    print(f"\n[train] epochs={epochs}  LR scheduler=CosineAnnealing")
    print(f"[train] lambda: scalar={W_SCALAR}  regime={W_REGIME}  "
          f"physics={W_PHYSICS}  gating={W_GATING}  "
          f"balance={W_BALANCE}  bounds={W_BOUNDS}")
    print(f"[train] channel_weights = [u:1, v:1, p:1, alpha:{alpha_weight}]")
    print()

    # -- Training loop --
    hist = {k: [] for k in
            ['total', 'field', 'scalar', 'regime', 'physics',
             'gating', 'balance', 'bounds']}
    hist['lr'] = []

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {k: 0.0 for k in hist if k != 'lr'}

        for batch in train_loader:
            x_hist, y_target, global_cond, scalar_target, scalar_mask, \
                regime_label = batch

            x_hist       = x_hist.to(device)
            y_target     = y_target.to(device)
            global_cond  = global_cond.to(device)
            scalar_target = scalar_target.to(device)
            scalar_mask  = scalar_mask.to(device)
            regime_label = regime_label.to(device)

            optimizer.zero_grad()

            y_pred, scalar_pred, regime_logits, gates = \
                model(x_hist, global_cond)

            # Loss 1: IMPROVEMENT 2 -- channel-weighted field loss
            field_loss = channel_weighted_mse(
                y_pred, y_target, channel_weights)

            # Loss 2: scalar (CL, CD)
            sd = (scalar_pred - scalar_target) ** 2
            sp = sd.mean(dim=1)
            nv = scalar_mask.sum()
            scalar_loss = (sp * scalar_mask).sum() / (nv + 1e-8)

            # Loss 3: regime classification (supervised)
            regime_loss = criterion_regime(regime_logits, regime_label)

            # Loss 4: regime-specific physics
            physics_loss = compute_regime_physics_loss(
                y_target, y_pred, regime_label)

            # Loss 5: gating entropy
            gating_loss = gating_entropy_loss(gates)

            # Loss 6: IMPROVEMENT 1 -- load balancing
            balance_loss = load_balancing_loss(gates)

            # Loss 7: IMPROVEMENT 5 -- alpha bounds penalty
            bounds_loss = alpha_bounds_penalty(y_pred)

            # Total
            loss = (field_loss
                    + W_SCALAR  * scalar_loss
                    + W_REGIME  * regime_loss
                    + W_PHYSICS * physics_loss
                    + W_GATING  * gating_loss
                    + W_BALANCE * balance_loss
                    + W_BOUNDS  * bounds_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = x_hist.size(0)
            sums['total']   += loss.item()          * n
            sums['field']   += field_loss.item()     * n
            sums['scalar']  += scalar_loss.item()    * n
            sums['regime']  += regime_loss.item()     * n
            sums['physics'] += physics_loss.item()    * n
            sums['gating']  += gating_loss.item()     * n
            sums['balance'] += balance_loss.item()    * n
            sums['bounds']  += bounds_loss.item()     * n

        # IMPROVEMENT 3: Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        hist['lr'].append(current_lr)

        # Epoch averages
        N = len(train_ds)
        for k in sums:
            v = sums[k] / N
            hist[k].append(v)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"total {hist['total'][-1]:.6f} | "
                  f"field {hist['field'][-1]:.6f} | "
                  f"scalar {hist['scalar'][-1]:.6f} | "
                  f"regime {hist['regime'][-1]:.6f} | "
                  f"balance {hist['balance'][-1]:.6f} | "
                  f"bounds {hist['bounds'][-1]:.6f} | "
                  f"LR {current_lr:.2e}")

    # -- Save model --
    torch.save(model.state_dict(), 'outputs/prano_cav_moe_model.pt')
    print(f"\n[train] model saved -> outputs/prano_cav_moe_model.pt")

    # -- Loss curves (8 panels) --
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    titles = ['Field Loss', 'Scalar Loss', 'Regime Loss',
              'Physics Loss', 'Gating Loss', 'Balance Loss',
              'Bounds Loss', 'Total Loss']
    keys   = ['field', 'scalar', 'regime', 'physics',
              'gating', 'balance', 'bounds', 'total']
    for ax, title, key in zip(axes.flat, titles, keys):
        ax.semilogy(range(1, len(hist[key])+1), hist[key],
                    marker='o', markersize=1.5, linewidth=1.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('outputs/loss_curves_moe.png', dpi=300)
    fig.savefig('outputs/loss_curves_moe.pdf', dpi=300)
    plt.close(fig)

    # -- LR curve --
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(1, epochs+1), hist['lr'], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Cosine Annealing LR Schedule')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('outputs/lr_schedule.png', dpi=300)
    plt.close(fig)

    print(f"[train] loss curves saved -> outputs/loss_curves_moe.png")

    # -- Test evaluation --
    model.eval()
    test_field = 0.0
    correct = 0
    total_samp = 0
    expert_counts = [0] * NUM_REGIMES

    with torch.no_grad():
        for batch in test_loader:
            x_hist, y_target, global_cond, scalar_target, scalar_mask, \
                regime_label = batch
            x_hist       = x_hist.to(device)
            y_target     = y_target.to(device)
            global_cond  = global_cond.to(device)
            regime_label = regime_label.to(device)

            y_pred, _, regime_logits, gates = model(x_hist, global_cond)
            test_field += nn.MSELoss()(y_pred, y_target).item() * x_hist.size(0)

            pred = regime_logits.argmax(dim=-1)
            correct    += (pred == regime_label).sum().item()
            total_samp += x_hist.size(0)

            # Track expert usage
            top_expert = gates.argmax(dim=-1)
            for i in range(NUM_REGIMES):
                expert_counts[i] += (top_expert == i).sum().item()

    test_field /= len(test_ds)
    test_acc    = correct / total_samp if total_samp > 0 else 0

    print(f"\n[train] test field-loss = {test_field:.6f}")
    print(f"[train] test regime accuracy = {test_acc:.3f} "
          f"({correct}/{total_samp})")
    print(f"[train] expert usage on test set:")
    for i in range(NUM_REGIMES):
        pct = 100 * expert_counts[i] / total_samp if total_samp > 0 else 0
        print(f"  {REGIME_NAMES[i]:18s}: {expert_counts[i]:5d} "
              f"({pct:.1f}%)")
    print(f"\n[train] done")


# =====================================================================
if __name__ == '__main__':
    config = load_config('configs/config.yaml')
    print(f"[train] batch={config['batch_size']}  "
          f"window={config['window_size']}  "
          f"hidden={config['hidden_size']}  "
          f"epochs={config.get('epochs', 200)}  "
          f"alpha_weight={config.get('alpha_weight', 3.0)}")
    train_moe(config)
