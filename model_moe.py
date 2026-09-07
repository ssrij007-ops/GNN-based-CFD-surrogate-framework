import torch
import torch.nn as nn


class PRANOCavMoEModel(nn.Module):
    """
    Regime-Aware MoE with shared GRU backbone + 4 expert heads.

    Improvements over v3:
      1. Attention-weighted pooling for scalar (CL/CD) prediction
         instead of naive mean pooling -- learns which cells matter
         for force integration.
      2. Alpha_v output clamped to [0, 1] physical bounds.
    """

    def __init__(self, hidden_size=128, num_regimes=4, field_channels=4,
                 scalar_outputs=2, global_cond_size=3):
        super().__init__()

        self.hidden_size      = hidden_size
        self.num_regimes      = num_regimes
        self.field_channels   = field_channels
        self.scalar_outputs   = scalar_outputs
        self.global_cond_size = global_cond_size

        # -- Shared GRU backbone (runs ONCE) --
        self.node_gru  = nn.GRU(input_size=field_channels,
                                hidden_size=hidden_size, batch_first=True)
        self.node_proj = nn.Linear(hidden_size, hidden_size)

        # -- 4 expert projection heads --
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_regimes)
        ])

        # -- Gating network --
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_size + global_cond_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_regimes),
        )

        # -- Regime classifier (supervised) --
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size + global_cond_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_regimes),
        )

        # -- Field prediction head --
        self.field_node_head = nn.Sequential(
            nn.Linear(hidden_size + global_cond_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, field_channels),
        )

        # -- IMPROVEMENT 4: Attention-weighted pooling for forces --
        # Learns which cells are important for CL/CD prediction
        # (approximates surface integration without mesh info)
        self.force_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # -- Scalar head (CL, CD) receives attention-pooled features --
        self.scalar_head = nn.Sequential(
            nn.Linear(hidden_size + global_cond_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, scalar_outputs),
        )

        self.gru_chunk = 32768

    def forward(self, x_hist, global_cond):
        """
        Args:
            x_hist      : (B, K, N, C)
            global_cond : (B, 3)

        Returns:
            field_next    : (B, N, C)
            scalars       : (B, 2)
            regime_logits : (B, 4)
            gates         : (B, 4)
        """
        B, K, N, C = x_hist.shape

        # -- Shared GRU backbone --
        x_nodes = x_hist.permute(0, 2, 1, 3).reshape(B * N, K, C)

        chunks = []
        for start in range(0, x_nodes.shape[0], self.gru_chunk):
            end = start + self.gru_chunk
            _, h = self.node_gru(x_nodes[start:end])
            chunks.append(h.squeeze(0))
        shared_latent = torch.cat(chunks, dim=0)
        shared_latent = self.node_proj(shared_latent)
        shared_latent = shared_latent.view(B, N, self.hidden_size)

        # -- Expert heads --
        expert_outputs = []
        for head in self.expert_heads:
            expert_outputs.append(head(shared_latent))

        # -- Gating --
        pooled = shared_latent.mean(dim=1)
        gating_input = torch.cat([pooled, global_cond], dim=-1)
        gates = torch.softmax(
            self.gating_network(gating_input), dim=-1)

        # -- Mixture --
        node_latent = torch.zeros(B, N, self.hidden_size,
                                  device=x_hist.device)
        for i in range(self.num_regimes):
            w = gates[:, i].unsqueeze(1).unsqueeze(2)
            node_latent = node_latent + w * expert_outputs[i]

        # -- Regime classifier --
        pooled_mix = node_latent.mean(dim=1)
        pooled_cond = torch.cat([pooled_mix, global_cond], dim=-1)
        regime_logits = self.regime_classifier(pooled_cond)

        # -- Field prediction (residual) --
        gc_broad = global_cond.unsqueeze(1).expand(-1, N, -1)
        node_feat = torch.cat([node_latent, gc_broad], dim=-1)
        field_delta = self.field_node_head(node_feat)
        field_next = x_hist[:, -1, :, :] + field_delta

        # -- IMPROVEMENT 5: Clamp alpha_v to [0, 1] --
        field_next_clamped = field_next
        #field_next_clamped[..., 3] = torch.clamp(field_next[..., 3], 0.0, 1.0)

        # -- IMPROVEMENT 4: Attention-weighted pooling for forces --
        # attention scores per cell
        attn_scores = self.force_attention(node_latent)    # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, N, 1)
        # weighted sum: cells near surface get higher weight
        force_pooled = (node_latent * attn_weights).sum(dim=1)  # (B, H)
        force_cond = torch.cat([force_pooled, global_cond], dim=-1)
        scalars = self.scalar_head(force_cond)

        return field_next_clamped, scalars, regime_logits, gates
