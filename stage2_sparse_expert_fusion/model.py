"""
model.py — FineX cross-attentive latent sparse expert fusion (Stage 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D              = 512
NUM_EXPERTS    = 16
TOP_K          = 8
NUM_HEADS      = 8
NUM_ATT_LAYERS = 3
DROPOUT        = 0.3
HIDDEN_MUL     = 0.5


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        h = max(64, int(D * HIDDEN_MUL))
        self.net = nn.Sequential(
            nn.Linear(D, h),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(h, D),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.router  = nn.Linear(D, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([Expert() for _ in range(NUM_EXPERTS)])

    def forward(self, x):
        B, D = x.shape
        logits = self.router(x)                                # [B, E]        e.g. [4, 16]
        top_k_logits, top_k_idx = logits.topk(TOP_K, dim=-1)  # [B, K]        e.g. [4, 8]
        weights = F.softmax(top_k_logits, dim=-1)              # [B, K]        e.g. [4, 8]

        # tile each token K times — every (sample, slot) pair becomes one row
        tokens  = x.unsqueeze(1).expand(-1, TOP_K, -1).reshape(B * TOP_K, D)  # [B*K, D]  e.g. [32, 512]
        indices = top_k_idx.reshape(-1)                        # [B*K]         e.g. [32]  — expert id per row
        outputs = torch.zeros_like(tokens)                     # [B*K, D]      e.g. [32, 512]

        for e, expert in enumerate(self.experts):
            mask = indices == e                                # [B*K]  True where this expert is assigned
            if mask.any():
                outputs[mask] = expert(tokens[mask])           # each expert runs once on all its rows

        outputs = outputs.reshape(B, TOP_K, D)                 # [B, K, D]     e.g. [4, 8, 512]
        out     = (weights.unsqueeze(-1) * outputs).sum(dim=1) # [B, D]        e.g. [4, 512]
        return out, logits, top_k_idx, outputs                 # top_k_idx: [B,K]  outputs: [B,K,D]


class PairwiseCrossAttnMoE(nn.Module):
    """FineX Stage 2: pairwise cross-attention (Sec. 3.1) + streamwise latent sparse MoE (Sec. 3.2)."""
    def __init__(self, num_classes):
        super().__init__()
        self.proj_r = nn.Identity()
        self.proj_s = nn.Identity()
        self.proj_g = nn.Linear(256, D)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(D, NUM_HEADS, dropout=0.1, batch_first=True)
            for _ in range(NUM_ATT_LAYERS)
        ])
        self.norms    = nn.ModuleList([nn.LayerNorm(D) for _ in range(NUM_ATT_LAYERS)])
        self.moe      = SparseMoELayer()
        self.moe_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(DROPOUT),
            nn.Linear(D, num_classes),
        )

    def _cross(self, q, kv, attn, norm):
        out, _ = attn(q.unsqueeze(1), kv, kv)
        return norm(q + out.squeeze(1))

    def forward(self, fr, fs, fg):
        r, s, g = self.proj_r(fr), self.proj_s(fs), self.proj_g(fg)
        for attn, norm in zip(self.cross_attns, self.norms):
            r_new = self._cross(r, torch.stack([s, g], dim=1), attn, norm)
            s_new = self._cross(s, torch.stack([r, g], dim=1), attn, norm)
            g_new = self._cross(g, torch.stack([r, s], dim=1), attn, norm)
            r, s, g = r_new, s_new, g_new
        B = fr.size(0)
        flat = torch.stack([r, s, g], dim=1).reshape(B * 3, D)
        moe_flat, rl_flat, tk_flat, eo_flat = self.moe(flat)
        moe_out = moe_flat.reshape(B, 3, D).mean(dim=1)
        # router logits for all 3B (sample, stream) routing decisions, as in Eq. (9)
        return self.moe_head(moe_out), rl_flat, tk_flat, eo_flat


def load_balance_loss(router_logits):
    """Eq. (9): N * sum_i f_i q_i over all (sample, stream) routing decisions. router_logits: [3B, N]."""
    probs      = F.softmax(router_logits, dim=-1)
    top_k_mask = torch.zeros_like(probs)
    top_k_mask.scatter_(1, probs.topk(TOP_K, dim=-1).indices, 1.0)
    return NUM_EXPERTS * (top_k_mask.mean(0) * probs.mean(0)).sum()

