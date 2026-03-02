import torch
import torch.nn as nn


class AttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        attn_out, attn_w = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x, attn_w


class FusionMultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        morph_dim: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.vis_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.morph_proj = nn.Sequential(
            nn.LayerNorm(morph_dim),
            nn.Linear(morph_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([AttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_emb: torch.Tensor, x_morph: torch.Tensor, return_attn: bool = False):
        bsz = x_emb.size(0)

        v = self.vis_proj(x_emb).unsqueeze(1)
        m = self.morph_proj(x_morph).unsqueeze(1)
        cls = self.cls_token.expand(bsz, -1, -1)

        x = torch.cat([cls, v, m], dim=1)

        attn_all = []
        for blk in self.blocks:
            x, attn_w = blk(x)
            if return_attn:
                attn_all.append(attn_w)

        logits = self.head(x[:, 0])
        if return_attn:
            return logits, attn_all
        return logits