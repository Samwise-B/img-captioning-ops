import torch
import torch.nn.functional as F
from torch import nn
import sys
from pathlib import Path

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.positional import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_patches: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Linear(patch_size, embed_dim)
        self.embed_pos = PositionalEncoding(embed_dim, num_patches)
        self.attn_block = nn.ModuleList(
            [
                EncoderAttentionBlock(embed_dim, ff_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, patches: torch.LongTensor):
        embeddings = self.embedding(patches)
        embeddings = self.embed_pos(embeddings)
        # batch_size, seq_len, embedding_dim
        for layer in self.attn_block:
            embeddings = layer(embeddings)
        # batch_size, seq_len, embedding_dim

        return embeddings


class EncoderAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ff_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.scaling_fac = self.embed_dim ** (1 / 2)
        self.add_norm = add_norm
        # self.M_q = nn.Linear(embedding_dim, embedding_dim)
        # self.M_k = nn.Linear(embedding_dim, embedding_dim)
        # self.M_v = nn.Linear(embedding_dim, embedding_dim)
        assert embedding_dim % num_heads == 0
        self.head_dim = int(embedding_dim // num_heads)
        self.num_heads = int(num_heads)
        self.qkv_combined = nn.Linear(embedding_dim, embedding_dim * 3)
        self.concat_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, img_emb: torch.Tensor):
        batch_size, seq_len, _ = img_emb.size()
        qkv = self.qkv_combined(img_emb)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)
        Qs, Ks, Vs = qkv.unbind(dim=1)

        # [batch_size, num_heads, seq_len, seq_len]
        As = (Qs @ Ks.transpose(-1, -2)) / self.scaling_fac

        # [batch_size, num_heads, seq_len, seq_len]
        As = F.softmax(As, dim=-1)

        # [batch_size, num_heads, seq_len, head_dim]
        attn_emb = As @ Vs

        attn_emb = attn_emb.view(batch_size, seq_len, self.embed_dim)
        # batch_size, seq_len, embed_dim

        attn_emb = self.concat_proj(attn_emb)

        # batch_size, seq_len, embed_dim
        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + img_emb)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(attn_emb)
