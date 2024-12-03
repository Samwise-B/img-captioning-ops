import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        self.scaling_fac = self.embed_dim ** (1 / 2)
        self.add_norm = add_norm
        if embedding_dim % num_heads:
            raise Exception("Embed dim not divisible by num of heads")
        self.head_dim = embedding_dim // num_heads

        self.concat_proj = nn.Linear(embedding_dim, embedding_dim)
        self.qkv_concat = nn.Linear(embedding_dim, embedding_dim * 3)

        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        embeddings = x
        batch_size, seq_len, _ = x.size()
        # Embeddings: [batch_size, seq_len, embedding_dim]

        # [num_heads, batch_size, seq_len, head_dim]
        # Qs = [M_q(embeddings) for M_q in self.M_qs]
        # Ks = [M_k(embeddings) for M_k in self.M_ks]
        qkv = self.qkv_concat(x)
        qkv = qkv.view(batch_size, 3, self.num_heads, seq_len, self.head_dim)
        Qs, Ks, Vs = qkv.unbind(dim=1)

        # [num_heads, batch_size, seq_len, seq_len]
        As = (Qs @ Ks.transpose(-1, -2)) / self.scaling_fac

        masks = torch.full_like(As, float("-inf"))
        masks = torch.triu(masks, diagonal=1)

        As_masked = As + masks

        # num_heads[batch_size, seq_len, seq_len]
        # As = [F.softmax(As_masked, dim=-1) for A in A_primes]
        As = F.softmax(As_masked, dim=-1)

        # Vs = [M_v(embeddings) for M_v in self.M_vs]

        # num_heads[batch_size, seq_len, head_dim]
        # Hs = [torch.bmm(A, V) for A, V in zip(As, Vs)]
        Hs = As @ Vs

        # [batch_size, seq_len, num_heads*head_dim = embed_dim]
        # H = torch.cat(Hs, dim=-1)
        H = Hs.view(batch_size, seq_len, self.embed_dim)

        H = self.concat_proj(H)
        H = self.attn_dropout(H)

        if self.add_norm:
            H = self.norm(H + embeddings)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(H)
