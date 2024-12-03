import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.positional import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_embed_dim: int,
        img_embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.word_emb_dim = word_embed_dim
        # Must return tensor of the same shape
        self.attn_layers = nn.ModuleList(
            DecoderAttentionBlock(
                word_embed_dim, img_embed_dim, num_heads, ff_dim, dropout
            )
            for _ in range(num_layers)
        )

        self.embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.projection = nn.Linear(word_embed_dim, vocab_size)

    def forward(self, tokens: torch.LongTensor, img_emb: torch.Tensor):
        self.embed_pos = PositionalEncoding(self.word_emb_dim, tokens.size(1))
        # tokens: [batch_size, seq_len]
        word_emb = self.embedding(tokens)
        word_emb = self.embed_pos(word_emb)
        # [batch_size, seq_len, embedding_dim]

        # [batch_size, seq_len, embedding_dim]
        for layer in self.attn_layers:
            word_emb = layer((word_emb, img_emb))

        # [seq_len, vocab_size]
        projected = self.projection(word_emb)

        return projected


class DecoderAttentionBlock(nn.Module):
    def __init__(
        self,
        word_embed_dim: int,
        img_embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: int = 0.1,
    ):
        super().__init__()
        # self.combined_dim = max(word_embed_dim, img_embed_dim)
        self.combined_dim = word_embed_dim
        self.masked_attn = MaskedAttentionBlock(word_embed_dim, num_heads)
        self.cross_attn = CrossAttentionBlock(word_embed_dim, img_embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(self.combined_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, word_embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: tuple[torch.Tensor]):
        word_emb, img_emb = x

        word_emb = self.masked_attn(word_emb)
        combined_embeddings = self.cross_attn(word_emb, img_emb)

        return self.ff(combined_embeddings)


class MaskedAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.scaling_fac = self.embed_dim ** (1 / 2)
        self.add_norm = add_norm

        assert embedding_dim % num_heads == 0
        self.num_heads = int(num_heads)
        self.head_dim = int(embedding_dim // num_heads)
        self.qkv_combined = nn.Linear(embedding_dim, embedding_dim * 3)
        self.concat_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, word_emb: torch.Tensor):

        batch_size, seq_len, _ = word_emb.size()

        qkv = self.qkv_combined(word_emb)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)
        Qs, Ks, Vs = qkv.unbind(dim=1)

        # [seq_len, seq_len]
        attn = (Qs @ Ks.transpose(-1, -2)) / self.scaling_fac

        # seq_len = attn.shape[1]

        mask = torch.full_like(attn, float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        masked_attn = attn + mask

        # [batch_size, seq_len, seq_len]
        As = F.softmax(masked_attn, dim=-1)

        # [batch_size, seq_len, embedding_dim]
        attn_emb = As @ Vs

        attn_emb = attn_emb.view(batch_size, seq_len, self.embed_dim)
        attn_emb = self.concat_proj(attn_emb)

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + word_emb)

        # [batch_size, seq_len, embedding_dim]
        return attn_emb


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        word_emb_dim: int,
        img_emb_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.word_emb_dim = word_emb_dim
        self.img_emb_dim = img_emb_dim
        # self.combined_dim = min(word_emb_dim, img_emb_dim)
        self.scaling_fac = self.word_emb_dim ** (1 / 2)
        self.add_norm = add_norm

        assert word_emb_dim % num_heads == 0
        assert img_emb_dim % num_heads == 0
        self.num_heads = int(num_heads)
        # self.combined_dim = max(self.word_emb_dim, self.img_emb_dim)
        self.combined_dim = self.word_emb_dim
        self.head_dim = int(self.combined_dim // num_heads)

        self.concat_proj = nn.Linear(self.combined_dim, self.combined_dim)
        self.M_q = nn.Linear(word_emb_dim, self.combined_dim)
        self.M_k = nn.Linear(img_emb_dim, self.combined_dim)
        self.M_v = nn.Linear(img_emb_dim, self.combined_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(word_emb_dim)

    def forward(self, word_emb: torch.Tensor, img_emb: torch.Tensor):
        batch_size, seq_len, _ = word_emb.size()
        _, num_patches, _ = img_emb.size()

        Qs = self.M_q(word_emb).view(batch_size, seq_len, self.num_heads, self.head_dim)
        Qs = Qs.permute(0, 2, 1, 3)
        # [batch_size, seq_len, num_heads, word_head_dim]

        Ks = self.M_k(img_emb).view(
            batch_size, num_patches, self.num_heads, self.head_dim
        )
        Ks = Ks.permute(0, 2, 1, 3)
        # [batch_size, seq_len, num_heads, img_head_dim]

        As = (Qs @ Ks.transpose(-1, -2)) / self.scaling_fac
        # [batch_size, seq_len, seq_len]

        As = F.softmax(As, dim=-1)
        # [batch_size, seq_len, seq_len]

        Vs = self.M_v(img_emb).view(
            batch_size, num_patches, self.num_heads, self.head_dim
        )
        Vs = Vs.permute(0, 2, 1, 3)
        # [batch_size, seq_len, num_heads, img_head_dim]

        attn_emb = As @ Vs
        # [batch_size, seq_len, embedding_dim]

        attn_emb = attn_emb.view(batch_size, seq_len, self.combined_dim)
        attn_emb = self.concat_proj(attn_emb)

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + word_emb)

        # [batch_size, seq_len, embedding_dim]
        return attn_emb


if __name__ == "__main__":
    pass
