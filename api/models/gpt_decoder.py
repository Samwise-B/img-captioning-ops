import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.positional import PositionalEncoding
from transformers import GPT2Tokenizer, GPT2Model


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        word_embed_dim: int = 768,
        img_embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 4 * 768,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.word_emb_dim = word_embed_dim
        gpt = GPT2Model.from_pretrained("gpt2")

        self.masked_attn_layers = nn.ModuleList(
            nn.Sequential(layer.ln_1, layer.attn) for layer in gpt.h
        )
        self.ln_2s = nn.ModuleList(layer.ln_2 for layer in gpt.h)
        self.cross_attn_layers = nn.ModuleList(
            CrossAttentionBlock(word_embed_dim, img_embed_dim, num_heads)
            for x in range(num_layers)
        )
        self.num_layers = num_layers
        self.new_vocab_size = vocab_size + 3

        self.embedding = nn.Embedding(self.new_vocab_size, word_embed_dim)
        with torch.no_grad():
            self.embedding.weight[:vocab_size] = gpt.wte.weight

        # self.positional = self.gpt.wpe
        self.projection = nn.Linear(word_embed_dim, self.new_vocab_size)

        self.ffs = nn.ModuleList(
            nn.Sequential(
                nn.Linear(word_embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, word_embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        )

        self.norm = nn.LayerNorm(word_embed_dim)

    def forward(self, tokens: torch.LongTensor, img_emb: torch.Tensor):
        positional = PositionalEncoding(self.word_emb_dim, tokens.size(1))
        word_emb = self.embedding(tokens)
        out_emb = positional(word_emb)
        # [batch_size, seq_len, embedding_dim]

        # [batch_size, seq_len, embedding_dim]
        for i in range(self.num_layers):
            masked_emb, _, _ = self.masked_attn_layers[i](out_emb)
            ln2_emb = self.ln_2s[i](masked_emb)
            cross_emb = self.cross_attn_layers[i](ln2_emb, img_emb)
            out_emb = self.ffs[i](cross_emb)

        out_emb = self.norm(out_emb + word_emb)

        # [seq_len, vocab_size]
        projected = self.projection(word_emb)

        return projected


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
    model = Decoder()
    pass
