import torch
import sys
from pathlib import Path

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

# from models.encoder import Encoder
from models.vit import ViT
from models.gpt_decoder import Decoder


class DoubleTrouble(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ViT()
        self.decoder = Decoder()

    def forward(self, tokens, patches):
        img_emb = self.encoder(patches)
        label_prediction = self.decoder(tokens, img_emb)
        return label_prediction

    def get_captions(self, pred_tokens, target_tokens, tokeniser):
        return tokeniser.decode(pred_tokens.tolist()), tokeniser.decode(
            target_tokens.tolist()
        )
