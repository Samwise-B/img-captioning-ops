import torch
from tqdm import tqdm
from pathlib import Path
import sys
from torchvision.transforms import ToPILImage
from PIL import Image
import random
import wandb
import torch.nn.functional as F

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.gpt_transformer import DoubleTrouble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, tokeniser, patches, temperature=1):
    bos_token = tokeniser.bos_token_id
    eos_token = tokeniser.eos_token_id
    max_len = 50

    with torch.inference_mode():
        # idx = random.randint(0, ds.__len__())
        inpt = torch.tensor([bos_token], device=device).unsqueeze(0)
        for i in range(max_len):
            next_pred = model(inpt, patches)
            logits = next_pred[:, -1, :]
            logits = logits / temperature
            top_k_logits, top_k_indices = torch.topk(logits, 50)
            probs = F.softmax(top_k_logits, dim=-1)

            sample = torch.multinomial(probs, num_samples=1)
            # print("sample", sample.shape)
            next_token = top_k_indices[0, sample.squeeze()].view(1, 1)
            # print("next_t", next_token.shape)
            inpt = torch.cat([inpt, next_token], dim=-1)
            if inpt[0, -1] == eos_token:
                break

            # print(f"Target: {tokeniser.decode(target, skip_special_tokens=True)}")
            # print(f"Prediction: {tokeniser.decode(inpt.squeeze(), skip_special_tokens=True)}")
            # print(inpt.shape)

        return tokeniser.decode(inpt.squeeze(), skip_special_tokens=True)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from sets.flickr import Flickr

    val_dataset = Flickr("val", num_rows=10)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, collate_fn=Flickr.collate_fn
    )

    model = DoubleTrouble()
    model = model.to(device)
    patches, inpt, targ = val_dataset[0]
    out, targ = run_inference(model, val_dataset.tokeniser, patches)
    pass
