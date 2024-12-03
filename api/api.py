from fastapi import FastAPI, UploadFile, Form
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import torch
import sys
from pathlib import Path
from transformers import GPT2Tokenizer
import torchvision
from typing import Union
from PIL import Image
from io import BytesIO

repo_dir = Path(__file__).parent
sys.path.append(str(repo_dir))

from models.gpt_transformer import DoubleTrouble
from utils.inference import run_inference

app = FastAPI()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device is {device}")

model = DoubleTrouble()
state_dict = torch.load(
    repo_dir / "weights/flickr-vit-transformer-full-0.pt", map_location=device
)
model.load_state_dict(state_dict)

tokeniser = GPT2Tokenizer.from_pretrained("gpt2")
tokeniser.add_special_tokens(
    {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
)
preprocess = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()


@app.post("/caption")
async def generate_caption(image: UploadFile, temperature: float = Form(...)):
    """
    Endpoint to generate a caption for an uploaded image.

    Parameters:
        image (UploadFile): The uploaded image file.
        temperature (float): The temperature parameter for caption generation.

    Returns:
        JSONResponse: Contains the generated caption.
    """
    if image:
        # Read the uploaded image into memory
        image_bytes = await image.read()

        # Open the image using PIL from the byte data
        img = Image.open(BytesIO(image_bytes))

        # Preprocess the image
        img = preprocess(img).unsqueeze(0)
        print("img shape:", img.shape)
        caption = run_inference(model, tokeniser, img, temperature)
        print(caption)

    return JSONResponse(content={"caption": caption})
