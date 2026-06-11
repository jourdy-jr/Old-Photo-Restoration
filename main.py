from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.responses import FileResponse

import torch
import torch.nn as nn

import torchvision.transforms as transforms

from PIL import Image

import numpy as np

import os

from model import UNet

# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "best_unet.pth"

UPLOAD_DIR = "uploads"

RESULT_DIR = "results"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# =====================================================
# LOAD MODEL
# =====================================================

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded.")

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

# =====================================================
# PADDING
# =====================================================

def pad_to_multiple(
    image,
    multiple=16
):

    w, h = image.size

    new_w = (
        (w + multiple - 1)
        // multiple
    ) * multiple

    new_h = (
        (h + multiple - 1)
        // multiple
    ) * multiple

    padded = Image.new(
        "RGB",
        (new_w, new_h),
        (0, 0, 0)
    )

    padded.paste(
        image,
        (0, 0)
    )

    return padded, w, h

# =====================================================
# RESTORE
# =====================================================

def restore_pil_image(image):

    padded, original_w, original_h = (
        pad_to_multiple(image)
    )

    tensor = transforms.ToTensor()(
        padded
    )

    tensor = tensor.unsqueeze(0).to(
        DEVICE
    )

    with torch.no_grad():

        pred = model(
            tensor
        )

    pred = pred.squeeze(0)

    pred = (
        pred.cpu()
        .permute(1,2,0)
        .numpy()
    )

    pred = np.clip(
        pred,
        0,
        1
    )

    pred = pred[
        :original_h,
        :original_w
    ]

    pred = (
        pred * 255
    ).astype(np.uint8)

    return Image.fromarray(pred)

# =====================================================
# API
# =====================================================

@app.post("/restore")
async def restore_image(
    image: UploadFile = File(...)
):

    input_path = os.path.join(
        UPLOAD_DIR,
        image.filename
    )

    with open(
        input_path,
        "wb"
    ) as f:

        f.write(
            await image.read()
        )

    pil_image = Image.open(
        input_path
    ).convert("RGB")

    restored = restore_pil_image(
        pil_image
    )

    output_path = os.path.join(
        RESULT_DIR,
        f"restored_{image.filename}"
    )

    restored.save(
        output_path
    )

    return FileResponse(
        output_path,
        media_type="image/jpeg"
    )