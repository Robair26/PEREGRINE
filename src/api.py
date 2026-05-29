import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.capsnet import CapsNet

DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

app = FastAPI(
    title="PEREGRINE API",
    description="Aerospace anomaly detection using CapsNet — Hinton 2017",
    version="1.0.0"
)

# Global model
model = None
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    global model
    model = CapsNet(num_classes=20, in_channels=3, img_size=64)

    model_path = os.path.expanduser("~/PEREGRINE/best_capsnet_dior.pth")
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        print(f"Loaded trained model from {model_path}")
    else:
        print("No trained model found — using random weights")
        print("Train first with: python src/train_dior.py")

    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    load_model()
    print("PEREGRINE API is live")


@app.get("/")
async def root():
    return {
        "system": "PEREGRINE",
        "description": "Aerospace anomaly detection — CapsNet",
        "version": "1.0.0",
        "status": "operational",
        "model": "CapsNet — Hinton 2017 dynamic routing",
        "classes": len(DIOR_CLASSES),
        "endpoints": {
            "health": "/health",
            "detect": "/detect",
            "classes": "/classes",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": len(DIOR_CLASSES)
    }


@app.get("/classes")
async def get_classes():
    return {
        "total": len(DIOR_CLASSES),
        "classes": {i: cls for i, cls in enumerate(DIOR_CLASSES)}
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an aerial or satellite image.
    PEREGRINE returns the detected object class
    with confidence scores for all 20 aerospace classes.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        start_time = time.time()
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = (time.time() - start_time) * 1000

        predicted_class = DIOR_CLASSES[predicted.item()]
        confidence_score = confidence.item()

        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
        top5 = [
            {
                "class": DIOR_CLASSES[idx.item()],
                "confidence": round(prob.item(), 4)
            }
            for prob, idx in zip(top5_probs[0], top5_indices[0])
        ]

        all_scores = {
            DIOR_CLASSES[i]: round(probabilities[0][i].item(), 4)
            for i in range(len(DIOR_CLASSES))
        }

        return JSONResponse({
            "system": "PEREGRINE",
            "detected": predicted_class,
            "confidence": round(confidence_score, 4),
            "inference_ms": round(inference_time, 2),
            "top5": top5,
            "all_scores": all_scores,
            "image": {
                "filename": file.filename,
                "size": f"{img.size[0]}x{img.size[1]}"
            }
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Upload multiple images for batch detection.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )

    results = []
    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            results.append({
                "filename": file.filename,
                "detected": DIOR_CLASSES[predicted.item()],
                "confidence": round(confidence.item(), 4)
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse({
        "system": "PEREGRINE",
        "batch_size": len(files),
        "results": results
    })


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
