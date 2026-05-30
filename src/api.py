import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
import sys
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.capsnet import CapsNet

DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

def get_or_create_metric(metric_class, name, description, labels=None, buckets=None):
    try:
        if labels and buckets:
            return metric_class(name, description, labels, buckets=buckets)
        elif labels:
            return metric_class(name, description, labels)
        elif buckets:
            return metric_class(name, description, buckets=buckets)
        else:
            return metric_class(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

DETECTION_COUNTER = get_or_create_metric(Counter, 'peregrine_detections_total', 'Total detections', ['class_name'])
INFERENCE_HISTOGRAM = get_or_create_metric(Histogram, 'peregrine_inference_ms', 'Inference time ms', buckets=[5,10,25,50,100,250,500])
CONFIDENCE_GAUGE = get_or_create_metric(Gauge, 'peregrine_last_confidence', 'Last confidence score')
REQUEST_COUNTER = get_or_create_metric(Counter, 'peregrine_requests_total', 'Total requests', ['endpoint', 'status'])
MODEL_LOADED_GAUGE = get_or_create_metric(Gauge, 'peregrine_model_loaded', 'Model loaded status')

app = FastAPI(
    title="PEREGRINE API",
    description="Aerospace anomaly detection using CapsNet — Hinton 2017",
    version="1.0.0"
)

model = None
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    global model
    model = CapsNet(num_classes=20, in_channels=3, img_size=32)
    model_path = os.path.expanduser("~/PEREGRINE/best_capsnet_dior_jetson.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
        MODEL_LOADED_GAUGE.set(1)
    else:
        print("No trained model found — using random weights")
        MODEL_LOADED_GAUGE.set(0)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    load_model()
    print("PEREGRINE API is live")


@app.get("/")
async def root():
    REQUEST_COUNTER.labels(endpoint="/", status="200").inc()
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
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    REQUEST_COUNTER.labels(endpoint="/health", status="200").inc()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": len(DIOR_CLASSES)
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/classes")
async def get_classes():
    REQUEST_COUNTER.labels(endpoint="/classes", status="200").inc()
    return {
        "total": len(DIOR_CLASSES),
        "classes": {i: cls for i, cls in enumerate(DIOR_CLASSES)}
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        REQUEST_COUNTER.labels(endpoint="/detect", status="400").inc()
        raise HTTPException(status_code=400, detail="File must be an image")

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

        DETECTION_COUNTER.labels(class_name=predicted_class).inc()
        INFERENCE_HISTOGRAM.observe(inference_time)
        CONFIDENCE_GAUGE.set(confidence_score)
        REQUEST_COUNTER.labels(endpoint="/detect", status="200").inc()

        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
        top5 = [
            {"class": DIOR_CLASSES[idx.item()], "confidence": round(prob.item(), 4)}
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
            "image": {"filename": file.filename, "size": f"{img.size[0]}x{img.size[1]}"}
        })

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="/detect", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

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

            predicted_class = DIOR_CLASSES[predicted.item()]
            DETECTION_COUNTER.labels(class_name=predicted_class).inc()
            REQUEST_COUNTER.labels(endpoint="/detect/batch", status="200").inc()

            results.append({
                "filename": file.filename,
                "detected": predicted_class,
                "confidence": round(confidence.item(), 4)
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return JSONResponse({"system": "PEREGRINE", "batch_size": len(files), "results": results})


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


@app.get("/reports/latest")
async def latest_report():
    """Returns the most recent agent incident report"""
    reports_dir = os.path.expanduser("~/PEREGRINE/reports")
    if not os.path.exists(reports_dir):
        return JSONResponse({"error": "No reports yet"})
    
    reports = sorted([
        f for f in os.listdir(reports_dir)
        if f.endswith('.json')
    ])
    
    if not reports:
        return JSONResponse({"error": "No reports yet"})
    
    with open(os.path.join(reports_dir, reports[-1])) as f:
        import json as _json
        return JSONResponse(_json.load(f))


@app.get("/stats")
async def stats():
    """Returns live detection statistics"""
    try:
        import sqlite3 as _sqlite3
        db_path = os.path.expanduser("~/PEREGRINE/drift_monitor.db")
        if not os.path.exists(db_path):
            return JSONResponse({"error": "No data yet"})
        
        conn = _sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM detections")
        total = c.fetchone()[0]
        
        c.execute("""
            SELECT detected_class, COUNT(*) as cnt
            FROM detections
            GROUP BY detected_class
            ORDER BY cnt DESC
        """)
        class_counts = {row[0]: row[1] for row in c.fetchall()}
        
        c.execute("""
            SELECT AVG(confidence), AVG(entropy)
            FROM detections
            ORDER BY id DESC
            LIMIT 100
        """)
        avg_conf, avg_entropy = c.fetchone()
        
        c.execute("""
            SELECT COUNT(*) FROM drift_alerts
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        alerts = c.fetchone()[0]
        
        conn.close()
        
        return JSONResponse({
            "total_detections": total,
            "class_distribution": class_counts,
            "avg_confidence": round(avg_conf or 0, 4),
            "avg_entropy": round(avg_entropy or 0, 4),
            "drift_alerts_24h": alerts,
            "status": "ALERT" if alerts > 0 else "NOMINAL"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})
