import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import time
import sqlite3
from scipy import stats
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.capsnet import CapsNet

DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

DB_PATH = os.path.expanduser("~/PEREGRINE/drift_monitor.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            detected_class TEXT,
            confidence REAL,
            top1_score REAL,
            entropy REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS drift_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            metric TEXT,
            value REAL,
            threshold REAL,
            message TEXT
        )
    ''')
    conn.commit()
    conn.close()

def compute_entropy(probabilities):
    probs = np.array(probabilities)
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs + 1e-10))

def log_detection(detected_class, confidence, all_scores):
    entropy = compute_entropy(list(all_scores.values()))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections
        (timestamp, detected_class, confidence, top1_score, entropy)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), detected_class,
          confidence, confidence, entropy))
    conn.commit()
    conn.close()

def log_alert(alert_type, metric, value, threshold, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO drift_alerts
        (timestamp, alert_type, metric, value, threshold, message)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), alert_type,
          metric, value, threshold, message))
    conn.commit()
    conn.close()
    print(f"DRIFT ALERT [{alert_type}]: {message}")

def get_recent_stats(window=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT confidence, entropy, detected_class
        FROM detections
        ORDER BY id DESC
        LIMIT ?
    ''', (window,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_baseline_stats(window=500):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT confidence, entropy
        FROM detections
        ORDER BY id ASC
        LIMIT ?
    ''', (window,))
    rows = c.fetchall()
    conn.close()
    return rows

def check_drift():
    recent = get_recent_stats(100)
    baseline = get_baseline_stats(500)

    if len(recent) < 50 or len(baseline) < 100:
        print(f"Not enough data yet — {len(recent)} recent, {len(baseline)} baseline")
        return

    recent_conf = [r[0] for r in recent]
    recent_entropy = [r[1] for r in recent]
    baseline_conf = [r[0] for r in baseline]
    baseline_entropy = [r[1] for r in baseline]

    # KS test on confidence scores
    ks_stat, ks_pval = stats.ks_2samp(baseline_conf, recent_conf)
    print(f"Confidence KS stat: {ks_stat:.4f} p-value: {ks_pval:.4f}")

    if ks_pval < 0.05:
        log_alert(
            "DATA_DRIFT",
            "confidence_ks_test",
            ks_stat,
            0.05,
            f"Confidence score distribution shifted — KS stat: {ks_stat:.4f}, p-value: {ks_pval:.4f}"
        )

    # KS test on entropy
    ks_stat_e, ks_pval_e = stats.ks_2samp(baseline_entropy, recent_entropy)
    print(f"Entropy KS stat: {ks_stat_e:.4f} p-value: {ks_pval_e:.4f}")

    if ks_pval_e < 0.05:
        log_alert(
            "DATA_DRIFT",
            "entropy_ks_test",
            ks_stat_e,
            0.05,
            f"Prediction entropy shifted — KS stat: {ks_stat_e:.4f}, p-value: {ks_pval_e:.4f}"
        )

    # Low confidence alert
    avg_conf = np.mean(recent_conf)
    print(f"Average confidence: {avg_conf:.4f}")
    if avg_conf < 0.04:
        log_alert(
            "LOW_CONFIDENCE",
            "avg_confidence",
            avg_conf,
            0.04,
            f"Average confidence critically low: {avg_conf:.4f} — model may need retraining"
        )

    # Class distribution shift
    recent_classes = [r[2] for r in recent]
    class_counts = {c: recent_classes.count(c) for c in DIOR_CLASSES}
    dominant = max(class_counts, key=class_counts.get)
    dominant_pct = class_counts[dominant] / len(recent_classes)
    print(f"Dominant class: {dominant} ({dominant_pct:.1%})")

    if dominant_pct > 0.5:
        log_alert(
            "CLASS_IMBALANCE",
            "dominant_class_pct",
            dominant_pct,
            0.5,
            f"Class '{dominant}' dominates {dominant_pct:.1%} of recent detections — possible distribution shift"
        )

    print(f"Drift check complete — {len(recent)} recent samples analyzed")


def get_drift_report():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]

    c.execute("""
        SELECT detected_class, COUNT(*) as cnt
        FROM detections
        GROUP BY detected_class
        ORDER BY cnt DESC
        LIMIT 5
    """)
    top_classes = c.fetchall()

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
    alerts_24h = c.fetchone()[0]

    conn.close()

    report = {
        "total_detections": total,
        "top_classes": [{"class": c, "count": n} for c, n in top_classes],
        "avg_confidence_last_100": round(avg_conf or 0, 4),
        "avg_entropy_last_100": round(avg_entropy or 0, 4),
        "drift_alerts_24h": alerts_24h,
        "status": "ALERT" if alerts_24h > 0 else "NOMINAL"
    }
    return report


if __name__ == "__main__":
    print("=" * 55)
    print("PEREGRINE — Drift Detection System")
    print("=" * 55)

    init_db()

    model = CapsNet(num_classes=20, in_channels=3, img_size=32)
    model_path = os.path.expanduser("~/PEREGRINE/best_capsnet_dior_jetson.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("\nRunning batch detection on test images...")
    test_dir = os.path.expanduser(
        "~/PEREGRINE/data/DIOR/JPEGImages-test"
    )
    images = [f for f in os.listdir(test_dir)
              if f.endswith('.jpg')][:200]

    for i, img_file in enumerate(images):
        try:
            img = Image.open(
                os.path.join(test_dir, img_file)
            ).convert('RGB')
            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                import torch.nn.functional as F
                probs = F.softmax(output, dim=1)[0]
                conf, pred = torch.max(probs, 0)

            scores = {DIOR_CLASSES[j]: probs[j].item()
                      for j in range(len(DIOR_CLASSES))}
            log_detection(
                DIOR_CLASSES[pred.item()],
                conf.item(),
                scores
            )

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(images)} images")

        except Exception as e:
            continue

    print("\nChecking for drift...")
    check_drift()

    print("\nDrift Report:")
    report = get_drift_report()
    print(json.dumps(report, indent=2))
    print("=" * 55)
