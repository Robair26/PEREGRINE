import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import json
import time
from datetime import datetime
from drift_detection import log_detection, init_db, get_drift_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.capsnet import CapsNet

DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

THREAT_CLASSES = ['airplane', 'ship', 'vehicle', 'harbor', 'airport']

SEVERITY_MAP = {
    'airplane': 'HIGH',
    'ship': 'HIGH',
    'vehicle': 'MEDIUM',
    'harbor': 'MEDIUM',
    'airport': 'MEDIUM',
    'bridge': 'LOW',
    'dam': 'LOW',
    'stadium': 'LOW',
    'storagetank': 'MEDIUM',
    'trainstation': 'LOW',
    'overpass': 'LOW',
    'chimney': 'LOW',
    'windmill': 'LOW',
    'golffield': 'LOW',
    'groundtrackfield': 'LOW',
    'baseballfield': 'LOW',
    'basketballcourt': 'LOW',
    'tenniscourt': 'LOW',
    'Expressway-Service-area': 'LOW',
    'Expressway-toll-station': 'LOW'
}

CONTEXT_MAP = {
    'airplane': 'Fixed-wing aircraft detected. Possible military or commercial aviation asset. Cross-reference with known airfield locations.',
    'ship': 'Maritime vessel detected. Assess vessel class, size, and heading. Cross-reference with known shipping lanes and naval patrol zones.',
    'vehicle': 'Ground vehicle detected. Multiple vehicles may indicate convoy or staging activity. Assess road network proximity.',
    'harbor': 'Harbor facility detected. Assess vessel presence, loading activity, and infrastructure condition.',
    'airport': 'Airport facility detected. Assess runway condition, aircraft presence, and ground support equipment.',
    'bridge': 'Bridge infrastructure detected. Assess structural condition and traffic presence.',
    'dam': 'Dam structure detected. Assess water levels and structural integrity.',
    'storagetank': 'Storage tank facility detected. Possible fuel or chemical storage. Assess proximity to other infrastructure.',
    'trainstation': 'Rail facility detected. Assess track condition and rolling stock presence.',
    'stadium': 'Large assembly facility detected. Assess occupancy and surrounding activity.',
    'windmill': 'Wind energy infrastructure detected. Nominal observation.',
    'golffield': 'Recreational facility detected. Nominal observation.',
    'groundtrackfield': 'Athletic facility detected. Nominal observation.',
    'overpass': 'Road overpass detected. Assess structural condition and traffic flow.',
    'chimney': 'Industrial chimney detected. Possible manufacturing or energy facility nearby.',
    'baseballfield': 'Recreational facility detected. Nominal observation.',
    'basketballcourt': 'Recreational facility detected. Nominal observation.',
    'tenniscourt': 'Recreational facility detected. Nominal observation.',
    'Expressway-Service-area': 'Highway service area detected. Assess vehicle density and fuel infrastructure.',
    'Expressway-toll-station': 'Highway toll station detected. Assess traffic flow and operational status.'
}

RECOMMENDED_ACTIONS = {
    'HIGH': [
        'Escalate to watch officer immediately',
        'Request additional imagery collection',
        'Cross-reference with signals intelligence',
        'Initiate pattern of life analysis',
        'Notify relevant command authority'
    ],
    'MEDIUM': [
        'Flag for analyst review within 2 hours',
        'Schedule follow-up imagery collection',
        'Update activity log',
        'Compare with baseline imagery'
    ],
    'LOW': [
        'Log for routine reporting',
        'No immediate action required',
        'Include in next scheduled summary'
    ]
}


class PEREGRINEAgent:
    """
    PEREGRINE autonomous investigation agent.
    Detects objects, assesses severity, generates
    structured incident reports automatically.
    """

    def __init__(self):
        self.device = torch.device('cpu')
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        init_db()
        print("PEREGRINE Agent initialized")

    def _load_model(self):
        model = CapsNet(num_classes=20, in_channels=3, img_size=32)
        model_path = os.path.expanduser(
            "~/PEREGRINE/best_capsnet_dior_jetson.pth"
        )
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print("Model loaded for agent")
        model.eval()
        return model

    def detect(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        start = time.time()
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)[0]
            conf, pred = torch.max(probs, 0)
        inference_ms = (time.time() - start) * 1000

        top5_probs, top5_idx = torch.topk(probs, 5)
        top5 = [
            {"class": DIOR_CLASSES[i.item()],
             "confidence": round(p.item(), 4)}
            for p, i in zip(top5_probs, top5_idx)
        ]

        all_scores = {
            DIOR_CLASSES[i]: round(probs[i].item(), 4)
            for i in range(len(DIOR_CLASSES))
        }

        detected_class = DIOR_CLASSES[pred.item()]
        confidence = conf.item()

        log_detection(detected_class, confidence, all_scores)

        return {
            "detected": detected_class,
            "confidence": round(confidence, 4),
            "inference_ms": round(inference_ms, 2),
            "top5": top5,
            "all_scores": all_scores,
            "image_size": f"{img.size[0]}x{img.size[1]}"
        }

    def investigate(self, detection_result, image_path):
        detected = detection_result["detected"]
        confidence = detection_result["confidence"]
        top5 = detection_result["top5"]

        severity = SEVERITY_MAP.get(detected, "LOW")
        context = CONTEXT_MAP.get(detected, "Object detected.")
        actions = RECOMMENDED_ACTIONS.get(severity, [])

        ambiguity_score = 1.0 - (
            top5[0]["confidence"] - top5[1]["confidence"]
        ) if len(top5) > 1 else 0.0

        secondary_detections = [
            t for t in top5[1:]
            if t["confidence"] > 0.04
        ]

        investigation = {
            "primary_detection": detected,
            "severity": severity,
            "confidence_score": confidence,
            "ambiguity_score": round(ambiguity_score, 4),
            "contextual_assessment": context,
            "secondary_candidates": secondary_detections,
            "is_threat_class": detected in THREAT_CLASSES,
            "recommended_actions": actions,
            "requires_escalation": severity == "HIGH"
        }

        return investigation

    def generate_report(self, image_path):
        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        print("-" * 50)

        detection = self.detect(image_path)
        investigation = self.investigate(detection, image_path)
        drift_report = get_drift_report()

        report = {
            "report_id": f"PRG-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "system": "PEREGRINE",
            "version": "1.0.0",
            "image": {
                "path": os.path.basename(image_path),
                "size": detection["image_size"]
            },
            "detection": {
                "primary_class": detection["detected"],
                "confidence": detection["confidence"],
                "inference_ms": detection["inference_ms"],
                "top5": detection["top5"]
            },
            "assessment": {
                "severity": investigation["severity"],
                "is_threat_class": investigation["is_threat_class"],
                "requires_escalation": investigation["requires_escalation"],
                "ambiguity_score": investigation["ambiguity_score"],
                "contextual_assessment": investigation["contextual_assessment"],
                "secondary_candidates": investigation["secondary_candidates"],
                "recommended_actions": investigation["recommended_actions"]
            },
            "system_health": {
                "total_detections": drift_report["total_detections"],
                "drift_alerts_24h": drift_report["drift_alerts_24h"],
                "system_status": drift_report["status"]
            }
        }

        self._print_report(report)
        self._save_report(report)
        return report

    def _print_report(self, report):
        sev_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        sev = report["assessment"]["severity"]
        icon = sev_colors.get(sev, "⚪")

        print(f"\n{'='*55}")
        print(f"PEREGRINE INCIDENT REPORT — {report['report_id']}")
        print(f"{'='*55}")
        print(f"Timestamp:    {report['timestamp']}")
        print(f"Image:        {report['image']['path']}")
        print(f"\nDETECTION RESULT")
        print(f"Primary:      {report['detection']['primary_class'].upper()}")
        print(f"Confidence:   {report['detection']['confidence']:.4f}")
        print(f"Inference:    {report['detection']['inference_ms']:.1f}ms")
        print(f"\nTOP 5 CANDIDATES")
        for i, t in enumerate(report['detection']['top5']):
            print(f"  {i+1}. {t['class']:<30} {t['confidence']:.4f}")
        print(f"\nASSESSMENT")
        print(f"Severity:     {icon} {sev}")
        print(f"Threat class: {'YES' if report['assessment']['is_threat_class'] else 'NO'}")
        print(f"Escalate:     {'YES — IMMEDIATE ACTION REQUIRED' if report['assessment']['requires_escalation'] else 'NO'}")
        print(f"Ambiguity:    {report['assessment']['ambiguity_score']:.4f}")
        print(f"\nCONTEXT")
        print(f"  {report['assessment']['contextual_assessment']}")
        print(f"\nRECOMMENDED ACTIONS")
        for action in report['assessment']['recommended_actions']:
            print(f"  • {action}")
        print(f"\nSYSTEM STATUS: {report['system_health']['system_status']}")
        print(f"Total detections logged: {report['system_health']['total_detections']}")
        print(f"{'='*55}")

    def _save_report(self, report):
        reports_dir = os.path.expanduser("~/PEREGRINE/reports")
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(
            reports_dir,
            f"{report['report_id']}.json"
        )
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {path}")

    def batch_analyze(self, image_dir, max_images=5):
        print(f"\nBatch analysis — {image_dir}")
        images = [
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg')
        ][:max_images]

        reports = []
        high_severity = []

        for img_file in images:
            path = os.path.join(image_dir, img_file)
            report = self.generate_report(path)
            reports.append(report)
            if report['assessment']['severity'] == 'HIGH':
                high_severity.append(report)

        print(f"\n{'='*55}")
        print(f"BATCH ANALYSIS SUMMARY")
        print(f"{'='*55}")
        print(f"Images analyzed: {len(reports)}")
        print(f"High severity:   {len(high_severity)}")
        print(f"Escalations:     {sum(1 for r in reports if r['assessment']['requires_escalation'])}")

        classes = [r['detection']['primary_class'] for r in reports]
        unique = set(classes)
        print(f"Classes found:   {', '.join(unique)}")
        print(f"{'='*55}")

        return reports


if __name__ == "__main__":
    print("="*55)
    print("PEREGRINE — Autonomous Investigation Agent")
    print("="*55)

    agent = PEREGRINEAgent()

    test_dir = os.path.expanduser(
        "~/PEREGRINE/data/DIOR/JPEGImages-test"
    )

    print("\nRunning single image analysis...")
    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_image = os.path.join(test_dir, images[0])
    agent.generate_report(test_image)

    print("\nRunning batch analysis...")
    agent.batch_analyze(test_dir, max_images=5)
