"""PEREGRINE v4: reproducible DIOR validation evaluation (no official test access).

Usage from repository root:
 PYTHONPATH=src python evaluate_dior_v4.py --output-dir experiments/resnet_v4_validation
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data.dior_dataloader import DIORDataset, DIOR_CLASSES
from models.resnet_baseline import ResNetBaseline
from train_dior_v2 import choose_indices


def calculate_metrics(actual, predicted, classes):
    labels = list(range(len(classes)))
    if not actual:
        raise ValueError('Cannot score an empty evaluation set')
    matrix = confusion_matrix(actual, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    per_class = [
        {'class_id': i, 'class_name': name, 'precision': float(precision[i]),
         'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
        for i, name in enumerate(classes)
    ]
    return {
        'accuracy': float(accuracy_score(actual, predicted)),
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'num_samples': len(actual),
        'per_class': per_class,
        'confusion_matrix': matrix.tolist(),
    }


def write_csv(path, fields, records):
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description='Independent reproducible validation diagnostics (DIOR test data untouched).')
    parser.add_argument('--manifest', default='dior_split_manifest_v1.json')
    parser.add_argument('--data-root', default='data/DIOR')
    parser.add_argument('--checkpoint', default='experiments/resnet_v2_experiment_3/best_resnet_v2_state_dict.pth')
    parser.add_argument('--per-class', type=int, default=20, help='Validation crops per class, 20 reproduces Experiment 3')
    parser.add_argument('--seed', type=int, default=42, help='Original experiment seed')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', default='experiments/resnet_v4_validation')
    args = parser.parse_args()
    if min(args.per_class, args.batch_size, args.threads) < 1:
        parser.error('per-class, batch-size and threads must be positive')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'{output} already exists. Refusing to overwrite reports.')
    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    train_ids = set(manifest['train_ids'])
    val_ids = set(manifest['validation_ids'])
    test_ids = set(manifest['test_ids'])
    if not (train_ids and val_ids and test_ids):
        raise ValueError('Empty partition in split manifest')
    if len(train_ids) != len(manifest['train_ids']) or len(val_ids) != len(manifest['validation_ids']) or len(test_ids) != len(manifest['test_ids']):
        raise ValueError('Duplicate image IDs found in manifest')
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('Source-image leakage detected in manifest')
    torch.set_num_threads(args.threads)
    transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Intentionally only load the trainval image directory; never instantiate split='test'.
    dataset = DIORDataset(args.data_root, split='train', transform=transform, img_size=64)
    indices = choose_indices(dataset, val_ids, args.per_class, args.seed + 10000)
    actual_ids = {Path(dataset.samples[i]['img_path']).stem for i in indices}
    if not actual_ids <= val_ids or actual_ids & (train_ids | test_ids):
        raise ValueError('Selected samples fall outside validation partition')
    if len(indices) != len(DIOR_CLASSES) * args.per_class:
        raise ValueError('Unexpected number of validation samples')
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    model = ResNetBaseline(num_classes=len(DIOR_CLASSES), in_channels=3)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(targets.tolist())
    metrics = calculate_metrics(actual, predicted, DIOR_CLASSES)
    predictions = []
    for i, truth, guess in zip(indices, actual, predicted):
        item = dataset.samples[i]
        predictions.append({
            'image_id': Path(item['img_path']).stem,
            'bbox': ','.join(map(str, item['bbox'])),
            'true_class': DIOR_CLASSES[truth],
            'predicted_class': DIOR_CLASSES[guess],
            'correct': int(truth == guess),
        })
    # Write only after all validation and scoring has succeeded.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'per_class.csv',
              ['class_id', 'class_name', 'precision', 'recall', 'f1', 'support'], metrics['per_class'])
    write_csv(output / 'predictions.csv',
              ['image_id', 'bbox', 'true_class', 'predicted_class', 'correct'], predictions)
    matrix_rows = [dict(actual_class=name, **{cls: count for cls, count in zip(DIOR_CLASSES, row)})
                   for name, row in zip(DIOR_CLASSES, metrics['confusion_matrix'])]
    write_csv(output / 'confusion_matrix.csv', ['actual_class', *DIOR_CLASSES], matrix_rows)
    report = {
        'evaluation_split': 'validation_only',
        'official_test_set_accessed': False,
        'checkpoint': str(args.checkpoint),
        'manifest': str(args.manifest),
        'seed': args.seed,
        'per_class': args.per_class,
        'class_order': DIOR_CLASSES,
        'selected_source_images': len(actual_ids),
        'metrics': metrics,
        'note': 'Checkpoint selected using validation macro-F1. These results are NOT independent test results.'
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Validation only: {len(actual)} crops / {len(actual_ids)} source images')
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f'Reports saved to: {output}')
    print('Official DIOR test images were NOT accessed.')


if __name__ == '__main__':
    main()
