"""PEREGRINE v2: deterministic, image-disjoint CPU-friendly ResNet pilot.

Run from repository root: PYTHONPATH=src python train_dior_v2.py --smoke
This script never touches original checkpoints or the official DIOR test images.
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data.dior_dataloader import DIORDataset, DIOR_CLASSES
from models.resnet_baseline import ResNetBaseline


def choose_indices(dataset, allowed_ids, per_class, seed):
    allowed = set(allowed_ids)
    by_class = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        if Path(sample['img_path']).stem in allowed:
            key = (Path(sample['img_path']).stem, tuple(sample['bbox']), sample['label'])
            by_class[sample['label']].append((key, index))
    selected = []
    for cls in range(len(DIOR_CLASSES)):
        candidates = sorted(by_class[cls])
        if len(candidates) < per_class:
            raise ValueError(f'Only {len(candidates)} objects for {DIOR_CLASSES[cls]}, need {per_class}')
        rng = random.Random(seed + cls)
        picked = rng.sample(candidates, per_class)
        selected.extend(index for _, index in picked)
    return selected


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            actual.extend(targets.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return {'accuracy': accuracy_score(actual, predicted),
            'macro_f1': f1_score(actual, predicted, labels=list(range(20)), average='macro', zero_division=0),
            'class_support': dict(Counter(DIOR_CLASSES[x] for x in actual))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='dior_split_manifest_v1.json')
    parser.add_argument('--data-root', default='data/DIOR')
    parser.add_argument('--train-per-class', type=int, default=30)
    parser.add_argument('--val-per-class', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='experiments/resnet_v2_cpu_pilot')
    parser.add_argument('--smoke', action='store_true', help='Tiny end-to-end run; not a performance benchmark')
    args = parser.parse_args()
    if args.smoke:
        args.train_per_class, args.val_per_class, args.epochs = 2, 2, 1
    if min(args.train_per_class, args.val_per_class, args.epochs, args.batch_size, args.threads) < 1:
        parser.error('Counts, epochs, batch size, and threads must be positive')

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'{output} already exists. Refusing to overwrite experiment artifacts.')
    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    train_ids = set(manifest['train_ids'])
    val_ids = set(manifest['validation_ids'])
    test_ids = set(manifest['test_ids'])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('Manifest contains overlapping source image IDs')
    if len(train_ids) != len(manifest['train_ids']) or len(val_ids) != len(manifest['validation_ids']):
        raise ValueError('Manifest contains duplicate IDs')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])

    # Both objects index trainval only; the official test directory is never loaded.
    train_dataset = DIORDataset(args.data_root, split='train', transform=train_transform, img_size=64)
    val_dataset = DIORDataset(args.data_root, split='train', transform=val_transform, img_size=64)
    train_indices = choose_indices(train_dataset, train_ids, args.train_per_class, args.seed)
    val_indices = choose_indices(val_dataset, val_ids, args.val_per_class, args.seed + 10000)
    actual_train_ids = {Path(train_dataset.samples[i]['img_path']).stem for i in train_indices}
    actual_val_ids = {Path(val_dataset.samples[i]['img_path']).stem for i in val_indices}
    assert not actual_train_ids & actual_val_ids
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Train: {len(train_indices)} object crops, validation: {len(val_indices)}, 20 classes each')
    print('This pilot does not touch the official test split or original checkpoints.')

    model = ResNetBaseline(num_classes=20, in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0., 0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        metrics = evaluate(model, val_loader, device)
        epoch_result = {'epoch': epoch, 'train_loss': total_loss / batches, **metrics}
        history.append(epoch_result)
        print(f"Epoch {epoch}: loss={epoch_result['train_loss']:.4f}, val_acc={metrics['accuracy']:.4f}, val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            torch.save(model.state_dict(), output / 'best_resnet_v2_state_dict.pth')
    details = {'config': vars(args), 'manifest': args.manifest, 'seed': args.seed,
               'source_image_overlap': 0, 'train_objects': len(train_indices),
               'validation_objects': len(val_indices), 'duration_seconds': round(time.time()-started, 1),
               'history': history, 'experimental_warning': 'CPU pilot; NOT independent test accuracy. Do not compare directly with historical reported test scores.'}
    (output / 'results.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print('Results saved under:', output)

if __name__ == '__main__':
    main()
