"""PEREGRINE v3: CPU ResNet experiment with resumable, trusted-local checkpoints.

Run from repo root with PYTHONPATH=src. Uses only DIOR trainval images.
Full training checkpoints use PyTorch weights-only compatible contents.
"""
import argparse
import hashlib
import json
import os
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
    by_class = defaultdict(list)
    allowed = set(allowed_ids)
    for index, sample in enumerate(dataset.samples):
        image_id = Path(sample['img_path']).stem
        if image_id in allowed:
            key = (image_id, tuple(sample['bbox']), sample['label'])
            by_class[sample['label']].append((key, index))
    chosen = []
    for cls in range(len(DIOR_CLASSES)):
        candidates = sorted(by_class[cls])
        if len(candidates) < per_class:
            raise ValueError(f'Insufficient samples for {DIOR_CLASSES[cls]}')
        chosen += [index for _, index in random.Random(seed + cls).sample(candidates, per_class)]
    return chosen


def evaluate(model, loader):
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            actual.extend(targets.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
    return {
        'accuracy': accuracy_score(actual, predicted),
        'macro_f1': f1_score(actual, predicted, labels=list(range(20)), average='macro', zero_division=0),
        'class_support': dict(Counter(DIOR_CLASSES[x] for x in actual)),
    }


def atomic_torch_save(value, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json_save(value, path):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def pack_numpy_rng(state):
    """Store NumPy legacy RNG state as safe Python primitives, not NumPy objects."""
    name, keys, position, has_gauss, cached_gaussian = state
    return (name, keys.astype(np.uint32).tolist(), int(position),
            int(has_gauss), float(cached_gaussian))


def unpack_numpy_rng(state):
    name, keys, position, has_gauss, cached_gaussian = state
    if name != 'MT19937' or len(keys) != 624:
        raise ValueError('Invalid NumPy MT19937 checkpoint state')
    if not all(isinstance(k, int) and 0 <= k <= 0xFFFFFFFF for k in keys):
        raise ValueError('Invalid NumPy RNG key value')
    return (name, np.asarray(keys, dtype=np.uint32), int(position),
            int(has_gauss), float(cached_gaussian))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='dior_split_manifest_v1.json')
    p.add_argument('--data-root', default='data/DIOR')
    p.add_argument('--train-per-class', type=int, default=30)
    p.add_argument('--val-per-class', type=int, default=10)
    p.add_argument('--epochs', type=int, default=1, help='Total target epochs, not additional epochs')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='experiments/resnet_v3_resume_smoke')
    p.add_argument('--resume', action='store_true', help='Resume v3 schema-2 latest_checkpoint.pth in output-dir')
    args = p.parse_args()
    if min(args.train_per_class, args.val_per_class, args.epochs, args.batch_size, args.threads) < 1:
        p.error('Counts, epochs, batch size, and threads must be positive')
    output = Path(args.output_dir)
    checkpoint_path = output / 'latest_checkpoint.pth'
    if args.resume:
        if not checkpoint_path.is_file():
            p.error(f'No resume checkpoint found: {checkpoint_path}')
    elif output.exists():
        p.error(f'Output directory already exists: {output}; choose a new directory')

    manifest_bytes = Path(args.manifest).read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    train_ids, val_ids, test_ids = (set(manifest[k]) for k in ('train_ids','validation_ids','test_ids'))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('Manifest overlaps across train/validation/test images')
    if any(len(manifest[k]) != len(set(manifest[k])) for k in ('train_ids','validation_ids','test_ids')):
        raise ValueError('Manifest contains duplicate IDs')

    config = {'manifest_sha256': manifest_hash, 'data_root': str(Path(args.data_root).resolve()),
              'train_per_class': args.train_per_class, 'val_per_class': args.val_per_class,
              'batch_size': args.batch_size, 'threads': args.threads, 'seed': args.seed,
              'architecture': 'ResNetBaseline:classes=20,channels=3', 'image_size': 64}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    # Dedicated generator: its state is saved so next epoch's shuffle can be restored.
    shuffle_generator = torch.Generator().manual_seed(args.seed + 7)
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    train_t = transforms.Compose([transforms.Resize((64,64)), transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(15), transforms.ToTensor(), norm])
    val_t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), norm])
    train_ds = DIORDataset(args.data_root, split='train', transform=train_t, img_size=64)
    val_ds = DIORDataset(args.data_root, split='train', transform=val_t, img_size=64)
    train_idx = choose_indices(train_ds, train_ids, args.train_per_class, args.seed)
    val_idx = choose_indices(val_ds, val_ids, args.val_per_class, args.seed + 10000)
    train_image_ids = {Path(train_ds.samples[i]['img_path']).stem for i in train_idx}
    val_image_ids = {Path(val_ds.samples[i]['img_path']).stem for i in val_idx}
    if train_image_ids & val_image_ids:
        raise ValueError('Selected samples share source images')
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, generator=shuffle_generator, num_workers=0)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    model = ResNetBaseline(num_classes=20, in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history, best_f1, completed_epoch = [], -1.0, 0

    if args.resume:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved.get('schema_version') != 2 or saved.get('config') != config:
            raise ValueError('Checkpoint schema/config mismatch; refusing resume')
        completed_epoch = saved['completed_epoch']
        if args.epochs <= completed_epoch:
            p.error(f'--epochs must be greater than already completed {completed_epoch}')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        history = saved['history']
        best_f1 = saved['best_macro_f1']
        random.setstate(saved['python_rng'])
        np.random.set_state(unpack_numpy_rng(saved['numpy_rng']))
        torch.set_rng_state(saved['torch_rng'])
        shuffle_generator.set_state(saved['shuffle_rng'])
        print(f'Resuming after epoch {completed_epoch}; target total epochs {args.epochs}')
    else:
        output.mkdir(parents=True, exist_ok=False)
        print(f'New experiment: {len(train_idx)} train, {len(val_idx)} validation crops')
    print('Official DIOR test images and original checkpoints remain untouched.', flush=True)

    run_started = time.monotonic()
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        metrics = evaluate(model, val_loader)
        record = {'epoch': epoch, 'train_loss': total_loss / batches, **metrics}
        history.append(record)
        print(f"Epoch {epoch}: loss={record['train_loss']:.4f}, val_acc={metrics['accuracy']:.4f}, val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            atomic_torch_save(model.state_dict(), output / 'best_resnet_v3_state_dict.pth')
        state = {'schema_version': 2, 'config': config, 'completed_epoch': epoch,
                 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                 'history': history, 'best_macro_f1': best_f1,
                 'python_rng': random.getstate(), 'numpy_rng': pack_numpy_rng(np.random.get_state()),
                 'torch_rng': torch.get_rng_state(), 'shuffle_rng': shuffle_generator.get_state()}
        atomic_torch_save(state, checkpoint_path)
        atomic_json_save({'config': config, 'target_epochs_this_invocation': args.epochs,
                          'completed_epoch': epoch, 'train_objects': len(train_idx),
                          'validation_objects': len(val_idx), 'source_image_overlap': 0,
                          'history': history, 'last_invocation_seconds': round(time.monotonic()-run_started, 2),
                          'warning': 'Validation pilot; NOT independent test accuracy'}, output / 'results.json')
    print('Results saved under:', output)

if __name__ == '__main__':
    main()
