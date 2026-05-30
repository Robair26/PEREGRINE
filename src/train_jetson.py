import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.capsnet import CapsNet
from models.resnet_baseline import ResNetBaseline
from data.dior_dataloader import DIORDataset, DIOR_CLASSES

mlflow.set_tracking_uri(
    "sqlite:///" + os.path.expanduser("~/PEREGRINE/mlflow.db")
)


def margin_loss(predictions, targets, num_classes=20):
    one_hot = torch.eye(num_classes).to(targets.device)[targets]
    loss = one_hot * torch.clamp(0.9 - predictions, min=0) ** 2 + \
           0.5 * (1 - one_hot) * torch.clamp(predictions - 0.1, min=0) ** 2
    return loss.sum(dim=1).mean()


def train_epoch(model, loader, optimizer, device, model_type):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, targets in tqdm(loader, desc=f"Training {model_type}"):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)

        if model_type == "capsnet":
            loss = margin_loss(output, targets)
        else:
            loss = nn.CrossEntropyLoss()(output, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, model_type):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 20
    class_total = [0] * 20

    with torch.no_grad():
        for data, targets in tqdm(loader, desc=f"Evaluating {model_type}"):
            data, targets = data.to(device), targets.to(device)
            output = model(data)

            if model_type == "capsnet":
                loss = margin_loss(output, targets)
            else:
                loss = nn.CrossEntropyLoss()(output, targets)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += pred[i].eq(targets[i]).item()
                class_total[label] += 1

    return total_loss / len(loader), 100. * correct / total, \
           class_correct, class_total


def main():
    BATCH_SIZE = 4
    EPOCHS = 15
    LR = 0.001
    NUM_CLASSES = 20
    IMG_SIZE = 32
    DEVICE = torch.device("cuda")

    print("=" * 60)
    print("PEREGRINE — Full DIOR Training on NVIDIA Jetson Orin")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | IMG: {IMG_SIZE}x{IMG_SIZE}")

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("\nLoading DIOR dataset...")
    train_dataset = DIORDataset(
        'data/DIOR', split='train',
        transform=transform_train
    )
    test_dataset = DIORDataset(
        'data/DIOR', split='test',
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    mlflow.set_experiment("PEREGRINE-DIOR-Jetson-Full")

    # Train CapsNet
    print("\n" + "=" * 60)
    print("Training CapsNet on full DIOR dataset...")
    print("=" * 60)

    capsnet = CapsNet(
        num_classes=NUM_CLASSES,
        in_channels=3,
        img_size=IMG_SIZE
    ).to(DEVICE)

    total_params = sum(p.numel() for p in capsnet.parameters())
    print(f"CapsNet parameters: {total_params:,}")

    with mlflow.start_run(run_name="CapsNet-DIOR-Jetson-Full"):
        mlflow.log_params({
            "model": "CapsNet",
            "dataset": "DIOR",
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "train_samples": len(train_dataset),
            "device": "Jetson-Orin-GPU",
            "parameters": total_params
        })

        optimizer = optim.Adam(capsnet.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
        best_acc = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            start = time.time()

            train_loss, train_acc = train_epoch(
                capsnet, train_loader, optimizer, DEVICE, "capsnet"
            )
            test_loss, test_acc, cc, ct = evaluate(
                capsnet, test_loader, DEVICE, "capsnet"
            )

            epoch_time = time.time() - start
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
            print(f"Epoch time: {epoch_time:.1f}s")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "epoch_time": epoch_time
            }, step=epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    capsnet.state_dict(),
                    "best_capsnet_dior_jetson.pth"
                )
                print(f"New best CapsNet: {best_acc:.2f}%")

        print("\nPer-class accuracy (CapsNet):")
        for i, cls in enumerate(DIOR_CLASSES):
            if ct[i] > 0:
                acc = 100. * cc[i] / ct[i]
                print(f"  {cls:<35} {acc:.1f}%")

        mlflow.log_metric("best_accuracy", best_acc)

    # Train ResNet
    print("\n" + "=" * 60)
    print("Training ResNet baseline on full DIOR dataset...")
    print("=" * 60)

    resnet = ResNetBaseline(
        num_classes=NUM_CLASSES,
        in_channels=3
    ).to(DEVICE)

    res_params = sum(p.numel() for p in resnet.parameters())
    print(f"ResNet parameters: {res_params:,}")

    with mlflow.start_run(run_name="ResNet-DIOR-Jetson-Full"):
        mlflow.log_params({
            "model": "ResNet",
            "dataset": "DIOR",
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "train_samples": len(train_dataset),
            "device": "Jetson-Orin-GPU",
            "parameters": res_params
        })

        optimizer = optim.Adam(resnet.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
        best_acc_res = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            start = time.time()

            train_loss, train_acc = train_epoch(
                resnet, train_loader, optimizer, DEVICE, "resnet"
            )
            test_loss, test_acc, cc, ct = evaluate(
                resnet, test_loader, DEVICE, "resnet"
            )

            epoch_time = time.time() - start
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
            print(f"Epoch time: {epoch_time:.1f}s")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "epoch_time": epoch_time
            }, step=epoch)

            if test_acc > best_acc_res:
                best_acc_res = test_acc
                torch.save(
                    resnet.state_dict(),
                    "best_resnet_dior_jetson.pth"
                )
                print(f"New best ResNet: {best_acc_res:.2f}%")

        print("\nPer-class accuracy (ResNet):")
        for i, cls in enumerate(DIOR_CLASSES):
            if ct[i] > 0:
                acc = 100. * cc[i] / ct[i]
                print(f"  {cls:<35} {acc:.1f}%")

        mlflow.log_metric("best_accuracy", best_acc_res)

    print("\n" + "=" * 60)
    print("PEREGRINE FULL DIOR RESULTS — JETSON GPU")
    print("=" * 60)
    print(f"CapsNet  best accuracy: {best_acc:.2f}%")
    print(f"ResNet   best accuracy: {best_acc_res:.2f}%")
    print(f"Winner: {'CapsNet' if best_acc > best_acc_res else 'ResNet'}")
    print(f"Delta: {abs(best_acc - best_acc_res):.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
