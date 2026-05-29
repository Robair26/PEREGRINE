import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
from tqdm import tqdm
from models.capsnet import CapsNet
from models.resnet_baseline import ResNetBaseline
from data.dior_dataloader import get_dior_loaders, DIOR_CLASSES

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

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, class_correct, class_total


def main():
    # Lightweight config for laptop CPU smoke test
    BATCH_SIZE = 16
    EPOCHS = 2
    LR = 0.001
    NUM_CLASSES = 20
    IMG_SIZE = 64
    TRAIN_LIMIT = 2000
    TEST_LIMIT = 500
    DEVICE = torch.device("cpu")

    print("=" * 55)
    print("PEREGRINE — DIOR Aerospace Training")
    print("CapsNet on real satellite and aerial imagery")
    print("=" * 55)
    print(f"Device: {DEVICE}")
    print(f"Training samples: {TRAIN_LIMIT}")
    print(f"Test samples: {TEST_LIMIT}")
    print(f"Classes: {NUM_CLASSES}")

    # Load DIOR data
    from torch.utils.data import Subset
    from data.dior_dataloader import DIORDataset
    from torchvision import transforms

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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

    from torch.utils.data import DataLoader

    train_data = Subset(
        DIORDataset('data/DIOR', split='train',
                    transform=transform_train),
        range(TRAIN_LIMIT)
    )
    test_data = Subset(
        DIORDataset('data/DIOR', split='test',
                    transform=transform_test),
        range(TEST_LIMIT)
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    mlflow.set_experiment("PEREGRINE-DIOR-Aerospace")

    # Train CapsNet
    print("\nTraining CapsNet on DIOR aerospace imagery...")
    capsnet = CapsNet(
        num_classes=NUM_CLASSES,
        in_channels=3,
        img_size=IMG_SIZE
    ).to(DEVICE)

    with mlflow.start_run(run_name="CapsNet-DIOR"):
        mlflow.log_params({
            "model": "CapsNet",
            "dataset": "DIOR",
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "train_samples": TRAIN_LIMIT
        })

        optimizer = optim.Adam(capsnet.parameters(), lr=LR)
        best_acc = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss, train_acc = train_epoch(
                capsnet, train_loader, optimizer, DEVICE, "capsnet"
            )
            test_loss, test_acc, cc, ct = evaluate(
                capsnet, test_loader, DEVICE, "capsnet"
            )

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc
            }, step=epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(capsnet.state_dict(), "best_capsnet_dior.pth")
                print(f"New best: {best_acc:.2f}%")

        # Per class accuracy
        print("\nPer-class accuracy:")
        for i, cls in enumerate(DIOR_CLASSES):
            if ct[i] > 0:
                acc = 100. * cc[i] / ct[i]
                print(f"  {cls:<30} {acc:.1f}%")

        mlflow.log_metric("best_accuracy", best_acc)

    # Train ResNet baseline
    print("\nTraining ResNet baseline on DIOR...")
    resnet = ResNetBaseline(
        num_classes=NUM_CLASSES,
        in_channels=3
    ).to(DEVICE)

    with mlflow.start_run(run_name="ResNet-DIOR"):
        mlflow.log_params({
            "model": "ResNet",
            "dataset": "DIOR",
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "train_samples": TRAIN_LIMIT
        })

        optimizer = optim.Adam(resnet.parameters(), lr=LR)
        best_acc_res = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss, train_acc = train_epoch(
                resnet, train_loader, optimizer, DEVICE, "resnet"
            )
            test_loss, test_acc, cc, ct = evaluate(
                resnet, test_loader, DEVICE, "resnet"
            )

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc
            }, step=epoch)

            if test_acc > best_acc_res:
                best_acc_res = test_acc
                torch.save(resnet.state_dict(), "best_resnet_dior.pth")
                print(f"New best: {best_acc_res:.2f}%")

        mlflow.log_metric("best_accuracy", best_acc_res)

    print("\n" + "=" * 55)
    print("PEREGRINE DIOR RESULTS")
    print("=" * 55)
    print(f"CapsNet  best accuracy: {best_acc:.2f}%")
    print(f"ResNet   best accuracy: {best_acc_res:.2f}%")
    print(f"Winner: {'CapsNet' if best_acc > best_acc_res else 'ResNet'}")
    print(f"Delta: {abs(best_acc - best_acc_res):.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
