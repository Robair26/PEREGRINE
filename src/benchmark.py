import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time
from models.capsnet import CapsNet
from models.resnet_baseline import ResNetBaseline


def margin_loss(predictions, targets, num_classes=10):
    one_hot = torch.eye(num_classes).to(targets.device)[targets]
    loss = one_hot * torch.clamp(0.9 - predictions, min=0) ** 2 + \
           0.5 * (1 - one_hot) * torch.clamp(predictions - 0.1, min=0) ** 2
    return loss.sum(dim=1).mean()


def train_epoch(model, loader, optimizer, device, model_type="capsnet"):
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


def evaluate(model, loader, device, model_type="capsnet"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

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

    return total_loss / len(loader), 100. * correct / total


def run_experiment(model, model_type, train_loader,
                   test_loader, device, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0
    history = []
    total_time = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{model_type.upper()} — Epoch {epoch}/{epochs}")

        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, model_type
        )
        test_loss, test_acc = evaluate(
            model, test_loader, device, model_type
        )
        epoch_time = time.time() - start
        total_time += epoch_time

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
        print(f"Epoch time: {epoch_time:.1f}s")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(),
                       f"../best_{model_type}.pth")
            print(f"New best: {best_accuracy:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time
        })

    return best_accuracy, total_time, history


def main():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 2
    LR = 0.001
    NUM_CLASSES = 10
    TRAIN_SAMPLES = 1000
    TEST_SAMPLES = 200
    DEVICE = torch.device("cpu")

    print("=" * 50)
    print("PEREGRINE — CapsNet vs ResNet Benchmark")
    print("=" * 50)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = Subset(
        datasets.MNIST(root='../data', train=True,
                       download=True, transform=transform),
        range(TRAIN_SAMPLES)
    )
    test_data = Subset(
        datasets.MNIST(root='../data', train=False,
                       download=True, transform=transform),
        range(TEST_SAMPLES)
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    mlflow.set_experiment("PEREGRINE-Benchmark")

    # --- CapsNet ---
    print("\nRunning CapsNet experiment...")
    capsnet = CapsNet(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
    capsnet_params = sum(p.numel() for p in capsnet.parameters())

    with mlflow.start_run(run_name="CapsNet-PEREGRINE"):
        mlflow.log_params({
            "model": "CapsNet",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "parameters": capsnet_params,
            "train_samples": TRAIN_SAMPLES
        })

        caps_acc, caps_time, caps_history = run_experiment(
            capsnet, "capsnet", train_loader,
            test_loader, DEVICE, EPOCHS, LR
        )

        for h in caps_history:
            mlflow.log_metrics({
                "capsnet_train_loss": h["train_loss"],
                "capsnet_train_acc": h["train_acc"],
                "capsnet_test_loss": h["test_loss"],
                "capsnet_test_acc": h["test_acc"]
            }, step=h["epoch"])

        mlflow.log_metrics({
            "best_accuracy": caps_acc,
            "total_training_time": caps_time,
            "parameters": capsnet_params
        })

    # --- ResNet ---
    print("\nRunning ResNet baseline experiment...")
    resnet = ResNetBaseline(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
    resnet_params = sum(p.numel() for p in resnet.parameters())

    with mlflow.start_run(run_name="ResNet-Baseline"):
        mlflow.log_params({
            "model": "ResNet",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "parameters": resnet_params,
            "train_samples": TRAIN_SAMPLES
        })

        res_acc, res_time, res_history = run_experiment(
            resnet, "resnet", train_loader,
            test_loader, DEVICE, EPOCHS, LR
        )

        for h in res_history:
            mlflow.log_metrics({
                "resnet_train_loss": h["train_loss"],
                "resnet_train_acc": h["train_acc"],
                "resnet_test_loss": h["test_loss"],
                "resnet_test_acc": h["test_acc"]
            }, step=h["epoch"])

        mlflow.log_metrics({
            "best_accuracy": res_acc,
            "total_training_time": res_time,
            "parameters": resnet_params
        })

    # --- Results ---
    print("\n" + "=" * 50)
    print("PEREGRINE BENCHMARK RESULTS")
    print("=" * 50)
    print(f"CapsNet  — Accuracy: {caps_acc:.2f}% | "
          f"Params: {capsnet_params:,} | Time: {caps_time:.1f}s")
    print(f"ResNet   — Accuracy: {res_acc:.2f}% | "
          f"Params: {resnet_params:,} | Time: {res_time:.1f}s")
    print(f"\nWinner: {'CapsNet' if caps_acc > res_acc else 'ResNet'}")
    print(f"Accuracy delta: {abs(caps_acc - res_acc):.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
