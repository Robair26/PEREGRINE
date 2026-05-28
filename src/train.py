import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from models.capsnet import CapsNet


def margin_loss(predictions, targets, num_classes=10):
    one_hot = torch.eye(num_classes).to(targets.device)[targets]
    loss = one_hot * torch.clamp(0.9 - predictions, min=0) ** 2 + \
           0.5 * (1 - one_hot) * torch.clamp(predictions - 0.1, min=0) ** 2
    return loss.sum(dim=1).mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, targets in tqdm(loader, desc="Training"):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = margin_loss(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = margin_loss(output, targets)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / len(loader), 100. * correct / total


def main():
    # Lightweight config — CPU laptop smoke test
    BATCH_SIZE = 32
    EPOCHS = 2
    LR = 0.001
    NUM_CLASSES = 10
    TRAIN_SAMPLES = 1000
    TEST_SAMPLES = 200
    DEVICE = torch.device("cpu")

    print(f"PEREGRINE smoke test on: {DEVICE}")
    print(f"Training samples: {TRAIN_SAMPLES}")
    print(f"Test samples: {TEST_SAMPLES}")

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

    model = CapsNet(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mlflow.set_experiment("PEREGRINE-CapsNet-MNIST")

    with mlflow.start_run(run_name="capsnet-smoke-test"):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "device": str(DEVICE)
        })

        best_accuracy = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, DEVICE
            )
            test_loss, test_acc = evaluate(
                model, test_loader, DEVICE
            )
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc
            }, step=epoch)

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), "../best_capsnet.pth")
                print(f"New best: {best_accuracy:.2f}%")

        mlflow.log_metric("best_accuracy", best_accuracy)
        print(f"\nSmoke test complete. Best accuracy: {best_accuracy:.2f}%")
        print(f"PEREGRINE confirmed working on CPU.")
        print(f"Ready to move to Jetson for full training.")


if __name__ == "__main__":
    main()
