import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import mlflow
import os
from tqdm import tqdm
from models.capsnet import CapsNet
from models.resnet_baseline import ResNetBaseline

mlflow.set_tracking_uri(
    "sqlite:///" + os.path.expanduser("~/PEREGRINE/mlflow.db")
)


def margin_loss(predictions, targets, num_classes=10):
    one_hot = torch.eye(num_classes).to(targets.device)[targets]
    loss = one_hot * torch.clamp(0.9 - predictions, min=0) ** 2 + \
           0.5 * (1 - one_hot) * torch.clamp(predictions - 0.1, min=0) ** 2
    return loss.sum(dim=1).mean()


def evaluate_rotated(model, rotation_degrees,
                     device, model_type, num_samples=200):
    """
    Test model accuracy on rotated images.
    This is where CapsNet should beat ResNet —
    it preserves spatial relationships through rotation.
    """
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(rotation_degrees,
                                           rotation_degrees)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_data = Subset(
        datasets.MNIST(root='data', train=False,
                       download=True, transform=transform),
        range(num_samples)
    )
    loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    return 100. * correct / total


def train_model(model, model_type, device,
                num_samples=1000, epochs=2):
    """Train model on clean data first."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = Subset(
        datasets.MNIST(root='data', train=True,
                       download=True, transform=transform),
        range(num_samples)
    )
    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for data, targets in tqdm(loader,
                                  desc=f"Training {model_type} epoch {epoch+1}"):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            if model_type == "capsnet":
                loss = margin_loss(output, targets)
            else:
                loss = nn.CrossEntropyLoss()(output, targets)
            loss.backward()
            optimizer.step()

    return model


def main():
    DEVICE = torch.device("cpu")
    ROTATIONS = [0, 15, 30, 45, 60, 90]

    print("=" * 55)
    print("PEREGRINE — Rotation Invariance Test")
    print("CapsNet vs ResNet on rotated aerospace imagery")
    print("=" * 55)

    # Train both models on clean data
    print("\nTraining CapsNet...")
    capsnet = CapsNet(num_classes=10, in_channels=1).to(DEVICE)
    capsnet = train_model(capsnet, "capsnet", DEVICE)

    print("\nTraining ResNet...")
    resnet = ResNetBaseline(num_classes=10, in_channels=1).to(DEVICE)
    resnet = train_model(resnet, "resnet", DEVICE)

    # Test both at each rotation angle
    print("\nTesting rotation invariance...")
    results = []

    mlflow.set_experiment("PEREGRINE-Rotation-Test")

    with mlflow.start_run(run_name="rotation-invariance-test"):
        mlflow.log_params({
            "rotations_tested": str(ROTATIONS),
            "train_samples": 1000,
            "test_samples": 200,
            "epochs": 2
        })

        for deg in ROTATIONS:
            caps_acc = evaluate_rotated(
                capsnet, deg, DEVICE, "capsnet"
            )
            res_acc = evaluate_rotated(
                resnet, deg, DEVICE, "resnet"
            )
            delta = caps_acc - res_acc
            results.append((deg, caps_acc, res_acc, delta))

            mlflow.log_metrics({
                f"capsnet_acc_rot{deg}": caps_acc,
                f"resnet_acc_rot{deg}": res_acc,
                f"delta_rot{deg}": delta
            })

            print(f"Rotation {deg:3d}° → "
                  f"CapsNet: {caps_acc:.1f}% | "
                  f"ResNet: {res_acc:.1f}% | "
                  f"Delta: {delta:+.1f}%")

    # Final summary
    print("\n" + "=" * 55)
    print("ROTATION INVARIANCE RESULTS")
    print("=" * 55)
    print(f"{'Rotation':<12} {'CapsNet':<12} {'ResNet':<12} {'Delta'}")
    print("-" * 55)
    for deg, caps, res, delta in results:
        winner = "← CAPS" if delta > 0 else "← RES"
        print(f"{deg:>6}°      {caps:>6.1f}%     "
              f"{res:>6.1f}%    {delta:>+6.1f}% {winner}")

    caps_wins = sum(1 for _, c, r, _ in results if c > r)
    print(f"\nCapsNet wins: {caps_wins}/{len(ROTATIONS)} rotation angles")
    print("=" * 55)


if __name__ == "__main__":
    main()
