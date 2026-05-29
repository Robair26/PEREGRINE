import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCapsules(nn.Module):
    """
    First capsule layer — extracts low level
    spatial features from input images.
    """
    def __init__(self, in_channels=256, out_channels=32,
                 capsule_dim=8, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.capsule_dim = capsule_dim
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * capsule_dim,
            kernel_size=kernel_size,
            stride=stride
        )

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm ** 2 / (1 + norm ** 2)) * (x / (norm + 1e-8))

    def forward(self, x):
        out = self.conv(x)
        batch = out.shape[0]
        out = out.view(batch, -1, self.capsule_dim)
        return self.squash(out)


class DigitCapsules(nn.Module):
    """
    Final capsule layer — preserves spatial
    relationships between detected features.
    This is Hinton's core idea — dynamic routing by agreement.
    """
    def __init__(self, num_capsules=20, num_routes=1152,
                 in_dim=8, out_dim=16, num_iterations=3):
        super(DigitCapsules, self).__init__()
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.out_dim = out_dim
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules,
                        out_dim, in_dim) * 0.01
        )

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm ** 2 / (1 + norm ** 2)) * (x / (norm + 1e-8))

    def forward(self, x):
        batch = x.shape[0]
        x = x.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, x).squeeze(-1)
        b = torch.zeros(batch, self.num_routes,
                        self.num_capsules, 1).to(x.device)

        for i in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s)
            if i < self.num_iterations - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)

        return v


class CapsNet(nn.Module):
    """
    PEREGRINE CapsNet — full capsule network
    for aerospace anomaly detection.
    Implements Hinton 2017 dynamic routing by agreement.
    Supports both MNIST (1ch, 28x28) and DIOR (3ch, 64x64).
    """
    def __init__(self, num_classes=20, in_channels=3, img_size=64):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 256,
            kernel_size=9, stride=1
        )
        self.primary = PrimaryCapsules(
            in_channels=256,
            out_channels=32,
            capsule_dim=8,
            kernel_size=9,
            stride=2
        )
        # Calculate num_routes dynamically based on image size
        conv_out = img_size - 8
        primary_out = (conv_out - 8) // 2
        num_routes = 32 * primary_out * primary_out

        print(f"CapsNet routing: img_size={img_size} "
              f"conv_out={conv_out} "
              f"primary_out={primary_out} "
              f"num_routes={num_routes}")

        self.digit = DigitCapsules(
            num_capsules=num_classes,
            num_routes=num_routes,
            in_dim=8,
            out_dim=16,
            num_iterations=3
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary(x)
        x = self.digit(x)
        lengths = torch.norm(x, dim=-1)
        return lengths


if __name__ == "__main__":
    print("=" * 50)
    print("Testing MNIST mode (1ch, 28x28)...")
    model_mnist = CapsNet(num_classes=10, in_channels=1, img_size=28)
    x = torch.randn(2, 1, 28, 28)
    out = model_mnist(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model_mnist.parameters())
    print(f"Params: {total:,}")

    print("=" * 50)
    print("Testing DIOR mode (3ch, 64x64)...")
    model_dior = CapsNet(num_classes=20, in_channels=3, img_size=64)
    x = torch.randn(2, 3, 64, 64)
    out = model_dior(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model_dior.parameters())
    print(f"Params: {total:,}")
    print("PEREGRINE CapsNet ready for aerospace data")
