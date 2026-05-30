import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels=64, out_channels=8,
                 capsule_dim=8, kernel_size=5, stride=2):
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
    def __init__(self, num_capsules=20, num_routes=512,
                 in_dim=8, out_dim=16, num_iterations=1):
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
    PEREGRINE CapsNet — Hinton 2017 dynamic routing.
    Lightweight version optimized for Jetson Orin 8GB.
    """
    def __init__(self, num_classes=20, in_channels=3, img_size=32):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=5, stride=1
        )
        self.primary = PrimaryCapsules(
            in_channels=64,
            out_channels=8,
            capsule_dim=8,
            kernel_size=5,
            stride=2
        )
        conv_out = img_size - 4
        primary_out = (conv_out - 4) // 2
        num_routes = 8 * primary_out * primary_out

        print(f"CapsNet routing: img_size={img_size} "
              f"primary_out={primary_out} "
              f"num_routes={num_routes}")

        self.digit = DigitCapsules(
            num_capsules=num_classes,
            num_routes=num_routes,
            in_dim=8,
            out_dim=16,
            num_iterations=1
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary(x)
        x = self.digit(x)
        lengths = torch.norm(x, dim=-1)
        return lengths


if __name__ == "__main__":
    print("Testing MNIST mode (1ch, 28x28)...")
    model = CapsNet(num_classes=10, in_channels=1, img_size=28)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"Input:  {x.shape} Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Testing DIOR mode (3ch, 32x32)...")
    model = CapsNet(num_classes=20, in_channels=3, img_size=32)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Input:  {x.shape} Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("PEREGRINE CapsNet ready for Jetson training")
