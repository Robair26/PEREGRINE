import torch
import torch.onnx
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.capsnet import CapsNet

def export_capsnet_onnx():
    print("=" * 50)
    print("PEREGRINE — ONNX Export")
    print("=" * 50)

    device = torch.device("cpu")
    model = CapsNet(num_classes=20, in_channels=3, img_size=32)

    checkpoint = os.path.expanduser(
        "~/PEREGRINE/best_capsnet_dior_jetson.pth"
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    dummy_input = torch.randn(1, 3, 32, 32)

    output_path = os.path.expanduser(
        "~/PEREGRINE/peregrine_capsnet.onnx"
    )

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"ONNX model saved: {output_path}")
    print(f"Model size: {size_mb:.1f} MB")

    print("\nVerifying ONNX model...")
    import onnxruntime as ort
    import numpy as np
    import time

    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name

    test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)

    times = []
    for _ in range(50):
        start = time.time()
        result = session.run(None, {input_name: test_input})
        times.append((time.time() - start) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    print(f"Output shape: {result[0].shape}")
    print(f"Average inference: {avg_ms:.2f}ms")
    print(f"Best inference:    {min_ms:.2f}ms")
    print(f"PEREGRINE ONNX ready for edge deployment")
    print("=" * 50)


if __name__ == "__main__":
    export_capsnet_onnx()
