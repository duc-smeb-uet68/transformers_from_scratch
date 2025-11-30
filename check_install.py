import torch
import torchtext
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torchtext version: {torchtext.__version__}")

# Kiểm tra GPU
if torch.cuda.is_available():
    print("--------------------------------------------------")
    print(f"✅ THÀNH CÔNG! Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("--------------------------------------------------")
else:
    print("--------------------------------------------------")
    print("❌ CẢNH BÁO: Chưa nhận diện được GPU. PyTorch đang chạy bằng CPU.")
    print("--------------------------------------------------")