import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

if torch.cuda.is_available():
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("❌ GPU Check: FAILED (PyTorch ไม่เห็น GPU)")
    
print("-" * 30)
print("คำแนะนำ:")
print("ถ้า GPU Name ขึ้นว่า RTX 3050/4050/4060 -> Driver ต้องใหม่มาก")