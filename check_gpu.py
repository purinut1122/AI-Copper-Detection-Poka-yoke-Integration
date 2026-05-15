import torch
print(f"Pytorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"✅ GPU Ready: {torch.cuda.get_device_name(0)}")
    print("เครื่องนี้แรงพร้อมลุย AI แล้วครับ! 🚀")
else:
    print("❌ ยังไม่เจอ GPU (อาจจะลงผิด)")