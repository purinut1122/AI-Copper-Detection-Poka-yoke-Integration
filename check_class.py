from ultralytics import YOLO

# ใส่ชื่อไฟล์โมเดลของคุณ
model = YOLO('my_model_n.onnx')

# สั่งปริ้นรายชื่อ Class ทั้งหมดออกมาดู
print("====================================")
print("รายชื่อ Class ในโมเดลของคุณ:")
print(model.names)
print("====================================")