import cv2
import numpy as np
import serial
import time
import onnxruntime as ort
import traceback

# =========================================================
# ⚙️ ส่วนตั้งค่า (CONFIGURATION)
# =========================================================
MODEL_PATH = 'model.onnx'       
ESP32_PORT = 'COM3'              # <--- อิงตาม Log ล่าสุดของคุณคือ COM5
BAUD_RATE = 115200

# ขนาดภาพที่โมเดลต้องการ (Fixed ตาม Error ที่เคยเจอ)
MODEL_WIDTH = 224
MODEL_HEIGHT = 224

# ชื่อ Class (จาก JSON ของอาจารย์)
CLASS_NAMES = ['COPPER', 'NO_COPPER', 'NO_PART'] 
TARGET_CLASS_ID = 0              # 0 คือ COPPER (OK)
CONF_THRESHOLD = 0.6             # ความมั่นใจ 60%

# =========================================================
# 1. เชื่อมต่อ ESP32
# =========================================================
ser = None
try:
    print(f"🔄 Connecting to ESP32 ({ESP32_PORT})...")
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("✅ Serial Connected!")
except Exception as e:
    print(f"⚠️ Serial Offline (ทำงานโหมด Offline): {e}")

# =========================================================
# 2. โหลดโมเดล (ONNX Runtime)
# =========================================================
try:
    print(f"🚀 Loading Model: {MODEL_PATH} ...")
    ort_session = ort.InferenceSession(MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    print("✅ Model Loaded Success!")
except Exception as e:
    print(f"❌ Model Error: {e}")
    print("👉 ตรวจสอบว่าไฟล์ model.onnx อยู่โฟลเดอร์เดียวกับโค้ดไหม?")
    input("Press Enter to exit...")
    exit()

# =========================================================
# 3. เปิดกล้อง (ระบบค้นหาอัตโนมัติ + CAP_DSHOW)
# =========================================================
print("📷 Searching for camera...")
cap = None
found_camera = False

# วนหาตั้งแต่ Index 0 ถึง 3
for i in range(4):
    print(f"   > Testing Camera Index {i}...")
    try:
        # ใช้ CAP_DSHOW เพื่อแก้บั๊ก Windows Tablet
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                print(f"   ✅ Found working camera at Index {i}!")
                cap = temp_cap
                found_camera = True
                break
            else:
                temp_cap.release()
    except:
        pass

if not found_camera:
    print("❌ Critical Error: ไม่เจอกล้องเลย!")
    print("👉 ลองเช็ก: 1.สาย USB หลวม? 2.Privacy Settings บล็อก Python ไว้ไหม?")
    input("Press Enter to exit...")
    exit()

# ตั้งค่าความละเอียดแสดงผล (Display Resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_NAME = "Professor Model Test (Final)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# =========================================================
# 🔁 MAIN LOOP
# =========================================================
print("--- SYSTEM READY ---")
running = True
try:
    while running:
        ret, frame = cap.read()
        if not ret:
            # กันกล้องหลุดกลางอากาศ
            print("⚠️ Camera frame drop...")
            time.sleep(0.5)
            continue

        height, width, _ = frame.shape

        # =================================================
        # 🧠 เตรียมภาพ (Preprocessing)
        # =================================================
        # 1. Resize เป็น 224x224 (ตามที่โมเดลต้องการ)
        img = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        
        # 2. Convert Color (BGR -> RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize (0-1) & Standardize (ImageNet Mean/Std)
        img = img.astype(np.float32) / 255.0
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - MEAN) / STD
        
        # 4. Transpose (H,W,C -> 1,C,H,W)
        img = img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img, axis=0)

        # =================================================
        # 🔮 ทำนายผล (Inference)
        # =================================================
        outputs = ort_session.run([output_name], {input_name: input_tensor})
        
        preds = outputs[0][0]
        # Softmax logic
        exp_preds = np.exp(preds - np.max(preds))
        softmax_preds = exp_preds / np.sum(exp_preds)
        
        class_id = np.argmax(softmax_preds)
        confidence = softmax_preds[class_id]

        # =================================================
        # 📝 แสดงผล & สั่งงาน
        # =================================================
        is_copper = False
        
        if confidence > CONF_THRESHOLD:
            # ตรวจสอบว่าเป็น Copper (ID 0) หรือไม่
            if class_id == TARGET_CLASS_ID:
                is_copper = True
                color = (0, 255, 0) # เขียว
                status_text = f"OK: {CLASS_NAMES[class_id]}"
            else:
                color = (0, 0, 255) # แดง
                # กันเหนียวเผื่อ Index เกิน
                c_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ID {class_id}"
                status_text = f"NG: {c_name}"

            # วาดกรอบ
            cv2.rectangle(frame, (0, 0), (width, height), color, 40)
            cv2.putText(frame, f"{status_text} ({confidence*100:.0f}%)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # สั่ง ESP32
            if ser:
                ser.write(b'1' if is_copper else b'0')
                
        else:
            # ไม่มั่นใจ -> สีเหลือง/ฟ้า
            cv2.putText(frame, f"Checking... ({confidence*100:.0f}%)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if ser: ser.write(b'0')

        # ปุ่ม EXIT
        cv2.putText(frame, "Press 'q' to Exit", (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

except Exception as e:
    print(f"\n❌ Runtime Error: {e}")
    traceback.print_exc()
    input("Press Enter to close...")

finally:
    if cap: cap.release()
    if ser: ser.close()
    cv2.destroyAllWindows()