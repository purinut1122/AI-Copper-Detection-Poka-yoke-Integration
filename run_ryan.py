import cv2
import numpy as np
import serial
import time
import os
import traceback
import json
from datetime import datetime

# =========================================================
# ⚙️ ส่วนตั้งค่า (CONFIGURATION)
# =========================================================
ESP32_PORT = 'COM3'              
BAUD_RATE = 115200               
MODEL_PATH = 'model.onnx'  
JSON_PATH = 'preprocessor_config.json'       # 📌 ไฟล์นี้มีข้อมูลครบถ้วนสำหรับ Preprocess

# 👉 ตั้งค่าพื้นที่ Crop (x, y, กว้าง, สูง)
CROP_X = 195
CROP_Y = 0
CROP_W = 224
CROP_H = 224

# 👉 โฟลเดอร์เก็บรูป
BASE_PATH = "D:/IT_Deb/IMG_COPPER_LOG/IMG_LOG_" 
PATH_OK = BASE_PATH + "OK_Images_Classify/" 
PATH_NG = BASE_PATH + "NG_Images_Classify/"

if not os.path.exists(PATH_OK): os.makedirs(PATH_OK)
if not os.path.exists(PATH_NG): os.makedirs(PATH_NG)

# =========================================================
# 📁 โหลดค่า Configuration ทั้งหมดจาก JSON อัตโนมัติ
# =========================================================
CLASS_NAMES = {0: 'Copper', 1: 'No_Copper', 2: 'No_part'}  
INPUT_SIZE = (224, 224)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        prep_data = json.load(f)
        
        # 1. ดึงชื่อ Class
        if 'id2label' in prep_data:
            CLASS_NAMES = {int(k): v for k, v in prep_data['id2label'].items()}
            
        # 2. ดึงขนาดภาพ Input
        if 'input_size' in prep_data:
            # สมมติ JSON ให้มาเป็น [224, 224]
            INPUT_SIZE = (prep_data['input_size'][0], prep_data['input_size'][1])
            
        # 3. ดึงค่า Mean และ Std
        if 'image_mean' in prep_data:
            IMG_MEAN = np.array(prep_data['image_mean'], dtype=np.float32)
        if 'image_std' in prep_data:
            IMG_STD = np.array(prep_data['image_std'], dtype=np.float32)

    print(f"✅ โหลดข้อมูลจาก JSON สำเร็จ!")
    print(f"   - Classes: {CLASS_NAMES}")
    print(f"   - Input Size: {INPUT_SIZE}")
    print(f"   - Mean: {IMG_MEAN}, Std: {IMG_STD}")
except Exception as e:
    print(f"⚠️ อ่านไฟล์ JSON ไม่สำเร็จ (ใช้ค่าเริ่มต้น): {e}")

# =========================================================
# 🖱️ ฟังก์ชันจัดการเมาส์ (ปุ่ม EXIT)
# =========================================================
running = True 

def mouse_callback(event, x, y, flags, param):
    global running
    if event == cv2.EVENT_LBUTTONDOWN:
        if 520 <= x <= 620 and 420 <= y <= 460:
            print("🛑 EXIT BUTTON CLICKED!")
            running = False

# =========================================================
# 🔌 เชื่อมต่อ ESP32
# =========================================================
print(f"🔄 กำลังเชื่อมต่อ ESP32 ที่ {ESP32_PORT}...")
ser = None
try:
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=0.1, dsrdtr=False)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(2) 
    print("✅ เชื่อมต่อสำเร็จ!")
except Exception as e:
    print(f"⚠️ ไม่พบ ESP32 (Offline Mode): {e}")

# =========================================================
# 🧠 โหลดโมเดล (OpenCV DNN)
# =========================================================
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print("✅ โหลดโมเดลด้วย OpenCV DNN สำเร็จ!")
except Exception as e:
    print(f"❌ Error โหลดโมเดล: {e}")
    input("กด Enter เพื่อจบโปรแกรม...")
    exit()

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_NAME = "QC System - Classification Mode"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

is_logged = False
process_timer_start = None
PROCESS_DELAY = 0.2 

try:
    while running:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            continue

        # 1. วาดกรอบตำแหน่ง Crop
        cv2.rectangle(frame, (CROP_X, CROP_Y), (CROP_X + CROP_W, CROP_Y + CROP_H), (0, 255, 255), 2)
        cv2.putText(frame, "PLACE PART HERE", (CROP_X, CROP_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2. ทำการ Crop ภาพ
        cropped_img = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]

        # 3. เตรียมภาพเข้าโมเดล (อ่านค่าจาก JSON อัตโนมัติ)
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) 
        
        # 📌 ใช้ขนาดภาพ (INPUT_SIZE) ที่อ่านมาจาก JSON
        resized_img = cv2.resize(gray_3ch, INPUT_SIZE) 
        
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 📌 ใช้ค่า Mean / Std ที่อ่านมาจาก JSON
        img_float = rgb_img.astype(np.float32) / 255.0
        norm_img = (img_float - IMG_MEAN) / IMG_STD
        
        blob = np.transpose(norm_img, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # 4. ส่งให้โมเดลทำนาย
        net.setInput(blob)
        preds = net.forward()

        preds_flat = preds[0]
        exp_preds = np.exp(preds_flat - np.max(preds_flat))
        probs = exp_preds / np.sum(exp_preds)
        
        top_class_id = int(np.argmax(probs))
        confidence = float(probs[top_class_id])

        predicted_class_name = "UNKNOWN"
        final_status = "WAITING"

        if confidence > 0.6: 
            predicted_class_name = CLASS_NAMES.get(top_class_id, "UNKNOWN")
            
            if predicted_class_name == "Copper":
                final_status = "OK"
            elif predicted_class_name == "No_Copper":
                final_status = "NG"
            elif predicted_class_name == "No_part":
                final_status = "WAITING"

        # ============================================
        # 5. การส่งค่าและการบันทึกผล
        # ============================================
        if final_status != "WAITING":
            if process_timer_start is None:
                process_timer_start = time.time()
            
            color_text = (0, 255, 0) if final_status == "OK" else (0, 0, 255)
            cv2.putText(frame, f"STATUS: {final_status} ({confidence:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_text, 3)
            cv2.putText(frame, f"CLASS: {predicted_class_name}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            if time.time() - process_timer_start > PROCESS_DELAY:
                try:
                    if ser is not None:
                        if final_status == "OK": ser.write(b'1')
                        else: ser.write(b'0')
                except: pass

                if not is_logged:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = f"{PATH_OK if final_status=='OK' else PATH_NG}{predicted_class_name}_{timestamp}.jpg"
                    try:
                        cv2.imwrite(fname, frame) 
                        print(f"📸 Saved: {fname} [Class: {predicted_class_name}]")
                    except: pass
                    is_logged = True
        else:
            process_timer_start = None
            is_logged = False
            cv2.putText(frame, "WAITING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            if ser is not None: ser.write(b'0')

        # ปุ่ม EXIT
        btn_x1, btn_y1 = 520, 420
        btn_x2, btn_y2 = 620, 460
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 255), -1) 
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv2.putText(frame, "EXIT", (btn_x1 + 15, btn_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Crop View (Gray Resized)", resized_img)
        cv2.imshow(WINDOW_NAME, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

except Exception as e:
    print("❌ CRASHED")
    traceback.print_exc()

finally:
    if ser is not None:
        try:
            print("🛑 Stopping Machine...")
            ser.write(b'0') 
            time.sleep(0.5) 
            ser.close()
        except: pass
        
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Program Closed.")