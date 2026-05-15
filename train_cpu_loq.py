import cv2
import sys
import os
import time
import winsound
from ultralytics import YOLO

# ฟังก์ชันช่วยหยุดหน้าจอเวลา Error (แทน sys.exit)
def error_pause(msg):
    print("\n" + "!"*50)
    print(f"❌ ERROR: {msg}")
    print("!"*50)
    print("\n👉 กด Enter เพื่อปิดโปรแกรม...")
    input()
    sys.exit()

# ==========================================
# ⚙️ CONFIG
# ==========================================
SCREEN_W = 1920
SCREEN_H = 1080
BORDER_THICKNESS = 150
CONF_THRESHOLD = 0.70
DISPLAY_NAME = "Copper"
SOUND_DELAY = 2.0

# ==========================================
# 1. 📷 เช็คกล้อง (Debug Mode)
# ==========================================
print(f"\n🚀 Starting System ({SCREEN_W}x{SCREEN_H})...")
print("📷 กำลังค้นหากล้อง...")

cap = None
active_index = -1

# ลองวนหาตั้งแต่กล้อง 0 ถึง 5 (Tablet บางทีกล้องหลังอยู่เลขแปลกๆ)
for i in range(5):
    print(f"   - Testing Camera Index {i}...", end="")
    temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        print(" ✅ เจอแล้ว!")
        # ลองอ่านภาพ
        ret, frame = temp_cap.read()
        if ret:
            cap = temp_cap
            active_index = i
            break
        else:
            print(" (แต่จอดำ)")
            temp_cap.release()
    else:
        print(" ❌")

if cap is None:
    error_pause("หาไม่เจอสักกล้อง! (เช็ค Driver หรือ Privacy Settings บน Tablet)")

# ตั้งค่ากล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

# ==========================================
# 2. 🧠 เช็คไฟล์โมเดล (จุดที่คนลืมบ่อยสุด)
# ==========================================
print("\n🧠 กำลังโหลด AI...")

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
MODEL_ONNX = os.path.join(BASE_DIR, "copper_model.onnx")
MODEL_PT = os.path.join(BASE_DIR, "copper_model.pt")
MODEL_BEST = os.path.join(BASE_DIR, "best.pt")

print(f"📂 โฟลเดอร์ปัจจุบัน: {BASE_DIR}")
print(f"🔍 กำลังหาไฟล์: copper_model.onnx")

model_path = None
task = None

if os.path.exists(MODEL_ONNX):
    model_path = MODEL_ONNX
    task = 'detect'
elif os.path.exists(MODEL_PT):
    model_path = MODEL_PT
    task = None
elif os.path.exists(MODEL_BEST):
    model_path = MODEL_BEST
    task = None
else:
    # 🚨 ถ้าไม่เจอไฟล์ จะค้างหน้านี้ให้ดู
    files_in_dir = os.listdir(BASE_DIR)
    error_pause(f"ไม่เจอไฟล์โมเดล! \n   ไฟล์ที่มีในโฟลเดอร์นี้: {files_in_dir}")

try:
    model = YOLO(model_path, task=task)
    print(f"✅ AI Loaded! ({os.path.basename(model_path)})")
except Exception as e:
    error_pause(f"โหลดโมเดลพัง: {e}")

# ==========================================
# 3. 🚀 เริ่มทำงาน
# ==========================================
# (ส่วน UI เหมือนเดิม)
should_exit = False
def handle_click(event, x, y, flags, param):
    global should_exit
    if event == cv2.EVENT_LBUTTONDOWN:
        if (SCREEN_W - 200) < x < SCREEN_W and 0 < y < BORDER_THICKNESS:
            should_exit = True

window_name = "Copper Detection System"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, handle_click)

last_beep_time = 0

while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Camera frame dropped")
        continue

    img = cv2.resize(frame, (SCREEN_W, SCREEN_H))
    scale_x = SCREEN_W / frame.shape[1]
    scale_y = SCREEN_H / frame.shape[0]

    found_object = False

    try:
        results = model(frame, device='cpu', verbose=False, conf=CONF_THRESHOLD, iou=0.5)
        for r in results:
            for box in r.boxes:
                found_object = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{DISPLAY_NAME} {int(conf*100)}%"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    except Exception as e:
        print(f"AI Runtime Error: {e}")

    if found_object:
        current_time = time.time()
        if current_time - last_beep_time > SOUND_DELAY:
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            last_beep_time = current_time

    if found_object:
        color = (0, 255, 0)
        status_text = "DETECTED"
    else:
        color = (0, 0, 255)
        status_text = "SEARCHING..."

    # UI วาด
    cv2.rectangle(img, (0, 0), (SCREEN_W, BORDER_THICKNESS), color, -1)
    cv2.rectangle(img, (0, SCREEN_H - BORDER_THICKNESS), (SCREEN_W, SCREEN_H), color, -1)
    cv2.rectangle(img, (0, 0), (BORDER_THICKNESS, SCREEN_H), color, -1)
    cv2.rectangle(img, (SCREEN_W - BORDER_THICKNESS, 0), (SCREEN_W, SCREEN_H), color, -1)

    text_y = int((BORDER_THICKNESS + 30) / 2)
    cv2.putText(img, status_text, (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    btn_w = 200
    cv2.rectangle(img, (SCREEN_W - btn_w, 0), (SCREEN_W, BORDER_THICKNESS), (0, 0, 180), -1)
    cv2.line(img, (SCREEN_W - btn_w, 0), (SCREEN_W - btn_w, BORDER_THICKNESS), (255, 255, 255), 3)
    cv2.putText(img, "EXIT", (SCREEN_W - 160, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow(window_name, img)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or should_exit:
        break

cap.release()
cv2.destroyAllWindows()