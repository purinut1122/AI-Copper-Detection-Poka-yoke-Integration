import cv2
import time

print("1. กำลังเริ่มระบบ...")

# 🔧 บังคับใช้ DirectShow (ยาแก้แพ้สำหรับ Lenovo LOQ)
# ลองเปลี่ยนเลข 0 เป็น 1 ถ้า 0 ไม่ติด
cam_index = 0 
print(f"2. กำลังสั่งเปิดกล้องหมายเลข {cam_index} แบบ DirectShow...")

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

# ตั้งค่าความละเอียด (บางทีถ้าไม่ตั้ง มันจะเอ๋อ)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("3. สั่งเปิดไปแล้ว รอดูผลลัพธ์...")
time.sleep(2) # รอแป๊บนึง ให้เวลากล้องตื่น

if not cap.isOpened():
    print("❌ เปิดไม่ติด! (Windows หากล้องไม่เจอ)")
    print("คำแนะนำ: ลองเปลี่ยน cam_index เป็น 1 หรือเช็ค Antivirus")
else:
    print("✅ เปิดติดแล้ว! กำลังลองอ่านภาพ...")
    ret, frame = cap.read()
    
    if ret:
        print("🎉 สำเร็จ! เห็นภาพแล้ว (กด q เพื่อปิด)")
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Test Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("⚠️ เปิดติด แต่ภาพมืด (Black Screen)")

cap.release()
cv2.destroyAllWindows()