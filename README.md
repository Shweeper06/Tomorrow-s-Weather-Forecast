# พยากรณ์อากาศ ฝนสะสม และ PM2.5 จังหวัดเชียงใหม่ (Streamlit)

แอปพยากรณ์ล่วงหน้า 1–7 วัน ได้แก่ อุณหภูมิสูงสุด/ต่ำสุด โอกาสเกิดฝน ปริมาณฝนรายวันและฝนสะสม
และ PM2.5 เฉลี่ยรายวัน ด้วยโมเดล Machine Learning (Gradient Boosting, Random Forest, Ridge/Logistic)
ข้อมูลจาก [Open-Meteo](https://open-meteo.com/) (ไม่ต้องใช้ API Key)

## โครงสร้างไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | หน้าเว็บ Streamlit |
| `weather_ml.py` | ดึงข้อมูล สร้างฟีเจอร์ ฝึกและประเมินโมเดล |
| `requirements.txt` | รายการไลบรารี |

## รันบนเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy บน Streamlit Community Cloud
1. อัปโหลดไฟล์ทั้งหมดขึ้น GitHub (ให้ `app.py` อยู่ที่รากของ Repository)
2. เข้า https://share.streamlit.io แล้วเข้าสู่ระบบด้วยบัญชี GitHub
3. เลือก **Create app** → เลือก Repository, Branch และ Main file path เป็น `app.py`
4. กด **Deploy**

## ข้อจำกัด
- PM2.5 เป็นข้อมูลจากแบบจำลอง (CAMS) ไม่ใช่ค่าตรวจวัดจากสถานี อาจต่างจากค่าของกรมควบคุมมลพิษ/Air4Thai
- ปริมาณฝนรายวันเป็นค่าคาดหมายเชิงสถิติ ฝนสะสมจึงอาจต่ำกว่าเหตุการณ์ฝนตกหนักจริง
- โมเดลใช้ข้อมูลย้อนหลังของจุดเดียว ความแม่นยำลดลงเมื่อพยากรณ์ล่วงหน้าหลายวัน
- เพื่อการศึกษาและสาธิตเท่านั้น
