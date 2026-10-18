"""app.py — พยากรณ์อากาศ ฝนสะสม และ PM2.5 จังหวัดเชียงใหม่ ด้วย Machine Learning"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import weather_ml as wm

st.set_page_config(page_title="พยากรณ์อากาศและ PM2.5 เชียงใหม่", page_icon="🌦️", layout="wide")

# พิกัดโดยประมาณของที่ว่าการอำเภอ/ตัวเมือง (ผลจริงอิงตามกริดของแบบจำลอง)
AREAS = {
    "อ.เมืองเชียงใหม่": (18.7883, 98.9853),
    "อ.แม่ริม": (18.9167, 98.9333),
    "อ.สันกำแพง": (18.7500, 99.1167),
    "อ.เชียงดาว": (19.3667, 98.9667),
    "อ.ฝาง": (19.9167, 99.2139),
    "อ.จอมทอง": (18.4167, 98.6667),
    "อ.ฮอด": (18.1833, 98.6000),
}
PM_COLORS = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#ef4444"]


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_history(lat, lon, years, today_str):
    return wm.build_history(lat, lon, years, pd.Timestamp(today_str))


@st.cache_data(ttl=3600, show_spinner=False)
def load_nwp(lat, lon, days):
    return wm.fetch_nwp(lat, lon, days)


@st.cache_data(ttl=3600, show_spinner=False)
def load_pm_nwp(lat, lon, days):
    return wm.fetch_pm25_forecast(lat, lon, days)


@st.cache_resource(show_spinner=False)
def get_models(df, horizons, model_name):
    return wm.train_all(df, list(horizons), model_name)


# ------------------------------------------------------------------ Sidebar
st.sidebar.header("การตั้งค่า")
place = st.sidebar.selectbox("พื้นที่ (จังหวัดเชียงใหม่)", list(AREAS) + ["กำหนดพิกัดเอง"])
if place == "กำหนดพิกัดเอง":
    lat = st.sidebar.number_input("ละติจูด", -90.0, 90.0, 18.7883, format="%.4f")
    lon = st.sidebar.number_input("ลองจิจูด", -180.0, 180.0, 98.9853, format="%.4f")
else:
    lat, lon = AREAS[place]
years = st.sidebar.slider("ข้อมูลอากาศย้อนหลัง (ปี)", 2, 10, 5)
horizon = st.sidebar.slider("พยากรณ์ล่วงหน้า (วัน)", 1, 7, 7)
model_name = st.sidebar.selectbox("โมเดล", wm.MODEL_NAMES)

st.title("🌦️ พยากรณ์อากาศ ฝนสะสม และ PM2.5 จังหวัดเชียงใหม่")
st.caption(f"พื้นที่: {place} (ละติจูด {lat:.4f}, ลองจิจูด {lon:.4f}) | แหล่งข้อมูล: Open-Meteo")

# ------------------------------------------------------------------ ข้อมูล
today = pd.Timestamp.now(tz=wm.TZ).tz_localize(None).normalize()
try:
    with st.spinner("กำลังดึงข้อมูลย้อนหลัง..."):
        df, info = load_history(lat, lon, years, today.strftime("%Y-%m-%d"))
except Exception as e:  # noqa: BLE001
    st.error(f"ไม่สามารถดึงข้อมูลจาก Open-Meteo ได้: {e}")
    st.stop()

gap = (today - df.index[-1]).days
horizons = tuple(gap + k for k in range(1, horizon + 1))  # พยากรณ์ตั้งแต่พรุ่งนี้เป็นต้นไป

with st.spinner("กำลังฝึกโมเดล (ครั้งแรกอาจใช้เวลาสักครู่)..."):
    models, metrics = get_models(df, horizons, model_name)
fc = wm.predict(df, models, horizons)
has_pm = fc["pm25"].notna().any()

try:
    nwp = load_nwp(lat, lon, horizon + 1).reindex(fc.index)
except Exception:  # noqa: BLE001
    nwp = None
pm_nwp = None
if has_pm:
    try:
        pm_nwp = load_pm_nwp(lat, lon, horizon + 1).reindex(fc.index)
    except Exception:  # noqa: BLE001
        pm_nwp = None

# ------------------------------------------------------------------ ฝนสะสมจริง
rs = wm.rain_summary(df, today)
st.subheader("ปริมาณฝนสะสมจากข้อมูลจริง")
r1, r2, r3 = st.columns(3)
r1.metric("7 วันล่าสุด (มม.)", f"{rs['d7']:.1f}")
r2.metric("30 วันล่าสุด (มม.)", f"{rs['d30']:.1f}")
r3.metric(f"สะสมตั้งแต่ {rs['season_start'].strftime('%d/%m/%Y')} (มม.)", f"{rs['season']:.1f}")
st.caption(f"นับถึงวันที่ {rs['last'].strftime('%d/%m/%Y')} (ข้อมูลล่าสุดที่มี)")

# ------------------------------------------------------------------ สรุปพยากรณ์
st.subheader(f"ผลการพยากรณ์ {horizon} วันข้างหน้า")
c1, c2, c3, c4 = st.columns(4)
c1.metric("พรุ่งนี้: สูงสุด/ต่ำสุด (°C)", f"{fc.tmax.iloc[0]:.1f} / {fc.tmin.iloc[0]:.1f}")
c2.metric("พรุ่งนี้: โอกาสฝน", f"{fc.rain_prob.iloc[0] * 100:.0f}%")
c3.metric(f"ฝนสะสมพยากรณ์ {horizon} วัน (มม.)", f"{fc.rain_cum.iloc[-1]:.1f}")
if has_pm:
    _, lv = wm.pm25_level(fc.pm25.iloc[0])
    c4.metric("พรุ่งนี้: PM2.5 (µg/m³)", f"{fc.pm25.iloc[0]:.0f}", lv, delta_color="off")
else:
    c4.metric("พรุ่งนี้: PM2.5 (µg/m³)", "-")

left, right = st.columns(2)

fig_t = go.Figure()
fig_t.add_scatter(x=fc.index, y=fc.tmax, name="สูงสุด (ML)", mode="lines+markers", line=dict(color="#d62728"))
fig_t.add_scatter(x=fc.index, y=fc.tmin, name="ต่ำสุด (ML)", mode="lines+markers", line=dict(color="#1f77b4"))
if nwp is not None:
    fig_t.add_scatter(x=nwp.index, y=nwp.tmax, name="สูงสุด (Open-Meteo)", mode="lines", line=dict(color="#d62728", dash="dot"))
    fig_t.add_scatter(x=nwp.index, y=nwp.tmin, name="ต่ำสุด (Open-Meteo)", mode="lines", line=dict(color="#1f77b4", dash="dot"))
fig_t.update_layout(title="อุณหภูมิ (°C)", height=370, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
left.plotly_chart(fig_t, use_container_width=True)

fig_r = make_subplots(specs=[[{"secondary_y": True}]])
fig_r.add_bar(x=fc.index, y=fc.rain_mm, name="ฝนรายวัน (ML)", marker_color="#60a5fa", secondary_y=False)
fig_r.add_scatter(x=fc.index, y=fc.rain_cum, name="ฝนสะสม (ML)", mode="lines+markers", line=dict(color="#1d4ed8"), secondary_y=True)
if nwp is not None:
    fig_r.add_scatter(x=nwp.index, y=nwp.rain.cumsum(), name="ฝนสะสม (Open-Meteo)", mode="lines", line=dict(color="#1d4ed8", dash="dot"), secondary_y=True)
fig_r.update_yaxes(title_text="รายวัน (มม.)", secondary_y=False)
fig_r.update_yaxes(title_text="สะสม (มม.)", secondary_y=True)
fig_r.update_layout(title="ปริมาณฝนคาดการณ์และฝนสะสม", height=370, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
right.plotly_chart(fig_r, use_container_width=True)

left2, right2 = st.columns(2)
if has_pm:
    colors = [PM_COLORS[max(wm.pm25_level(v)[0], 0)] for v in fc.pm25]
    fig_p = go.Figure(go.Bar(x=fc.index, y=fc.pm25, name="PM2.5 (ML)", marker_color=colors))
    if pm_nwp is not None:
        fig_p.add_scatter(x=pm_nwp.index, y=pm_nwp, name="PM2.5 (Open-Meteo)", mode="lines+markers", line=dict(color="#374151", dash="dot"))
    fig_p.add_hline(y=37.5, line_dash="dash", line_color="#f97316", annotation_text="37.5")
    fig_p.add_hline(y=75, line_dash="dash", line_color="#ef4444", annotation_text="75")
    fig_p.update_layout(title="PM2.5 เฉลี่ยรายวัน (µg/m³)", height=370, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
    left2.plotly_chart(fig_p, use_container_width=True)
else:
    left2.info("ไม่มีข้อมูล PM2.5 เพียงพอสำหรับฝึกโมเดล (ต้องมีอย่างน้อย "
               f"{wm.MIN_PM_ROWS} วัน) จึงแสดงเฉพาะอุณหภูมิและฝน")

fig_pr = go.Figure(go.Bar(x=fc.index, y=fc.rain_prob * 100, marker_color="#2ca02c"))
fig_pr.update_layout(title=f"โอกาสเกิดฝน ≥ {wm.RAIN_MM:g} มม. (%)", yaxis=dict(range=[0, 100]), height=370, margin=dict(t=50, b=20))
right2.plotly_chart(fig_pr, use_container_width=True)

# ------------------------------------------------------------------ ตาราง
table = pd.DataFrame(
    {
        "สูงสุด (°C)": fc.tmax.round(1),
        "ต่ำสุด (°C)": fc.tmin.round(1),
        "โอกาสฝน (%)": (fc.rain_prob * 100).round(0),
        "ฝนรายวัน (มม.)": fc.rain_mm.round(1),
        "ฝนสะสม (มม.)": fc.rain_cum.round(1),
    }
)
if has_pm:
    table["PM2.5 (µg/m³)"] = fc.pm25.round(0)
    table["ระดับ PM2.5"] = [wm.pm25_level(v)[1] for v in fc.pm25]
if nwp is not None:
    table["ฝนสะสม Open-Meteo (มม.)"] = nwp.rain.cumsum().round(1)
if pm_nwp is not None:
    table["PM2.5 Open-Meteo (µg/m³)"] = pm_nwp.round(0)
table.index = table.index.strftime("%d/%m/%Y")
table.index.name = "วันที่"
st.dataframe(table, use_container_width=True)

with st.expander("ผลการประเมินโมเดล (ชุดทดสอบ 20% ล่าสุดของข้อมูล)"):
    m = metrics.copy()
    m["เป้าหมาย"] = m["target"].map(
        {
            "tmax": "อุณหภูมิสูงสุด (MAE, °C)",
            "tmin": "อุณหภูมิต่ำสุด (MAE, °C)",
            "rain": "การเกิดฝน (Brier score)",
            "rain_mm": "ปริมาณฝนรายวัน (MAE, มม.)",
            "pm25": "PM2.5 (MAE, µg/m³)",
        }
    )
    m = m.rename(columns={"horizon": "ล่วงหน้า (วัน)", "model": "โมเดล", "baseline": "ฐานเปรียบเทียบ"})
    st.dataframe(m[["ล่วงหน้า (วัน)", "เป้าหมาย", "โมเดล", "ฐานเปรียบเทียบ"]].round(3), hide_index=True, use_container_width=True)
    st.caption(
        "ค่ายิ่งต่ำยิ่งดี | ฐานเปรียบเทียบ: อุณหภูมิและ PM2.5 ใช้ค่า Persistence (ค่าวันล่าสุด) "
        "ส่วนฝน (รายวัน/โอกาส) ใช้ค่าเฉลี่ยจากข้อมูลฝึก (Climatology)"
    )

# ------------------------------------------------------------------ ข้อจำกัด
notes = [
    "โมเดลเรียนรู้จากข้อมูลย้อนหลังของจุดเดียว ไม่ได้จำลองฟิสิกส์ของบรรยากาศ จึงแม่นยำลดลงเมื่อพยากรณ์ล่วงหน้าหลายวัน "
    "และปริมาณฝนรายวันเป็นค่าคาดหมายเชิงสถิติ ซึ่งมักต่ำกว่าฝนตกหนักจริง",
    "ค่า PM2.5 เป็นข้อมูลจากแบบจำลอง (CAMS ผ่าน Open-Meteo) ไม่ใช่ค่าตรวจวัดจากสถานี อาจต่างจากค่าของกรมควบคุมมลพิษ/Air4Thai "
    "โดยเฉพาะช่วงหมอกควันรุนแรง จึงควรตรวจสอบกับแหล่งทางการก่อนตัดสินใจ",
    "เกณฑ์ระดับ PM2.5 อ้างอิงดัชนีคุณภาพอากาศของกรมควบคุมมลพิษ โปรดตรวจสอบเกณฑ์ล่าสุดอีกครั้ง",
    "ผลลัพธ์เพื่อการศึกษาและสาธิตเท่านั้น ไม่ควรใช้ตัดสินใจในงานที่เกี่ยวข้องกับความปลอดภัยหรือสุขภาพ",
]
if has_pm:
    st.caption(
        f"ข้อมูล PM2.5 ที่ใช้ฝึกโมเดล: {info['pm25_days']:,} วัน (เริ่ม {info['pm25_start'].strftime('%d/%m/%Y')})"
    )
    if info["pm25_days"] < 730:
        notes.append("ข้อมูล PM2.5 ครอบคลุมน้อยกว่า 2 ปี อาจยังเรียนรู้ฤดูหมอกควันได้ไม่ครบถ้วน")
if info["pm25_failed_chunks"]:
    notes.append(f"ดึงข้อมูล PM2.5 ไม่สำเร็จ {info['pm25_failed_chunks']} ช่วง จึงมีข้อมูลบางช่วงขาดหาย")
if not info["bridged"]:
    notes.append("ไม่สามารถเติมข้อมูลช่วงล่าสุดจาก Forecast API ได้ จึงใช้เฉพาะข้อมูล Archive ซึ่งล่าช้าประมาณ 5 วัน")
for n in notes:
    st.caption("• " + n)
