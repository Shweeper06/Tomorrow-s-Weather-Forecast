"""
weather_ml.py
ส่วนตรรกะ: ดึงข้อมูลอากาศและ PM2.5 รายวันจาก Open-Meteo, สร้างฟีเจอร์ และฝึกโมเดลพยากรณ์
(ไม่ผูกกับ Streamlit เพื่อให้ทดสอบและนำไปใช้ซ้ำได้ง่าย)

แนวทางโมเดล: Direct multi-horizon (ฝึกโมเดลแยกตามจำนวนวันล่วงหน้า h และตามเป้าหมาย)
เป้าหมาย:
  tmax, tmin : อุณหภูมิสูงสุด/ต่ำสุด (ถดถอย)
  rain       : โอกาสเกิดฝน >= RAIN_MM มม. (จำแนก)
  rain_mm    : ปริมาณฝนรายวันที่คาดการณ์ (ถดถอย) ใช้คำนวณฝนสะสม
  pm25       : PM2.5 เฉลี่ยรายวัน (ถดถอยบน log1p)
แบ่งข้อมูลตามลำดับเวลา 80/20 เพื่อประเมินผล แล้วฝึกซ้ำด้วยข้อมูลทั้งหมดเพื่อใช้งานจริง
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
TZ = "Asia/Bangkok"
RAIN_MM = 1.0  # เกณฑ์ "วันที่มีฝน" (มม./วัน)
AIR_START = pd.Timestamp("2022-08-01")  # ช่วงต้นของข้อมูล PM2.5 รายชั่วโมงที่คาดว่าใช้ได้ (ตรวจสอบกับเอกสาร API)
MIN_PM_ROWS = 300  # จำนวนวันขั้นต่ำของ PM2.5 ที่จะฝึกโมเดล

VARS = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "rain",
    "relative_humidity_2m_mean": "rh",
    "wind_speed_10m_max": "wind",
    "surface_pressure_mean": "pressure",
}
BASE = list(VARS.values())
MODEL_NAMES = ["Gradient Boosting", "Random Forest", "Ridge / Logistic"]

# เกณฑ์ PM2.5 เฉลี่ย 24 ชม. (µg/m³) ตามดัชนีคุณภาพอากาศของกรมควบคุมมลพิษ (ฉบับปรับปรุงปี 2566)
# หมายเหตุ: ควรตรวจสอบเกณฑ์ล่าสุดกับกรมควบคุมมลพิษก่อนนำไปใช้อ้างอิง
PM25_LEVELS = [
    (15.0, "ดีมาก"),
    (25.0, "ดี"),
    (37.5, "ปานกลาง"),
    (75.0, "เริ่มมีผลกระทบต่อสุขภาพ"),
    (float("inf"), "มีผลกระทบต่อสุขภาพ"),
]


def pm25_level(v: float) -> tuple[int, str]:
    if v is None or np.isnan(v):
        return -1, "-"
    for i, (limit, label) in enumerate(PM25_LEVELS):
        if v <= limit:
            return i, label
    return len(PM25_LEVELS) - 1, PM25_LEVELS[-1][1]


# ---------------------------------------------------------------- ข้อมูลอากาศ
def _daily_frame(url: str, params: dict) -> pd.DataFrame:
    r = requests.get(url, params={**params, "timezone": TZ}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date", **VARS})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def fetch_weather_history(lat: float, lon: float, years: int, today: pd.Timestamp):
    """ข้อมูลย้อนหลังจาก Archive API (ล่าช้าประมาณ 5 วัน) และพยายามเติมช่วงล่าสุดจาก Forecast API
    คืนค่า (DataFrame, bridged)"""
    end = today - pd.Timedelta(days=5)
    start = end - pd.DateOffset(years=years)
    hist = _daily_frame(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ",".join(VARS),
        },
    ).dropna(subset=["tmax", "tmin", "rain"])

    bridged = False
    try:
        recent = _daily_frame(
            FORECAST_URL,
            {"latitude": lat, "longitude": lon, "daily": ",".join(VARS), "past_days": 10, "forecast_days": 1},
        )
        recent = recent[(recent.index > hist.index[-1]) & (recent.index < today)]
        recent = recent.dropna(subset=["tmax", "tmin", "rain"])
        if len(recent):
            hist = pd.concat([hist, recent])
            bridged = True
    except Exception:
        pass
    return hist[BASE].ffill().bfill(), bridged


def fetch_nwp(lat: float, lon: float, days: int) -> pd.DataFrame:
    """พยากรณ์เชิงตัวเลขของ Open-Meteo (ใช้เป็นค่าเปรียบเทียบเท่านั้น)"""
    return _daily_frame(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": days,
        },
    )[["tmax", "tmin", "rain"]]


# ---------------------------------------------------------------- PM2.5
def _pm25_daily(params: dict) -> pd.Series:
    r = requests.get(AIR_URL, params={**params, "hourly": "pm2_5", "timezone": TZ}, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    s = pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"]), dtype=float)
    g = s.groupby(s.index.normalize())
    return g.mean().where(g.count() >= 18).rename("pm25")  # ต้องมีข้อมูลอย่างน้อย 18 ชม./วัน


def fetch_pm25_history(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp):
    """PM2.5 เฉลี่ยรายวัน (µg/m³) ดึงเป็นช่วง ๆ ช่วงละ 180 วัน คืนค่า (Series, จำนวนช่วงที่ดึงไม่สำเร็จ)"""
    cur, parts, failed = max(start, AIR_START), [], 0
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=179), end)
        try:
            parts.append(
                _pm25_daily(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": cur.strftime("%Y-%m-%d"),
                        "end_date": nxt.strftime("%Y-%m-%d"),
                    }
                )
            )
        except Exception:
            failed += 1
        cur = nxt + pd.Timedelta(days=1)
    if not parts:
        return pd.Series(dtype=float, name="pm25"), failed
    return pd.concat(parts).sort_index(), failed


def fetch_pm25_forecast(lat: float, lon: float, days: int) -> pd.Series:
    """พยากรณ์ PM2.5 ของ Open-Meteo (ข้อมูลแบบจำลอง CAMS) ใช้เป็นค่าเปรียบเทียบ"""
    return _pm25_daily({"latitude": lat, "longitude": lon, "forecast_days": days})


def build_history(lat: float, lon: float, years: int, today: pd.Timestamp):
    """รวมข้อมูลอากาศและ PM2.5 คืนค่า (DataFrame, info)"""
    df, bridged = fetch_weather_history(lat, lon, years, today)
    pm, failed = pd.Series(dtype=float), 0
    try:
        pm, failed = fetch_pm25_history(lat, lon, df.index[0], df.index[-1])
    except Exception:
        failed = 1
    df["pm25"] = pm.reindex(df.index).ffill(limit=3) if len(pm) else np.nan
    n_pm = int(df["pm25"].notna().sum())
    info = {
        "bridged": bridged,
        "pm25_days": n_pm,
        "pm25_start": df["pm25"].first_valid_index(),
        "pm25_failed_chunks": failed,
    }
    return df, info


def rain_summary(df: pd.DataFrame, today: pd.Timestamp) -> dict:
    """ฝนสะสมจากข้อมูลจริงย้อนหลัง"""
    year = today.year if today.month >= 5 else today.year - 1
    season_start = pd.Timestamp(year=year, month=5, day=1)
    since = df.loc[df.index >= season_start, "rain"]
    return {
        "last": df.index[-1],
        "d7": float(df["rain"].iloc[-7:].sum()),
        "d30": float(df["rain"].iloc[-30:].sum()),
        "season_start": season_start,
        "season": float(since.sum()),
        "season_days": int(len(since)),
    }


# ---------------------------------------------------------------- ฟีเจอร์
def make_features(df: pd.DataFrame, with_pm: bool = False) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in BASE:
        for lag in (0, 1, 2, 3, 7):
            X[f"{c}_lag{lag}"] = df[c].shift(lag)
    for c in ("tmax", "tmin", "rain", "rh"):
        for w in (3, 7, 14):
            X[f"{c}_ma{w}"] = df[c].rolling(w).mean()
    X["rain_sum7"] = df["rain"].rolling(7).sum()
    X["rain_sum30"] = df["rain"].rolling(30).sum()
    X["pressure_diff1"] = df["pressure"].diff()
    doy = df.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if with_pm:
        for lag in (0, 1, 2, 3, 7):
            X[f"pm25_lag{lag}"] = df["pm25"].shift(lag)
        for w in (3, 7):
            X[f"pm25_ma{w}"] = df["pm25"].rolling(w).mean()
    return X


# ---------------------------------------------------------------- โมเดล
def model_factory(name: str):
    """คืนค่า (ตัวถดถอย, ตัวจำแนก)"""
    if name == "Ridge / Logistic":
        return (
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        )
    if name == "Random Forest":
        kw = dict(n_estimators=150, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42)
        return RandomForestRegressor(**kw), RandomForestClassifier(**kw)
    kw = dict(max_iter=200, learning_rate=0.05, random_state=42)
    return HistGradientBoostingRegressor(**kw), HistGradientBoostingClassifier(**kw)


def _proba(model, X) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def _targets(df: pd.DataFrame, h: int) -> dict:
    fut = {c: df[c].shift(-h) for c in ("tmax", "tmin", "rain", "pm25")}
    return {
        "tmax": fut["tmax"],
        "tmin": fut["tmin"],
        "rain": (fut["rain"] >= RAIN_MM).astype(float).where(fut["rain"].notna()),
        "rain_mm": fut["rain"],
        "pm25": np.log1p(fut["pm25"].clip(lower=0)),
    }


def train_all(df: pd.DataFrame, horizons: list[int], model_name: str):
    """ฝึกโมเดลทุก (horizon, target) คืนค่า (models, metrics)"""
    Xw = make_features(df).dropna()
    Xp = make_features(df, with_pm=True).dropna()
    has_pm = df["pm25"].notna().sum() >= MIN_PM_ROWS
    reg, clf = model_factory(model_name)
    models, rows = {}, []

    for h in horizons:
        for name, y in _targets(df, h).items():
            if name == "pm25" and not has_pm:
                continue
            Xs = Xp if name == "pm25" else Xw
            data = Xs.join(y.rename("y")).dropna()
            if len(data) < 150:
                continue
            feats = list(Xs.columns)
            cut = int(len(data) * 0.8)
            tr, te = data.iloc[:cut], data.iloc[cut:]
            is_clf = name == "rain"
            cast = (lambda s: s.astype(int)) if is_clf else (lambda s: s)
            proto = clf if is_clf else reg

            est = clone(proto).fit(tr[feats], cast(tr["y"]))
            if is_clf:
                yt = te["y"].astype(int)
                rows.append((h, name, brier_score_loss(yt, _proba(est, te[feats])),
                             brier_score_loss(yt, np.full(len(te), tr["y"].mean()))))
            elif name == "pm25":
                pred = np.clip(np.expm1(est.predict(te[feats])), 0, None)
                rows.append((h, name, mean_absolute_error(np.expm1(te["y"]), pred),
                             mean_absolute_error(np.expm1(te["y"]), te["pm25_lag0"])))
            elif name == "rain_mm":
                pred = np.clip(est.predict(te[feats]), 0, None)
                rows.append((h, name, mean_absolute_error(te["y"], pred),
                             mean_absolute_error(te["y"], np.full(len(te), tr["y"].mean()))))
            else:
                rows.append((h, name, mean_absolute_error(te["y"], est.predict(te[feats])),
                             mean_absolute_error(te["y"], te[f"{name}_lag0"])))

            models[(h, name)] = clone(proto).fit(data[feats], cast(data["y"]))

    metrics = pd.DataFrame(rows, columns=["horizon", "target", "model", "baseline"])
    return models, metrics


def predict(df: pd.DataFrame, models: dict, horizons: list[int]) -> pd.DataFrame:
    xw = make_features(df).iloc[[-1]]
    xp = make_features(df, with_pm=True).iloc[[-1]]
    pm_ready = not xp.isna().values.any()
    last = df.index[-1]
    out = []
    for h in horizons:
        tmax = float(models[(h, "tmax")].predict(xw)[0])
        tmin = float(models[(h, "tmin")].predict(xw)[0])
        pm = np.nan
        if pm_ready and (h, "pm25") in models:
            pm = max(0.0, float(np.expm1(models[(h, "pm25")].predict(xp)[0])))
        out.append(
            {
                "date": last + pd.Timedelta(days=h),
                "tmax": tmax,
                "tmin": min(tmin, tmax),
                "rain_prob": float(_proba(models[(h, "rain")], xw)[0]),
                "rain_mm": max(0.0, float(models[(h, "rain_mm")].predict(xw)[0])),
                "pm25": pm,
            }
        )
    res = pd.DataFrame(out).set_index("date")
    res["rain_cum"] = res["rain_mm"].cumsum()  # ฝนสะสมนับจากวันแรกของช่วงพยากรณ์
    return res
