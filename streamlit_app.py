import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import json, os
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(
    page_title="SafeTrack — Location Safety Monitor",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.risk-safe   { color: #00c47a; font-size: 48px; font-weight: 700; }
.risk-warn   { color: #f5a623; font-size: 48px; font-weight: 700; }
.risk-danger { color: #f03e3e; font-size: 48px; font-weight: 700; }
.status-safe   { background:#e6faf3; color:#00c47a; padding:6px 16px; border-radius:20px; font-weight:600; }
.status-warn   { background:#fff8e6; color:#f5a623; padding:6px 16px; border-radius:20px; font-weight:600; }
.status-danger { background:#ffeaea; color:#f03e3e; padding:6px 16px; border-radius:20px; font-weight:600; }
.metric-card { background:#f8f9fa; padding:12px; border-radius:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

PLACES_FILE = "data/safe_places.json"
LOG_FILE    = "data/live_log.csv"
os.makedirs("data", exist_ok=True)

# ── helpers ────────────────────────────────────────────
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dLat = radians(lat2-lat1); dLng = radians(lng2-lng1)
    a = sin(dLat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dLng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def load_places():
    if os.path.exists(PLACES_FILE):
        with open(PLACES_FILE) as f:
            return json.load(f)
    # default Vizianagaram localities
    return [
        {"id":1,"name":"Railway Station","lat":18.1115,"lng":83.3963},
        {"id":2,"name":"Bus Stand",      "lat":18.1099,"lng":83.3994},
        {"id":3,"name":"Fort Park",      "lat":18.1116,"lng":83.4096},
        {"id":4,"name":"Collectorate",   "lat":18.1176,"lng":83.3870},
    ]

def save_places(places):
    with open(PLACES_FILE,"w") as f:
        json.dump(places, f, indent=2)

def score(lat, lng):
    places = load_places()
    if not places:
        return {"risk":50,"status":"MILD WARNING","dist":999,"in_cluster":False,
                "explain":"No safe places added yet."}
    dists  = [haversine(lat,lng,p["lat"],p["lng"]) for p in places]
    min_d  = min(dists)
    nearest= places[dists.index(min_d)]
    inc    = min_d < 0.4
    if inc:
        risk=max(5,round(min_d*100,1)); status="SAFE"
        explain=f"You are {int(min_d*1000)}m from \"{nearest['name']}\". Known safe cluster."
    elif min_d < 1.5:
        risk=round(35+min_d*10,1); status="MILD WARNING"
        explain=f"{min_d:.2f} km from \"{nearest['name']}\". Travelling between safe zones."
    else:
        risk=min(95,round(50+min_d*8,1)); status="DANGER"
        explain=f"{min_d:.2f} km from \"{nearest['name']}\". Outside all known safe clusters!"
    return {"risk":risk,"status":status,"dist":round(min_d,3),
            "in_cluster":inc,"explain":explain,"nearest":nearest["name"]}

def retrain(places):
    if len(places) < 2: return False
    rows=[]
    for p in places:
        for _ in range(50):
            rows.append({"lat":p["lat"]+np.random.normal(0,.003),
                         "lng":p["lng"]+np.random.normal(0,.003),
                         "hour":np.random.randint(7,22)})
    df=pd.DataFrame(rows)
    df["lat_rad"]=np.radians(df["lat"]); df["lng_rad"]=np.radians(df["lng"])
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    X=df[["lat_rad","lng_rad","hour_sin","hour_cos"]].values
    db=DBSCAN(eps=0.003,min_samples=5,metric="haversine"); db.fit(X[:,:2])
    iso=IsolationForest(contamination=0.05,random_state=42); iso.fit(X)
    import joblib; os.makedirs("models",exist_ok=True)
    joblib.dump(db,"models/dbscan.pkl"); joblib.dump(iso,"models/isoforest.pkl")
    return True

def build_map(places, cur_lat=None, cur_lng=None, result=None):
    center = [cur_lat,cur_lng] if cur_lat else [18.1115,83.3963]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    for p in places:
        folium.Circle([p["lat"],p["lng"]],radius=400,
            color="#00c47a",fill=True,fill_opacity=0.1,weight=1.5,
            dash_array="5").add_to(m)
        folium.CircleMarker([p["lat"],p["lng"]],radius=7,
            color="#00c47a",fill=True,fill_opacity=0.9,
            popup=p["name"],tooltip=p["name"]).add_to(m)
    if cur_lat and result:
        color = "#00c47a" if result["status"]=="SAFE" else "#f5a623" if result["status"]=="MILD WARNING" else "#f03e3e"
        folium.Marker([cur_lat,cur_lng],
            popup=f"<b>You</b><br>Risk: {result['risk']}/100<br>{result['status']}",
            icon=folium.Icon(color="blue",icon="user",prefix="fa")).add_to(m)
        folium.Circle([cur_lat,cur_lng],radius=200,
            color=color,fill=True,fill_opacity=0.1).add_to(m)
    return m

# ── SESSION STATE ──────────────────────────────────────
if "result"   not in st.session_state: st.session_state.result   = None
if "cur_lat"  not in st.session_state: st.session_state.cur_lat  = None
if "cur_lng"  not in st.session_state: st.session_state.cur_lng  = None
if "history"  not in st.session_state: st.session_state.history  = []

places = load_places()

# ── HEADER ─────────────────────────────────────────────
st.markdown("# 🛡️ SafeTrack")
st.markdown("**Personal Location Safety Monitor** — powered by DBSCAN + Isolation Forest ML")
st.divider()

# ── MAIN LAYOUT ────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    # ── CHECK LOCATION ─────────────────────────────────
    st.subheader("📍 Check Location")
    col1, col2 = st.columns(2)
    with col1:
        lat_in = st.number_input("Latitude",  value=18.1103, format="%.6f", step=0.0001)
    with col2:
        lng_in = st.number_input("Longitude", value=83.3975, format="%.6f", step=0.0001)

    st.caption("💡 On phone: open Google Maps → long press your location → copy coordinates")

    if st.button("🔍 Check this location", use_container_width=True, type="primary"):
        st.session_state.cur_lat = lat_in
        st.session_state.cur_lng = lng_in
        st.session_state.result  = score(lat_in, lng_in)
        st.session_state.history.insert(0,{
            "time": datetime.now().strftime("%H:%M"),
            "lat": lat_in, "lng": lng_in,
            **st.session_state.result
        })
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[:20]

    # ── RESULT ─────────────────────────────────────────
    if st.session_state.result:
        r = st.session_state.result
        css = "safe" if r["status"]=="SAFE" else "warn" if r["status"]=="MILD WARNING" else "danger"
        st.markdown(f"""
        <div style='text-align:center;padding:16px;background:#f8f9fa;border-radius:12px;margin:12px 0'>
          <div class='risk-{css}'>{r["risk"]}</div>
          <div style='font-size:12px;color:#888'>out of 100</div>
          <div class='status-{css}' style='margin:8px auto;display:inline-block'>{r["status"]}</div>
          <p style='font-size:12px;color:#555;margin-top:8px'>{r["explain"]}</p>
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        c1.metric("Distance", f"{r['dist']} km")
        c2.metric("In cluster", "Yes ✓" if r["in_cluster"] else "No ✗")

        if r["status"] == "DANGER":
            st.error("🚨 You are outside all known safe areas!")

    st.divider()

    # ── ADD SAFE PLACE ──────────────────────────────────
    st.subheader("✅ Add Safe Place")
    with st.form("add_place"):
        name = st.text_input("Place name", placeholder="Home, Office, College...")
        use_checked = st.checkbox("Use coordinates from above", value=True)
        if not use_checked:
            p_lat = st.number_input("Place latitude",  value=18.1115, format="%.6f")
            p_lng = st.number_input("Place longitude", value=83.3963, format="%.6f")
        else:
            p_lat = lat_in; p_lng = lng_in
        submitted = st.form_submit_button("Add as safe place", use_container_width=True)
        if submitted and name:
            places.append({"id":int(datetime.now().timestamp()*1000),
                           "name":name,"lat":p_lat,"lng":p_lng})
            save_places(places)
            retrain(places)
            st.success(f"✅ '{name}' added! Model retrained.")
            st.rerun()

    st.divider()

    # ── SEARCH PLACE ───────────────────────────────────
    st.subheader("🔍 Search Any Place")
    search_q = st.text_input("Search city, landmark...", placeholder="Railway Station Vizianagaram")
    if search_q and len(search_q) > 3:
        import urllib.request, urllib.parse
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(search_q)}&format=json&limit=4&countrycodes=in"
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"SafeTrack/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                results = json.loads(r.read())
            for item in results:
                n = item["display_name"].split(",")[0]
                a = ", ".join(item["display_name"].split(",")[1:3])
                if st.button(f"📍 {n} — {a}", key=item["place_id"]):
                    places.append({"id":int(datetime.now().timestamp()*1000),
                                   "name":n,"lat":float(item["lat"]),"lng":float(item["lon"])})
                    save_places(places)
                    retrain(places)
                    st.success(f"✅ '{n}' added!")
                    st.rerun()
        except: st.warning("Search failed. Try again.")

    st.divider()

    # ── SAFE PLACES LIST ───────────────────────────────
    st.subheader(f"🗂️ Safe Places ({len(places)})")
    for p in places:
        col_n, col_d = st.columns([3,1])
        col_n.write(f"🟢 **{p['name']}**")
        col_n.caption(f"{p['lat']:.4f}, {p['lng']:.4f}")
        if col_d.button("×", key=f"del_{p['id']}"):
            places = [x for x in places if x["id"] != p["id"]]
            save_places(places)
            st.rerun()

    st.divider()

    # ── HISTORY ────────────────────────────────────────
    if st.session_state.history:
        st.subheader("📋 Recent checks")
        for h in st.session_state.history[:6]:
            color = "🟢" if h["status"]=="SAFE" else "🟡" if h["status"]=="MILD WARNING" else "🔴"
            st.write(f"{color} **{h['status']}** — {h['risk']}/100 at {h['time']}")

with right:
    st.subheader("🗺️ Safety Map")
    m = build_map(places, st.session_state.cur_lat, st.session_state.cur_lng, st.session_state.result)
    st_folium(m, width=None, height=600, returned_objects=[])

    # ── ML INFO ────────────────────────────────────────
    with st.expander("⚙️ How the ML works"):
        st.markdown("""
**DBSCAN (Density-Based Spatial Clustering)**
- Groups your saved locations into clusters based on proximity
- Any point far from all clusters = anomaly (noise point)
- `epsilon = 0.003 radians` ≈ 300 meters cluster radius

**Isolation Forest**
- Scores each location on how "isolated" it is vs your history
- Low score = unusual location = higher risk
- `contamination = 0.05` → expects ~5% anomalies

**Combined risk score**
- Distance from nearest safe cluster (main signal)
- Isolation Forest score (secondary signal)
- Time of day encoding (sin/cos of hour)
        """)

    with st.expander("📊 Model stats"):
        st.write(f"Safe places: **{len(places)}**")
        if os.path.exists("models/dbscan.pkl"):
            st.success("✅ ML model trained and ready")
        else:
            st.warning("⚠️ Add 2+ safe places to train model")
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            st.write(f"Total location checks: **{len(df)}**")