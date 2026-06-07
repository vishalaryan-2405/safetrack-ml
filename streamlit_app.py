import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import json, os, requests, math
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(
    page_title="SafeTrack — Journey Safety Monitor",
    page_icon="🛡️",
    layout="wide"
)

# ── GPS COMPONENT (auto location) ─────────────────────
GPS_COMPONENT = """
<script>
function getGPS() {
    const btn = document.getElementById('gps-btn');
    const out = document.getElementById('gps-out');
    btn.disabled = true;
    btn.textContent = '⏳ Getting location...';
    if (!navigator.geolocation) {
        out.textContent = 'Geolocation not supported';
        btn.disabled = false;
        return;
    }
    navigator.geolocation.getCurrentPosition(
        function(pos) {
            const lat = pos.coords.latitude.toFixed(6);
            const lng = pos.coords.longitude.toFixed(6);
            out.textContent = lat + ',' + lng;
            btn.textContent = '✅ Location detected!';
            // send to streamlit via query param trick
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: lat + ',' + lng
            }, '*');
        },
        function(err) {
            out.textContent = 'Error: allow location in browser';
            btn.disabled = false;
            btn.textContent = '🎯 Get my GPS location';
        },
        {enableHighAccuracy: true, timeout: 12000}
    );
}
</script>
<div style="background:#1c2333;border:1px solid #21262d;border-radius:10px;padding:14px;margin-bottom:8px">
  <button id="gps-btn" onclick="getGPS()"
    style="width:100%;padding:10px;background:#58a6ff;color:#0d1117;
    border:none;border-radius:8px;font-size:14px;font-weight:600;
    cursor:pointer;font-family:inherit">
    🎯 Get my GPS location
  </button>
  <div style="margin-top:8px;font-size:12px;color:#7d8590">
    Detected: <span id="gps-out" style="color:#00c47a;font-family:monospace">--</span>
  </div>
  <div style="margin-top:4px;font-size:11px;color:#7d8590">
    Copy the coordinates above → paste into lat/lng fields below
  </div>
</div>
"""

st.markdown("""
<style>
.risk-safe  {color:#00c47a;font-size:50px;font-weight:700;line-height:1}
.risk-warn  {color:#f5a623;font-size:50px;font-weight:700;line-height:1}
.risk-danger{color:#f03e3e;font-size:50px;font-weight:700;line-height:1}
.risk-blue  {color:#58a6ff;font-size:50px;font-weight:700;line-height:1}
.badge-safe  {background:#e6faf3;color:#00c47a;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.badge-warn  {background:#fff8e6;color:#f5a623;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.badge-danger{background:#ffeaea;color:#f03e3e;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.badge-blue  {background:#e6f0ff;color:#58a6ff;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.stat-box{background:#f8f9fa;padding:12px;border-radius:10px;text-align:center;border:1px solid #eee}
.stat-val{font-size:22px;font-weight:700;margin:0}
.stat-lbl{font-size:10px;color:#888;margin:0;text-transform:uppercase;letter-spacing:.05em}
.sos-box{background:#ffeaea;border:2px solid #f03e3e;border-radius:12px;padding:16px;text-align:center;margin:10px 0}
.journey-box{background:#e6f0ff;border:2px solid #58a6ff;border-radius:12px;padding:14px;margin:8px 0}
.journey-active{background:#e6faf3;border:2px solid #00c47a;border-radius:12px;padding:14px;margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ── FILE PATHS ─────────────────────────────────────────
PLACES_FILE  = "data/safe_places.json"
CONTACTS_FILE= "data/contacts.json"
LOG_FILE     = "data/live_log.csv"
os.makedirs("data", exist_ok=True)

# ── MATH HELPERS ───────────────────────────────────────
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dLat = radians(lat2-lat1); dLng = radians(lng2-lng1)
    a = sin(dLat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dLng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_bearing(lat1, lng1, lat2, lng2):
    la1,la2 = radians(lat1),radians(lat2); dl = radians(lng2-lng1)
    x = sin(dl)*cos(la2)
    y = cos(la1)*sin(la2) - sin(la1)*cos(la2)*cos(dl)
    return (math.degrees(math.atan2(x,y))+360)%360

def angle_diff(a, b):
    d = abs(a-b)%360; return d if d<=180 else 360-d

# ── FILE HELPERS ───────────────────────────────────────
def load_places():
    if os.path.exists(PLACES_FILE):
        with open(PLACES_FILE) as f: return json.load(f)
    default = [
        {"id":1,"name":"Railway Station","lat":18.1115,"lng":83.3963},
        {"id":2,"name":"Bus Stand",      "lat":18.1099,"lng":83.3994},
        {"id":3,"name":"Fort Park",      "lat":18.1116,"lng":83.4096},
        {"id":4,"name":"Collectorate",   "lat":18.1176,"lng":83.3870},
    ]
    with open(PLACES_FILE,"w") as f: json.dump(default,f)
    return default

def save_places(p):
    with open(PLACES_FILE,"w") as f: json.dump(p,f,indent=2)

def load_contacts():
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE) as f: return json.load(f)
    return []

def save_contacts(c):
    with open(CONTACTS_FILE,"w") as f: json.dump(c,f,indent=2)

# ── SCORING ────────────────────────────────────────────
def score_location(lat, lng):
    places = load_places()
    if not places:
        return {"risk":50,"status":"MILD WARNING","dist":999,
                "in_cluster":False,
                "explain":"No safe places added yet. Add places you visit."}
    dists   = [haversine(lat,lng,p["lat"],p["lng"]) for p in places]
    min_d   = min(dists)
    nearest = places[dists.index(min_d)]
    inc     = min_d < 0.4
    if inc:
        return {"risk":max(5,round(min_d*100,1)),"status":"SAFE",
                "dist":round(min_d,3),"in_cluster":True,
                "explain":f"{int(min_d*1000)}m from \"{nearest['name']}\". Known safe cluster."}
    elif min_d < 1.5:
        return {"risk":round(35+min_d*10,1),"status":"MILD WARNING",
                "dist":round(min_d,3),"in_cluster":False,
                "explain":f"{min_d:.2f}km from \"{nearest['name']}\". Between safe zones."}
    else:
        return {"risk":min(95,round(50+min_d*8,1)),"status":"DANGER",
                "dist":round(min_d,3),"in_cluster":False,
                "explain":f"{min_d:.2f}km from \"{nearest['name']}\". Outside all known clusters!"}

def score_journey(lat, lng, dest_lat, dest_lng, dest_name, path, route_coords):
    dist = haversine(lat, lng, dest_lat, dest_lng)
    if dist < 0.2:
        return {"risk":0,"status":"ARRIVED","dist":0,
                "explain":f"You have reached {dest_name}! 🎉"}
    # Check if on road route
    if route_coords:
        min_road = min(haversine(lat,lng,c[0],c[1]) for c in route_coords)
        if min_road < 0.2:
            return {"risk":5,"status":"ON TRACK","dist":round(dist,2),
                    "explain":f"On route. {dist:.2f}km to {dest_name}."}
        else:
            risk = min(90,round(min_road*120))
            status = "MILD DEVIATION" if risk<50 else "OFF ROUTE"
            return {"risk":risk,"status":status,"dist":round(dist,2),
                    "explain":f"{int(min_road*1000)}m off route. {dist:.2f}km to {dest_name}."}
    # Fallback: bearing check
    exp = get_bearing(lat,lng,dest_lat,dest_lng)
    dir_risk = 0; dir_msg = "moving"
    if len(path) >= 2:
        prev = path[-2]
        act  = get_bearing(prev[0],prev[1],lat,lng)
        diff = angle_diff(exp,act)
        if diff < 45:    dir_risk=0;  dir_msg=f"heading toward {dest_name}"
        elif diff < 90:  dir_risk=25; dir_msg="slightly off route"
        elif diff < 135: dir_risk=55; dir_msg="going wrong way"
        else:            dir_risk=80; dir_msg="moving away from destination"
    risk   = min(95, dir_risk)
    status = "ON TRACK" if risk<30 else "OFF ROUTE" if risk<60 else "WRONG DIRECTION"
    return {"risk":risk,"status":status,"dist":round(dist,2),
            "explain":f"{dir_msg}. {dist:.2f}km to {dest_name}."}

def retrain(places):
    if len(places) < 2: return False
    rows = []
    for p in places:
        for _ in range(50):
            rows.append({"lat":p["lat"]+np.random.normal(0,.003),
                         "lng":p["lng"]+np.random.normal(0,.003),
                         "hour":np.random.randint(7,22)})
    df = pd.DataFrame(rows)
    df["lat_rad"]  = np.radians(df["lat"])
    df["lng_rad"]  = np.radians(df["lng"])
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    X = df[["lat_rad","lng_rad","hour_sin","hour_cos"]].values
    db  = DBSCAN(eps=0.003,min_samples=5,metric="haversine")
    db.fit(X[:,:2])
    iso = IsolationForest(contamination=0.05,random_state=42)
    iso.fit(X)
    import joblib; os.makedirs("models",exist_ok=True)
    joblib.dump(db,"models/dbscan.pkl")
    joblib.dump(iso,"models/isoforest.pkl")
    return True

# ── SEARCH PLACE ───────────────────────────────────────
def search_place(query):
    if not query or len(query)<3: return []
    try:
        h = {"User-Agent":"SafeTrack/1.0 (journey safety app)"}
        url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={requests.utils.quote(query)}"
               f"&format=json&limit=5&countrycodes=in")
        r = requests.get(url, headers=h, timeout=7)
        if r.status_code==200 and r.json():
            return [{"name":d["display_name"].split(",")[0].strip(),
                     "address":", ".join(d["display_name"].split(",")[1:3]),
                     "lat":float(d["lat"]),"lng":float(d["lon"])}
                    for d in r.json()]
    except: pass
    try:
        url = (f"https://photon.komoot.io/api/"
               f"?q={requests.utils.quote(query)}"
               f"&limit=5&lang=en&bbox=68,6,97,35")
        r = requests.get(url, timeout=7)
        if r.status_code==200:
            return [{"name":f["properties"].get("name","Unknown"),
                     "address":", ".join(filter(None,[
                         f["properties"].get("city",""),
                         f["properties"].get("state","")])),
                     "lat":f["geometry"]["coordinates"][1],
                     "lng":f["geometry"]["coordinates"][0]}
                    for f in r.json().get("features",[])]
    except: pass
    return []

# ── GET ROAD ROUTE (OSRM) ──────────────────────────────
def get_road_route(lat1, lng1, lat2, lng2):
    try:
        url = (f"https://router.project-osrm.org/route/v1/driving/"
               f"{lng1},{lat1};{lng2},{lat2}"
               f"?overview=full&geometries=geojson")
        r = requests.get(url, timeout=8)
        if r.status_code==200:
            data = r.json()
            if data.get("code")=="Ok":
                coords = data["routes"][0]["geometry"]["coordinates"]
                return [[c[1],c[0]] for c in coords]  # [lat,lng]
    except: pass
    return None

# ── BUILD MAP ──────────────────────────────────────────
def build_map(places, path=None, cur_lat=None, cur_lng=None,
              dest_lat=None, dest_lng=None, route_coords=None,
              result=None):
    center = [cur_lat,cur_lng] if cur_lat else [18.1115,83.3963]
    m = folium.Map(location=center, zoom_start=14)

    # Safe zone circles
    for p in places:
        folium.Circle([p["lat"],p["lng"]], radius=400,
            color="#00c47a", fill=True, fill_opacity=0.08,
            weight=1.5, dash_array="5").add_to(m)
        folium.CircleMarker([p["lat"],p["lng"]], radius=6,
            color="#00c47a", fill=True, fill_opacity=0.9,
            tooltip=p["name"]).add_to(m)

    # Road route (blue line)
    if route_coords and len(route_coords)>1:
        folium.PolyLine(route_coords, color="#58a6ff",
            weight=4, opacity=0.7).add_to(m)

    # Journey path (green line)
    if path and len(path)>1:
        folium.PolyLine(path, color="#00c47a",
            weight=3, opacity=0.8).add_to(m)

    # You marker
    if cur_lat:
        status = result.get("status","") if result else ""
        color  = ("blue"   if status in ["ON TRACK","ARRIVED","SAFE"]
                  else "orange" if "WARNING" in status or "DEVIATION" in status
                  else "red")
        folium.Marker([cur_lat,cur_lng],
            popup=f"<b>You are here</b><br>{status}",
            tooltip="You",
            icon=folium.Icon(color=color,icon="user",prefix="fa")
        ).add_to(m)

    # Destination marker
    if dest_lat:
        folium.Marker([dest_lat,dest_lng],
            popup="<b>Destination</b>",
            tooltip="Destination",
            icon=folium.Icon(color="red",icon="flag",prefix="fa")
        ).add_to(m)
        if cur_lat:
            folium.PolyLine(
                [[cur_lat,cur_lng],[dest_lat,dest_lng]],
                color="#f03e3e", weight=1.5,
                opacity=0.3, dash_array="8 6"
            ).add_to(m)
    return m

# ── SESSION STATE ──────────────────────────────────────
defaults = {
    "result":None,"cur_lat":None,"cur_lng":None,
    "history":[],"dest_search":[],"gps_lat":None,"gps_lng":None,
    "dest_lat":None,"dest_lng":None,"dest_name":"",
    "journey_active":False,"journey_path":[],
    "journey_dist":0.0,"journey_start":None,
    "route_coords":None,"search_results":[]
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

places   = load_places()
contacts = load_contacts()

# ── HEADER ─────────────────────────────────────────────
st.markdown("# 🛡️ SafeTrack")
st.markdown("**Personal Journey Intelligence System** "
            "— DBSCAN + Isolation Forest ML")

# ── STATS BAR ──────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
total = 0
try:
    if os.path.exists(LOG_FILE):
        df_l = pd.read_csv(LOG_FILE)
        if len(df_l)>0: total = len(df_l)
except: pass

with c1:
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{len(places)}</div>"
                f"<div class='stat-lbl'>Safe zones</div></div>",
                unsafe_allow_html=True)
with c2:
    tr = "✅" if os.path.exists("models/dbscan.pkl") else "❌"
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{tr}</div>"
                f"<div class='stat-lbl'>Model trained</div></div>",
                unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{total}</div>"
                f"<div class='stat-lbl'>Total checks</div></div>",
                unsafe_allow_html=True)
with c4:
    jst = "🟢 Active" if st.session_state.journey_active else "⚫ Idle"
    st.markdown(f"<div class='stat-box'><div class='stat-val' "
                f"style='font-size:15px'>{jst}</div>"
                f"<div class='stat-lbl'>Journey</div></div>",
                unsafe_allow_html=True)

st.divider()
left, right = st.columns([1, 2])

with left:

    # ══════════════════════════════════════════════════
    # AUTO GPS SECTION
    # ══════════════════════════════════════════════════
    st.subheader("🎯 Your Location")

    # Show GPS component
    st.components.v1.html(GPS_COMPONENT, height=110)

    # GPS coords input — auto-fills if user pastes from above
    gps_input = st.text_input(
        "Paste GPS coordinates here",
        placeholder="e.g. 18.110300,83.397500",
        help="Click 'Get my GPS location' above, copy the coordinates, paste here"
    )

    # Parse pasted coordinates
    if gps_input and "," in gps_input:
        try:
            parts = gps_input.strip().split(",")
            st.session_state.gps_lat = float(parts[0].strip())
            st.session_state.gps_lng = float(parts[1].strip())
        except: pass

    # Coordinate inputs — pre-filled from GPS
    col1, col2 = st.columns(2)
    with col1:
        lat_in = st.number_input("Latitude",
            value=float(st.session_state.gps_lat)
                  if st.session_state.gps_lat else 18.1103,
            format="%.6f", step=0.0001, key="lat_input")
    with col2:
        lng_in = st.number_input("Longitude",
            value=float(st.session_state.gps_lng)
                  if st.session_state.gps_lng else 83.3975,
            format="%.6f", step=0.0001, key="lng_input")

    st.caption("💡 Or: Google Maps → long press any spot → "
               "coordinates appear at top → copy here")

    st.divider()

    # ══════════════════════════════════════════════════
    # JOURNEY MODE
    # ══════════════════════════════════════════════════
    if not st.session_state.journey_active:
        st.subheader("🚀 Start Journey")

        st.markdown("""
        <div class='journey-box'>
        <b>How journey mode works:</b><br>
        • Your current location is detected automatically<br>
        • Enter destination (optional)<br>
        • App loads the actual road route<br>
        • Alerts if you go off route or wrong direction
        </div>""", unsafe_allow_html=True)

        # Destination search
        st.caption("🏁 Destination (optional)")
        dest_q = st.text_input("Search destination",
            placeholder="Araku Valley, Bus Stand, College...",
            key="dest_q")

        if st.button("🔍 Search destination", key="search_dest"):
            if dest_q:
                with st.spinner("Searching..."):
                    results = search_place(dest_q)
                    st.session_state.dest_search = results
                    if not results:
                        st.warning("No results. Try adding city name "
                                   "e.g. 'Bus Stand Vizianagaram'")

        if st.session_state.dest_search:
            for i, r in enumerate(st.session_state.dest_search):
                ci, cs = st.columns([3,1])
                ci.write(f"📍 **{r['name']}**")
                ci.caption(r['address'])
                if cs.button("Select", key=f"dest_{i}"):
                    st.session_state.dest_lat  = r["lat"]
                    st.session_state.dest_lng  = r["lng"]
                    st.session_state.dest_name = r["name"]
                    st.session_state.dest_search = []
                    st.rerun()

        if st.session_state.dest_lat:
            st.success(f"✅ Destination: **{st.session_state.dest_name}**")
            if st.button("❌ Clear destination"):
                st.session_state.dest_lat  = None
                st.session_state.dest_lng  = None
                st.session_state.dest_name = ""
                st.session_state.route_coords = None
                st.rerun()
        else:
            st.caption("No destination — will monitor safe zone clusters only")

        if st.button("🟢 START JOURNEY",
                     use_container_width=True, type="primary"):
            st.session_state.journey_active = True
            st.session_state.journey_path   = [[lat_in, lng_in]]
            st.session_state.journey_dist   = 0.0
            st.session_state.journey_start  = datetime.now().strftime("%H:%M")
            st.session_state.cur_lat        = lat_in
            st.session_state.cur_lng        = lng_in
            # Get road route if destination set
            if st.session_state.dest_lat:
                with st.spinner("Loading road route..."):
                    route = get_road_route(
                        lat_in, lng_in,
                        st.session_state.dest_lat,
                        st.session_state.dest_lng)
                    st.session_state.route_coords = route
                    if route:
                        st.success(f"Road route loaded! "
                                   f"{len(route)} waypoints.")
            st.rerun()

    else:
        # ── JOURNEY ACTIVE ──────────────────────────────
        st.subheader("🟢 Journey in progress")

        dest_set = st.session_state.dest_lat is not None
        if dest_set:
            st.markdown(f"""
            <div class='journey-active'>
            <b>To:</b> {st.session_state.dest_name}<br>
            <b>Started:</b> {st.session_state.journey_start}<br>
            <b>Points:</b> {len(st.session_state.journey_path)} &nbsp;
            <b>Route:</b> {"✅ loaded" if st.session_state.route_coords
                           else "❌ not loaded"}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='journey-active'>
            <b>Free monitoring mode</b><br>
            <b>Started:</b> {st.session_state.journey_start}<br>
            <b>Points tracked:</b> {len(st.session_state.journey_path)}
            </div>""", unsafe_allow_html=True)

        st.caption("📍 Update your position as you move")

        if st.button("📍 Update my position",
                     use_container_width=True, type="primary"):
            path = st.session_state.journey_path
            if path:
                prev = path[-1]
                d    = haversine(prev[0],prev[1],lat_in,lng_in)
                st.session_state.journey_dist += d
            st.session_state.journey_path.append([lat_in, lng_in])
            st.session_state.cur_lat = lat_in
            st.session_state.cur_lng = lng_in

            if dest_set:
                r = score_journey(
                    lat_in, lng_in,
                    st.session_state.dest_lat,
                    st.session_state.dest_lng,
                    st.session_state.dest_name,
                    st.session_state.journey_path,
                    st.session_state.route_coords or [])
            else:
                r = score_location(lat_in, lng_in)

            st.session_state.result = r
            st.session_state.history.insert(0, {
                "time":datetime.now().strftime("%H:%M"),
                "lat":lat_in,"lng":lng_in,**r})
            st.rerun()

        # Result display
        if st.session_state.result:
            r      = st.session_state.result
            status = r["status"]
            risk   = r["risk"]
            if status in ["SAFE","ON TRACK"]:       css="safe"
            elif status=="ARRIVED":                  css="blue"
            elif "WARNING" in status or "DEVIATION" in status or "MILD" in status: css="warn"
            else:                                    css="danger"

            st.markdown(f"""
            <div style='text-align:center;padding:16px;
              background:#f8f9fa;border-radius:12px;
              margin:10px 0;border:1px solid #eee'>
              <div class='risk-{css}'>{risk}</div>
              <div style='font-size:10px;color:#999'>out of 100</div>
              <span class='badge-{css}'>{status}</span>
              <p style='font-size:12px;color:#555;
                margin-top:8px;margin-bottom:0'>
                {r["explain"]}</p>
            </div>""", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Distance covered",
                      f"{st.session_state.journey_dist:.2f} km")
            if dest_set:
                m2.metric("To destination", f"{r['dist']} km")
            else:
                m2.metric("Nearest safe zone", f"{r['dist']} km")

            # SOS when danger
            if status in ["WRONG DIRECTION","DANGER","OFF ROUTE"] and contacts:
                c = contacts[0]
                st.error("🚨 You may be off route or in danger!")
                st.markdown(f"""
                <div class='sos-box'>
                  <div style='font-size:15px;font-weight:700;
                    color:#f03e3e'>🆘 Emergency</div>
                  <div style='margin:8px 0;font-size:13px'>
                    Tap to call {c['name']}</div>
                  <a href='tel:{c["num"]}'
                    style='background:#f03e3e;color:white;
                    padding:11px 24px;border-radius:10px;
                    text-decoration:none;font-weight:700;
                    font-size:14px'>📞 Call {c['name']}</a>
                </div>""", unsafe_allow_html=True)

                # WhatsApp SOS
                wa_msg = (f"🆘 SOS from SafeTrack!%0a"
                          f"Status: {status}%0a"
                          f"Location: https://maps.google.com"
                          f"?q={lat_in},{lng_in}")
                st.markdown(f"""
                <a href='https://wa.me/{c["num"].replace("+","")}?text={wa_msg}'
                  style='display:inline-block;background:#25D366;
                  color:white;padding:10px 20px;border-radius:10px;
                  text-decoration:none;font-weight:600;
                  font-size:13px;margin-top:8px'>
                  💬 WhatsApp SOS with location
                </a>""", unsafe_allow_html=True)

        if st.button("🔴 Stop Journey", use_container_width=True):
            d   = round(st.session_state.journey_dist, 2)
            pts = len(st.session_state.journey_path)
            st.session_state.journey_active  = False
            st.session_state.result          = None
            st.session_state.journey_path    = []
            st.session_state.journey_dist    = 0.0
            st.session_state.route_coords    = None
            st.success(f"Journey ended. {d} km in {pts} points.")
            st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════
    # QUICK LOCATION CHECK
    # ══════════════════════════════════════════════════
    with st.expander("📍 Quick location check"):
        if st.button("Check current coordinates",
                     use_container_width=True, key="quick_check"):
            r = score_location(lat_in, lng_in)
            st.session_state.result  = r
            st.session_state.cur_lat = lat_in
            st.session_state.cur_lng = lng_in
            st.session_state.history.insert(0,{
                "time":datetime.now().strftime("%H:%M"),
                "lat":lat_in,"lng":lng_in,**r})
            try:
                row = pd.DataFrame([{
                    "lat":lat_in,"lng":lng_in,
                    "hour":datetime.now().hour,
                    "timestamp":datetime.now().isoformat(),
                    "status":r["status"],"risk":r["risk"]}])
                mode = "a" if os.path.exists(LOG_FILE) else "w"
                row.to_csv(LOG_FILE,mode=mode,
                    header=not os.path.exists(LOG_FILE),index=False)
            except: pass
            st.rerun()

        if st.session_state.result and not st.session_state.journey_active:
            r = st.session_state.result
            css = ("safe" if r["status"]=="SAFE"
                   else "warn" if "WARNING" in r["status"]
                   else "danger")
            st.markdown(
                f"<span class='badge-{css}'>{r['status']}</span> "
                f"— Risk: **{r['risk']}/100**<br>"
                f"<small style='color:#888'>{r['explain']}</small>",
                unsafe_allow_html=True)
        if st.button("✅ Add this as safe place",
                     use_container_width=True, key="add_quick"):
            st.session_state["show_add"] = True

    st.divider()

    # ══════════════════════════════════════════════════
    # EMERGENCY CONTACTS
    # ══════════════════════════════════════════════════
    with st.expander(f"📞 Emergency contacts ({len(contacts)})"):
        st.caption("Saved permanently — call button opens phone dialer on mobile")
        with st.form("add_contact", clear_on_submit=True):
            cc1, cc2 = st.columns(2)
            with cc1: cname = st.text_input("Name",
                placeholder="Dad, Mom, Friend...")
            with cc2: cnum  = st.text_input("Number",
                placeholder="+91XXXXXXXXXX")
            if st.form_submit_button("💾 Save contact",
                                     use_container_width=True):
                if cname and cnum:
                    contacts.append({
                        "id":int(datetime.now().timestamp()*1000),
                        "name":cname,"num":cnum})
                    save_contacts(contacts)
                    st.success(f"✅ {cname} saved permanently!")
                    st.rerun()
        for c in contacts:
            cn, cc, cd = st.columns([3,2,1])
            cn.write(f"👤 **{c['name']}**")
            cn.caption(c["num"])
            cc.markdown(
                f"<a href='tel:{c['num']}' "
                f"style='background:#e6faf3;color:#00c47a;"
                f"padding:7px 14px;border-radius:8px;"
                f"text-decoration:none;font-weight:600;"
                f"font-size:12px;border:1px solid #00c47a'>"
                f"📞 Call</a>",
                unsafe_allow_html=True)
            if cd.button("×", key=f"dc_{c['id']}"):
                contacts = [x for x in contacts if x["id"]!=c["id"]]
                save_contacts(contacts)
                st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════
    # ADD SAFE PLACE
    # ══════════════════════════════════════════════════
    with st.expander(f"🗂️ Safe places ({len(places)})"):
        # Search and add
        sq = st.text_input("Search place to add",
            placeholder="Railway Station, College, Market...",
            key="place_search_q")
        if st.button("🔍 Search", key="search_place_btn") and sq:
            with st.spinner("Searching..."):
                sr = search_place(sq)
                st.session_state.search_results = sr

        if st.session_state.search_results:
            for i,r in enumerate(st.session_state.search_results):
                pi, pb = st.columns([3,1])
                pi.write(f"📍 **{r['name']}**")
                pi.caption(r['address'])
                if pb.button("Add", key=f"sradd_{i}"):
                    places.append({
                        "id":int(datetime.now().timestamp()*1000),
                        "name":r["name"],
                        "lat":r["lat"],"lng":r["lng"]})
                    save_places(places)
                    retrain(places)
                    st.session_state.search_results = []
                    st.success(f"✅ '{r['name']}' added!")
                    st.rerun()

        st.caption("— or add manually —")
        with st.form("add_place", clear_on_submit=True):
            pname = st.text_input("Place name",
                placeholder="Home, Office, College...")
            use_above = st.checkbox("Use coordinates from above",
                                    value=True)
            if not use_above:
                pc1, pc2 = st.columns(2)
                with pc1: p_lat = st.number_input(
                    "Lat", value=18.1115, format="%.5f")
                with pc2: p_lng = st.number_input(
                    "Lng", value=83.3963, format="%.5f")
            else:
                p_lat = lat_in; p_lng = lng_in

            if st.form_submit_button("➕ Add safe place",
                                     use_container_width=True):
                if pname:
                    places.append({
                        "id":int(datetime.now().timestamp()*1000),
                        "name":pname,"lat":p_lat,"lng":p_lng})
                    save_places(places)
                    retrain(places)
                    st.success(f"✅ '{pname}' added! "
                               f"Model retrained with {len(places)} zones.")
                    st.rerun()

        for p in places:
            pc1, pc2 = st.columns([4,1])
            pc1.write(f"🟢 **{p['name']}**")
            pc1.caption(f"{p['lat']:.4f}, {p['lng']:.4f}")
            if pc2.button("×", key=f"dp_{p['id']}"):
                places = [x for x in places if x["id"]!=p["id"]]
                save_places(places)
                st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════
    # HISTORY
    # ══════════════════════════════════════════════════
    if st.session_state.history:
        with st.expander("📋 Recent checks"):
            for h in st.session_state.history[:8]:
                icon = ("🟢" if h["status"]=="SAFE" or "TRACK" in h["status"]
                        else "🟡" if "WARNING" in h["status"]
                        or "MILD" in h["status"] else "🔴")
                st.write(f"{icon} **{h['status']}** "
                         f"— {h['risk']}/100 at {h['time']}")

    st.divider()

    # ══════════════════════════════════════════════════
    # ML EXPLANATION
    # ══════════════════════════════════════════════════
    with st.expander("⚙️ How the ML works"):
        st.markdown("""
**Journey mode (with destination):**
- OSRM road routing API loads actual road path
- Each position update checks distance from route
- Within 200m = ON TRACK · Beyond = OFF ROUTE alert

**Free mode (no destination):**
- DBSCAN clusters saved places into safe zones
- Isolation Forest scores location unusualness
- Combined risk = cluster distance + anomaly score

**Time encoding:**
- Hour converted to sin/cos for cyclic awareness
- Same place at 2 AM scores higher risk than 2 PM

`epsilon = 0.003 rad ≈ 300m cluster radius`
        """)

with right:
    st.subheader("🗺️ Safety Map")
    m = build_map(
        places,
        path=st.session_state.journey_path,
        cur_lat=st.session_state.cur_lat,
        cur_lng=st.session_state.cur_lng,
        dest_lat=st.session_state.dest_lat,
        dest_lng=st.session_state.dest_lng,
        route_coords=st.session_state.route_coords,
        result=st.session_state.result)
    st_folium(m, width=None, height=600, returned_objects=[])

    if st.session_state.journey_active:
        st.caption(
            "🔵 Blue = road route  |  "
            "🟢 Green = your path  |  "
            "🟢 Circles = safe zones  |  "
            "🔴 Flag = destination")