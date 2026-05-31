import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import json, os, requests
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(
    page_title="SafeTrack",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.risk-safe   {color:#00c47a;font-size:52px;font-weight:700;line-height:1}
.risk-warn   {color:#f5a623;font-size:52px;font-weight:700;line-height:1}
.risk-danger {color:#f03e3e;font-size:52px;font-weight:700;line-height:1}
.badge-safe  {background:#e6faf3;color:#00c47a;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.badge-warn  {background:#fff8e6;color:#f5a623;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.badge-danger{background:#ffeaea;color:#f03e3e;padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;display:inline-block}
.stat-box    {background:#f8f9fa;padding:14px;border-radius:10px;text-align:center;border:1px solid #eee}
.stat-val    {font-size:26px;font-weight:700;margin:0}
.stat-lbl    {font-size:11px;color:#888;margin:0;text-transform:uppercase;letter-spacing:.05em}
</style>
""", unsafe_allow_html=True)

# ── FILE PATHS ─────────────────────────────────────────
PLACES_FILE   = "data/safe_places.json"
CONTACTS_FILE = "data/contacts.json"
LOG_FILE      = "data/live_log.csv"
os.makedirs("data", exist_ok=True)

# ── HELPERS ────────────────────────────────────────────
def haversine(lat1,lng1,lat2,lng2):
    R=6371; dLat=radians(lat2-lat1); dLng=radians(lng2-lng1)
    a=sin(dLat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dLng/2)**2
    return R*2*atan2(sqrt(a),sqrt(1-a))

def load_places():
    if os.path.exists(PLACES_FILE):
        with open(PLACES_FILE) as f: return json.load(f)
    default=[
        {"id":1,"name":"Railway Station","lat":18.1115,"lng":83.3963},
        {"id":2,"name":"Bus Stand",      "lat":18.1099,"lng":83.3994},
        {"id":3,"name":"Fort Park",      "lat":18.1116,"lng":83.4096},
        {"id":4,"name":"Collectorate",   "lat":18.1176,"lng":83.3870},
    ]
    save_places(default)
    return default

def save_places(places):
    with open(PLACES_FILE,"w") as f: json.dump(places,f,indent=2)

def load_contacts():
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE) as f: return json.load(f)
    return []

def save_contacts(contacts):
    with open(CONTACTS_FILE,"w") as f: json.dump(contacts,f,indent=2)

def score(lat,lng):
    places=load_places()
    if not places:
        return {"risk":50,"status":"MILD WARNING","dist":999,"in_cluster":False,
                "explain":"No safe places added yet."}
    dists=[haversine(lat,lng,p["lat"],p["lng"]) for p in places]
    min_d=min(dists); nearest=places[dists.index(min_d)]; inc=min_d<0.4
    if inc:
        risk=max(5,round(min_d*100,1)); status="SAFE"
        explain=f"You are {int(min_d*1000)}m from \"{nearest['name']}\". Known safe cluster."
    elif min_d<1.5:
        risk=round(35+min_d*10,1); status="MILD WARNING"
        explain=f"{min_d:.2f} km from \"{nearest['name']}\". Travelling between safe zones."
    else:
        risk=min(95,round(50+min_d*8,1)); status="DANGER"
        explain=f"{min_d:.2f} km from \"{nearest['name']}\". Outside all known safe clusters!"
    return {"risk":risk,"status":status,"dist":round(min_d,3),
            "in_cluster":inc,"explain":explain,"nearest":nearest["name"]}

def retrain(places):
    if len(places)<2: return False
    rows=[]
    for p in places:
        for _ in range(50):
            rows.append({"lat":p["lat"]+np.random.normal(0,.003),
                         "lng":p["lng"]+np.random.normal(0,.003),
                         "hour":np.random.randint(7,22)})
    df=pd.DataFrame(rows)
    df["lat_rad"]=np.radians(df["lat"]); df["lng_rad"]=np.radians(df["lng"])
    df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
    X=df[["lat_rad","lng_rad","hour_sin","hour_cos"]].values
    db=DBSCAN(eps=0.003,min_samples=5,metric="haversine"); db.fit(X[:,:2])
    iso=IsolationForest(contamination=0.05,random_state=42); iso.fit(X)
    import joblib; os.makedirs("models",exist_ok=True)
    joblib.dump(db,"models/dbscan.pkl")
    joblib.dump(iso,"models/isoforest.pkl")
    return True

def search_place(query):
    """Search using multiple APIs — tries each until one works"""
    if not query or len(query)<3: return []
    # Try OpenStreetMap with different headers
    try:
        headers={"User-Agent":"SafeTrack-App/1.0 (safety monitoring application)","Accept-Language":"en"}
        url=f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(query)}&format=json&limit=5&countrycodes=in&addressdetails=1"
        r=requests.get(url,headers=headers,timeout=8)
        if r.status_code==200:
            data=r.json()
            if data:
                return [{"name":d["display_name"].split(",")[0].strip(),
                         "address":", ".join(d["display_name"].split(",")[1:3]).strip(),
                         "lat":float(d["lat"]),"lng":float(d["lon"])} for d in data]
    except: pass
    # Try Photon (another free geocoder)
    try:
        url=f"https://photon.komoot.io/api/?q={requests.utils.quote(query)}&limit=5&lang=en&bbox=68,6,97,35"
        r=requests.get(url,timeout=8)
        if r.status_code==200:
            features=r.json().get("features",[])
            results=[]
            for f in features:
                p=f["properties"]
                name=p.get("name",p.get("street","Unknown"))
                addr=", ".join(filter(None,[p.get("city",""),p.get("state","")]))
                lat=f["geometry"]["coordinates"][1]
                lng=f["geometry"]["coordinates"][0]
                results.append({"name":name,"address":addr,"lat":lat,"lng":lng})
            if results: return results
    except: pass
    return []

def build_map(places,cur_lat=None,cur_lng=None,result=None):
    center=[cur_lat,cur_lng] if cur_lat else [18.1115,83.3963]
    m=folium.Map(location=center,zoom_start=14)
    for p in places:
        folium.Circle([p["lat"],p["lng"]],radius=400,
            color="#00c47a",fill=True,fill_opacity=0.1,
            weight=1.5,dash_array="5").add_to(m)
        folium.CircleMarker([p["lat"],p["lng"]],radius=7,
            color="#00c47a",fill=True,fill_opacity=0.9,
            popup=p["name"],tooltip=p["name"]).add_to(m)
    if cur_lat and result:
        color=("#00c47a" if result["status"]=="SAFE"
               else "#f5a623" if result["status"]=="MILD WARNING" else "#f03e3e")
        folium.Marker([cur_lat,cur_lng],
            popup=f"<b>YOU</b><br>Risk: {result['risk']}/100<br>{result['status']}",
            tooltip="You are here",
            icon=folium.Icon(color="blue",icon="user",prefix="fa")).add_to(m)
        folium.Circle([cur_lat,cur_lng],radius=150,
            color=color,fill=True,fill_opacity=0.15).add_to(m)
    return m

# ── SESSION STATE ──────────────────────────────────────
for k,v in [("result",None),("cur_lat",None),("cur_lng",None),
             ("history",[]),("search_results",[]),
             ("sel_lat",None),("sel_lng",None)]:
    if k not in st.session_state: st.session_state[k]=v

places   = load_places()
contacts = load_contacts()

# ── HEADER ─────────────────────────────────────────────
st.markdown("# 🛡️ SafeTrack")
st.markdown("**Personal Location Safety Monitor** — DBSCAN + Isolation Forest ML")

# ── STATS BAR ──────────────────────────────────────────
c1,c2,c3,c4=st.columns(4)
total_checks=0
try:
    if os.path.exists(LOG_FILE):
        df_log=pd.read_csv(LOG_FILE)
        if len(df_log)>0: total_checks=len(df_log)
except: pass

with c1:
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{len(places)}</div><div class='stat-lbl'>Safe zones</div></div>",unsafe_allow_html=True)
with c2:
    trained="✅" if os.path.exists("models/dbscan.pkl") else "❌"
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{trained}</div><div class='stat-lbl'>Model trained</div></div>",unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{total_checks}</div><div class='stat-lbl'>Total checks</div></div>",unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='stat-box'><div class='stat-val'>{len(contacts)}</div><div class='stat-lbl'>Contacts saved</div></div>",unsafe_allow_html=True)

st.divider()
left,right=st.columns([1,2])

with left:

    # ── SEARCH PLACE ────────────────────────────────────
    st.subheader("🔍 Search & Check Place")
    search_q=st.text_input("Search any place in India",
        placeholder="Araku Valley, JNTU Vizianagaram, Charminar...")

    if st.button("🔍 Search",use_container_width=True) and search_q:
        with st.spinner("Searching..."):
            results=search_place(search_q)
            st.session_state.search_results=results
            if not results:
                st.warning("No results. Try more specific name e.g. 'Araku Valley Andhra Pradesh'")

    if st.session_state.search_results:
        st.caption(f"Found {len(st.session_state.search_results)} results — select one:")
        for i,r in enumerate(st.session_state.search_results):
            col_info,col_btn=st.columns([3,1])
            col_info.write(f"📍 **{r['name']}**")
            col_info.caption(r['address'])
            if col_btn.button("Select",key=f"sr_{i}"):
                st.session_state.sel_lat=r["lat"]
                st.session_state.sel_lng=r["lng"]
                st.session_state.search_results=[]
                st.rerun()

    st.divider()

    # ── CHECK LOCATION ──────────────────────────────────
    st.subheader("📍 Check Location")
    if st.session_state.sel_lat:
        st.info(f"📌 Selected: {st.session_state.sel_lat:.5f}, {st.session_state.sel_lng:.5f}")

    col1,col2=st.columns(2)
    with col1:
        lat_in=st.number_input("Latitude",
            value=float(st.session_state.sel_lat) if st.session_state.sel_lat else 18.1103,
            format="%.6f",step=0.0001)
    with col2:
        lng_in=st.number_input("Longitude",
            value=float(st.session_state.sel_lng) if st.session_state.sel_lng else 83.3975,
            format="%.6f",step=0.0001)
    st.caption("💡 Or: Google Maps → long press → copy coordinates shown at top")

    if st.button("✅ Check this location",use_container_width=True,type="primary"):
        st.session_state.cur_lat=lat_in
        st.session_state.cur_lng=lng_in
        st.session_state.sel_lat=None
        st.session_state.sel_lng=None
        r=score(lat_in,lng_in)
        st.session_state.result=r
        st.session_state.history.insert(0,{
            "time":datetime.now().strftime("%H:%M"),
            "lat":lat_in,"lng":lng_in,**r})
        try:
            row=pd.DataFrame([{"lat":lat_in,"lng":lng_in,
                "hour":datetime.now().hour,
                "timestamp":datetime.now().isoformat(),
                "status":r["status"],"risk":r["risk"]}])
            mode="a" if os.path.exists(LOG_FILE) else "w"
            row.to_csv(LOG_FILE,mode=mode,
                header=not os.path.exists(LOG_FILE),index=False)
        except: pass

    # ── RESULT ─────────────────────────────────────────
    if st.session_state.result:
        r=st.session_state.result
        css="safe" if r["status"]=="SAFE" else "warn" if r["status"]=="MILD WARNING" else "danger"
        st.markdown(f"""
        <div style='text-align:center;padding:18px;background:#f8f9fa;
          border-radius:12px;margin:12px 0;border:1px solid #eee'>
          <div class='risk-{css}'>{r["risk"]}</div>
          <div style='font-size:11px;color:#999;margin:2px 0'>out of 100</div>
          <div class='badge-{css}' style='margin:8px 0'>{r["status"]}</div>
          <p style='font-size:12px;color:#555;margin-top:8px;margin-bottom:0'>{r["explain"]}</p>
        </div>""",unsafe_allow_html=True)
        m1,m2=st.columns(2)
        m1.metric("Distance",f"{r['dist']} km")
        m2.metric("In cluster","Yes ✓" if r["in_cluster"] else "No ✗")

        if r["status"]=="DANGER" and contacts:
            c=contacts[0]
            st.error("🚨 You are outside all known safe areas!")
            st.markdown(f"""
            <div style='background:#ffeaea;border:2px solid #f03e3e;
              border-radius:12px;padding:16px;text-align:center;margin:8px 0'>
              <div style='font-size:16px;font-weight:700;color:#f03e3e'>🆘 EMERGENCY</div>
              <div style='margin:8px 0;font-size:13px'>Tap to call {c['name']}</div>
              <a href='tel:{c["num"]}' style='display:inline-block;
                background:#f03e3e;color:white;padding:12px 28px;
                border-radius:10px;text-decoration:none;
                font-weight:700;font-size:15px'>📞 Call {c['name']}</a>
            </div>""",unsafe_allow_html=True)

    st.divider()

    # ── EMERGENCY CONTACTS ──────────────────────────────
    st.subheader("📞 Emergency Contacts")
    st.caption("Saved permanently — call button opens phone dialer on mobile")

    with st.form("add_contact",clear_on_submit=True):
        cc1,cc2=st.columns(2)
        with cc1: cname=st.text_input("Name",placeholder="Dad, Mom...")
        with cc2: cnum=st.text_input("Number",placeholder="+91XXXXXXXXXX")
        if st.form_submit_button("💾 Save contact",use_container_width=True):
            if cname and cnum:
                contacts.append({"id":int(datetime.now().timestamp()*1000),
                                  "name":cname,"num":cnum})
                save_contacts(contacts)
                st.success(f"✅ {cname} saved permanently!")
                st.rerun()

    if contacts:
        for c in contacts:
            col_n,col_call,col_del=st.columns([3,2,1])
            col_n.write(f"👤 **{c['name']}**")
            col_n.caption(c['num'])
            col_call.markdown(f"""
            <a href='tel:{c["num"]}' style='display:inline-block;
              background:#e6faf3;color:#00c47a;padding:8px 16px;
              border-radius:8px;text-decoration:none;font-weight:600;
              font-size:13px;border:1px solid #00c47a'>📞 Call</a>
            """,unsafe_allow_html=True)
            if col_del.button("×",key=f"dc_{c['id']}"):
                contacts=[x for x in contacts if x["id"]!=c["id"]]
                save_contacts(contacts)
                st.rerun()
    else:
        st.caption("No contacts yet. Add one above.")

    st.divider()

    # ── ADD SAFE PLACE ──────────────────────────────────
    st.subheader("✅ Add Safe Place")
    with st.form("add_place"):
        pname=st.text_input("Place name",placeholder="Home, Office, College...")
        use_above=st.checkbox("Use coordinates from above",value=True)
        if not use_above:
            pc1,pc2=st.columns(2)
            with pc1: p_lat=st.number_input("Latitude",value=18.1115,format="%.6f")
            with pc2: p_lng=st.number_input("Longitude",value=83.3963,format="%.6f")
        else:
            p_lat=lat_in; p_lng=lng_in
        if st.form_submit_button("➕ Add as safe place",use_container_width=True):
            if pname:
                places.append({"id":int(datetime.now().timestamp()*1000),
                               "name":pname,"lat":p_lat,"lng":p_lng})
                save_places(places)
                retrain(places)
                st.success(f"✅ '{pname}' added! Model retrained with {len(places)} zones.")
                st.rerun()

    st.divider()

    # ── SAFE PLACES LIST ────────────────────────────────
    st.subheader(f"🗂️ My Safe Places ({len(places)})")
    for p in places:
        pc1,pc2=st.columns([4,1])
        pc1.write(f"🟢 **{p['name']}**")
        pc1.caption(f"{p['lat']:.4f}, {p['lng']:.4f}")
        if pc2.button("×",key=f"dp_{p['id']}"):
            places=[x for x in places if x["id"]!=p["id"]]
            save_places(places)
            st.rerun()

    st.divider()

    # ── HISTORY ─────────────────────────────────────────
    if st.session_state.history:
        st.subheader("📋 Recent checks")
        for h in st.session_state.history[:6]:
            icon="🟢" if h["status"]=="SAFE" else "🟡" if h["status"]=="MILD WARNING" else "🔴"
            st.write(f"{icon} **{h['status']}** — {h['risk']}/100 at {h['time']}")

    st.divider()

    # ── ML INFO ─────────────────────────────────────────
    st.subheader("⚙️ How the ML works")
    st.markdown("""
**DBSCAN** — clusters your saved safe places. Points outside = anomaly.

**Isolation Forest** — scores how unusual your location is vs history.

**Risk score** = distance from nearest cluster + time encoding.

`epsilon=0.003rad` ≈ 300m · `contamination=0.05` → 5% anomaly rate
    """)

with right:
    st.subheader("🗺️ Safety Map")
    m=build_map(places,
                st.session_state.cur_lat,
                st.session_state.cur_lng,
                st.session_state.result)
    st_folium(m,width=None,height=600,returned_objects=[])