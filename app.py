from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import haversine_distances

app = Flask(__name__)

DATA_FILE   = "data/live_log.csv"
PLACES_FILE = "data/safe_places.json"
MODEL_DIR   = "models"
DBSCAN_EPS  = 0.003
ISO_CONT    = 0.05

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Safe places: load/save to JSON ──────────────────────
def load_places():
    if os.path.exists(PLACES_FILE):
        with open(PLACES_FILE, "r") as f:
            return json.load(f)
    return []

def save_places(places):
    with open(PLACES_FILE, "w") as f:
        json.dump(places, f, indent=2)

# ── Models ───────────────────────────────────────────────
def load_models():
    db_path  = os.path.join(MODEL_DIR, "dbscan.pkl")
    iso_path = os.path.join(MODEL_DIR, "isoforest.pkl")
    db  = joblib.load(db_path)  if os.path.exists(db_path)  else None
    iso = joblib.load(iso_path) if os.path.exists(iso_path) else None
    return db, iso

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLng = np.radians(lng2 - lng1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLng/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def score_point(lat, lng, hour):
    places = load_places()

    # ── Score from safe places (always works) ───────────
    if places:
        dists = [haversine_km(lat, lng, p["lat"], p["lng"]) for p in places]
        min_dist = min(dists)
        nearest  = places[dists.index(min_dist)]
        in_cluster = min_dist < 0.4

        if in_cluster:
            risk = max(5, min_dist * 100)
            status = "SAFE"
            explain = f'You are near "{nearest["name"]}" ({int(min_dist*1000)}m away). This is a known safe cluster.'
        elif min_dist < 1.5:
            risk = 35 + min_dist * 10
            status = "MILD WARNING"
            explain = f'You are {min_dist:.2f} km from "{nearest["name"]}". You may be travelling between safe zones.'
        else:
            risk = min(95, 50 + min_dist * 8)
            status = "DANGER"
            explain = f'You are {min_dist:.2f} km from "{nearest["name"]}". You are outside all known safe clusters!'
    else:
        min_dist = 999
        in_cluster = False
        risk = 50
        status = "MILD WARNING"
        explain = "No safe places saved yet. Go to a safe location and tap 'I am safe here'."
        nearest = None

    # ── Also check ML model if available ────────────────
    db, iso = load_models()
    iso_score = None
    if iso is not None:
        lat_r = np.radians(lat); lng_r = np.radians(lng)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        point = np.array([[lat_r, lng_r, hour_sin, hour_cos]])
        iso_score = float(iso.decision_function(point)[0])
        iso_pred  = int(iso.predict(point)[0])
        # if model says anomaly AND you are far, elevate risk
        if iso_pred == -1 and min_dist > 0.5:
            risk = min(95, risk + 15)

    return {
        "lat"       : lat,
        "lng"       : lng,
        "hour"      : hour,
        "dist_km"   : round(min_dist, 3),
        "in_cluster": in_cluster,
        "risk_score": round(risk, 1),
        "status"    : status,
        "explain"   : explain,
        "iso_score" : iso_score
    }

def retrain(places):
    if len(places) < 2:
        return False, "Need at least 2 safe places to train"

    rows = []
    for p in places:
        # simulate 50 points around each safe place
        for _ in range(50):
            rows.append({
                "lat" : p["lat"] + np.random.normal(0, 0.003),
                "lng" : p["lng"] + np.random.normal(0, 0.003),
                "hour": np.random.randint(7, 22)
            })

    # also add any logged history
    if os.path.exists(DATA_FILE):
        df_log = pd.read_csv(DATA_FILE).tail(200)
        rows.extend(df_log[["lat","lng","hour"]].to_dict("records"))

    df = pd.DataFrame(rows)
    df["lat_rad"]  = np.radians(df["lat"])
    df["lng_rad"]  = np.radians(df["lng"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    X = df[["lat_rad","lng_rad","hour_sin","hour_cos"]].values

    db  = DBSCAN(eps=DBSCAN_EPS, min_samples=5, metric="haversine")
    db.fit(X[:, :2])
    iso = IsolationForest(contamination=ISO_CONT, random_state=42)
    iso.fit(X)

    joblib.dump(db,  os.path.join(MODEL_DIR, "dbscan.pkl"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "isoforest.pkl"))
    return True, f"Model trained on {len(places)} safe places ({len(df)} points)"

# ── Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()

@app.route("/api/score", methods=["POST"])
def api_score():
    data = request.json
    lat  = float(data["lat"])
    lng  = float(data["lng"])
    hour = int(datetime.now().hour)
    result = score_point(lat, lng, hour)

    # log to CSV
    row = pd.DataFrame([{
        "lat": lat, "lng": lng, "hour": hour,
        "timestamp": datetime.now().isoformat(),
        "status": result["status"],
        "risk": result["risk_score"]
    }])
    if os.path.exists(DATA_FILE):
        row.to_csv(DATA_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(DATA_FILE, index=False)

    return jsonify(result)

@app.route("/api/places", methods=["GET"])
def api_get_places():
    return jsonify(load_places())

@app.route("/api/places", methods=["POST"])
def api_add_place():
    data   = request.json
    places = load_places()
    place  = {
        "id"     : int(datetime.now().timestamp() * 1000),
        "name"   : data["name"],
        "lat"    : float(data["lat"]),
        "lng"    : float(data["lng"]),
        "addedAt": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }
    places.append(place)
    save_places(places)
    # auto retrain when new place added
    retrain(places)
    return jsonify({"success": True, "place": place, "total": len(places)})

@app.route("/api/places/<int:place_id>", methods=["DELETE"])
def api_delete_place(place_id):
    places = [p for p in load_places() if p["id"] != place_id]
    save_places(places)
    return jsonify({"success": True, "remaining": len(places)})

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    places = load_places()
    success, msg = retrain(places)
    return jsonify({"success": success, "message": msg})

@app.route("/api/history", methods=["GET"])
def api_history():
    if not os.path.exists(DATA_FILE):
        return jsonify([])
    return jsonify(pd.read_csv(DATA_FILE).tail(50).to_dict(orient="records"))

@app.route("/api/stats", methods=["GET"])
def api_stats():
    places = load_places()
    if not os.path.exists(DATA_FILE):
        return jsonify({"total": 0, "safe": 0, "anomalies": 0, "places": len(places)})
    df = pd.read_csv(DATA_FILE)
    return jsonify({
        "total"    : len(df),
        "safe"     : int((df["status"] == "SAFE").sum()),
        "anomalies": int((df["status"] == "DANGER").sum()),
        "places"   : len(places)
    })

if __name__ == "__main__":
    print("SafeTrack starting...")
    print(f"Safe places loaded: {len(load_places())}")
    db, iso = load_models()
    print(f"ML model: {'loaded' if iso else 'not found (will use distance scoring)'}")
    app.run(debug=True, port=5000)
