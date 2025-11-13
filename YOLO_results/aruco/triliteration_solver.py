import json
import os
import numpy as np
from pyproj import Transformer
import csv
import time

MARKER_CSV = "markers.csv"
META_DIR = "flight_data/metadata"
OUT_CSV = "corrected_positions.csv"
OUT_JSON = "corrected_positions.json"

# Load markers
markers = {}
with open(MARKER_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mid = int(row["id"])
        markers[mid] = {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "alt": float(row["alt"])
        }

# Convert markers to local (origin = marker #1)
origin_id = sorted(markers.keys())[0]
origin = markers[origin_id]

transformer = Transformer.from_crs("epsg:4326", "epsg:32643", always_xy=True)

def gps_to_local(m):
    x, y = transformer.transform(m["lon"], m["lat"])
    ox, oy = transformer.transform(origin["lon"], origin["lat"])
    return np.array([x - ox, y - oy, m["alt"] - origin["alt"]], float)

marker_locals = {mid: gps_to_local(m) for mid, m in markers.items()}

# Trilateration solver
def solve_position(marker_pts, distances):
    A = []
    b = []

    keys = list(marker_pts.keys())
    x1 = marker_pts[keys[0]]
    d1 = distances[keys[0]]

    for k in keys[1:]:
        xi = marker_pts[k]
        di = distances[k]

        A.append(2*(xi - x1))
        b.append(d1*d1 - di*di - np.dot(x1, x1) + np.dot(xi, xi))

    A = np.array(A)
    b = np.array(b)

    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p

results = []

for file in sorted(os.listdir(META_DIR)):
    if not file.endswith(".json"):
        continue

    with open(f"{META_DIR}/{file}") as f:
        meta = json.load(f)

    if "distances_to_markers" not in meta:
        continue

    d = {int(k): float(v) for k, v in meta["distances_to_markers"].items()}

    try:
        pos_local = solve_position(marker_locals, d)
    except:
        continue

    # Convert BACK to GPS
    ox, oy = transformer.transform(origin["lon"], origin["lat"])
    lon, lat = transformer.transform(pos_local[0] + ox, pos_local[1] + oy, direction="INVERSE")

    # --- Add local coordinates ---
    x_local = float(pos_local[0])
    y_local = float(pos_local[1])
    z_local = float(pos_local[2])

    # --- Add timestamp (taken from metadata file) ---
    timestamp = meta.get("timestamp", None)

    results.append({
        "image": file.replace(".json", ""),
        "lat": lat,
        "lon": lon,
        "alt": pos_local[2],
        "x_local": x_local,
        "y_local": y_local,
        "z_local": z_local,
        "timestamp": timestamp
    })

# Save JSON
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

# Save CSV
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "image", "lat", "lon", "alt", "x_local", "y_local", "z_local", "timestamp"
    ])
    w.writeheader()
    for r in results:
        w.writerow(r)

print("✅ Trilateration completed.")
print("📁 JSON:", OUT_JSON)
print("📁 CSV:", OUT_CSV)
