import json
import os
import math
import simplekml

MARKER_CSV = "markers.csv"
ARUCO_DIR = "flight_data"
OUTPUT_KML = "flight_visual.kml"

# Load markers
import csv
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

# Load all metadata JSON files
metadatas = []
for file in sorted(os.listdir(f"{ARUCO_DIR}/metadata")):
    if file.endswith(".json"):
        with open(f"{ARUCO_DIR}/metadata/{file}") as f:
            meta = json.load(f)
            meta["name"] = file.replace(".json", "")
            metadatas.append(meta)

kml = simplekml.Kml()

# Add marker points
for mid, m in markers.items():
    p = kml.newpoint(name=f"Marker {mid}", coords=[(m["lon"], m["lat"], m["alt"])])

# Add drone flight path
coords_path = []
for m in metadatas:
    if m["lat"] is not None and m["lon"] is not None:
        coords_path.append((m["lon"], m["lat"], m["alt"]))

line = kml.newlinestring(name="Drone Path", coords=coords_path)
line.style.linestyle.color = simplekml.Color.red
line.style.linestyle.width = 3

# Add distance circles
for meta in metadatas:
    if "distances_to_markers" not in meta:
        continue

    for mid, dist in meta["distances_to_markers"].items():
        m = markers[int(mid)]
        circle = kml.newpolygon(name=f"{meta['name']} to marker {mid}")
        circle.outerboundaryis = [
            (
                m["lon"] + (dist/111320)*math.cos(math.radians(a)),
                m["lat"] + (dist/110540)*math.sin(math.radians(a))
            )
            for a in range(0, 360, 5)
        ]
        circle.style.polystyle.color = simplekml.Color.changealphaint(40, simplekml.Color.blue)

# Save
kml.save(OUTPUT_KML)
print("✅ KML exported:", OUTPUT_KML)
