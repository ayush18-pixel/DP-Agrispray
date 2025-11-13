import cv2
import os
import time
import threading
import json
import math
import csv
from pymavlink import mavutil
import sys
import serial.tools.list_ports

# >>> NEW: local conversion
try:
    from pyproj import Transformer
except ImportError:
    print("❌ pyproj not installed. Run: pip install pyproj")
    sys.exit(1)

# ==========================
# CONFIGURATION
# ==========================
SAVE_DIR = "flight_data"
IMAGE_RES = (1280, 720)
FREQ = 0.033                     # capture interval in seconds (≈30 FPS)
BAUD = 57600
SHOW_TELEMETRY = True
SHOW_ARUCO_BOXES = True
MARKER_CSV_PATH = "markers.csv"  # put your CSV next to this script

# ArUco dictionary setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

latest = {
    "lat": None, "lon": None, "alt": None,
    "yaw": None, "pitch": None, "roll": None,
    "timestamp": None
}

aruco_events = []

os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/metadata", exist_ok=True)

# ==========================
# UTILS
# ==========================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# >>> NEW: GPS <-> Local conversion (IIT Mandi = UTM 43N, EPSG:32643)
transformer = Transformer.from_crs("epsg:4326", "epsg:32643", always_xy=True)

def gps_to_local(lon, lat, origin_lon, origin_lat):
    """Return (x_local, y_local) in meters relative to origin."""
    x, y = transformer.transform(lon, lat)
    x0, y0 = transformer.transform(origin_lon, origin_lat)
    return x - x0, y - y0

# ==========================
# LOAD MARKERS (lat, lon, alt) + compute local XY
# ==========================
if not os.path.exists(MARKER_CSV_PATH):
    print(f"❌ {MARKER_CSV_PATH} not found. Create it with your marker coordinates.")
    sys.exit(1)

markers = {}            # id -> dict(lat, lon, alt)
with open(MARKER_CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mid = int(row["id"])
        markers[mid] = {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "alt": float(row.get("alt", 0.0))
        }

if not markers:
    print("❌ No markers loaded from CSV.")
    sys.exit(1)

# Use the smallest ID as origin (you can change to any fixed ID)
origin_id = sorted(markers.keys())[0]
origin_lat = markers[origin_id]["lat"]
origin_lon = markers[origin_id]["lon"]
origin_alt = markers[origin_id]["alt"]

# Precompute local coords for markers
marker_locals = {}
for mid, m in markers.items():
    mx, my = gps_to_local(m["lon"], m["lat"], origin_lon, origin_lat)
    mz = m["alt"] - origin_alt
    marker_locals[mid] = {"x": mx, "y": my, "z": mz}

print("✅ Markers loaded (local coords, origin = ID {}):".format(origin_id))
for mid in sorted(marker_locals.keys()):
    ml = marker_locals[mid]
    print(f"  ID {mid}: x={ml['x']:.2f} m, y={ml['y']:.2f} m, z={ml['z']:.2f} m")

# ==========================
# AUTODETECT PIXHAWK PORT
# ==========================
def find_pixhawk_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if ("PX4" in p.description) or ("FMU" in p.description) or ("Pixhawk" in p.description):
            return p.device
    # fallback
    for port in ["COM11"]:
        try:
            s = serial.Serial(port)
            s.close()
            return port
        except:
            pass
    return None

# ==========================
# CONNECT PIXHAWK
# ==========================
PX4_PORT = find_pixhawk_port()
pixhawk = None

if PX4_PORT:
    print(f"✅ Connecting to Pixhawk on {PX4_PORT}...")
    try:
        pixhawk = mavutil.mavlink_connection(PX4_PORT, baud=BAUD)
        pixhawk.wait_heartbeat(timeout=10)
        print("✅ Pixhawk connected")
        pixhawk.mav.request_data_stream_send(
            pixhawk.target_system, pixhawk.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1
        )
    except Exception as e:
        print("❌ Failed to connect to Pixhawk:", e)
        pixhawk = None
else:
    print("⚠️ Pixhawk NOT found → Running camera-only mode")

# ==========================
# TELEMETRY THREAD
# ==========================
def telemetry_listener():
    global latest
    if pixhawk is None:
        return

    while True:
        msg = pixhawk.recv_match(blocking=True, timeout=1)
        if not msg:
            continue

        t = time.time()

        if msg.get_type() == "GLOBAL_POSITION_INT":
            latest["lat"] = msg.lat / 1e7
            latest["lon"] = msg.lon / 1e7
            latest["alt"] = msg.relative_alt / 1000.0  # AGL-ish
            latest["timestamp"] = t

        elif msg.get_type() == "GPS_RAW_INT":
            # If you prefer absolute alt from GPS ellipsoid, uncomment:
            # latest["alt"] = msg.alt / 1000.0
            latest["lat"] = msg.lat / 1e7
            latest["lon"] = msg.lon / 1e7
            # Keep alt from GLOBAL_POSITION_INT if available
            latest["timestamp"] = t

        elif msg.get_type() == "ATTITUDE":
            latest["yaw"] = msg.yaw
            latest["pitch"] = msg.pitch
            latest["roll"] = msg.roll

if pixhawk:
    threading.Thread(target=telemetry_listener, daemon=True).start()

# ==========================
# CAMERA SETUP
# ==========================
print("\n🎥 Searching for camera...")
cam = None
for i in range(5):
    test = cv2.VideoCapture(i)
    if test.isOpened():
        cam = test
        print(f"✅ Camera found at index {i}")
        break
    test.release()

if cam is None:
    print("❌ No camera found. Exiting.")
    sys.exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_RES[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_RES[1])
time.sleep(1)

# ==========================
# MAIN LOOP
# ==========================
idx, last = 0, 0
print("\n✅ Capture started (press 'q' to stop)\n")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # ---------- ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            for cid, corner in zip(ids.flatten(), corners):
                cx = int(corner[:, 0].mean())
                cy = int(corner[:, 1].mean())
                aruco_events.append({
                    "id": int(cid),
                    "pixel_center": [cx, cy],
                    "timestamp": time.time(),
                    "telemetry": latest.copy()
                })
            if SHOW_ARUCO_BOXES:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # ---------- HUD
        if SHOW_TELEMETRY:
            t = latest
            text = f"Lat:{t['lat']} Lon:{t['lon']} Alt:{t['alt']} | Y:{t['yaw']} P:{t['pitch']} R:{t['roll']} | ArUco:{len(aruco_events)}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

        # ---------- SAVE per FREQ
        now = time.time()
        if now - last >= FREQ:
            img_name = f"img_{idx:04d}.jpg"
            cv2.imwrite(f"{SAVE_DIR}/images/{img_name}", frame)

            # Build per-image metadata
            meta = latest.copy()

            # >>> NEW: compute drone local coords + distances to each marker
            x_local = y_local = z_local = None
            distances = {}

            if meta["lat"] is not None and meta["lon"] is not None:
                # Drone local XY relative to origin marker
                x_local, y_local = gps_to_local(meta["lon"], meta["lat"], origin_lon, origin_lat)
                # Use current AGL-ish as z_local (markers z=0)
                if meta["alt"] is not None:
                    z_local = meta["alt"]
                else:
                    z_local = None

                # Distances to each marker in LOCAL 3D (meters)
                for mid, ml in marker_locals.items():
                    if z_local is None:
                        d = math.hypot(x_local - ml["x"], y_local - ml["y"])
                    else:
                        dx = x_local - ml["x"]
                        dy = y_local - ml["y"]
                        dz = z_local - ml["z"]  # ml['z'] is 0 if markers on ground
                        d = math.sqrt(dx*dx + dy*dy + dz*dz)
                    distances[str(mid)] = round(d, 3)

            meta["x_local"] = None if x_local is None else round(x_local, 3)
            meta["y_local"] = None if y_local is None else round(y_local, 3)
            meta["z_local"] = None if z_local is None else round(z_local, 3)
            meta["distances_to_markers"] = distances  # {"1": d1, ..., "5": d5}

            with open(f"{SAVE_DIR}/metadata/{img_name.replace('.jpg','.json')}", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"📸 Saved {img_name} | locals: ({meta['x_local']}, {meta['y_local']}, {meta['z_local']}) | dists: {distances}")
            idx += 1
            last = now

        # ---------- PREVIEW
        cv2.imshow("Drone Camera", cv2.resize(frame, (800, 450)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped manually")

finally:
    cam.release()
    cv2.destroyAllWindows()

    with open(f"{SAVE_DIR}/aruco_events.json", "w") as f:
        json.dump(aruco_events, f, indent=2)

    print("\n✅ Capture stopped")
    print(f"✅ Total ArUco events: {len(aruco_events)}")
    print(f"📁 Saved at: {SAVE_DIR}")
