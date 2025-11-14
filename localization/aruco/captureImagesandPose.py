import cv2
import os
import time
import threading
import math
from datetime import datetime
from pymavlink import mavutil

# --- Configuration ---
OUTPUT_DIR = "RTData"
RESIZE = (1024, 1024)
EXT = ".jpg"

CAPTURE_INTERVAL_SEC = 2
SHOW_PREVIEW = True
PX4_PORT = "COM11"       # Fixed Pixhawk port
PX4_BAUDRATE = 57600

latest_gps = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'timestamp': 0.0}
gps_lock = threading.Lock()

# Reference GPS (first image)
reference_gps = None

# Haversine formula to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c  # Distance in meters

def setup_camera():
    print("🔧 Setting up camera...")
    for i in [1, 2, 3]:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        cap.release()
    print("❌ No camera found")
    return None

def connect_to_pixhawk():
    print(f"🔧 Connecting to Pixhawk on {PX4_PORT}...")
    try:
        master = mavutil.mavlink_connection(PX4_PORT, baud=PX4_BAUDRATE)
        master.wait_heartbeat(timeout=10)
        print("✅ Pixhawk connected (Heartbeat OK)")
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1
        )
        return master
    except Exception as e:
        print(f"❌ Pixhawk connection error: {e}")
        return None

def pixhawk_gps_listener(master):
    """Thread to continuously read GPS data."""
    global latest_gps
    try:
        while True:
            msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
            if msg:
                with gps_lock:
                    latest_gps['lat'] = msg.lat / 1e7
                    latest_gps['lon'] = msg.lon / 1e7
                    latest_gps['alt'] = msg.alt / 1000.0
                    latest_gps['timestamp'] = time.time()
    except Exception as e:
        print(f"⚠️ GPS listener stopped: {e}")

def save_gps_data(csv_path, filename, gps_data, distance=None):
    """Append GPS data and distance to CSV."""
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if new_file:
            f.write("image_filename,lat,lon,alt,distance_from_first_m,timestamp\n")
        dist_str = f"{distance:.2f}" if distance is not None else ""
        f.write(f"{filename},{gps_data['lat']:.7f},{gps_data['lon']:.7f},{gps_data['alt']:.2f},{dist_str},{gps_data['timestamp']:.3f}\n")

def capture_images():
    global reference_gps
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "gps_data.csv")

    cap = setup_camera()
    if cap is None:
        return

    master = connect_to_pixhawk()
    pixhawk_connected = master is not None

    if pixhawk_connected:
        listener_thread = threading.Thread(target=pixhawk_gps_listener, args=(master,), daemon=True)
        listener_thread.start()

    print(f"📁 Saving images in: {OUTPUT_DIR}")
    print(f"⏰ Capturing every {CAPTURE_INTERVAL_SEC} seconds")
    if pixhawk_connected:
        print("📡 Logging GPS (lat, lon, alt) and distance")
    print("🛑 Press 'q' to quit\n")

    img_idx = 1
    last_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()
            if now - last_time >= CAPTURE_INTERVAL_SEC:
                filename = f"img_{img_idx:04d}{EXT}"
                path = os.path.join(OUTPUT_DIR, filename)

                resized = cv2.resize(frame, RESIZE)
                cv2.imwrite(path, resized)
                print(f"💾 Saved: {filename}")

                distance = None
                if pixhawk_connected:
                    with gps_lock:
                        gps_data = latest_gps.copy()

                    # Set reference for first image
                    if reference_gps is None:
                        reference_gps = gps_data.copy()
                        distance = 0.0
                    else:
                        distance = haversine(
                            reference_gps['lat'], reference_gps['lon'],
                            gps_data['lat'], gps_data['lon']
                        )

                    save_gps_data(csv_path, filename, gps_data, distance)
                    print(f"   🌍 Lat: {gps_data['lat']:.7f}, Lon: {gps_data['lon']:.7f}, Alt: {gps_data['alt']:.2f} m, Dist: {distance:.2f} m")

                img_idx += 1
                last_time = now

            if SHOW_PREVIEW:
                cv2.imshow("Camera Preview", cv2.resize(frame, (640, 480)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (Ctrl+C)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera released")
        print(f"📊 Total images captured: {img_idx - 1}")

if __name__ == "__main__":
    capture_images()
