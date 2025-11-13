import pandas as pd
from pyproj import Transformer

# Load simple marker GPS file
df = pd.read_csv("markers_raw.csv")

# ✅ UTM Zone 43N for IIT Mandi
transformer = Transformer.from_crs("epsg:4326", "epsg:32643", always_xy=True)

# Use first marker as local origin
origin_lat = df.loc[0, "lat"]
origin_lon = df.loc[0, "lon"]

origin_x, origin_y = transformer.transform(origin_lon, origin_lat)

x_list = []
y_list = []

for _, r in df.iterrows():
    lon = r["lon"]
    lat = r["lat"]

    x, y = transformer.transform(lon, lat)

    x_list.append(x - origin_x)
    y_list.append(y - origin_y)

df["x_local"] = x_list
df["y_local"] = y_list
df["z_local"] = 0.0        # All markers on ground

df.to_csv("markers.csv", index=False)
print("✅ markers.csv generated (no size, no heading)")
