import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ----------------------------
# Load your trained YOLO11m model
# ----------------------------
MODEL_PATH = "best.pt"  # Since app.py and best.pt are in the same folder
model = YOLO(MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Leaf Disease Detection 🌿")
st.write("Upload an image to detect diseased leaves.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Open uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Save temporarily for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img.save(tmp_file.name)
        tmp_path = tmp_file.name

    # Predict
    results = model.predict(source=tmp_path, save=False, conf=0.25, imgsz=640)
    
    # Show prediction
    annotated_img = results[0].plot()  # Image with bounding boxes
    st.image(annotated_img, caption="Predictions", use_column_width=True)

    # Clean up
    os.remove(tmp_path)