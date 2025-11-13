import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ----------------------------
# Load YOLO model
# ----------------------------
MODEL_PATH = "best.pt"
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

    # Show annotated image
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Predictions", use_column_width=True)

    # Show prediction results
    st.subheader("Prediction Results")
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        predictions = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            conf = float(box.conf[0].item())
            predictions.append(f"{label} ({conf:.2f})")
            st.markdown(f"**Predicted Label:** {label} ({conf:.2f})")

        # Join all predictions into one hidden span for Selenium
        joined_preds = "; ".join(predictions)
        st.markdown(
            f"<span id='prediction-text'>Predicted Labels: {joined_preds}</span>",
            unsafe_allow_html=True
        )
    else:
        st.write("No objects detected.")
        st.markdown(
            "<span id='prediction-text'>Predicted Labels: unknown (0.00)</span>",
            unsafe_allow_html=True
        )


    # Clean up
    os.remove(tmp_path)
