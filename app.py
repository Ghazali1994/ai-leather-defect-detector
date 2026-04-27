```python
# app.py
# Real YOLOv8 Leather Defect Detector (Streamlit)
# Run: streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Leather Defect Detector",
    page_icon="👜",
    layout="wide"
)

st.title("👜 Leather Defect Detector (YOLOv8)")
st.write("Upload a leather image and detect defects using a trained YOLOv8 model.")

# ---------------------------------
# Load model
# Replace with your trained model
# ---------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained weights file

model = load_model()

# ---------------------------------
# Upload image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload Leather Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # -----------------------------
    # Prediction
    # -----------------------------
    with st.spinner("Detecting defects..."):
        results = model.predict(
            source=img_array,
            conf=0.30,
            save=False
        )

    result = results[0]

    plotted = result.plot()   # image with boxes

    with col2:
        st.subheader("Detected Defects")
        st.image(plotted, use_container_width=True)

    # -----------------------------
    # Inspection Report
    # -----------------------------
    st.subheader("Inspection Report")

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        total_defects = len(boxes)

        for i, box in enumerate(boxes):

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            st.write(
                f"{i+1}. {label} | Confidence: {conf*100:.1f}%"
            )

        # Grade logic
        if total_defects == 0:
            grade = "A"
        elif total_defects <= 2:
            grade = "B"
        else:
            grade = "C"

        st.success(f"Leather Grade: {grade}")

    else:
        st.success("No defects detected. Leather Grade: A")

else:
    st.info("Upload an image to start inspection.")
```
