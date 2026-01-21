import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
# from PIL import Image


# ---------------- CONFIG ----------------
MODEL_PATH = "Microplastic_Yolov8_Model.pt"
EXAMPLE_DIR = "Example_images"
PIXEL_TO_NM = 100
RISK_THRESHOLD = 15

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- UI ----------------
st.set_page_config(page_title="Microplastic Detection System", layout="wide")
st.title("🧪 Microplastic Detection System (YOLOv8)")

st.markdown("### 📥 Choose Input Method")

input_mode = st.radio(
    "Select Input Type:",
    ["Upload Image", "Use Example Image", "Capture from Camera"]
)

img = None

# ---------------- CAMERA INPUT ----------------
if input_mode == "Capture from Camera":
    camera_image = st.camera_input("Capture image from microscope / camera")

    if camera_image:
        img_bytes = np.frombuffer(camera_image.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        st.image(img, channels="BGR", caption="Captured Image")

# ---------------- EXAMPLE IMAGE ----------------
elif input_mode == "Use Example Image":
    example_images = sorted(os.listdir(EXAMPLE_DIR))
    selected_image = st.selectbox("Select an example image:", example_images)

    img_path = os.path.join(EXAMPLE_DIR, selected_image)
    img = cv2.imread(img_path)
    st.image(img, channels="BGR", caption=f"Example Image: {selected_image}")

# ---------------- UPLOAD IMAGE ----------------
else:
    uploaded_file = st.file_uploader(
        "Upload Microscopic Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        st.image(img, channels="BGR", caption="Uploaded Image")

# ---------------- YOLO PROCESSING ----------------
if img is not None:
    results = model(img)
    boxes = results[0].boxes
    total_count = len(boxes)

    annotated = results[0].plot()
    st.image(annotated, channels="BGR", caption="Detected Microplastics")

    st.subheader("📊 Detection Summary")
    st.write(f"Total Microplastics Detected: **{total_count}**")

    sizes_nm = []

    st.subheader("📐 Individual Microplastic Sizes (nm)")

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0]

        width_px = float(x2 - x1)
        height_px = float(y2 - y1)

        width_nm = width_px * PIXEL_TO_NM
        height_nm = height_px * PIXEL_TO_NM
        size_nm = np.sqrt(width_nm * height_nm)

        sizes_nm.append(size_nm)

        st.write(
            f"Microplastic {i}: "
            f"Width = {width_nm:.1f} nm | "
            f"Height = {height_nm:.1f} nm | "
            f"Size ≈ {size_nm:.1f} nm"
        )

    # ---------------- STATS & RISK ----------------
    if sizes_nm:
        min_size = min(sizes_nm)
        max_size = max(sizes_nm)
        avg_size = sum(sizes_nm) / len(sizes_nm)

        min_thresh = min_size * 1.10
        max_thresh = max_size * 0.90

        min_count = sum(s <= min_thresh for s in sizes_nm)
        max_count = sum(s >= max_thresh for s in sizes_nm)
        avg_count = total_count - min_count - max_count

        risk_score = (min_count * 3) + (avg_count * 2) + (max_count * 1)
        status = "UNSAFE ⚠️" if risk_score >= RISK_THRESHOLD else "SAFE ✅"

        st.subheader("📦 Size Category Counts")
        st.write(f"Min Size: **{min_count}**")
        st.write(f"Average Size: **{avg_count}**")
        st.write(f"Max Size: **{max_count}**")

        st.subheader("🚦 Safety Status")
        st.write(f"Risk Score: **{risk_score}**")
        st.write(f"Final Status: **{status}**")

        # ---------------- BAR GRAPH ----------------
        labels = ["Min Size", "Average Size", "Max Size"]
        counts = [min_count, avg_count, max_count]

        fig, ax = plt.subplots()
        ax.bar(labels, counts)
        ax.set_ylabel("Count")
        ax.set_title("Microplastic Size Distribution (Count-Based)")

        for i, v in enumerate(counts):
            ax.text(i, v, str(v), ha="center", va="bottom")

        st.pyplot(fig)
    else:
        st.info("No microplastics detected.")
