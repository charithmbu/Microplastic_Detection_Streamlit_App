import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from PIL import Image

# ---------------- CONFIG ----------------
API_URL = "https://microplastic-detection-backend.onrender.com/detect"
EXAMPLE_DIR = "Example_images"
PIXEL_TO_NM = 100
RISK_THRESHOLD = 15

# ---------------- UI ----------------
st.set_page_config(page_title="Microplastic Detection System", layout="wide")
st.title("🧪 Microplastic Detection System (YOLOv8)")

st.markdown("### 📥 Choose Input Method")

input_mode = st.radio(
    "Select Input Type:",
    ["Upload Image", "Use Example Image", "Capture from Camera"]
)

img = None
img_bytes = None

# ---------------- CAMERA INPUT ----------------
if input_mode == "Capture from Camera":
    camera_image = st.camera_input("Capture image from microscope / camera")
    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        img_bytes = camera_image.getvalue()
        st.image(img, caption="Captured Image")

# ---------------- EXAMPLE IMAGE ----------------
elif input_mode == "Use Example Image":
    if not os.path.exists(EXAMPLE_DIR):
        st.error("Example_images folder not found.")
    else:
        example_images = sorted(os.listdir(EXAMPLE_DIR))
        selected_image = st.selectbox("Select an example image:", example_images)

        img_path = os.path.join(EXAMPLE_DIR, selected_image)
        img = Image.open(img_path).convert("RGB")
        st.image(img, caption=f"Example Image: {selected_image}")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

# ---------------- UPLOAD IMAGE ----------------
else:
    uploaded_file = st.file_uploader(
        "Upload Microscopic Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_bytes = uploaded_file.read()
        st.image(img, caption="Uploaded Image")

# ---------------- API CALL ----------------
if img_bytes is not None:
    st.subheader("🚀 Running Detection...")

    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            timeout=60
        )
    except Exception as e:
        st.error("❌ Could not connect to Detection API")
        st.exception(e)
        st.stop()

    if response.status_code != 200:
        st.error("❌ Error from detection API")
        st.text(response.text)
        st.stop()

    data = response.json()

    # ---------------- RESPONSE PARSING ----------------
    total_count = data.get("total_count", 0)
    boxes = data.get("boxes", [])
    status = data.get("status", "UNKNOWN")
    risk_score = data.get("risk_score", 0)

    # ---------------- SUMMARY ----------------
    st.subheader("📊 Detection Summary")
    st.write(f"Total Microplastics Detected: **{total_count}**")
    st.write(f"Risk Score: **{risk_score}**")
    st.write(f"Final Status: **{status}**")

    # ---------------- SIZE CALCULATION ----------------
    sizes_nm = []

    st.subheader("📐 Individual Microplastic Sizes (nm)")

    for i, box in enumerate(boxes, start=1):
        width_px = box.get("width", 0)
        height_px = box.get("height", 0)

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

    # ---------------- SIZE STATS & GRAPH ----------------
    if sizes_nm:
        min_size = min(sizes_nm)
        max_size = max(sizes_nm)
        avg_size = sum(sizes_nm) / len(sizes_nm)

        min_thresh = min_size * 1.10
        max_thresh = max_size * 0.90

        min_count = sum(s <= min_thresh for s in sizes_nm)
        max_count = sum(s >= max_thresh for s in sizes_nm)
        avg_count = total_count - min_count - max_count

        st.subheader("📦 Size Category Counts")
        st.write(f"Min Size: **{min_count}**")
        st.write(f"Average Size: **{avg_count}**")
        st.write(f"Max Size: **{max_count}**")

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
