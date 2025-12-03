import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Image Preprocessing App",
    layout="wide"
)

st.title("🖼 Image Preprocessing App")
st.write("Upload an image and apply preprocessing techniques")

# ---------------- HELPER FUNCTIONS ----------------
def pil_to_cv2(img):
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.warning("Please upload an image.")
    st.stop()

# Load image
image = Image.open(uploaded_file).convert("RGB")
img = pil_to_cv2(image)

st.subheader("Original Image")
st.image(image, use_container_width=True)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙ Image Controls")

width = st.sidebar.slider("Width", 100, 1500, img.shape[1])
height = st.sidebar.slider("Height", 100, 1500, img.shape[0])

angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)

flip = st.sidebar.selectbox(
    "Flip",
    ["None", "Horizontal", "Vertical"]
)

brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast = st.sidebar.slider("Contrast", -50, 50, 0)

grayscale = st.sidebar.checkbox("Grayscale")
blur = st.sidebar.slider("Blur", 0, 25, 0, step=2)
edges = st.sidebar.checkbox("Edge Detection")

# ---------------- IMAGE PROCESSING ----------------
processed = cv2.resize(img, (width, height))

# Rotation
if angle != 0:
    h, w = processed.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    processed = cv2.warpAffine(processed, matrix, (w, h))

# Flip
if flip == "Horizontal":
    processed = cv2.flip(processed, 1)
elif flip == "Vertical":
    processed = cv2.flip(processed, 0)

# Brightness & Contrast
processed = cv2.convertScaleAbs(
    processed,
    alpha=1 + (contrast / 50),
    beta=brightness
)

# Grayscale
if grayscale:
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Blur
if blur > 0:
    processed = cv2.GaussianBlur(
        processed, (blur + 1, blur + 1), 0
    )

# Edge Detection
if edges:
    edge_img = cv2.Canny(processed, 100, 200)
    processed = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)

# ---------------- OUTPUT ----------------
st.subheader("Processed Image")
output_image = cv2_to_pil(processed)
st.image(output_image, use_container_width=True)

# ---------------- DOWNLOAD ----------------
buffer = io.BytesIO()
output_image.save(buffer, format="PNG")

st.download_button(
    label="⬇ Download Image",
    data=buffer.getvalue(),
    file_name="processed_image.png",
    mime="image/png"
)
