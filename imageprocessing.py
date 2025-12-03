import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

st.title("📸 Basic Image Processing App – Deep Learning Practical")
st.write("Upload an image and perform basic image processing operations.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original Image", use_column_width=True)

    st.subheader("1️⃣ Convert to Black & White")
    if st.button("Convert"):
        gray = img.convert("L")
        st.image(gray, caption="Black & White Image")

    st.subheader("2️⃣ Show Image Properties")
    if st.button("Show Properties"):
        st.write("Format: RGB (Converted)")
        st.write("Size:", img.size)
        st.write("Mode:", img.mode)

    st.subheader("3️⃣ Rotate Image")
    angle = st.selectbox("Select rotation", [90, 180, 270])
    if st.button("Rotate"):
        rotated = img.rotate(angle, expand=True)
        st.image(rotated, caption=f"Rotated {angle}°")

    st.subheader("4️⃣ Mirror the Image")
    if st.button("Mirror Image"):
        mirror = ImageOps.mirror(img)
        st.image(mirror, caption="Mirror Image")

    st.subheader("5️⃣ Object Detection (Edges)")
    if st.button("Detect Edges"):
        gray = img.convert("L")
        arr = np.array(gray)
        sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
        edges = np.abs(np.convolve(arr.flatten(), sobel_x.flatten(), mode='same'))
        edges = edges.reshape(arr.shape)
        st.image(edges, caption="Detected Edges (Sobel Filter)", clamp=True)

    st.subheader("6️⃣ Vertical Cut 70–30")
    if st.button("Vertical 70–30"):
        arr = np.array(img)
        h, w, c = arr.shape
        left_70 = arr[:, :int(0.7 * w), :]
        right_30 = arr[:, int(0.7 * w):, :]
        st.image(left_70, caption="Left 70%")
        st.image(right_30, caption="Right 30%")

    st.subheader("7️⃣ Horizontal Cut 70–30")
    if st.button("Horizontal 70–30"):
        arr = np.array(img)
        h, w, c = arr.shape
        top_70 = arr[:int(0.7 * h), :, :]
        bottom_30 = arr[int(0.7 * h):, :, :]
        st.image(top_70, caption="Top 70%")
        st.image(bottom_30, caption="Bottom 30%")

    st.subheader("8️⃣ Create 3×3 Image Grid")
    if st.button("Create Grid"):
        arr = np.array(img)
        fig = plt.figure(figsize=(6, 6))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(arr)
            plt.axis("off")
        st.pyplot(fig)

    st.success("✔ All operations executed successfully!")
