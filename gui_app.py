import os
import shutil
import streamlit as st
from PIL import Image

from image_compress import compress_image_wavelet
from Algorithms import analyze_image  # must return a string

# Create folders
for folder in ["uploads", "info", "Compressed"]:
    os.makedirs(folder, exist_ok=True)

st.set_page_config(page_title="Satellite Image Processor", layout="centered")

st.title("🛰️ Satellite Image Processor")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    filename = uploaded_file.name
    upload_path = os.path.join("uploads", filename)

    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded: {filename}")

    # Preview image
    img = Image.open(upload_path)
    st.image(img, caption="Preview", use_column_width=True)

    # --- Actions ---
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Get Image Properties"):
            result = analyze_image(upload_path)
            info_path = os.path.join("info", filename + "_info.txt")
            with open(info_path, "w") as f:
                f.write(result)
            st.text_area("Image Info", result, height=200)
            st.download_button("📥 Download Info File", result, file_name=os.path.basename(info_path))

    with col2:
        if st.button("📦 Compress Image"):
            output_name = os.path.splitext(filename)[0] + "_compressed.webp"
            output_path = os.path.join("Compressed", output_name)
            compress_image_wavelet(upload_path, output_path)

            st.success("✅ Image compressed")
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Compressed Image", f, file_name=output_name)
