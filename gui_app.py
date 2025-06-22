import os
import shutil
import streamlit as st
from PIL import Image

import importlib
import Algorithms
importlib.reload(Algorithms)
from Algorithms import analyze_image  # must return a string

from image_compress import compress_image_wavelet

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

    # --- Actions (VERTICAL) ---
    if "info_result" not in st.session_state:
        st.session_state.info_result = ""

    get_info = st.button("🔍 Get Image Properties")
    retest = st.button("🔄 Retest Image")

    if get_info or retest:
        result = analyze_image(upload_path)
        info_path = os.path.join("info", filename + "_info.txt")
        with open(info_path, "w") as f:
            f.write(result)
        st.session_state.info_result = result  # Store result in session state

    st.text_area("Image Info", st.session_state.info_result, height=350)
    if st.session_state.info_result:
        info_path = os.path.join("info", filename + "_info.txt")
        st.download_button("📥 Download Info File", st.session_state.info_result, file_name=os.path.basename(info_path))

    st.markdown("---")

    if st.button("📦 Compress Image"):
        output_name = os.path.splitext(filename)[0] + "_compressed.webp"
        output_path = os.path.join("Compressed", output_name)
        compress_image_wavelet(upload_path, output_path)

        st.success("✅ Image compressed")
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Compressed Image", f, file_name=output_name)
