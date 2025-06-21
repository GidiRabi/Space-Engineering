# 🛰️ Satellite Image Analyzer & Compressor

A full pipeline for automatic **satellite image classification** and **visual compression**. This tool was developed to assist in filtering space imagery—identifying high-quality star images while rejecting blurry, noisy, or non-sky images. It includes an interactive web interface for uploading, analyzing, and compressing images.

## 🔍 Features

- **Image Quality Analysis**:
  - Laplacian-based sharpness detection
  - Noise level estimation
  - Horizon line detection using Sobel & Hough transforms
  - Glitch detection via overexposure
  - Flicker detection via row brightness variation

- **Sky & Earth Segmentation**:
  - Masks Earth vs. sky regions using adaptive thresholding
  - Ignores false stars in Earth regions
  - Handles Earth at night

- **Star Detection**:
  - Custom star finder for bright circular objects in the sky mask
  - Filters by size, brightness, and location
  - Optional visualization with drawn circles

- **Image Compression**:
  - Wavelet-based lossy compression (PyWavelets)
  - Targets high-frequency detail for size reduction
  - Saves as WebP format with preserved quality

- **Streamlit GUI**:
  - Upload images
  - View image preview
  - Run analysis and download detailed report
  - Compress and download optimized image

## 🗂️ Project Structure

```
.
├── Algorithms.py           # Full pipeline: quality checks, star detection, classification
├── earth.py                # Sky/Earth segmentation and classification
├── star_finder.py          # Bright object (star) detection module
├── image_compress.py       # Wavelet compression utility
├── gui_app.py              # Streamlit-based interface
├── uploads/                # Uploaded images
├── Compressed/             # Output of compressed images
├── info/                   # Text reports of image analysis
```

## 🧪 Dataset

Used the **"Stars"** folder from the [Space Images Category dataset on Kaggle](https://www.kaggle.com/datasets/abhikalpsrivastava15/space-images-category), with 175 star field images containing both usable and low-quality examples.

- All images resized to 1280×720
- Grayscale conversion before analysis
- Earth masking applied before star detection

## 📊 Accuracy

After testing 175 images and comparing against manual review:

- **Accuracy**: ~84.6%
- **True Positives**: 127
- **False Positives**: 13
- **False Negatives**: 8
- **True Negatives**: 27

## 🧠 Key Technologies

- `OpenCV` – image processing (blur, edge detection, masking)
- `PyWavelets` – wavelet-based image compression
- `Streamlit` – interactive web app
- `NumPy`, `PIL`, `imageio` – core image handling

## ⚙️ How to Run

### Web App (Streamlit):

```bash
streamlit run gui_app.py
```

### CLI Example (optional):
```python
from image_compress import compress_image_wavelet
compress_image_wavelet("input.png", "output.webp")
```

## 📁 Requirements

```bash
pip install opencv-python numpy streamlit pillow imageio PyWavelets
```

## ✍️ Authors

- **Gidi Rabi** – Computer Vision, Streamlit GUI, Analysis Logic  
- **Roi Bruchim** – Star Detection, Evaluation Logic, Research

## 📃 Reference

This project was part of a research initiative at **Ariel University, Israel**.  
See the full research paper in [`Research_Article.pdf`](Research_Article.pdf)
