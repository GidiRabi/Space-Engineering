import cv2
import numpy as np
import pywt
import imageio
import os

def compress_image_wavelet(input_path, output_path, wavelet='haar', level=2, quant_level=30):
    # Load color image (BGR)
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to YUV (better for visual compression)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    channels = cv2.split(img_yuv)

    compressed_channels = []

    for channel in channels:
        # Perform multi-level wavelet decomposition
        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)
        coeffs_quant = []

        for i, c in enumerate(coeffs):
            if i == 0:
                # Approximation (low freq): keep as is
                coeffs_quant.append(c)
            else:
                # Details (high freq): quantize aggressively
                cH, cV, cD = c
                cH_q = np.round(cH / quant_level)
                cV_q = np.round(cV / quant_level)
                cD_q = np.round(cD / quant_level)
                coeffs_quant.append((cH_q, cV_q, cD_q))

        # Reconstruct
        compressed = pywt.waverec2(coeffs_quant, wavelet=wavelet)
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        compressed_channels.append(compressed)

    # Merge and convert back
    compressed_yuv = cv2.merge(compressed_channels)
    compressed_rgb = cv2.cvtColor(compressed_yuv, cv2.COLOR_YUV2RGB)

    # Save as high-efficiency format (WebP or PNG)
    imageio.imwrite(output_path, compressed_rgb, format='WEBP', quality=85)
    print(f"✅ Wavelet-compressed image saved to {output_path}")

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    compress_image_wavelet("Images/good5.png", "wavelet_compressed.webp")
