import cv2
import numpy as np
import os
from earth import earth
from star_finder import star_finder

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def check_quality(image_gray, sharp_thresh=100, noise_thresh=20):
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    sharpness = laplacian.var()
    blurred = cv2.medianBlur(image_gray, 3)
    residual = cv2.absdiff(image_gray, blurred)
    noise = residual.std()

    if sharpness < sharp_thresh:
        return False, f"Blurry – image sharpness is {round(sharpness, 1)}, expected ≥ {sharp_thresh}"
    if noise > noise_thresh:
        return False, f"Noisy – estimated noise is {round(noise, 1)}, allowed ≤ {noise_thresh}"
    return True, {"sharpness": round(sharpness, 1), "noise": round(noise, 1)}

def detect_horizon(image_gray):
    edges = cv2.Canny(image_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return False, "No horizon found (no strong edges detected)"
    horizontal_lines = [l for l in lines if abs(l[0][1]) < 0.15 or abs(l[0][1] - np.pi) < 0.15]
    if len(horizontal_lines) >= 1:
        return True, None
    return False, "No strong horizontal edge (Earth limb likely missing)"

def detect_stars_with_sky_mask(image_path, dim=(1280, 720), min_stars=10, sensitivity=50):
    try:
        e = earth(image_path, dim=dim)
        sky_gray = cv2.cvtColor(e.clear_sky, cv2.COLOR_BGR2GRAY)
        star_analysis = star_finder(image_path, gray_image=sky_gray, sensitivity=sensitivity, dim=dim)
        count = len(star_analysis.stars)
        if count >= min_stars:
            return True, count
        else:
            return False, f"Only {count} stars found, expected ≥ {min_stars}"
    except Exception as ex:
        return False, f"Star detection error: {str(ex)}"

def detect_glitch(image_gray, bright_pixel_threshold=240, max_allowed_ratio=0.1):
    bright_pixels = np.sum(image_gray > bright_pixel_threshold)
    total_pixels = image_gray.shape[0] * image_gray.shape[1]
    ratio = bright_pixels / total_pixels
    if ratio > max_allowed_ratio:
        return False, f"Glitch – {round(ratio*100, 2)}% pixels overexposed (allowed ≤ {max_allowed_ratio*100}%)"
    return True, None

def run_phase1_on_folder(folder_path='images'):
    results = []

    for filename in os.listdir(folder_path):
        if not is_image_file(filename):
            continue

        full_path = os.path.join(folder_path, filename)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{filename}: ERROR – Could not read image")
            continue

        passed_all = True
        reasons = []

        q_ok, q_result = check_quality(image)
        if not q_ok:
            passed_all = False
            reasons.append(q_result)

        h_ok, h_reason = detect_horizon(image)
        if not h_ok:
            reasons.append(f"Note: {h_reason} (allowed for sky-only images)")

        s_ok, s_result = detect_stars_with_sky_mask(full_path)
        if not s_ok:
            passed_all = False
            reasons.append(s_result)

        g_ok, g_reason = detect_glitch(image)
        if not g_ok:
            passed_all = False
            reasons.append(g_reason)

        if passed_all:
            star_count = s_result if isinstance(s_result, int) else "?"
            print(f"{filename}: ✅ PASSED – Detected {star_count} stars")
        else:
            print(f"{filename}: ❌ REJECTED – " + " | ".join(reasons))

        results.append({
            'image': filename,
            'passed': passed_all,
            'reasons': reasons if not passed_all else [f"{s_result} stars"]
        })

    return results

if __name__ == '__main__':
    run_phase1_on_folder('images')
