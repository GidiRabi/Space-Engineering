import cv2
import numpy as np
import os
import re
from earth import earth
from star_finder import star_finder

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def check_quality(image_gray, sharp_thresh=400, noise_thresh=15):
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    sharpness = laplacian.var()
    blurred = cv2.medianBlur(image_gray, 3)
    residual = cv2.absdiff(image_gray, blurred)
    noise = residual.std()

    tags = []
    passed = True

    tags.append(f"Image sharpness: {round(sharpness, 1)}")
    if sharpness < sharp_thresh:
        tags.append("→ Blurry")
        passed = False
    else:
        tags.append("→ Sharp")

    tags.append(f"Noise level: {round(noise, 1)}")
    if noise > noise_thresh:
        tags.append("→ Noisy")
        passed = False
    else:
        tags.append("→ Clean")

    return passed, tags

def detect_horizon(image_gray):
    edges = cv2.Canny(image_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    projection = np.sum(np.abs(sobel_y), axis=1)

    found_by_sobel = np.max(projection) > 8000
    found_by_hough = any(abs(l[0][1]) < 0.2 or abs(l[0][1] - np.pi) < 0.2 for l in lines) if lines is not None else False

    if found_by_sobel and found_by_hough:
        return True, "Horizon: Strong horizontal line detected (Sobel & Hough)"
    elif found_by_sobel:
        return True, "Horizon: Weak horizontal line (Sobel only)"
    elif found_by_hough:
        return True, "Horizon: Weak horizontal line (Hough only)"
    return False, "Horizon: No strong horizontal edge"

def detect_stars_with_sky_mask(image_path, dim=(1280, 720), min_stars=40, sensitivity=60):
    try:
        e = earth(image_path, dim=dim)
        sky_gray = cv2.cvtColor(e.clear_sky, cv2.COLOR_BGR2GRAY)
        star_analysis = star_finder(image_path, gray_image=sky_gray, sensitivity=sensitivity, dim=dim)
        count = len(star_analysis.stars)
        tags = [f"Star count: {count}"]
        if count >= min_stars:
            tags.append("→ Sufficient stars")
        else:
            tags.append("→ Low star count")
        if count > 1000:
            tags.append("→ Warning: Excessive stars (possible noise)")

        return count >= min_stars, tags
    except Exception as ex:
        return False, [f"Star detection error: {str(ex)}"]

def detect_glitch(image_gray, bright_pixel_threshold=240, max_allowed_ratio=0.08):
    bright_pixels = np.sum(image_gray > bright_pixel_threshold)
    total_pixels = image_gray.shape[0] * image_gray.shape[1]
    ratio = bright_pixels / total_pixels
    if ratio > max_allowed_ratio:
        return False, f"Glitch: {round(ratio*100, 2)}% pixels overexposed"
    return True, "Glitch: No overexposure detected"

def detect_flicker(image_gray, flicker_threshold=12, row_brightness_variation=10):
    diffs = np.abs(np.diff(image_gray.astype(np.int16), axis=0))
    row_std = np.std(np.mean(diffs, axis=1))
    row_means = np.mean(image_gray, axis=1)
    mean_var = np.std(row_means)

    tags = [f"Flicker – row diff std: {round(row_std, 2)}", f"Flicker – brightness row variation: {round(mean_var, 2)}"]

    passed = True
    if mean_var > 15:
        tags.append("→ Moderate flicker")
    if mean_var > 30:
        tags.append("→ Severe flicker")
        passed = False

    return passed, " | ".join(tags)

def natural_key(string):
    # Split string into list of strings and integers: "10.jpg" -> ["", 10, ".jpg"]
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def run_phase1_on_folder(folder_path='stars'):
    results = []

    # Natural sort the filenames
    for filename in sorted(os.listdir(folder_path), key=natural_key):
        if not is_image_file(filename):
            continue

        full_path = os.path.join(folder_path, filename)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"{filename}: ERROR – Could not read image")
            continue

        passed_all = True
        tags = []

        # Quality
        q_ok, q_tags = check_quality(image)
        tags.extend(q_tags)
        if not q_ok:
            passed_all = False

        # Stars
        s_ok, s_tags = detect_stars_with_sky_mask(full_path)
        tags.extend(s_tags)
        if not s_ok:
            passed_all = False

        # Horizon
        h_ok, h_tag = detect_horizon(image)
        tags.append(h_tag)
        if not h_ok and not s_ok:
            tags.append("→ No sky detected")

        # Glitch
        g_ok, g_tag = detect_glitch(image)
        tags.append(g_tag)
        if not g_ok:
            passed_all = False

        # Flicker
        f_ok, f_tag = detect_flicker(image)
        tags.append(f_tag)
        if not f_ok:
            passed_all = False

        # Output
        status = "✅ PASSED" if passed_all else "❌ REJECTED"
        print(f"\n{filename}: {status}")
        for t in tags:
            print(f"  - {t}")

        results.append({
            'image': filename,
            'passed': passed_all,
            'tags': tags
        })

    return results

def analyze_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return f"{os.path.basename(image_path)}: ERROR – Could not read image"

    passed_all = True
    tags = []

    # Quality
    q_ok, q_tags = check_quality(image)
    tags.extend(q_tags)
    if not q_ok:
        passed_all = False

    # Stars
    s_ok, s_tags = detect_stars_with_sky_mask(image_path)
    tags.extend(s_tags)
    if not s_ok:
        passed_all = False

    # Horizon
    h_ok, h_tag = detect_horizon(image)
    tags.append(h_tag)
    if not h_ok and not s_ok:
        tags.append("→ No sky detected")

    # Glitch
    g_ok, g_tag = detect_glitch(image)
    tags.append(g_tag)
    if not g_ok:
        passed_all = False

    # Flicker
    f_ok, f_tag = detect_flicker(image)
    tags.append(f_tag)
    if not f_ok:
        passed_all = False

    status = "✅ PASSED" if passed_all else "❌ REJECTED"

    result = f"{os.path.basename(image_path)}: {status}\n"
    result += "\n".join(f"  - {t}" for t in tags)
    return result


if __name__ == '__main__':
    from contextlib import redirect_stdout

    output_path = "Output.txt"
    with open(output_path, "w") as f, redirect_stdout(f):
        run_phase1_on_folder('stars')
