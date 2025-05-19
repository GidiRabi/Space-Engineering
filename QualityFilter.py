import cv2
import numpy as np
import os

# Evaluate if image is blurry or noisy based on Laplacian variance and residual std
def evaluate_image_quality(image_path, sharpness_thresh=100.0, noise_thresh=20.0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {
            'image': os.path.basename(image_path),
            'sharpness': None,
            'noise': None,
            'passed': False,
            'reason': 'Unreadable image'
        }

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = laplacian.var()

    blurred = cv2.medianBlur(image, 3)
    residual = cv2.absdiff(image, blurred)
    noise = residual.std()

    passed = True
    reason = None

    if sharpness < sharpness_thresh:
        passed = False
        reason = 'Blurry'
    elif noise > noise_thresh:
        passed = False
        reason = 'Noisy'

    return {
        'image': os.path.basename(image_path),
        'sharpness': round(sharpness, 2),
        'noise': round(noise, 2),
        'passed': passed,
        'reason': reason
    }

# Run the image quality check on all images in a folder
def run_quality_filter_on_folder(folder_path='images', sharpness_thresh=100.0, noise_thresh=20.0):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            full_path = os.path.join(folder_path, filename)
            result = evaluate_image_quality(full_path, sharpness_thresh, noise_thresh)
            results.append(result)
            print(f"{filename}: {'PASSED' if result['passed'] else 'REJECTED'} ({result['reason']})")

    return results

if __name__ == '__main__':
    run_quality_filter_on_folder('images')
