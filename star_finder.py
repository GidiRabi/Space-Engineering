import cv2
import numpy as np
from copy import deepcopy
from cv2 import (
    cvtColor, GaussianBlur, threshold, adaptiveThreshold,
    bitwise_and, circle, findContours,
    THRESH_TOZERO, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,
    RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY
)

# The star_finder class identifies stars in an image based on brightness and size.
# It applies thresholding, contour detection, and brightness filtering to extract star-like regions.
# Parameters:
# - path_or_image: input file path or NumPy array of the image.
# - gray_image: optional precomputed grayscale image (saves computation if already available).
# - sensitivity: brightness threshold for detecting stars.
# - N_stars: maximum number of stars to keep (sorted by brightness).
# - dim: optional resize dimensions for input image.
# - draw: if True, generates an image with star annotations.
class star_finder:
    def __init__(self, path_or_image, gray_image=None, sensitivity=None, N_stars=None, dim=None, draw=False):
        self.image = self._get_image(path_or_image, dim)
        self.gray_image = gray_image if isinstance(gray_image, np.ndarray) else cvtColor(self.image, COLOR_BGR2GRAY)
        if sensitivity is None:
            avg_brightness = np.mean(self.gray_image)
            # Map average brightness [0,255] to sensitivity [70,120]
            self.sensitivity = int(70 + (avg_brightness / 255) * 50)
        else:
            self.sensitivity = sensitivity
        self.mask = self.get_threshold()
        self.stars = self.find_stars()
        self.N_stars = min(N_stars or len(self.stars), len(self.stars)) if self.stars else 0
        self.stars = self.stars[:self.N_stars]
        self.draw_image = None
        if draw:
            self.draw()

    # Loads an image from a path or directly uses a NumPy array.
    # Resizes if dimensions are provided.
    def _get_image(self, path_or_image, dim):
        if isinstance(path_or_image, str):
            image = cv2.imread(path_or_image, -1)
            if dim:
                image = cv2.resize(image, dim)
            return image
        elif isinstance(path_or_image, np.ndarray):
            return path_or_image
        else:
            raise TypeError("Must provide image path or np.ndarray")

    # Applies thresholding and adaptive thresholding to generate a binary mask
    # for star detection based on brightness differences.
    # Returns:
    # - Binary mask highlighting star candidate regions.
    def get_threshold(self):
        blurred = GaussianBlur(self.gray_image, (5, 5), 0)
        _, thresh = threshold(blurred, self.sensitivity, 255, THRESH_TOZERO)
        adaptive = adaptiveThreshold(thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, blockSize=11, C=3)
        return adaptive

    # Detects contours in the binary mask and filters them by area and brightness.
    # Extracts valid star regions, computes their brightness, and stores metadata.
    # Returns:
    # - List of stars as tuples: (cropped region, center coordinates, radius, brightness).
    def find_stars(self):
        contours, _ = findContours(self.mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        stars = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2 or area > 180:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)

            x, y, r = int(x), int(y), int(r) + 1
            center = (x, y)
            star_crop = self.extract_star(y, x, r)
            brightness = np.amax(star_crop)
            if brightness < self.sensitivity - 8:
                continue
            stars.append((star_crop, center, r, brightness))
        stars.sort(key=lambda s: s[3], reverse=True)
        return stars

    # Extracts the circular region around a detected star from the grayscale image.
    # Parameters:
    # - y, x: center coordinates of the star.
    # - r: radius of the star.
    # Returns:
    # - Circular masked region of the star from the grayscale image.
    def extract_star(self, y, x, r):
        region = self.gray_image[max(y - r, 0): y + r, max(x - r, 0): x + r]
        mask = np.zeros_like(self.gray_image)
        circle(mask, (x, y), r, 255, -1)
        cropped_mask = mask[max(y - r, 0): y + r, max(x - r, 0): x + r]
        return bitwise_and(region, region, mask=cropped_mask)

    # Draws a circle around each detected star on a copy of the original image.
    # Only executed if `draw=True` in the constructor.
    def draw(self):
        self.draw_image = deepcopy(self.image)
        for _, center, r, _ in self.stars[:self.N_stars]:
            circle(self.draw_image, center, r + 4, (0, 255, 0), 2)

    # Utility method for printing star metadata: index, center position, radius, brightness.
    # Only for debugging or inspection purposes.
    def get_data(self):
        for i, (img, center, r, b) in enumerate(self.stars[:self.N_stars]):
            print(f"Star {i}: center={center}, radius={r}, brightness={b}")
 