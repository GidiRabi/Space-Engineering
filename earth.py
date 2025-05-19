import cv2
import numpy as np
import os
from cv2 import (
    cvtColor, morphologyEx, GaussianBlur, threshold, adaptiveThreshold,
    bitwise_and, bitwise_not, Sobel, circle, findContours, dilate,
    minEnclosingCircle, THRESH_TOZERO, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,
    MORPH_CLOSE, MORPH_OPEN, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY
)


def get_image(path_or_image, x=-1, dim=None):
    if isinstance(path_or_image, str) and os.path.exists(path_or_image):
        image = cv2.imread(path_or_image, x)
        if dim:
            image = cv2.resize(image, dim)
    elif isinstance(path_or_image, np.ndarray):
        image = path_or_image
    else:
        raise TypeError("Input must be a file path or an image array.")
    return image


class earth:
    def __init__(self, path_or_image, dim=None):
        self.image = get_image(path_or_image, dim=dim)
        self.gray_image = cvtColor(self.image, COLOR_BGR2GRAY)
        self.clear_sky, self.clear_earth, self.earth_mask = self.get_mask()
        self.is_earth = self._is_earth()
        self.earth_precent = self._earth_precent()
        self.curve = self.get_curve()
        self.is_dark_earth = False if self.is_earth else self._is_earth_at_night()
        self.is_stars = not self.is_earth and not self.is_dark_earth
        self.mask = None

    def _is_earth_at_night(self):
        gray = GaussianBlur(self.gray_image, (3, 3), 0)
        _, gray = threshold(gray, 80, 255, THRESH_TOZERO)
        mask = adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_MEAN_C,
                                 THRESH_BINARY_INV, 3, 5)
        mask = morphologyEx(mask, MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        self.mask = mask
        for cnt in contours:
            (_, _), r = minEnclosingCircle(cnt)
            if r > 60:
                return True
        return False

    def _get_earth_mask(self):
        gauss = GaussianBlur(self.gray_image, (5, 5), 0)
        _, thresh = threshold(gauss, 30, 255, THRESH_TOZERO)
        thresh = Sobel(thresh, ddepth=-1, dx=1, dy=1, ksize=11)

        for size in [10, 20, 40, 70]:
            kernel = np.ones((size, size), np.uint8)
            morph_type = MORPH_CLOSE if size % 20 == 0 else MORPH_OPEN
            thresh = morphologyEx(thresh, morph_type, kernel)

        return thresh

    def draw_black_cir(self, contours):
        for center, radius in contours:
            circle(self.mask, center, radius, (0, 0, 0), -1)

    def _get_cir_contours(self):
        contours, _ = findContours(self.mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        h, w = self.gray_image.shape
        return [((int(x), int(y)), int(r)) for (x, y), r in
                (minEnclosingCircle(cnt) for cnt in contours)
                if r < max(h, w) * 0.3]

    def fill_earth(self):
        self.mask = bitwise_not(self.mask)
        self.draw_black_cir(self._get_cir_contours())
        self.mask = bitwise_not(self.mask)

    def get_mask(self):
        self.mask = self._get_earth_mask()
        self.draw_black_cir(self._get_cir_contours())
        self.mask = dilate(self.mask, np.ones((5, 5), np.uint8), iterations=3)
        self.fill_earth()
        self.mask = dilate(self.mask, np.ones((15, 15), np.uint8), iterations=1)

        clear_earth = bitwise_and(self.image, self.image, mask=self.mask)
        clear_sky = bitwise_and(self.image, self.image, mask=bitwise_not(self.mask))

        return clear_sky, clear_earth, self.mask

    def _is_earth(self):
        white_pix = np.count_nonzero(self.earth_mask == 255)
        total = self.gray_image.shape[0] * self.gray_image.shape[1]
        return white_pix / total > 0.1

    def _earth_precent(self):
        white_pix = np.count_nonzero(self.earth_mask == 255)
        total = self.gray_image.shape[0] * self.gray_image.shape[1]
        return white_pix / total

    def get_curve(self):
        gauss = GaussianBlur(self.gray_image, (7, 7), 0)
        _, gauss = threshold(gauss, 80, 255, THRESH_TOZERO)
        adaptive = adaptiveThreshold(gauss, 255, ADAPTIVE_THRESH_MEAN_C,
                                     THRESH_BINARY_INV, 201, 3)
        kernel = np.ones((5, 5), np.uint8)
        adaptive = morphologyEx(adaptive, MORPH_CLOSE, kernel)
        adaptive = morphologyEx(adaptive, MORPH_OPEN, kernel)
        return adaptive
