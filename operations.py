import cv2
import numpy as np

from config import config
from merging_helper import merge_hough_lines, filter_lines

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_image(image, dim=(720, 480)):
    return cv2.resize(image, dim)


def dilation(image, kernel_size=3, iterations=1):
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8), iterations)


def erosion(image, kernel_size, iterations=1):
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8), iterations)


def opening(image, kernel_size):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))


def closing(image, kernel_size):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))


def gaussian(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), cv2.BORDER_CONSTANT)


def thresholding(gray_image):
    ret, thresh = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def canny(image, th1, th2, apertureSize):
    return cv2.Canny(image, th1, th2, apertureSize=apertureSize)


def draw_contour(image, min_area, max_area, max_contours=None, mode=cv2.RETR_EXTERNAL):
    contours, hierarchy = cv2.findContours(image, mode, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    if max_contours is not None:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
    return cv2.fillPoly(image, pts=contours, color=(255, 255, 255))


def region_selection(image, max_height_from_bottom):
    mask = np.zeros_like(image)
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # We could have used fixed numbers as the vertices of the polygon,
    # but they will not be applicable to images with different dimensions.
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0, rows * 1]
    top_left = [cols * 0, rows * (1 - max_height_from_bottom)]
    bottom_right = [cols * 1, rows * 1]
    top_right = [cols * 1, rows * (1 - max_height_from_bottom)]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_skeleton(image):
    # Step 1: Create an empty skeleton
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    # Get a Cross Shaped Kernel
    kernel_size = 2
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # Repeat steps 2-4
    for i in range(1):
        # Step 2: Open the image
        open = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(image, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(image, element)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

    return image


def fast_line_detector(image):
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(image)
    line_on_image = fld.drawSegments(image, lines)
    return line_on_image, len(lines)


def hough_transform(image, masked_image, threshold, minLineLength, maxLineGap, merge_lines=False):
    rho = 1  # Distance resolution of the accumulator in pixels.
    theta = np.pi / 180  # Angle resolution of the accumulator in radians.
    hough_lines = []
    modified_image = image.copy()
    hough_lines = cv2.HoughLinesP(masked_image, rho=rho, theta=theta, threshold=threshold,
                                  minLineLength=minLineLength, maxLineGap=maxLineGap)

    if hough_lines is None:
        return image, 0

    # if merge_lines:
    #     for i in range(config.LINE_MERGE_COUNT):
    #         hough_lines = merge_hough_lines(hough_lines)

    hough_lines = np.squeeze(hough_lines, axis=1)
    filtered_lines = filter_lines(hough_lines)

    for line in filtered_lines:
        (x1, y1, x2, y2) = line
        cv2.line(modified_image, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=2, lineType=cv2.FILLED)

    return modified_image, len(filtered_lines)
