import cv2
import numpy as np
import argparse
import sys
from os import listdir
from os.path import isfile, join

import config
from operations import *
from util import *


def extract_hsl_mask(image):
    """
    First extract the gray component of the road from the image, then apply white, yellow and red masks on the image to obtain lane lines.
    """
    # Convert the input RGB image to HSL
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # White color mask
    lower_threshold = create_hls_from_hsl(0, 0, 75)
    upper_threshold = create_hls_from_hsl(360, 100, 100)
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = create_hls_from_hsl(20, 20, 20)
    upper_threshold = create_hls_from_hsl(54, 100, 100)
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Red color mask
    # Red lower threshold
    lower_threshold = create_hls_from_hsl(0, 50, 20)
    upper_threshold = create_hls_from_hsl(30, 100, 80)
    red_mask1 = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    # Red upper threshold
    lower_threshold = create_hls_from_hsl(320, 50, 20)
    upper_threshold = create_hls_from_hsl(360, 100, 80)
    red_mask2 = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Gray color mask
    # Gray lower threshold
    copy_image = np.copy(image)
    copy_image = gaussian(copy_image, kernel_size=15)
    converted_image = cv2.cvtColor(copy_image, cv2.COLOR_RGB2HLS)

    lower_threshold = create_hls_from_hsl(0, 0, 10)
    upper_threshold = create_hls_from_hsl(70, 10.5, 70.5)
    gray_mask1 = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    # Gray upper threshold
    lower_threshold = create_hls_from_hsl(206, 0, 10.5)
    upper_threshold = create_hls_from_hsl(360, 10, 70.5)
    gray_mask2 = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    gray_mask = cv2.bitwise_or(gray_mask1, gray_mask2)

    gray_mask = gaussian(gray_mask, kernel_size=5)
    gray_mask = draw_contour(gray_mask, 100, 100000, mode=cv2.RETR_EXTERNAL)
    gray_mask = gaussian(gray_mask, kernel_size=5)
    gray_mask = dilation(gray_mask, kernel_size=5, iterations=1)

    # Combine the three color masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    mask = cv2.bitwise_or(mask, red_mask)

    # Apply gray mask on image
    masked_image = cv2.bitwise_and(image, image, mask=gray_mask)

    # Apply color mask on image
    masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask)

    return masked_image

def detect_stop_sign(filename, image):
    stop_sign = cv2.CascadeClassifier("stopsign_cascade_classifier.xml")
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stop_signs = stop_sign.detectMultiScale(gray, 1.02, 100)

    if len(stop_signs) > 0:
        print("Stop signs detected in image:{} {}", filename, len(stop_signs))
        for (x, y, w, h) in stop_signs:
            x1 = x + w
            y1 = y + h
            cv2.circle(image, ((x + x1) // 2, (y + y1) // 2), (w // 2), (0, 0, 255), 4)

    return image


def detect_lane(filename, frame):
    result = 0
    # Resize of image to apply results uniformly
    new_image = image = resize_image(frame)

    # Extract HSL - Gray + White + Yellow + Red Mask
    new_image = extract_hsl_mask(new_image)

    # Select the region of interest
    new_image = region_selection(new_image, max_height_from_bottom=config.MAX_REGION_HEIGHT)

    # Perform canny edge detection to detect edges
    new_image = canny(new_image, config.CANNY_TH1, config.CANNY_TH2, apertureSize=config.CANNY_APERTURE_SIZE)

    # Perform closing to narrow down near boundary regions
    new_image = closing(new_image, 15)

    # Drawing contours to fill internal regions, so as to give better boundaries.
    new_image = draw_contour(new_image, 100, 5000000, mode=cv2.RETR_EXTERNAL)

    # Perform canny edge detection to detect edges
    new_image = canny(new_image, config.CANNY_TH1, config.CANNY_TH2, apertureSize=config.CANNY_APERTURE_SIZE)

    # Perform dilation to increase border width
    new_image = dilation(new_image, kernel_size=config.DILATION_KERNEL_SIZE, iterations=1)

    # Apply hough transforms to detect lane lines
    new_image, result = hough_transform(image, new_image,
                                        threshold=config.HOUGH_THRESHOLD,
                                        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
                                        maxLineGap=config.HOUGH_MAX_LINE_GAP,
                                        merge_lines=True)

    # Apply stop sign detection on the image
    new_image = detect_stop_sign(filename, new_image)

    print("Detected lines on {}: {}".format(filename, result))
    return new_image, result


###########################################################################

def runon_image(path):
    frame = cv2.imread(path)
    # Change the codes here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filename = os.path.basename(path)
    new_image, detections_in_frame = detect_lane(filename, frame)
    frame = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    # Change the codes above
    cv2.imshow("one image", frame)
    cv2.waitKey(0)
    return detections_in_frame


def runon_folder(path):
    files = None
    if (path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    for f in files:
        print(f)
        f_detections = runon_image(f)
        all_detections += f_detections
    return all_detections


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None:
        print("Folder path must be given \n Example: python proj1.py -folder images")
        sys.exit()

    if folder is not None:
        all_detections = runon_folder(folder)
        print("total of ", all_detections, " detections")

    cv2.destroyAllWindows()
