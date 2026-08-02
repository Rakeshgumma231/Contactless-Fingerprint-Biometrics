import cv2
import numpy as np
import time


def check_blur(image_bgr, threshold=10.0):
    """
    Check whether an image is blurry using
    Variance of Laplacian.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Input image in OpenCV BGR format.

    threshold : float
        Blur threshold.
        Default = 10.0

    Returns
    -------
    dict
    """

    start_time = time.perf_counter()

    # Convert image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Calculate Laplacian variance
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Determine blur
    is_blurry = blur_score < threshold

    end_time = time.perf_counter()

    processing_time = (
        end_time - start_time
    ) * 1000

    return {
        "blur_score": round(float(blur_score), 2),
        "is_blurry": is_blurry,
        "processing_time_ms": round(processing_time, 2)
    }

# =====================================================
# Brightness Detection
# =====================================================

def check_brightness(image_bgr,
                     dark_threshold=50,
                     bright_threshold=210):

    """
    Check image brightness.
    """

    start_time = time.perf_counter()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    too_dark = brightness < dark_threshold
    too_bright = brightness > bright_threshold

    end_time = time.perf_counter()

    processing_time = (end_time - start_time) * 1000

    return {
        "brightness": round(float(brightness), 2),
        "too_dark": too_dark,
        "too_bright": too_bright,
        "processing_time_ms": round(processing_time, 2)
    }