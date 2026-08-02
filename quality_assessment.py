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

# =====================================================
# Glare Detection
# =====================================================

def check_glare(image_bgr, glare_threshold=240, max_glare_percentage=5.0):
    """
    Detect glare (overexposed regions) in the fingerprint image.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Input image in BGR format.

    glare_threshold : int
        Pixel intensity considered as glare.

    max_glare_percentage : float
        Maximum acceptable glare percentage.

    Returns
    -------
    dict
    """

    start_time = time.perf_counter()

    # Convert image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Count pixels above glare threshold
    glare_pixels = np.sum(gray > glare_threshold)

    total_pixels = gray.size

    glare_percentage = (glare_pixels / total_pixels) * 100

    has_glare = glare_percentage > max_glare_percentage

    end_time = time.perf_counter()

    processing_time = (end_time - start_time) * 1000

    return {
        "glare_percentage": round(float(glare_percentage), 2),
        "has_glare": has_glare,
        "processing_time_ms": round(processing_time, 2)
    }

# =====================================================
# ROI Completeness (Improved)
# =====================================================

def check_roi_completeness(image_bgr, min_roi_percentage=30.0):
    """
    Estimate fingerprint ROI completeness using contour detection.
    """

    start_time = time.perf_counter()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Automatic threshold
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove small holes
    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "roi_percentage": 0.0,
            "roi_complete": False,
            "processing_time_ms": 0.0
        }

    largest = max(contours, key=cv2.contourArea)

    roi_area = cv2.contourArea(largest)

    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    roi_percentage = (roi_area / image_area) * 100

    end_time = time.perf_counter()

    return {
        "roi_percentage": round(float(roi_percentage), 2),
        "roi_complete": roi_percentage >= min_roi_percentage,
        "processing_time_ms": round(
            (end_time - start_time) * 1000,
            2
        )
    }

# =====================================================
# Ridge Clarity
# =====================================================

def check_ridge_clarity(image_bgr, threshold=25.0):
    """
    Estimate fingerprint ridge clarity using Sobel gradients.
    """

    start_time = time.perf_counter()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    ridge_score = gradient.mean()

    end_time = time.perf_counter()

    processing_time = (
        end_time - start_time
    ) * 1000

    return {
        "ridge_score": round(float(ridge_score), 2),
        "ridge_clear": ridge_score >= threshold,
        "processing_time_ms": round(processing_time, 2)
    }

# =====================================================
# Composite Quality Score
# =====================================================

def calculate_composite_score(
    blur_result,
    brightness_result,
    glare_result,
    roi_result,
    ridge_result
):
    """
    Calculate overall fingerprint quality score.
    """

    score = 0

    # Blur (30)
    if not blur_result["is_blurry"]:
        score += 30

    # Brightness (15)
    if (
        not brightness_result["too_dark"]
        and
        not brightness_result["too_bright"]
    ):
        score += 15

    # Glare (10)
    if not glare_result["has_glare"]:
        score += 10

    # ROI (20)
    if roi_result["roi_complete"]:
        score += 20

    # Ridge (25)
    if ridge_result["ridge_clear"]:
        score += 25

    return {
        "quality_score": score,
        "max_score": 100
    }

# =====================================================
# Quality Gate
# =====================================================

def quality_gate(
    blur_result,
    brightness_result,
    glare_result,
    roi_result,
    ridge_result,
    composite_result
):
    """
    Final decision based on all quality metrics.
    """

    reasons = []
    suggestions = []

    if blur_result["is_blurry"]:
        reasons.append("Image is blurry")
        suggestions.append("Hold the camera steady and improve focus.")

    if brightness_result["too_dark"]:
        reasons.append("Image is too dark")
        suggestions.append("Increase lighting while capturing.")

    if brightness_result["too_bright"]:
        reasons.append("Image is too bright")
        suggestions.append("Reduce lighting or avoid direct light.")

    if glare_result["has_glare"]:
        reasons.append("Glare detected")
        suggestions.append("Tilt the finger slightly to reduce reflections.")

    if not roi_result["roi_complete"]:
        reasons.append("Fingerprint area is incomplete")
        suggestions.append("Capture the entire fingerprint inside the frame.")

    if not ridge_result["ridge_clear"]:
        reasons.append("Poor ridge clarity")
        suggestions.append("Move closer and ensure the fingerprint is in focus.")

    score = composite_result["quality_score"]

    if score >= 80:
        decision = "PASS"
    elif score >= 60:
        decision = "ACCEPTABLE"
    else:
        decision = "FAIL"

    return {
        "decision": decision,
        "quality_score": score,
        "reasons": reasons,
        "suggestions": suggestions
    }