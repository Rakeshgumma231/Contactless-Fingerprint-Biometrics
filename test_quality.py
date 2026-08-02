import cv2

from quality_assessment import (
    check_blur,
    check_brightness,
    check_glare,
    check_roi_completeness,
    check_ridge_clarity
)

# Image Path
# Change extension if your image is .png
image_path = "../data/good/sample.jpeg"

# Read Image
image = cv2.imread(image_path)

# Check whether image loaded successfully
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# ---------------------------------------------------
# Blur Detection
# ---------------------------------------------------

blur_result = check_blur(image)

print("\n==============================")
print("      BLUR DETECTION")
print("==============================")

print(f"Blur Score         : {blur_result['blur_score']}")
print(f"Is Blurry          : {blur_result['is_blurry']}")
print(f"Processing Time    : {blur_result['processing_time_ms']} ms")


# ---------------------------------------------------
# Brightness Detection
# ---------------------------------------------------

brightness_result = check_brightness(image)

print("\n==============================")
print("   BRIGHTNESS DETECTION")
print("==============================")

print(f"Brightness         : {brightness_result['brightness']}")
print(f"Too Dark           : {brightness_result['too_dark']}")
print(f"Too Bright         : {brightness_result['too_bright']}")
print(f"Processing Time    : {brightness_result['processing_time_ms']} ms")


# ---------------------------------------------------
# Glare Detection
# ---------------------------------------------------

glare_result = check_glare(image)

print("\n==============================")
print("      GLARE DETECTION")
print("==============================")

print(f"Glare Percentage   : {glare_result['glare_percentage']}%")
print(f"Has Glare          : {glare_result['has_glare']}")
print(f"Processing Time    : {glare_result['processing_time_ms']} ms")

# ---------------------------------------------------
# ROI Completeness
# ---------------------------------------------------

roi_result = check_roi_completeness(image)

print("\n==============================")
print("   ROI COMPLETENESS")
print("==============================")

print(f"ROI Percentage     : {roi_result['roi_percentage']}%")
print(f"ROI Complete       : {roi_result['roi_complete']}")
print(f"Processing Time    : {roi_result['processing_time_ms']} ms")


# ---------------------------------------------------
# Ridge Clarity
# ---------------------------------------------------

ridge_result = check_ridge_clarity(image)

print("\n==============================")
print("   RIDGE CLARITY")
print("==============================")

print(f"Ridge Score        : {ridge_result['ridge_score']}")
print(f"Ridge Clear        : {ridge_result['ridge_clear']}")
print(f"Processing Time    : {ridge_result['processing_time_ms']} ms")

