import cv2

from quality_assessment import (
    check_blur,
    check_brightness
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

