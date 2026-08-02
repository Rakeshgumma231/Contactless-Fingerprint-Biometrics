import cv2

from quality_assessment import (
    check_blur,
    check_brightness,
    check_glare,
    check_roi_completeness,
    check_ridge_clarity,
    calculate_composite_score,
    quality_gate
)
image_paths = [
    "data/good/good1.jpg",
    "data/good/good2.jpg",
    "data/good/good3.jpg",
    "data/good/good4.jpg",
    "data/good/good5.jpg",

    "data/blurry/blur1.jpg",
    "data/blurry/blur2.jpg",
    "data/blurry/blur3.jpg",
    "data/blurry/blur4.jpg",
    "data/blurry/blur5.jpg",

    "data/dark/dark1.jpg",
    "data/dark/dark2.jpg",
    "data/dark/dark3.jpg",
    "data/dark/dark4.jpg",
    "data/dark/dark5.jpg",

    "data/glare/glare1.jpg",
    "data/glare/glare2.jpg",
    "data/glare/glare3.jpg",
    "data/glare/glare4.jpg",
    "data/glare/glare5.jpg",
]
for image_path in image_paths:

    print(f"\n{'=' * 60}")
    print(f"Testing: {image_path}")
    print(f"{'=' * 60}")

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        continue

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

    # ---------------------------------------------------
    # Composite Quality Score
    # ---------------------------------------------------

    composite_result = calculate_composite_score(
        blur_result,
        brightness_result,
        glare_result,
        roi_result,
        ridge_result
    )

    print("\n==============================")
    print("   COMPOSITE QUALITY SCORE")
    print("==============================")

    print(f"Quality Score      : {composite_result['quality_score']}/{composite_result['max_score']}")

    # ---------------------------------------------------
    # Quality Gate
    # ---------------------------------------------------

    quality_result = quality_gate(
        blur_result,
        brightness_result,
        glare_result,
        roi_result,
        ridge_result,
        composite_result
    )

    print("\n==============================")
    print("   QUALITY GATE")
    print("==============================")

    print(f"Decision           : {quality_result['decision']}")
    print(f"Reasons            : {', '.join(quality_result['reasons'])}")
    print(f"Suggestions        : {', '.join(quality_result['suggestions'])}")