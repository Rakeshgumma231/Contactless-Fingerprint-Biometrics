import cv2
import numpy as np
import streamlit as st

from quality_assessment import (
    check_blur,
    check_brightness,
    check_glare,
    check_roi_completeness,
    check_ridge_clarity,
    calculate_composite_score,
    quality_gate
)

st.set_page_config(
    page_title="Contactless Fingerprint Quality Assessment",
    layout="wide"
)

st.title("Contactless Fingerprint Quality Assessment")

uploaded_file = st.file_uploader(
    "Upload Fingerprint Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # ==========================================
    # Read Image
    # ==========================================

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # ==========================================
    # Run All Quality Assessment Modules
    # ==========================================

    blur_result = check_blur(image)

    brightness_result = check_brightness(image)

    glare_result = check_glare(image)

    roi_result = check_roi_completeness(image)

    ridge_result = check_ridge_clarity(image)

    composite_result = calculate_composite_score(
        blur_result,
        brightness_result,
        glare_result,
        roi_result,
        ridge_result
    )

    quality_result = quality_gate(
        blur_result,
        brightness_result,
        glare_result,
        roi_result,
        ridge_result,
        composite_result
    )

    # ==========================================
    # IMAGE + METRICS
    # ==========================================

    left, right = st.columns([1, 2], gap="large")

    # -------------------------
    # Uploaded Image
    # -------------------------

    with left:

        st.subheader("Uploaded Image")

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width=300
        )

    # -------------------------
    # Quality Metrics
    # -------------------------

    with right:

        st.subheader("Quality Metrics")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Blur Score",
                blur_result["blur_score"]
            )

            st.metric(
                "Brightness",
                brightness_result["brightness"]
            )

            st.metric(
                "Glare (%)",
                glare_result["glare_percentage"]
            )

        with col2:

            st.metric(
                "ROI (%)",
                roi_result["roi_percentage"]
            )

            st.metric(
                "Ridge Score",
                ridge_result["ridge_score"]
            )

            st.metric(
                "Quality Score",
                f"{composite_result['quality_score']}/100"
            )

            st.progress(
                composite_result["quality_score"] / 100
            )

    st.divider()

    # ==========================================
    # QUALITY GATE
    # ==========================================

    st.header("Decision")

    decision = quality_result["decision"]

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric(
            "Decision",
            decision
        )

    with col2:

        if decision == "PASS":
            st.success("✅ PASS")

        elif decision == "ACCEPTABLE":
            st.warning("⚠️ ACCEPTABLE")

        else:
            st.error("❌ FAIL")

    # ==========================================
    # PROCESSING TIME
    # ==========================================

    st.subheader("Processing Time")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"Blur : {blur_result['processing_time_ms']} ms"
        )

        st.write(
            f"Brightness : {brightness_result['processing_time_ms']} ms"
        )

        st.write(
            f"Glare : {glare_result['processing_time_ms']} ms"
        )

    with col2:

        st.write(
            f"ROI : {roi_result['processing_time_ms']} ms"
        )

        st.write(
            f"Ridge : {ridge_result['processing_time_ms']} ms"
        )

    # ==========================================
    # REASONS
    # ==========================================

    st.subheader("Reasons")

    if quality_result["reasons"]:

        for reason in quality_result["reasons"]:

            st.write(f"• {reason}")

    else:

        st.success("No quality issues detected.")

    # ==========================================
    # SUGGESTIONS
    # ==========================================

    st.subheader("Suggestions")

    if quality_result["suggestions"]:

        for suggestion in quality_result["suggestions"]:

            st.write(f"• {suggestion}")

    else:

        st.success("No suggestions. Image quality is good.")

    

st.divider()

st.caption(
    "Contactless Fingerprint Quality Assessment using OpenCV and Streamlit"
)