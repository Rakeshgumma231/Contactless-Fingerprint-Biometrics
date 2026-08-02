# Contactless Fingerprint Biometrics

A Python and OpenCV-based Contactless Fingerprint Quality Assessment Pipeline for evaluating blur, brightness, glare, ROI completeness, and ridge clarity with a composite quality score and Streamlit interface.

## Features

- Blur Detection using Laplacian Variance
- Brightness Analysis
- Glare Detection
- ROI (Region of Interest) Completeness Check
- Ridge Clarity Assessment using Gabor Filter
- Composite Quality Score (0–100)
- PASS / ACCEPTABLE / FAIL Decision
- User Guidance for Image Retake
- Interactive Streamlit Web Application

## Project Structure

```
Assignment 4/
│
├── data/
│   ├── good/
│   ├── blurry/
│   ├── dark/
│   └── glare/
│
├── screenshots/
├── quality_assessment.py
├── quality_app.py
├── test_quality.py
├── report.pdf
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Run the Streamlit application:

```bash
py -m streamlit run quality_app.py
```

Run the testing script:

```bash
py test_quality.py
```

## Technologies Used

- Python 3.11
- OpenCV
- NumPy
- Streamlit
- Matplotlib
- Pandas

## Author

Rakesh Gumma