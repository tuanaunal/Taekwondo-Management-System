# Taekwondo Impact Analysis System (Ghost Hit Detection)

This repository contains a computer vision-based analysis system developed to differentiate between genuine impacts (Real Hits) and sensor-triggered helmet tremors (Ghost Hits) in Taekwondo competitions.

## Project Description

The system utilizes digital image processing and object tracking techniques to analyze the movement patterns of helmets. By calculating velocity and acceleration data from video footage, the project aims to provide objective decision support for competition scoring and referee evaluations.

## Current Development Phase: Analysis and Tracking

The project has successfully completed the data preparation and preprocessing stages. Current efforts are focused on the integration of object tracking algorithms and kinematic data extraction.

## Technical Specifications

- **Programming Language:** Python
- **Primary Frameworks:** OpenCV, NumPy, MoviePy
- **Data Standardization:** Automatic conversion to 720p resolution and 30 FPS.
- **Image Preprocessing:** - Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination stabilization.
  - Region of Interest (ROI) filtering to enhance processing efficiency and reduce noise.

## Installation and Setup

1. Clone the repository:
   git clone https://github.com/tuanaunal/Taekwondo-Management-System.git

2. Install the required dependencies:
   pip install opencv-python numpy moviepy

3. Usage:
   - Run `standardize.py` to prepare the dataset.
   - Run `main.py` to execute the analysis pipeline.

## Data Privacy and Constraints

Due to privacy regulations and file size limitations, the video dataset used for this project is not included in the public repository.
