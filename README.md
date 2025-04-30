# Pose-Based Stress Level Monitoring using MediaPipe (IoT Project)

![image](https://github.com/user-attachments/assets/6bd3c3e2-e97e-46c5-b99f-ca48d71e13ed)


## Overview

This project is an exploration into using computer vision for understanding human poses and potentially correlating them with stress levels. It utilizes Google's MediaPipe library to detect and analyze human pose structures from a live camera feed. By calculating key body points and joint angles, the system classifies the detected pose into predefined categories. The long-term goal is to associate these poses with corresponding stress indicators and generate an overall stress report based on observations over time.

This repository represents my first venture into combining IoT concepts with computer vision for human behavior analysis.

## Features

* **Real-time Pose Detection:** Leverages MediaPipe Pose for accurate detection of body landmarks from a camera stream.
* **Pose Classification:** Analyzes landmark coordinates and angles to classify the current pose.
    * Currently recognizes: **Log** position, **Foetus** position.
* **(Work in Progress)** **Stress Level Association:** Aims to map recognized poses to potential stress levels.
* **(Work in Progress)** **Data Logging & Reporting:** Plans to collect pose data over a specified duration and generate a summary stress report.

## Technology Stack

* **Core Logic:** Python (Assumed - please specify version, e.g., Python 3.8+)
* **Pose Estimation:** Google MediaPipe (Pose)
* **Image/Video Handling:** OpenCV (Likely used for camera access - please confirm)
* **(Potential IoT Hardware):** (Specify if using Raspberry Pi, ESP32-CAM, Jetson Nano, etc.)
* **(Other Libraries):** NumPy (Likely used for calculations - please confirm), (Add any other significant libraries)

## Project Status & Capabilities

* **Core Pose Analysis Engine Complete:** Successfully implemented real-time human pose detection and landmark extraction using Google's MediaPipe framework, forming the foundation of the system.
* **Pose Classification Module Developed:** Created a robust module that analyzes pose landmarks and joint angles to categorize distinct body positions.
* **Initial Position Recognition Achieved:** The system has been successfully trained and validated to accurately identify and differentiate between key positions, specifically the 'Log' and 'Foetus' poses.
* **Foundation for Temporal Analysis:** The underlying structure is designed to capture and process pose data sequentially, laying the groundwork for future time-based analysis and pattern recognition.
* **Designed for Extensibility:** The modular architecture facilitates planned future enhancements, including:
    * **Expanding the Pose Library:** Incorporating a wider range of recognizable body positions.
    * **Stress Correlation Modeling:** Developing and integrating algorithms to explore potential correlations between observed pose patterns (frequency, duration) and stress indicators.
    * **Automated Reporting:** Implementing functionality for data aggregation over defined periods and generating comprehensive summary reports.
