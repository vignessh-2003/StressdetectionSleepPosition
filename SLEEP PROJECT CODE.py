import cv2
import mediapipe as mp
import numpy as np
import math
import os
import json
import time

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Create directories for storing results
os.makedirs('results/full_jsons/image', exist_ok=True)
os.makedirs('results/full_jsons/video', exist_ok=True)
os.makedirs('results/full_jsons/live', exist_ok=True)
os.makedirs('results/final_jsons/image', exist_ok=True)
os.makedirs('results/final_jsons/video', exist_ok=True)
os.makedirs('results/final_jsons/live', exist_ok=True)

def calculate_angle(landmark1, landmark2, landmark3):
    """
    Calculates the angle between three landmarks.
    """
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

# List of required landmarks
required_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

def get_landmarks(rgb_image):
    """
    Extracts landmarks from the given RGB image.
    """
    results = pose.process(rgb_image)
    if results.pose_landmarks:
        return results.pose_landmarks
    else:
        return None

def calculate_pose_angles(landmarks):
    """
    Calculates and returns a list of key pose angles from the landmarks.
    """
    left_elbow_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    )

    right_elbow_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    )

    left_shoulder_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    )

    right_shoulder_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    )

    left_knee_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
    )

    right_knee_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
    )

    right_hip_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    )

    left_hip_angle = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    )

    head_knee_right = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    )

    head_knee_left = calculate_angle(
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z],
        [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    )

    return [
        left_elbow_angle,
        right_elbow_angle,
        left_shoulder_angle,
        right_shoulder_angle,
        left_knee_angle,
        right_knee_angle,
        right_hip_angle,
        left_hip_angle,
        head_knee_right,
        head_knee_left
    ]

def classify_pose(angles):
    """
    Classifies the pose based on the calculated angles.
    """
    (
        left_elbow_angle,
        right_elbow_angle,
        left_shoulder_angle,
        right_shoulder_angle,
        left_knee_angle,
        right_knee_angle,
        right_hip_angle,
        left_hip_angle,
        head_knee_right,
        head_knee_left
    ) = angles

    label = "Unknown"

    if left_hip_angle >= 255.0 and left_hip_angle <= 270.0 and right_hip_angle >= 90.0 and right_hip_angle <= 110.0:
        if ((left_shoulder_angle >= 75.0 and left_shoulder_angle <= 125.0) or (right_shoulder_angle >= 70.0 and right_shoulder_angle <= 135.0)):
            label = 'Starfish'

    if (((left_shoulder_angle >= 0.0 and left_shoulder_angle <= 57.0) or (left_shoulder_angle >= 305.0 and left_shoulder_angle <= 360.0)) and ((right_shoulder_angle >= 0.0 and right_shoulder_angle <= 55.0) or (right_shoulder_angle >= 300.0 and right_shoulder_angle <= 360.0))):
        if left_knee_angle > 170.0 and left_knee_angle < 200.0 and right_knee_angle > 170.0 and right_knee_angle < 200.0:
            if head_knee_right > 155.0 and head_knee_right < 214.0 and head_knee_left >= 155.0 and head_knee_left <= 211.0:
                label = 'Log'

    if (((left_knee_angle > 40.0 and left_knee_angle < 100.0) or (left_knee_angle > 280.0 and left_knee_angle < 330.0)) and ((right_knee_angle > 40.0 and right_knee_angle < 100.0) or (right_knee_angle > 280.0 and right_knee_angle < 330.0))):
        label = 'Foetus'

    if left_hip_angle >= 80.0 and left_hip_angle <= 120.0 and right_hip_angle >= 235.0 and right_hip_angle <= 280.0:
        label = 'Free Fall'

    return label

def calculate_stress(pose_label):
    """
    Calculates stress level based on the pose label.
    """
    if pose_label == "Free Fall":
        return 75
    elif pose_label == "Foetus":
        return 65
    elif pose_label == "Starfish":
        return 55
    elif pose_label == "Log":
        return 45
    else:
        return 35

def display_annotations(image, landmarks, angles=None, avg_stress_values=None, title='Pose Detection with Annotations'):
    """
    Displays both landmarks and angles on the image.
    Displays average stress values for each interval without interfering with the 540x540 image area.
    """
    annotated_image = image.copy()
    canvas_height = 540
    canvas_width = 1380  # Extend width to add space for displaying annotations on both sides
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

    if landmarks is None:
        # Resize the original image to fit within 540x540 while preserving aspect ratio
        h, w = annotated_image.shape[:2]
        scale = min(540 / w, 540 / h)
        resized_image = cv2.resize(annotated_image, (int(w * scale), int(h * scale)))
        x_offset = (540 - resized_image.shape[1]) // 2
        y_offset = (540 - resized_image.shape[0]) // 2
        canvas[y_offset:y_offset+resized_image.shape[0], 420+x_offset:420+x_offset+resized_image.shape[1]] = resized_image

        # If no landmarks, display a message
        cv2.putText(canvas, "No landmarks detected", (canvas_width // 2 - 100, canvas_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        mp_drawing.draw_landmarks(annotated_image, landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize the annotated image to fit within 540x540 while preserving aspect ratio
        h, w = annotated_image.shape[:2]
        scale = min(540 / w, 540 / h)
        resized_image = cv2.resize(annotated_image, (int(w * scale), int(h * scale)))
        x_offset = (540 - resized_image.shape[1]) // 2
        y_offset = (540 - resized_image.shape[0]) // 2
        canvas[y_offset:y_offset+resized_image.shape[0], 420+x_offset:420+x_offset+resized_image.shape[1]] = resized_image

        if angles is not None:
            angle_names = [
                "Left Elbow Angle",
                "Right Elbow Angle",
                "Left Shoulder Angle",
                "Right Shoulder Angle",
                "Left Knee Angle",
                "Right Knee Angle",
                "Right Hip Angle",
                "Left Hip Angle",
                "Head to Right Knee Angle",
                "Head to Left Knee Angle"
            ]

            # Split annotations into two halves for displaying on both sides of the image
            half = len(required_landmarks) // 2
            left_landmarks = required_landmarks[:half]
            right_landmarks = required_landmarks[half:]
            left_angles = angles[:half]
            right_angles = angles[half:]
            left_angle_names = angle_names[:half]
            right_angle_names = angle_names[half:]

            # Add text for left landmarks and angles
            left_text_x_offset = 20
            left_text_y_offset = 20
            line_height = 20

            for landmark in left_landmarks:
                idx = landmark.value
                landmark_data = landmarks.landmark[idx]
                landmark_name = landmark.name
                text = f"{landmark_name}: [x: {landmark_data.x:.3f}, y: {landmark_data.y:.3f}, z: {landmark_data.z:.3f}]"
                cv2.putText(canvas, text, (left_text_x_offset, left_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                left_text_y_offset += line_height

            for angle_name, angle_value in zip(left_angle_names, left_angles):
                text = f"{angle_name}: {angle_value:.2f} degrees"
                cv2.putText(canvas, text, (left_text_x_offset, left_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                left_text_y_offset += line_height

            # Add text for right landmarks and angles
            right_text_x_offset = 960
            right_text_y_offset = 20

            for landmark in right_landmarks:
                idx = landmark.value
                landmark_data = landmarks.landmark[idx]
                landmark_name = landmark.name
                text = f"{landmark_name}: [x: {landmark_data.x:.3f}, y: {landmark_data.y:.3f}, z: {landmark_data.z:.3f}]"
                cv2.putText(canvas, text, (right_text_x_offset, right_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                right_text_y_offset += line_height

            for angle_name, angle_value in zip(right_angle_names, right_angles):
                text = f"{angle_name}: {angle_value:.2f} degrees"
                cv2.putText(canvas, text, (right_text_x_offset, right_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                right_text_y_offset += line_height

            # Classify the pose
            pose_label = classify_pose(angles)
            cv2.putText(canvas, f"Pose: {pose_label}", (canvas_width // 2 - 50, canvas_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Calculate stress level
            stress_level = calculate_stress(pose_label)
            cv2.putText(canvas, f"Stress Level: {stress_level}", (canvas_width // 2 - 50, canvas_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display interval average stress values
    if avg_stress_values is not None:
        interval_text_x_offset = 20
        interval_text_y_offset = 350
        line_height = 20

        for i, avg_stress in enumerate(avg_stress_values[-15:], start=1):
            text = f"Interval {i}: Average Stress: {avg_stress}"
            cv2.putText(canvas, text, (interval_text_x_offset, interval_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            interval_text_y_offset += line_height

    # Display the canvas using OpenCV
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, canvas_width, canvas_height)
    cv2.imshow(title, canvas)

def display_annotations(image, landmarks, angles=None, avg_stress_values=None, title='Pose Detection with Annotations', elapsed_time=None):
    """
    Displays both landmarks and angles on the image.
    Displays average stress values for each interval without interfering with the 540x540 image area.
    """
    annotated_image = image.copy()
    canvas_height = 540
    canvas_width = 1380  # Extend width to add space for displaying annotations on both sides
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

    if landmarks is None:
        # Resize the original image to fit within 540x540 while preserving aspect ratio
        h, w = annotated_image.shape[:2]
        scale = min(540 / w, 540 / h)
        resized_image = cv2.resize(annotated_image, (int(w * scale), int(h * scale)))
        x_offset = (540 - resized_image.shape[1]) // 2
        y_offset = (540 - resized_image.shape[0]) // 2
        canvas[y_offset:y_offset+resized_image.shape[0], 420+x_offset:420+x_offset+resized_image.shape[1]] = resized_image

        # If no landmarks, display a message
        cv2.putText(canvas, "No landmarks detected", (canvas_width // 2 - 100, canvas_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        mp_drawing.draw_landmarks(annotated_image, landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize the annotated image to fit within 540x540 while preserving aspect ratio
        h, w = annotated_image.shape[:2]
        scale = min(540 / w, 540 / h)
        resized_image = cv2.resize(annotated_image, (int(w * scale), int(h * scale)))
        x_offset = (540 - resized_image.shape[1]) // 2
        y_offset = (540 - resized_image.shape[0]) // 2
        canvas[y_offset:y_offset+resized_image.shape[0], 420+x_offset:420+x_offset+resized_image.shape[1]] = resized_image

        if angles is not None:
            angle_names = [
                "Left Elbow Angle",
                "Right Elbow Angle",
                "Left Shoulder Angle",
                "Right Shoulder Angle",
                "Left Knee Angle",
                "Right Knee Angle",
                "Right Hip Angle",
                "Left Hip Angle",
                "Head to Right Knee Angle",
                "Head to Left Knee Angle"
            ]

            # Split annotations into two halves for displaying on both sides of the image
            half = len(required_landmarks) // 2
            left_landmarks = required_landmarks[:half]
            right_landmarks = required_landmarks[half:]
            left_angles = angles[:half]
            right_angles = angles[half:]
            left_angle_names = angle_names[:half]
            right_angle_names = angle_names[half:]

            # Add text for left landmarks and angles
            left_text_x_offset = 20
            left_text_y_offset = 20
            line_height = 20

            for landmark in left_landmarks:
                idx = landmark.value
                landmark_data = landmarks.landmark[idx]
                landmark_name = landmark.name
                text = f"{landmark_name}: [x: {landmark_data.x:.3f}, y: {landmark_data.y:.3f}, z: {landmark_data.z:.3f}]"
                cv2.putText(canvas, text, (left_text_x_offset, left_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                left_text_y_offset += line_height

            for angle_name, angle_value in zip(left_angle_names, left_angles):
                text = f"{angle_name}: {angle_value:.2f} degrees"
                cv2.putText(canvas, text, (left_text_x_offset, left_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                left_text_y_offset += line_height

            # Add text for right landmarks and angles
            right_text_x_offset = 960
            right_text_y_offset = 20

            for landmark in right_landmarks:
                idx = landmark.value
                landmark_data = landmarks.landmark[idx]
                landmark_name = landmark.name
                text = f"{landmark_name}: [x: {landmark_data.x:.3f}, y: {landmark_data.y:.3f}, z: {landmark_data.z:.3f}]"
                cv2.putText(canvas, text, (right_text_x_offset, right_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                right_text_y_offset += line_height

            for angle_name, angle_value in zip(right_angle_names, right_angles):
                text = f"{angle_name}: {angle_value:.2f} degrees"
                cv2.putText(canvas, text, (right_text_x_offset, right_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                right_text_y_offset += line_height

            # Classify the pose
            pose_label = classify_pose(angles)
            cv2.putText(canvas, f"Pose: {pose_label}", (canvas_width // 2 - 50, canvas_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Calculate stress level
            stress_level = calculate_stress(pose_label)
            cv2.putText(canvas, f"Stress Level: {stress_level}", (canvas_width // 2 - 50, canvas_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display interval average stress values
    if avg_stress_values is not None:
        interval_text_x_offset = 20
        interval_text_y_offset = 350
        line_height = 20

        for i, avg_stress in enumerate(avg_stress_values[-15:], start=1):
            text = f"Interval {i}: Average Stress: {avg_stress}"
            cv2.putText(canvas, text, (interval_text_x_offset, interval_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            interval_text_y_offset += line_height

    # Display elapsed time at the top of the window
    if elapsed_time is not None:
        elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        cv2.putText(canvas, f"Elapsed Time: {elapsed_time_str}", (canvas_width // 2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the canvas using OpenCV
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, canvas_width, canvas_height)
    cv2.imshow(title, canvas)

def display_results(final_avg_stress, final_cumulative_stress, interval_stress_data, title='Final Stress Report'):
    """
    Display the final cumulative and average stress for each interval.
    Press 'q' to exit.
    """
    canvas_height = max(800, 50 + len(interval_stress_data) * 30)  # Dynamically adjust canvas height
    canvas_width = 800  # Increased width to fit long text
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

    text_x_offset = 20
    text_y_offset = 20
    line_height = 30

    for i, data in enumerate(interval_stress_data, start=1):
        text = f"Interval {i}: Cumulative Stress: {data['cumulative_stress']}, Average Stress: {data['average_stress']}"
        cv2.putText(canvas, text, (text_x_offset, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        text_y_offset += line_height

    final_text = f"Final Cumulative Stress: {final_cumulative_stress:.2f}, Final Average Stress: {final_avg_stress:.2f}"
    cv2.putText(canvas, final_text, (text_x_offset, text_y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, canvas_width, canvas_height)
    while True:
        cv2.imshow(title, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def save_to_json(file_path, data):
    return 0
    """
    Saves data to a JSON file.
    """
    #with open(file_path, 'w') as json_file:
        #json.dump(data, json_file, indent=4)

def calculate_interval_data(interval_number, cumulative_stress, frame_count, start_time, interval_stress_data):
    """
    Calculates and returns interval data.
    """
    avg_stress = cumulative_stress / frame_count if frame_count > 0 else 0
    interval_data = {
        'interval_number': interval_number,
        'time_stamp': time.strftime("%H:%M:%S", time.gmtime(start_time)),
        'cumulative_stress': cumulative_stress,
        'average_stress': avg_stress
    }
    interval_stress_data.append(interval_data)
    return interval_data

def eval_image(img_path, display=True, title='Pose Detection with Annotations'):
    """
    Evaluates the given image for pose landmarks and displays the result if specified.
    """
    interval_stress_data = []

    # Load the image
    image = cv2.imread(img_path)
    # Convert the image to RGB (since MediaPipe expects RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = get_landmarks(rgb_image)

    # Calculate angles if landmarks are detected
    angles = None
    if landmarks:
        angles = calculate_pose_angles(landmarks)
        pose_label = classify_pose(angles)
        stress_level = calculate_stress(pose_label)

        # Prepare data for JSON
        json_data = {
            'time_stamp': "00:00:00",
            'landmarks': {landmark.name: {
                'x': landmarks.landmark[landmark.value].x,
                'y': landmarks.landmark[landmark.value].y,
                'z': landmarks.landmark[landmark.value].z
            } for landmark in required_landmarks},
            'angles': {angle_name: angle for angle_name, angle in zip([
                "Left Elbow Angle",
                "Right Elbow Angle",
                "Left Shoulder Angle",
                "Right Shoulder Angle",
                "Left Knee Angle",
                "Right Knee Angle",
                "Right Hip Angle",
                "Left Hip Angle",
                "Head to Right Knee Angle",
                "Head to Left Knee Angle"
            ], angles)},
            'stress_level': stress_level
        }
        save_to_json(f'results/full_jsons/image/{os.path.basename(img_path).split(".")[0]}.json', json_data)

        # Save interval data to final JSON
        interval_data = {
        'interval_number': 0,
        'time_stamp': time.strftime("%H:%M:%S", time.gmtime(time.time())),
        'cumulative_stress': stress_level,
        'average_stress': stress_level
        }
        
        save_to_json(f'results/final_jsons/image/{os.path.basename(img_path).split(".")[0]}.json', interval_data)

    # Display annotations if needed
    if display:
        avg_stress_values=[stress_level]
        display_annotations(image, landmarks, angles, avg_stress_values, title)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    if interval_stress_data:
        display_results(stress_level, stress_level, [interval_data])

def eval_video(video_path, num_intervals=5, display=True, title='Pose Detection with Annotations'):
    """
    Evaluates the given video for pose landmarks and displays the result frame by frame.
    """
    interval_stress_data = []

    cap = cv2.VideoCapture(video_path)
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    interval_time = video_duration / num_intervals
    interval_start_time = time.time()
    json_data = []
    cumulative_stress = 0
    frame_count = 0
    interval_number = 1

    avg_stress_values = []
    start_time = time.time()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the frame to RGB (since MediaPipe expects RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = get_landmarks(rgb_frame)
            angles = None
            stress_level = 0

            if landmarks:
                angles = calculate_pose_angles(landmarks)
                pose_label = classify_pose(angles)
                stress_level = calculate_stress(pose_label)
                cumulative_stress += stress_level
                frame_count += 1

                frame_data = {
                    'time_stamp': time.strftime("%H:%M:%S", time.gmtime(time.time() - interval_start_time)),
                    'landmarks': {landmark.name: {
                        'x': landmarks.landmark[landmark.value].x,
                        'y': landmarks.landmark[landmark.value].y,
                        'z': landmarks.landmark[landmark.value].z
                    } for landmark in required_landmarks},
                    'angles': {angle_name: angle for angle_name, angle in zip([
                        "Left Elbow Angle",
                        "Right Elbow Angle",
                        "Left Shoulder Angle",
                        "Right Shoulder Angle",
                        "Left Knee Angle",
                        "Right Knee Angle",
                        "Right Hip Angle",
                        "Left Hip Angle",
                        "Head to Right Knee Angle",
                        "Head to Left Knee Angle"
                    ], angles)},
                    'stress_level': stress_level
                }
                json_data.append(frame_data)

            elapsed_time = time.time() - start_time

            if display:
                display_annotations(frame, landmarks, angles, avg_stress_values, title, elapsed_time)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if time.time() - interval_start_time >= interval_time:
                interval_data = calculate_interval_data(interval_number, cumulative_stress, frame_count, time.time() - interval_start_time, interval_stress_data)
                avg_stress_values.append(interval_data['average_stress'])
                json_data.append(interval_data)
                interval_number += 1
                interval_start_time = time.time()
                cumulative_stress = 0
                frame_count = 0

        save_to_json(f'results/full_jsons/video/{os.path.basename(video_path).split(".")[0]}.json', json_data)
        save_to_json(f'results/final_jsons/video/{os.path.basename(video_path).split(".")[0]}.json', [d for d in json_data if 'interval_number' in d])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if interval_stress_data:
            display_results(sum(avg_stress_values) / len(avg_stress_values), sum(avg_stress_values), interval_stress_data)

def eval_live(interval_time=60, display=True, title='Live Pose Detection with Annotations'):
    """
    Evaluates a live video feed for pose landmarks and displays the result frame by frame.
    """
    interval_stress_data = []

    cap = cv2.VideoCapture(0)
    live_files = [f for f in os.listdir('results/full_jsons/live') if f.startswith('live_')]
    n = max([int(f.split('_')[1].split('.')[0]) for f in live_files], default=-1) + 1
    json_data = []
    cumulative_stress = 0
    frame_count = 0
    interval_number = 1
    interval_start_time = time.time()
    start_time = time.time()

    avg_stress_values = []

    try:
        print(cap.isOpened())
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the frame to RGB (since MediaPipe expects RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = get_landmarks(rgb_frame)
            angles = None
            stress_level = 0

            if landmarks:
                angles = calculate_pose_angles(landmarks)
                pose_label = classify_pose(angles)
                stress_level = calculate_stress(pose_label)
                cumulative_stress += stress_level
                frame_count += 1

                frame_data = {
                    'time_stamp': time.strftime("%H:%M:%S", time.gmtime(time.time() - interval_start_time)),
                    'landmarks': {landmark.name: {
                        'x': landmarks.landmark[landmark.value].x,
                        'y': landmarks.landmark[landmark.value].y,
                        'z': landmarks.landmark[landmark.value].z
                    } for landmark in required_landmarks},
                    'angles': {angle_name: angle for angle_name, angle in zip([
                        "Left Elbow Angle",
                        "Right Elbow Angle",
                        "Left Shoulder Angle",
                        "Right Shoulder Angle",
                        "Left Knee Angle",
                        "Right Knee Angle",
                        "Right Hip Angle",
                        "Left Hip Angle",
                        "Head to Right Knee Angle",
                        "Head to Left Knee Angle"
                    ], angles)},
                    'stress_level': stress_level
                }
                json_data.append(frame_data)

            elapsed_time = time.time() - start_time

            if display:
                display_annotations(frame, landmarks, angles, avg_stress_values, title, elapsed_time)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if time.time() - interval_start_time >= interval_time:
                interval_data = calculate_interval_data(interval_number, cumulative_stress, frame_count, time.time() - interval_start_time, interval_stress_data)
                avg_stress_values.append(interval_data['average_stress'])
                json_data.append(interval_data)
                interval_number += 1
                interval_start_time = time.time()
                cumulative_stress = 0
                frame_count = 0

        save_to_json(f'results/full_jsons/live/live_{n}.json', json_data)
        save_to_json(f'results/final_jsons/live/live_{n}.json', [d for d in json_data if 'interval_number' in d])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if interval_stress_data:
            display_results(sum(avg_stress_values) / len(avg_stress_values), sum(avg_stress_values), interval_stress_data)

# Example usage
eval_image('sample_image.jpg', title='Pose Detection on Image')
eval_video('sample_video_60.mp4', num_intervals=5, title='Pose Detection on Video')
eval_live(interval_time=5, title='Live Pose Detection with Annotations')
