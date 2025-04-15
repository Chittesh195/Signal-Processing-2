from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Constants
FREQUENCY = 2500  # Beep frequency in Hz
DURATION = 5000   # Beep duration in milliseconds (5 seconds)
EAR_THRESH = 0.25  # Eye Aspect Ratio threshold
EAR_FRAMES = 9 # Consecutive frames for drowsiness detection
FFT_SIZE = 100    # Number of frames to collect for FFT

# Initialize Matplotlib figure
plt.ion()  # Interactive mode for real-time plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Setup real-time EAR plot with deque for rolling window
ear_buffer = deque(maxlen=FFT_SIZE)
line, = ax1.plot([], [], 'b-', label='EAR')
ax1.set_ylim([0, 0.5])
ax1.set_xlabel('Frames')
ax1.set_ylabel('EAR')
ax1.set_title('Real-time EAR Plot')

# Setup FFT plot
fft_line, = ax2.plot([], [], 'r-')
ax2.set_xlim(0, FFT_SIZE // 2)
ax2.set_xlabel("Frequency")
ax2.set_ylabel("Magnitude")
ax2.set_title("FFT of Eye Aspect Ratio (EAR)")
ax2.grid()

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Function to update the FFT plot
def update_fft_plot():
    ear_values = list(ear_buffer)
    fft_result = np.fft.fft(ear_values, n=FFT_SIZE)
    fft_magnitude = np.abs(fft_result[:FFT_SIZE // 2])

    fft_line.set_data(range(len(fft_magnitude)), fft_magnitude)
    ax2.set_ylim(0, max(fft_magnitude) + 1)  # Adjust y-axis based on max magnitude
    ax2.set_xlim(0, len(fft_magnitude))
    fig.canvas.draw()
    fig.canvas.flush_events()

def main():
    # Initialize webcam and dlib face detector
    cam = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Get coordinates for the left and right eye landmarks
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    count = 0  # Drowsiness counter
    drowsiness_detected = False  # Track drowsiness state

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        rects = detector(gray, 0)

        # If faces are detected, process for EAR
        if len(rects) > 0:
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Add EAR to buffer
                ear_buffer.append(ear)

                # Update real-time EAR plot with deque buffer
                line.set_data(range(len(ear_buffer)), list(ear_buffer))
                ax1.set_xlim(0, max(FFT_SIZE, len(ear_buffer)))

                # Draw contours around the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

                # Check if drowsiness is detected
                if ear < EAR_THRESH:
                    count += 1
                    if count >= EAR_FRAMES and not drowsiness_detected:
                        drowsiness_detected = True
                        cv2.putText(frame, "DROWSINESS DETECTED", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        winsound.Beep(FREQUENCY, DURATION)  # 5-second sound alert
                        update_fft_plot()  # FFT plot update on detection
                        time.sleep(2)  # Wait to avoid multiple alarms
                else:
                    count = 0
                    drowsiness_detected = False
        else:
            # If no face detected, skip EAR update for continuity in FFT plot
            if ear_buffer:
                line.set_data(range(len(ear_buffer)), list(ear_buffer))
                ax1.set_xlim(0, max(FFT_SIZE, len(ear_buffer)))

        # Display the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Redraw plots for real-time update
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Clean up
    cam.release()
    cv2.destroyAllWindows()
    plt.close("all")  # Close all Matplotlib figures

if __name__ == "__main__":
    main()
