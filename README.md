# Drowsiness Detection System  
*A Digital Signal Processing Approach for Driver Safety*

---

## 📝 Abstract

This project presents a real-time drowsiness detection system based on **Eye Aspect Ratio (EAR)** analysis and **Fast Fourier Transform (FFT)** techniques. Designed to monitor driver alertness, the system utilizes computer vision to track eye movements, analyze EAR patterns, and issue alerts upon detecting signs of fatigue. By integrating signal processing methods, this solution offers a reliable and non-intrusive approach to enhancing road safety.

---

## 📌 Project Objectives

- Monitor driver drowsiness through eye movement analysis.
- Utilize EAR to detect prolonged eye closure and blinking behavior.
- Apply FFT on EAR data to identify low-frequency fatigue patterns.
- Alert the driver through visual and audio cues when drowsiness is detected.

---

## ⚙️ System Workflow

1. **Video Acquisition**  
   - Captures live video from the computer's webcam using OpenCV.

2. **Facial Landmark Detection**  
   - Utilizes the `dlib` library to detect facial features, particularly eye landmarks.

3. **EAR Calculation**  
   - Computes Eye Aspect Ratio using six landmark points around each eye to quantify eye openness.

4. **Data Sampling and Windowing**  
   - EAR values are sampled every 10th frame.
   - Maintains a sliding window of 120 samples for real-time analysis.

5. **FFT-Based Drowsiness Detection**  
   - Performs frequency domain analysis on the EAR data.
   - Focuses on low-frequency components to detect slow blinks or prolonged closure.

6. **Alert Mechanism**  
   - Displays an on-screen warning and plays an audio alert (via `winsound`) when drowsiness is detected.

7. **Visualization**  
   - Real-time plotting of EAR values and their FFT spectrum to aid understanding and debugging.

---

## 📊 Digital Signal Processing Concepts

- **Sampling**: Reduces data load by analyzing every 10th video frame.
- **Windowing**: Ensures analysis is always based on the most recent 120 samples.
- **FFT (Fast Fourier Transform)**: Detects frequency patterns that suggest drowsiness (e.g., slow blinking).

---

## 🔧 Requirements

Ensure the following Python libraries are installed:

```bash
pip install opencv-python dlib scipy numpy matplotlib
```

---

## 💡 Recommendations for Future Enhancements

- Integrate more advanced facial detection models for increased robustness.
- Incorporate additional physiological signals (e.g., heart rate) for multimodal analysis.
- Improve performance in low-light environments for nighttime usability.

---

## 📄 License

This project is intended for educational and research use only. Feel free to modify or extend it with proper attribution.
