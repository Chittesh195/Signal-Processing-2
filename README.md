# 🚗 Drowsiness Detection System: A Digital Signal Processing Approach

A real-time, non-intrusive drowsiness detection system leveraging **Eye Aspect Ratio (EAR)** and **Fast Fourier Transform (FFT)** analysis to enhance driver safety through timely fatigue alerts.

---

## 🧠 Overview

This project monitors driver alertness using computer vision and signal processing. It continuously analyzes eye openness to detect signs of fatigue, issuing an **audio-visual alert** when drowsiness is detected.

Developed using Python libraries like `OpenCV`, `dlib`, `numpy`, and `scipy`, the system processes live video to extract facial landmarks, compute EAR, and perform FFT analysis for signal-based insight.

---

## ⚙️ How It Works

1. **Video Capture**: Uses webcam feed for real-time monitoring.
2. **Face and Eye Detection**: `dlib` detects 68 facial landmarks.
3. **EAR Calculation**: Measures eye openness using six key points around each eye.
4. **Drowsiness Detection**:
   - EAR is sampled every 10th frame.
   - A buffer stores the latest 120 EAR values.
   - FFT detects low-frequency patterns signaling fatigue.
5. **Alert System**: Triggers a visual warning and customizable beep sound when drowsiness is detected.
6. **Real-Time Visualization**:
   - Live EAR plot
   - FFT magnitude spectrum

---

## 📊 Digital Signal Processing Concepts

- **Sampling**: Downsamples video frames for efficient processing.
- **Windowing**: Maintains a sliding window of recent EAR values.
- **FFT Analysis**: Converts time-domain EAR data into frequency domain to identify slow eye blinks and closures.

---

## 🖥️ Output

- ✅ EAR and FFT plots in real-time
- ⚠️ Visual warning overlay
- 🔊 Audio alert to wake the driver

---

## 📦 Dependencies

Make sure to install the following Python libraries:

```bash
pip install opencv-python dlib scipy numpy matplotlib
```

---

## 🧪 Recommendations for Future Work

- Enhance accuracy using advanced facial landmark models.
- Integrate other physiological signals (e.g., heart rate).
- Optimize system for low-light or nighttime driving environments.

---


## 📜 License

Feel free to use or adapt this project for educational and research purposes.
