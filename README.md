# 🚗 Driver Drowsiness Detection System

An AI-powered **Driver Monitoring System** that detects driver fatigue in real time using computer vision.
The system analyzes facial landmarks to monitor eye movement, blinking patterns, and fatigue indicators, displaying results through an interactive dashboard.

---

## 📌 Features

* Real-time driver monitoring using webcam
* Eye Aspect Ratio (EAR) based drowsiness detection
* Blink detection and fatigue scoring
* Live face detection with bounding box
* Interactive dashboard built with Streamlit
* Driver fatigue history graph and alerts

---

## 🧠 Technologies Used

* Python
* OpenCV
* dlib
* Streamlit
* NumPy
* SciPy

---

## 📂 Project Structure

```
driver-drowsiness-detection
│
├── dashboard
│   └── app.py
│
├── models
│   └── shape_predictor_68_face_landmarks.dat
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

**Windows**

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 📥 Download Model

Download the facial landmark model:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract the file and place it inside:

```
models/
```

---

## ▶️ Run the Application

```
streamlit run dashboard/app.py
```

The dashboard will open in your browser and start the **AI driver monitoring system**.

---

## 📊 Dashboard Output

The system provides:

* Live driver camera feed
* Face detection with bounding box
* Eye Aspect Ratio (EAR) monitoring
* Blink count and fatigue score
* Fatigue history graph

---

## 🚀 Future Improvements

* Yawning detection
* Audio fatigue alerts
* Driver identity verification
* Cloud-based fleet monitoring

---

## 📜 License

This project is for educational and research purposes.
