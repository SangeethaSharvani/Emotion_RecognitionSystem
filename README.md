
# Facial Emotion Detection 🎭

This project is a **real-time facial emotion detection system** built using **Teachable Machine, TensorFlow/Keras, OpenCV, and Python**.
It uses a webcam feed to classify human emotions such as **Happy, Sad, Angry, Disgust, Surprise, Neutral, etc.** based on facial expressions.

---

## 🚀 Features

* Real-time webcam-based facial emotion recognition
* Trained using **Kaggle emotion dataset** via **Teachable Machine**
* Exports model as `.h5` and `labels.txt`
* Uses **TensorFlow/Keras** for predictions and **OpenCV** for camera access
* Custom training setup:

  * **65 epochs**
  * **Batch size:** 16
  * **Learning rate:** 0.00102

---

## 📂 Dataset

* Dataset: [Kaggle Facial Emotion Dataset](https://www.kaggle.com/datasets) (custom selection used)
* Classes trained: `Happy, Sad, Angry, Disgust, Surprise, Neutral` (modifiable in Teachable Machine).

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/facial-emotion-detection.git
   cd facial-emotion-detection
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow keras opencv-python numpy
   ```

3. Place the following files inside the project folder:

   * `keras_Model.h5` → trained model from Teachable Machine
   * `labels.txt` → class labels from Teachable Machine

---

## ▶️ Usage

Run the script to start webcam emotion detection:

```bash
python detect_emotion.py
```


    

## 🎯 Output Example

* Shows webcam window
* Prints detected emotion + confidence in console

Example:

```
Class: Happy  Confidence: 95 %
Class: Sad    Confidence: 88 %
```

---

## 📌 Future Improvements

* Add bounding box around detected faces
* Train with larger dataset for higher accuracy
* Deploy as a **web app** (Flask/Streamlit)
* Mobile version (TensorFlow Lite)

---

## 🤝 Contribution

Pull requests are welcome. For major changes, open an issue first to discuss.

---

## 📜 License

This project is licensed under the MIT License.

---

