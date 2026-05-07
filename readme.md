# 🔍 TruthLens AI  
### Detecting truth in a world of synthetic reality

TruthLens AI is a real-time deepfake detection system built using Convolutional Neural Networks (CNN) with transfer learning. The system analyzes facial images and predicts whether they are **real or AI-generated (deepfake)**.

---

## 🚀 Features

- 🧠 Deep Learning-based classification (Real vs Fake)
- ⚡ Real-time inference using Streamlit
- 📸 Image upload support
- 📊 Confidence score visualization
- 🧍 Face detection validation using OpenCV
- 🎯 ~90% accuracy on test dataset

---

## 🏗️ Tech Stack

- **Frontend/UI:** Streamlit  
- **Model:** MobileNetV2 (Transfer Learning)  
- **Framework:** TensorFlow / Keras  
- **Image Processing:** OpenCV, PIL  
- **Language:** Python  

---

## 🧠 Model Details

- Pretrained model: `MobileNetV2`
- Fine-tuned on deepfake dataset
- Input size: `224 x 224`
- Output: Binary classification (Real / Fake)
- Loss Function: Binary Crossentropy
- Optimizer: Adam (low learning rate for fine-tuning)

---

## 📂 Project Structure

```

.
├── app.py                  # Streamlit UI
├── deepfake_model.h5       # Trained model
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

````

---

## ▶️ How to Run Locally

```bash
# Clone repository
git clone https://github.com/RishiSharmapro/TruthLens-AI.git

# Navigate to project
cd TruthLens-AI

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
````

---

## 🌐 Live Demo

👉 (Add your Streamlit deployed link here)

---

## 🧪 Testing

The system was tested on:

* Real human face images
* AI-generated faces
* Various lighting and resolution conditions

---

## ⚠️ Limitations

* Performance may drop on:

  * Extremely low-quality images
  * Highly realistic deepfakes
* Face detection is required for accurate results

---

## 🔮 Future Improvements

* 🎥 Video-based deepfake detection
* 🔥 Grad-CAM visualization (explainability)
* 🌐 Full-stack deployment (MERN integration)
* 📱 Mobile compatibility

---

## 👨‍💻 Author

**Rishi Sharma**

🌐 [Portfolio](https://rishisharmapro.vercel.app)  
💼 [GitHub](https://github.com/rishisharmapro)


---

## 📜 License

This project is for academic and research purposes.
