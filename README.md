Face Mask Detection System

A real-time AI-powered face mask detection system that identifies whether a person is wearing a mask or not — built to support public safety and health monitoring.

Why This Project?

During and after the pandemic, monitoring mask compliance in public spaces became essential. This project automates that process using deep learning, reducing the need for manual monitoring.

Features

-  AI-powered mask detection using Deep Learning
-  Trained on 7000+ real face images
-  Achieved 98% validation accuracy
-  Upload any face image and get instant results
-  Web app built with Streamlit
-  Detects both masked and unmasked faces

Tech Stack

- Language: Python
- Deep Learning: TensorFlow, Keras
- Model: MobileNetV2 (Transfer Learning)
- Computer Vision: OpenCV
- Web App: Streamlit
- Training Platform: Google Colab

Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~98% |
| Epochs | 10 |
| Dataset Size | 7000+ images |

How to Run

```bash
# Install dependencies
pip install streamlit opencv-python numpy pillow

# Run the app
streamlit run app.py
```

Developer

Jamuna Devi R  
MCA Graduate 
Passionate about AI, Machine Learning and building technology that creates real-world impact.
