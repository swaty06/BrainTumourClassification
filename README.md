# 🧠 Brain Tumor Classification using ResNet18

This repository contains a deep learning-based image classification system to detect brain tumors from MRI scans. The model is trained using a modified ResNet18 architecture in PyTorch and deployed using Streamlit, offering an intuitive web interface for medical image classification.

---

## 📌 Motivation

Brain tumor detection using MRI imaging is a crucial step in early diagnosis and treatment planning. Manual interpretation can be time-consuming and subject to variability. This project leverages deep learning to automate classification, making the process faster and more consistent.

---

## 📷 Dataset

The dataset contains MRI brain scan images categorized into two classes:

- **Yes**: Brain tumor present  
- **No**: No tumor detected

> 📁 *Due to licensing restrictions, the dataset is not included. You can use publicly available brain MRI datasets from sources like [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).*

---

## 🧠 Model Architecture

- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Modified Layers**:
  - Final fully connected layer: `nn.Linear(in_features, 2)`
  - Optional: `nn.Dropout(p=0.5)` for regularization
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Input Size**: 224 x 224 RGB

---

## 🔍 Project Structure

```
brain-tumor-classification/
│
├── app.py # Streamlit web application
├── modal_helper.py # Model definition and prediction function
├── saved_modelbrain.pth # Trained model weights (handled via Git LFS or external link)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── assets/
└── example_image.jpg # Optional example image for README
```

## 📊 Results

- **Model Accuracy:** 90%
- **Evaluation Metric:** Accuracy, Precision, Recall, F1-score


## 🖼️ Streamlit Web App

A simple and interactive **Streamlit** interface allows users to upload an image and receive real-time predictions on the image. The app validates the image, resizes it, and shows the predicted class.

To run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**📦 Future Improvements**

- ✅ Add **Grad-CAM** or other interpretability tools to visualize tumor regions and understand model decisions  
- 🧪 Improve model performance using **data augmentation** (e.g., rotation, flipping) and **regularization** (e.g., dropout, weight decay)  
- 🔁 Experiment with **alternative architectures** like EfficientNet, DenseNet, or Vision Transformers for improved accuracy  
- ☁️ Deploy the model on **Streamlit Cloud**, **Hugging Face Spaces**, or package it in a **Docker container** for scalable access  
- 🌐 Add **multilingual support** in the Streamlit interface (e.g., English, German) to support international users and clinicians  
- 🗂️ Integrate with a secure backend **database** to store patient uploads, predictions, and timestamped logs for review  
- 📱 Develop a lightweight **mobile or web-based diagnostic app** for on-the-go MRI image classification in clinical settings  


---

**🙌 Acknowledgments**

- 🤖 Model inspired by the need for **automated waste segregation** to support recycling and sustainability  
- 🗃️ Dataset Source: *[[(https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)]]*  
- 📚 Special thanks to open-source contributors and the **Kaggle** community for insights and feedback  
- 💡 Built using frameworks like **PyTorch/TensorFlow** (choose one), **Streamlit**, and **PIL**
