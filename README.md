# 🧠 Brain Tumor Segmentation using U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?style=for-the-badge&logo=flask&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Brain Tumor Detector** is a deep learning-based web application designed to automatically segment and detect brain tumors from MRI images. Built using the **U-Net architecture**, this system provides real-time analysis, visual tumor overlays, and quantitative medical reports (tumor area & density) to assist medical professionals.

This project was developed as a Final Project for the **Computer Vision** course at **Politeknik Negeri Semarang (POLINES)**.

---

## ✨ Key Features

- 🤖 **Automated Segmentation** — Pixel-wise tumor segmentation using U-Net CNN  
- 🩺 **Medical Analysis Dashboard**
  - Tumor Area Estimation ($cm^2$)
  - Tumor Density Calculation (%)
  - Tumor Cluster Detection
- ⚡ **High Performance** — Inference time < 0.5 seconds (GPU RTX 3050)
- 🌐 **Interactive Web Interface** — Flask backend with Tailwind CSS frontend
- 🛡️ **Error Handling** — Validates file formats and minimizes false positives

---

## 📂 Project Structure

```text
Brain-Tumor-Segmentation-UNet/
├── dataset/              # Raw LGG MRI Dataset & Masks
├── eval_graphs/          # Training performance plots
├── training_zone/        # Jupyter Notebooks for training
├── web_app/              # Flask Web Application
│   ├── static/           # CSS, JS, Images, Uploads
│   ├── templates/        # HTML Templates
│   └── app.py            # Main Flask Server
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## 🛠️ Tech Stack

### 🧠 Deep Learning & AI
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![U-Net](https://img.shields.io/badge/Architecture-U--Net-blueviolet?style=for-the-badge&logo=google-colab&logoColor=white)

### 🖼️ Image Processing & Data
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### 🌐 Web Development
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

### 📊 Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## 📊 Model Performance

Model trained on **LGG MRI Segmentation Dataset** for **20 epochs**.

| Metric | Score | Description |
|-------|-------|-------------|
| Pixel Accuracy | **99.20%** | Overall pixel classification accuracy |
| Dice Coefficient | **0.8710** | Segmentation similarity score (F1) |
| Precision | **0.9450** | Low false positive rate |
| Recall | **0.8500** | Sensitivity to tumor regions |
| IoU Score | **0.7750** | Overlap accuracy |

---

## 🚀 Installation & Usage

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-Segmentation-UNet.git
cd Brain-Tumor-Segmentation-UNet
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Web Application
```bash
cd web_app
python app.py
```

Open your browser and go to:  
👉 **http://127.0.0.1:5000**

---

## 👤 Author

**Muhammad Rafa Enrico**  
Student ID: 4.33.24.2.15  
Major: Computer Engineering Technology  
State Polytechnic of Semarang

---

## 🤝 Acknowledgments

- Ir. Prayitno, S.ST., M.T., Ph.D. — Computer Vision Lecturer  
- Contributors of the **LGG MRI Segmentation Dataset (TCGA/Kaggle)**
