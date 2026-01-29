# 🧠 Brain Tumor Segmentation using U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Brain Tumor Detector** is a Deep Learning-based web application designed to automatically segment and detect brain tumors from MRI images. [cite_start]Built using the **U-Net architecture**, this system provides real-time analysis, visual overlays of tumor regions, and quantitative medical reports (tumor area & density) to assist medical professionals[cite: 38, 39, 171].

[cite_start]This project was developed as a Final Project for the **Computer Vision** course at **Politeknik Negeri Semarang (POLINES)**[cite: 4, 13].

---

## 📸 Demo & Screenshots

### 1. Hero Section & Landing Page
User-friendly interface with dark mode design for medical environments.
![Hero Section](web_app/static/images/screenshot_hero.png)
*(Note: Replace with your actual screenshot path inside your repository)*

### 2. Detection Result (Tumor Detected)
Accurate segmentation with red overlay and detailed quantitative dashboard.
![Detection Result](web_app/static/images/screenshot_detection.png)

### 3. Quantitative Analysis Dashboard
Real-time calculation of tumor area ($cm^2$) and density percentage.
![Dashboard](web_app/static/images/screenshot_dashboard.png)

---

## ✨ Key Features

* [cite_start]**🤖 Automated Segmentation:** Uses U-Net (CNN) to precisely pixel-wise segment tumor regions[cite: 35, 38].
* **🩺 Medical Analysis Dashboard:**
    * [cite_start]**Tumor Area Estimation:** Calculates physical size in $cm^2$[cite: 57].
    * [cite_start]**Density Calculation:** Percentage of tumor tissue vs. healthy brain tissue[cite: 57].
    * **Cluster Detection:** Counts separate tumor spots/lesions.
* [cite_start]**⚡ High Performance:** Inference speed **< 0.5 seconds** on GPU (RTX 3050)[cite: 199, 330].
* [cite_start]**🌐 Interactive Web Interface:** Built with **Flask** and **Tailwind CSS**, supporting Drag & Drop uploads[cite: 141, 189].
* [cite_start]**🛡️ Error Handling:** Validates file formats and handles non-tumor images accurately (avoiding False Positives)[cite: 220].

---

## 📂 Project Structure

The repository is organized as follows:

```text
Brain-Tumor-Segmentation-UNet/
├── dataset/             # Raw LGG MRI Segmentation Dataset & Masks
├── eval_graphs/         # Training performance plots (Loss, Accuracy, IoU, etc.)
├── training_zone/       # Jupyter Notebooks & Scripts for Model Training
├── web_app/             # Flask Application Source Code
│   ├── static/          # CSS, JS, Images, and Uploaded Files
│   ├── templates/       # HTML Templates (Hero, Detection, Result)
│   └── app.py           # Main Flask Application Server
├── requirements.txt     # Python Dependencies
└── README.md            # Project Documentation
