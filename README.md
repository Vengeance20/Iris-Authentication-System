# 👁️ Iris Recognition System with Presentation Attack Detection

This repository contains the official implementation of the two-stage iris authentication system proposed in our project. The system integrates a Presentation Attack Detection (PAD) module to intercept spoofing attempts (e.g., printed photos and patterned contact lenses) before performing deep embedding-based identity verification.

---

## ⚙️ Installation

### 1. Clone the Repository and Create a Virtual Environment

```bash
git clone https://github.com/Vengeance20/Iris-Authentication-System.git
cd Iris-Authentication-System

conda create -n iris_env python=3.10 -y
conda activate iris_env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

An interactive web interface is provided using Streamlit.

### Start the Application

```bash
# Ensure pretrained weights are available in the models/ directory
streamlit run app.py
```

### Open the Application

Navigate to:

```text
http://localhost:8501
```

The application supports:

* User registration
* Iris authentication
* Real-time PAD verification
* End-to-end identity verification workflow

The PAD module is executed automatically before authentication to prevent spoofing attacks.

---

## 📜 Disclaimer

This software is intended solely for **academic and research purposes**. We make no guarantees regarding production readiness, security compliance, or suitability for commercial deployment.

## Our team

[Bui Cong Minh](https://github.com/Vengeance20)

Special thanks:

[Tran Ngoc Minh](https://github.com/WiuHz)

[Pham Quang Minh](https://github.com/Hiiamming)

[Dang Duc Thinh](https://github.com/ThinhDang22)

Ph.D Tran Nguyen Ngoc & Ph.D Ngo Thanh Trung for guidance during the course
