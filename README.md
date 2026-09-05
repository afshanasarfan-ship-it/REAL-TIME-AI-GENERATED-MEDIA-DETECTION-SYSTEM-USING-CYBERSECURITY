# Real-Time AI-Generated Media Detection System

A multi-modal deep learning platform designed to detect synthetic manipulations (deepfakes) across **Image**, **Video**, and **Audio** formats within a unified, real-time web application.

---

## 📌 Project Overview

With the rapid advancement of generative AI, hyper-realistic media manipulations pose significant threats to digital trust and security. This system provides an end-to-end media verification pipeline that analyzes uploaded files, detects deepfake artifacts using deep learning models, and securely records detection logs for auditing.

* **Multi-Modal Analysis:** Inspects static images, video streams, and audio recordings.
* **Real-Time Verification:** Delivers instant prediction results with confidence percentage metrics and **REAL** / **AI FAKE** status badges.
* **Persistent Audit Logging:** Automatically logs all scan history, confidence scores, and execution metadata into a relational database.

---

## 🛠️ Tech Stack & Tools

* **Core Language:** Python
* **User Interface:** Streamlit
* **Deep Learning Framework:** PyTorch (ResNet-18)
* **Media Processing Utilities:**
  * **Image:** PIL (Pillow) & Torchvision
  * **Video:** OpenCV (`cv2`) for frame-by-frame extraction
  * **Audio:** Librosa for spectral feature analysis
* **Database Backend:** PostgreSQL (`DS_detection_db`)

---

## 🧱 System Architecture & Modules

The application is structured into four modular components:

1. `ui_dashboard.py` — The front-end user interface built with Streamlit for file uploads and live media previews.
2. `media_processor.py` — Handles domain-specific preprocessing (resizing images to $224 \times 224$, extracting video keyframes, and calculating audio spectral variance).
3. `ai_detector.py` — The machine learning inference engine executing the pre-trained ResNet-18 neural network (`deepfake_resnet18.pth`).
4. `result_engine.py` — Aggregates model probabilities, determines final classification verdicts, and writes persistent audit records to PostgreSQL via `psycopg2`.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your local environment:
* Python 3.8 or higher
* PostgreSQL Database Server

### Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
2. Install Dependencies:
   pip install -r requirements.txt
3.Configure Database:
  Ensure PostgreSQL is running locally and update your connection parameters in result_engine.py:
  Database Name: DS_detection_db
  Host: localhost
  Port: 5432
4.Run the Application:

Bash
streamlit run ui_dashboard.py

📋 System Workflow
[ User Upload ] ──► [ Streamlit UI ] ──► [ Media Processor (OpenCV / Librosa) ]
                                                   │
                                                   ▼
[ PostgreSQL DB ] ◄── [ Result Engine ] ◄── [ PyTorch ResNet-18 Model ]
