# 🤖 AI-Powered Infrared Wildlife Detection

This project utilizes a fine-tuned YOLOv12 object detection model to identify and classify 17 different species of wildlife from infrared (IR) camera trap images and videos. It features an interactive web application built with Streamlit that allows users to upload their own media for real-time analysis.

## ✨ Features

* **Real-time Detection:** Performs fast object detection on both static images and video streams.
* **Custom-Trained Model:** The YOLOv12s model has been fine-tuned on a custom dataset of 17 wildlife classes.
* **Interactive Web UI:** An easy-to-use interface built with Streamlit for uploading files and viewing results.
* **Detailed Predictions:** Displays bounding boxes, class labels, and confidence scores for each detected animal.

## 📈 Model Performance

The model was trained for 100 epochs and achieved excellent performance on the validation set:

* **mean Average Precision (mAP50-95):** **88.3%**
* **mean Average Precision (mAP50):** **98.7%**

These metrics indicate a high level of accuracy in both correctly classifying the animals and precisely locating them in the frame.

## 🛠️ Technologies Used

* Python
* PyTorch
* Ultralytics YOLO
* Streamlit
* OpenCV
* ImageIO & FFMPEG

## 🚀 Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the venv
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add Model and Dataset (Not included in repo):**
    > **Note:**The trained best.pt model is included in this repository. However, the full image and video dataset has been excluded due to its large size.
    >
    > To re-train the model or run formal evaluation, you will need to provide your own dataset and structure it according to the project's requirements.

## ▶️ Usage

To launch the Streamlit web application, make sure your virtual environment is activated and run the following command from the project's root directory:

```bash
streamlit run app.py
```
A new tab will open in your browser at `http://localhost:8501`.

## 📸 Demo

#### Image Prediction
![Image Prediction Demo](path/to/your/image_demo_screenshot.jpg)

#### Video Prediction
![Video Prediction Demo](path/to/your/video_demo_screenshot.jpg)
