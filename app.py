import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import shutil
import imageio
import cv2

# --- Configuration ---
MODEL_PATH = './Scripts/runs/detect/train5/weights/best.pt'
TEMP_DIR = "temp_uploads"
RESULTS_DIR = "prediction_outputs"

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)


# --- Main App ---
st.set_page_config(page_title="AI-Powered Infrared Wildlife Detection", layout="wide")
st.title("🤖 AI-Powered Infrared Wildlife Detection")
st.markdown("Upload an image or video, and the AI model will detect the animals present.")

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model(MODEL_PATH)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image or video file",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
)

if model is not None and uploaded_file is not None:
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)

    sanitized_filename = uploaded_file.name.replace(" ", "_")
    temp_file_path = os.path.join(TEMP_DIR, sanitized_filename)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_extension = os.path.splitext(sanitized_filename)[1].lower()
    
    # --- IMAGE PROCESSING ---
    if file_extension in ['.jpg', '.jpeg', '.png']:
        st.header("Results for Image")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(temp_file_path, caption="Original Image", use_container_width=True)

        with st.spinner('Detecting animals...'):
            results = model.predict(temp_file_path, conf=0.5)
            result = results[0]
            plotted_image = result.plot()
            
            with col2:
                st.image(plotted_image, caption="Predicted Image", channels="BGR", use_container_width=True)

            # --- TEXT SUMMARY SECTION REMOVED ---

    # --- VIDEO PROCESSING ---
    elif file_extension in ['.mp4', 'avi', 'mov']:
        st.header("Results for Video")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Video")
            st.video(temp_file_path)

        status_message = st.empty()
        status_message.info('Processing video frame by frame...')
        
        processed_frames = []
        results_generator = model.predict(temp_file_path, stream=True, conf=0.5)
        
        for result in results_generator:
            processed_frames.append(result.plot())
        
        if processed_frames:
            status_message.info('Creating output video...')
            
            cap = cv2.VideoCapture(temp_file_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            output_video_path = os.path.join(RESULTS_DIR, "processed_video.mp4")
            
            writer = imageio.get_writer(output_video_path, fps=original_fps, codec='libx264')
            for frame in processed_frames:
                writer.append_data(frame[..., ::-1])
            writer.close()

            with col2:
                st.subheader("Predicted Video")
                st.video(output_video_path)
            
            status_message.empty()
        else:
            with col2:
                st.info("No frames were processed.")
            status_message.empty()
        
        # --- TEXT SUMMARY SECTION REMOVED ---
    
    os.unlink(temp_file_path)

else:
    st.info("Please upload a file to get started.")