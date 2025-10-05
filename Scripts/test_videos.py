import os
import glob
from ultralytics import YOLO

# --- Configuration ---
# Path to the weights of your best trained model
TRAINED_MODEL_PATH = './runs/detect/train5/weights/best.pt' 

# Path to the folder containing your test videos
VIDEOS_DIR = '../test_videos/' 

# Path to the directory where you want to save the results
SAVE_DIR = '../video_results/'
# --- End Configuration ---

def run_video_inference():
    """
    Loads a trained model, finds all videos in a directory,
    runs inference on each, and saves the results.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load your custom-trained model
    try:
        model = YOLO(TRAINED_MODEL_PATH)
        print(f"Successfully loaded model from {TRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Find all video files in the specified directory
    video_files = glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')) + \
                  glob.glob(os.path.join(VIDEOS_DIR, '*.avi')) + \
                  glob.glob(os.path.join(VIDEOS_DIR, '*.mov'))
    
    if not video_files:
        print(f"No video files found in '{VIDEOS_DIR}'. Please check the path.")
        return
        
    print(f"Found {len(video_files)} videos to process...")
    
    # Loop through each video file and run prediction
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        print(f"\nProcessing video: {video_filename}...")
        
        # Run prediction
        # Each video's results will be saved in a uniquely named subfolder
        results = model.predict(
            source=video_path,
            save=True,
            project=SAVE_DIR,
            name=f'result_{os.path.splitext(video_filename)[0]}' # Creates a unique folder for each video
        )
    
    print("\n\nAll videos processed!")
    print(f"Result videos are saved in subfolders inside: {os.path.abspath(SAVE_DIR)}")


if __name__ == '__main__':
    run_video_inference()