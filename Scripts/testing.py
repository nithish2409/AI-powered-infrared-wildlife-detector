import os
from ultralytics import YOLO

# --- Configuration ---
# Path to the weights of your best trained model
TRAINED_MODEL_PATH = './runs/detect/train5/weights/best.pt' 

# Path to the folder containing your new test images
TEST_IMAGES_DIR = '../test_data/' 

# Path to the directory where you want to save the results
SAVE_DIR = '../test_results/'
# --- End Configuration ---

def run_inference():
    """
    Loads a trained model, runs inference on a directory of images,
    and saves the results.
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

    # Check if the test images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Error: Test images directory not found at {TEST_IMAGES_DIR}")
        return
        
    # Run prediction on the entire folder
    # 'project' and 'name' arguments control the save location.
    print(f"Running predictions on images in '{TEST_IMAGES_DIR}'...")
    results = model.predict(
        source=TEST_IMAGES_DIR,
        save=True,
        project=SAVE_DIR,
        name='results'  # This will create a 'results' subfolder in SAVE_DIR
    )
    
    # The 'results' object is a generator, so we can iterate to confirm
    file_count = len(os.listdir(TEST_IMAGES_DIR))
    print(f"\nPrediction complete for {file_count} images.")
    
    # The final save path will be something like 'test_results/results'
    final_save_path = os.path.join(SAVE_DIR, 'results')
    print(f"Result images are saved in: {os.path.abspath(final_save_path)}")


if __name__ == '__main__':
    run_inference()