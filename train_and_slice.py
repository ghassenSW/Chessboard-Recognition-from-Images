import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Inputs (Where your 2 big images are)
SRC_GOOD = os.path.join("source_images", "good.png")
SRC_BAD  = os.path.join("source_images", "bad.png")

# 2. Outputs (Where to save the slices and the model)
DATASET_DIR = "dataset_generated"
MODEL_FILE  = "tile_classifier.pkl"

# Settings
PATCH_SIZE = 32   # Pixel size of small squares
STEP_SIZE  = 20   # Overlap slice
# ==========================================

def slice_image_to_folder(img_path, output_subfolder):
    """Slices a big image into small patches and saves them."""
    if not os.path.exists(img_path):
        print(f"ERROR: Source image not found at: {img_path}")
        return False

    # Create output folder (e.g., dataset_generated/good)
    full_out_path = os.path.join(DATASET_DIR, output_subfolder)
    os.makedirs(full_out_path, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read image: {img_path}")
        return False

    h, w, c = img.shape
    count = 0
    
    print(f"--> Slicing '{img_path}'...")
    
    for y in range(0, h - PATCH_SIZE, STEP_SIZE):
        for x in range(0, w - PATCH_SIZE, STEP_SIZE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Save patch
            fname = f"{output_subfolder}_{count}.jpg"
            cv2.imwrite(os.path.join(full_out_path, fname), patch)
            count += 1
            
    print(f"    Created {count} samples in '{full_out_path}'")
    return True

def train_the_model():
    """Loads the sliced images and trains the SVM."""
    print("\n--> Loading dataset for training...")
    
    features = []
    labels = []
    
    # 0 = Bad, 1 = Good
    classes = {"bad": 0, "good": 1}
    
    found_data = False

    for label_name, label_idx in classes.items():
        folder_path = os.path.join(DATASET_DIR, label_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} is empty or missing.")
            continue
            
        for fname in os.listdir(folder_path):
            im_path = os.path.join(folder_path, fname)
            img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Resize just in case
            img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
            
            # Feature Extraction (Raw pixels + Variance)
            flat = img.flatten() / 255.0
            variance = np.var(img) / 1000.0
            feat_vec = np.append(flat, variance)
            
            features.append(feat_vec)
            labels.append(label_idx)
            found_data = True

    if not found_data:
        print("ERROR: No data found. Training aborted.")
        return

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    print(f"--> Training SVM on {len(X)} samples...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"--> Model Accuracy: {acc*100:.2f}%")
    
    # Save
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSUCCESS! Model saved to: {os.path.abspath(MODEL_FILE)}")
    print("You can now run your main chessboard detection script.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== Step 1: Slicing Images ===")
    
    # Check if sources exist
    if not os.path.exists("source_images"):
        os.makedirs("source_images")
        print("ALERT: Created 'source_images' folder.")
        print("Please put 'good.jpg' and 'bad.jpg' inside it and run this again.")
    else:
        # Run Slicing
        ok1 = slice_image_to_folder(SRC_GOOD, "good")
        ok2 = slice_image_to_folder(SRC_BAD, "bad")
        
        if ok1 and ok2:
            print("\n=== Step 2: Training Model ===")
            train_the_model()
        else:
            print("\nSkipping training because images were missing.")