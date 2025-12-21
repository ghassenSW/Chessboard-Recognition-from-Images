"""
Dataset Visualization Script
Visualizes images with bounding boxes from annotation labels
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_yolo_annotation(label_path, img_width, img_height):
    """
    Read YOLO format annotation file and convert to pixel coordinates.
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    
    Args:
        label_path: Path to the label file
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of bounding boxes: [(class_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to corner coordinates
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            boxes.append((class_id, x1, y1, x2, y2))
    
    return boxes


def read_class_names(classes_file):
    """
    Read class names from file.
    
    Args:
        classes_file: Path to classes.txt or _classes.txt
    
    Returns:
        List of class names
    """
    if not os.path.exists(classes_file):
        return []
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return classes


def visualize_image_with_boxes(image_path, label_path, class_names=None, 
                                box_color=(0, 255, 0), thickness=2):
    """
    Visualize a single image with bounding boxes.
    
    Args:
        image_path: Path to the image
        label_path: Path to the label file
        class_names: List of class names (optional)
        box_color: Color for bounding boxes (B, G, R) - default green
        thickness: Box line thickness
    
    Returns:
        Image with bounding boxes drawn
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Read annotations
    boxes = read_yolo_annotation(label_path, img_width, img_height)
    
    # Draw boxes
    for box in boxes:
        class_id, x1, y1, x2, y2 = box
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        
        # Add label text
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness=1
        )
        
        # Draw background for text
        cv2.rectangle(
            img, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width, y1), 
            box_color, 
            -1
        )
        
        # Put text
        cv2.putText(
            img, 
            label, 
            (x1, y1 - 5), 
            font, 
            font_scale, 
            (0, 0, 0),  # Black text
            thickness=1
        )
    
    return img


def visualize_dataset(image_dir, label_dir, classes_file=None, 
                       num_samples=9, save_path=None):
    """
    Visualize multiple images from dataset with bounding boxes.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing label files
        classes_file: Path to classes file (optional)
        num_samples: Number of images to display
        save_path: Path to save the visualization (optional)
    """
    # Read class names
    class_names = None
    if classes_file and os.path.exists(classes_file):
        class_names = read_class_names(classes_file)
        print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Limit to num_samples
    image_files = list(image_files)[:num_samples]
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # Visualize images
    for idx, image_path in enumerate(image_files):
        row = idx // cols
        col = idx % cols
        
        # Get corresponding label file
        label_path = Path(label_dir) / (image_path.stem + '.txt')
        
        # Visualize
        img_with_boxes = visualize_image_with_boxes(
            image_path, 
            label_path, 
            class_names
        )
        
        if img_with_boxes is not None:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(image_path.name, fontsize=10)
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    # Hide extra subplots
    for idx in range(len(image_files), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def main():
    """
    Main function to run dataset visualization.
    Update these paths according to your dataset structure.
    """
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # === CONFIGURE THESE PATHS ===
    image_dir = str(current_dir / "chessboard project" / "test" / "images")  # Directory with images
    label_dir = str(current_dir / "chessboard project" / "test" / "labels")  # Directory with YOLO format labels
    classes_file = str(current_dir / "chessboard project" / "_classes.txt")  # Optional: file with class names
    
    # Number of samples to visualize
    num_samples = 9
    
    # Optional: save visualization
    save_path = str(current_dir / "dataset_visualization.png")
    
    # === RUN VISUALIZATION ===
    print("Starting dataset visualization...")
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    
    visualize_dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        classes_file=classes_file,
        num_samples=num_samples,
        save_path=save_path
    )
    
    print("Done!")


if __name__ == "__main__":
    # Example usage with command line or direct configuration
    
    # Option 1: Run main() with configured paths
    main()
    
    # Option 2: Direct usage example
    # visualize_dataset(
    #     image_dir="./data/images",
    #     label_dir="./data/labels",
    #     classes_file="./data/_classes.txt",
    #     num_samples=9,
    #     save_path="visualization.png"
    # )
