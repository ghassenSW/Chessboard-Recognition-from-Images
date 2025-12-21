"""
YOLOv8 Training Script for Chess Piece Detection
Trains a YOLOv8 model to detect and classify chess pieces
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolov8_chess():
    """
    Train YOLOv8 model on chess piece dataset
    """
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Configuration
    data_yaml = str(current_dir / "chessboard project" / "data.yaml")
    
    # Training parameters
    epochs = 100  # Number of training epochs
    img_size = 640  # Image size (YOLOv8 default is 640)
    batch_size = 16  # Adjust based on your GPU memory
    model_size = 'n'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
    # Create results directory
    results_dir = current_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("YOLOv8 Chess Piece Detection Training")
    print("="*60)
    print(f"Dataset config: {data_yaml}")
    print(f"Model size: YOLOv8{model_size}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    print()
    
    # Load pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='chess_detection',
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        project='runs/chess',
        
        # Augmentation parameters (optional, adjust as needed)
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,    # Image HSV-Saturation augmentation
        hsv_v=0.4,    # Image HSV-Value augmentation
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,    # Image scale (+/- gain)
        shear=0.0,    # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction)
        flipud=0.0,   # Image flip up-down (probability)
        fliplr=0.5,   # Image flip left-right (probability)
        mosaic=1.0,   # Image mosaic (probability)
        mixup=0.0,    # Image mixup (probability)
    )
    
    print()
    print("="*60)
    print("Training completed!")
    print("="*60)
    print(f"Results saved to: {results.save_dir}")
    print()
    
    # Validate the model
    print("Validating model on validation set...")
    metrics = model.val()
    
    print()
    print("="*60)
    print("Validation Metrics:")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print()
    
    # Export model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    last_model_path = Path(results.save_dir) / "weights" / "last.pt"
    
    print("="*60)
    print("Model Saved:")
    print("="*60)
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print("="*60)
    
    return model, results


def test_model(model_path=None):
    """
    Test the trained model on test set
    
    Args:
        model_path: Path to trained model weights. If None, uses last trained model
    """
    # Get current script directory
    current_dir = Path(__file__).parent
    
    if model_path is None:
        # Use the best model from latest training run
        model_path = str(current_dir / "runs" / "chess" / "chess_detection" / "weights" / "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first or provide a valid model path")
        return
    
    print("="*60)
    print("Testing Model")
    print("="*60)
    print(f"Model: {model_path}")
    print()
    
    # Load trained model
    model = YOLO(model_path)
    
    # Test on test set
    data_yaml = str(current_dir / "chessboard project" / "data.yaml")
    metrics = model.val(data=data_yaml, split='test')
    
    print()
    print("="*60)
    print("Test Set Metrics:")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*60)
    
    return metrics


def predict_on_image(image_path, model_path=None):
    """
    Run inference on a single image
    
    Args:
        image_path: Path to image
        model_path: Path to trained model weights
    """
    # Get current script directory
    current_dir = Path(__file__).parent
    
    if model_path is None:
        model_path = str(current_dir / "runs" / "chess" / "chess_detection" / "weights" / "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Running inference on: {image_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        save=True,
        save_txt=True,
        conf=0.25,  # Confidence threshold
        iou=0.45,   # NMS IoU threshold
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    print(f"Results saved to: {results[0].save_dir}")
    
    return results


if __name__ == "__main__":
    # Train the model
    print("Starting YOLOv8 training for chess piece detection...")
    print()
    
    # Option 1: Train the model
    # model, results = train_yolov8_chess()
    
    # Option 2: Test an existing model (uncomment to use)
    # test_model(r"C:\Users\ghass\OneDrive\Desktop\practice projects\chessboard to FEN\runs\chess\chess_detection\weights\best.pt")
    
    # Option 3: Run inference on a single image (uncomment to use)
    predict_on_image(r"C:\Users\ghass\Downloads\php9v9aMF.jpeg")
    
    print()
    print("All done! 🎉")
    print()
    print("To use the trained model:")
    print("1. For validation: test_model('runs/chess/chess_detection/weights/best.pt')")
    print("2. For prediction: predict_on_image('image.jpg', 'runs/chess/chess_detection/weights/best.pt')")
