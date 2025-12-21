# Chess Piece Detection & Board Segmentation

A YOLOv8-based chess piece detection system with chessboard square segmentation for converting chess positions to FEN notation.

## Features

- **Chess Piece Detection**: YOLOv8 model trained to detect 12 chess piece classes
- **Dataset Visualization**: View training data with bounding boxes
- **Chessboard Segmentation**: Detect and extract 64 squares from chessboard images
- **FEN Conversion**: Convert chessboard images to FEN notation (work in progress)

## Project Structure

```
chessboard to FEN/
├── chessboard project/      # Dataset (Roboflow format)
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── data.yaml
│   └── _classes.txt
├── detect_classify.py       # Dataset visualization with bounding boxes
├── detect_chessboard.py     # Chessboard square detection pipeline
├── train_yolov8.py         # YOLOv8 training script
├── find_zero.py            # Utility to find specific class annotations
└── Roboflow_Yolov3.ipynb   # Original YOLOv3 notebook (TF2 updated)
```

## Classes

The model detects 13 classes (0-12):
- 0: none
- 1-6: Black pieces (bishop, king, knight, pawn, queen, rook)
- 7-12: White pieces (bishop, king, knight, pawn, queen, rook)

## Installation

```bash
pip install ultralytics opencv-python matplotlib numpy pathlib
```

## Usage

### 1. Visualize Dataset
```bash
python detect_classify.py
```

### 2. Train YOLOv8 Model
```bash
python train_yolov8.py
```

### 3. Detect Chessboard Squares
```bash
python detect_chessboard.py
```

## Pipeline

1. **Read Image** → Convert to grayscale
2. **Bilateral Filter** → Reduce noise while preserving edges
3. **Canny Edge Detection** → Find edges
4. **Hough Line Transform** → Detect chessboard lines
5. **Extract 64 Squares** → Get coordinates for each square
6. **Piece Classification** → Detect pieces in each square
7. **FEN Generation** → Convert to FEN notation

## Dataset

Chess piece dataset from Roboflow with:
- 292 training images
- 2894 annotations
- 12 piece classes
- 416x416 resolution

## Results

Training results will be saved to `runs/chess/chess_detection/`:
- `weights/best.pt` - Best model checkpoint
- `results.png` - Training metrics
- `confusion_matrix.png` - Class confusion matrix

## License

Public Domain

## Acknowledgments

- Dataset: [Roboflow Chess Pieces](https://universe.roboflow.com/joseph-nelson/chess-pieces-new/dataset/23)
- Framework: Ultralytics YOLOv8
