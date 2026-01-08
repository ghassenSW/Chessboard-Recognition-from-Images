"""
Chess Piece Detection - Simplified Version
Detects chess pieces from examples/chessboard.jpg

Usage: python detect_pieces.py

Model: 480M_pieces_float16 (LeYOLO-based)
Classes: 12 piece types (b, k, n, p, q, r, B, K, N, P, Q, R)
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Constants from CameraChessWeb
MODEL_WIDTH = 480
MODEL_HEIGHT = 288
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}
PIECE_NAMES = {
    'b': 'Black Bishop', 'k': 'Black King', 'n': 'Black Knight',
    'p': 'Black Pawn', 'q': 'Black Queen', 'r': 'Black Rook',
    'B': 'White Bishop', 'K': 'White King', 'N': 'White Knight',
    'P': 'White Pawn', 'Q': 'White Queen', 'R': 'White Rook'
}
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255)
]

SQUARE_NAMES = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'
]


class ChessPieceDetector:
    """Detect chess pieces using the CameraChess ONNX model."""
    
    def __init__(self, pieces_model_path: str, xcorners_model_path: Optional[str] = None):
        """
        Initialize the chess piece detector.
        
        Args:
            pieces_model_path: Path to the pieces ONNX model
            xcorners_model_path: Optional path to the xcorners ONNX model for board detection
        """
        self.pieces_session = self._load_model(pieces_model_path)
        self.xcorners_session = self._load_model(xcorners_model_path) if xcorners_model_path else None
        
        print(f"[OK] Loaded pieces model from {pieces_model_path}")
        if self.xcorners_session:
            print(f"[OK] Loaded xcorners model from {xcorners_model_path}")
    
    def _load_model(self, model_path: str) -> ort.InferenceSession:
        """Load ONNX model."""
        if not Path(model_path).exists():
            error_msg = f"\n{'='*60}\nERROR: Model not found: {model_path}\n{'='*60}\n"
            error_msg += "\nTo use this script, you need the ONNX models from CameraChessWeb.\n"
            error_msg += "\nDownload the pieces model:\n"
            error_msg += "  https://drive.google.com/file/d/1-80xp_nly9i6s3o0mF0mU9OZGEzUAlGj/view?usp=sharing\n"
            error_msg += "  Save as: 480M_leyolo_pieces.onnx\n"
            error_msg += "\nOptional - xcorners model:\n"
            error_msg += "  https://drive.google.com/file/d/1-2wodbiXag9UQ44e2AYAmoRN6jVpxy83/view?usp=sharing\n"
            error_msg += "  Save as: 480L_leyolo_xcorners.onnx\n"
            error_msg += "\nThen run:\n"
            error_msg += "  python detect_pieces_camerachess.py --pieces-model models/480M_leyolo_pieces.onnx --input examples/chessboard.jpg\n"
            error_msg += "="*60
            raise FileNotFoundError(error_msg)
        
        session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        return session
    
    def preprocess_image(self, image: np.ndarray, roi: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for model input (similar to getInput in CameraChessWeb).
        
        Args:
            image: Input image (BGR format)
            roi: Optional region of interest [x_min, y_min, x_max, y_max]
        
        Returns:
            Preprocessed image and metadata dictionary
        """
        original_height, original_width = image.shape[:2]
        
        # Crop to ROI if provided
        if roi:
            x_min, y_min, x_max, y_max = roi
            image = image[y_min:y_max, x_min:x_max]
        else:
            roi = [0, 0, original_width, original_height]
        
        height, width = image.shape[:2]
        
        # Calculate resize dimensions maintaining aspect ratio
        ratio = height / width
        desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
        
        if ratio > desired_ratio:
            resize_width = int(MODEL_HEIGHT / ratio)
            resize_height = MODEL_HEIGHT
        else:
            resize_width = MODEL_WIDTH
            resize_height = int(MODEL_WIDTH * ratio)
        
        # Resize
        resized = cv2.resize(image, (resize_width, resize_height))
        
        # Calculate padding
        dx = MODEL_WIDTH - resize_width
        dy = MODEL_HEIGHT - resize_height
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        
        # Pad with gray (114)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to (1, 3, 288, 480)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Convert to float16 if needed (CameraChess models use float16)
        input_tensor = input_tensor.astype(np.float16)
        
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'width': width,
            'height': height,
            'padding': [pad_left, pad_right, pad_top, pad_bottom],
            'roi': roi
        }
        
        return input_tensor, metadata
    
    def postprocess_predictions(self, preds: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-process model predictions (similar to getBoxesAndScores in CameraChessWeb).
        
        Args:
            preds: Raw model predictions (1, 17, N) where N is number of detections
            metadata: Preprocessing metadata
        
        Returns:
            boxes: Array of bounding boxes (N, 4) in format [x_min, y_min, x_max, y_max]
            scores: Array of class scores (N, 12)
        """
        # Transpose predictions to (1, N, 17)
        preds = np.transpose(preds, (0, 2, 1))
        
        # Extract box coordinates and dimensions
        xc = preds[0, :, 0]
        yc = preds[0, :, 1]
        w = preds[0, :, 2]
        h = preds[0, :, 3]
        
        # Convert center format to corner format
        l = xc - w / 2
        t = yc - h / 2
        r = xc + w / 2
        b = yc + h / 2
        
        # Remove padding
        pad_left, pad_right, pad_top, pad_bottom = metadata['padding']
        l -= pad_left
        r -= pad_left
        t -= pad_top
        b -= pad_top
        
        # Scale to original crop size
        scale_x = metadata['width'] / (MODEL_WIDTH - pad_left - pad_right)
        scale_y = metadata['height'] / (MODEL_HEIGHT - pad_top - pad_bottom)
        l *= scale_x
        r *= scale_x
        t *= scale_y
        b *= scale_y
        
        # Add ROI offset
        roi = metadata['roi']
        l += roi[0]
        r += roi[0]
        t += roi[1]
        b += roi[1]
        
        # Scale to model dimensions (for consistency with original project)
        l *= MODEL_WIDTH / metadata['original_width']
        r *= MODEL_WIDTH / metadata['original_width']
        t *= MODEL_HEIGHT / metadata['original_height']
        b *= MODEL_HEIGHT / metadata['original_height']
        
        # Combine into boxes array
        boxes = np.stack([l, t, r, b], axis=1)
        
        # Extract class scores (skip first 4 values which are box coordinates)
        scores = preds[0, :, 4:]
        
        return boxes, scores
    
    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray,
                           iou_threshold: float = 0.45,
                           score_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression to filter detections.
        
        Args:
            boxes: Bounding boxes (N, 4)
            scores: Class scores (N, 12)
            iou_threshold: IoU threshold for NMS
            score_threshold: Minimum score threshold
        
        Returns:
            filtered_boxes: Filtered boxes
            filtered_scores: Filtered scores
            filtered_classes: Predicted class indices
        """
        # Get max score and class for each detection
        max_scores = np.max(scores, axis=1)
        max_classes = np.argmax(scores, axis=1)
        
        # Filter by score threshold
        mask = max_scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        max_scores = max_scores[mask]
        max_classes = max_classes[mask]
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Apply NMS
        indices = self._nms(boxes, max_scores, iou_threshold)
        
        return boxes[indices], scores[indices], max_classes[indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect_pieces(self, image: np.ndarray,
                     roi: Optional[List[int]] = None,
                     score_threshold: float = 0.3,
                     iou_threshold: float = 0.45) -> List[Dict]:
        """
        Detect chess pieces in an image.
        Args:
            image: Input image (BGR format)
            roi: Optional region of interest [x_min, y_min, x_max, y_max]
            score_threshold: Minimum confidence score
            iou_threshold: IoU threshold for NMS
        
        Returns:
            List of detections, each containing:
                - bbox: [x_min, y_min, x_max, y_max]
                - class: piece class (e.g., 'P', 'r')
                - score: confidence score
                - name: piece name (e.g., 'White Pawn')
        """
        # Preprocess
        input_tensor, metadata = self.preprocess_image(image, roi)
        
        # Run inference
        input_name = self.pieces_session.get_inputs()[0].name
        output_name = self.pieces_session.get_outputs()[0].name
        preds = self.pieces_session.run([output_name], {input_name: input_tensor})[0]
        
        # Postprocess
        boxes, scores = self.postprocess_predictions(preds, metadata)
        
        # Apply NMS
        filtered_boxes, filtered_scores, filtered_classes = self.non_max_suppression(
            boxes, scores, iou_threshold, score_threshold
        )
        
        # Convert to detection format
        detections = []
        for box, score_vec, class_idx in zip(filtered_boxes, filtered_scores, filtered_classes):
            # Scale boxes back to original image coordinates
            scale_x = metadata['original_width'] / MODEL_WIDTH
            scale_y = metadata['original_height'] / MODEL_HEIGHT
            
            detection = {
                'bbox': [
                    float(box[0] * scale_x),
                    float(box[1] * scale_y),
                    float(box[2] * scale_x),
                    float(box[3] * scale_y)
                ],
                'class': LABELS[class_idx],
                'score': float(score_vec[class_idx]),
                'name': PIECE_NAMES[LABELS[class_idx]]
            }
            detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict],
                           show_labels: bool = True) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Input image (BGR format)
            detections: List of detections from detect_pieces()
            show_labels: Whether to show labels
        
        Returns:
            Image with drawn detections
        """
        vis_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            class_label = det['class']
            score = det['score']
            
            # Get color for this class
            color_idx = LABEL_MAP[class_label]
            color = COLORS[color_idx]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                # Draw label background
                label = f"{det['name']} {score:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return vis_image


def main():
    """Detect chess pieces from examples/chessboard.jpg"""
    
    # Hardcoded paths
    pieces_model_path = "models/480M_leyolo_pieces.onnx"
    input_image_path = "examples/test.jpg"
    output_image_path = "output/detected_board.jpg"
    
    print("=" * 60)
    print("Chess Piece Detector")
    print("=" * 60)
    
    # Check if input image exists
    if not Path(input_image_path).exists():
        print(f"\nError: Image not found: {input_image_path}")
        print("Please add a chess board image named 'chessboard.jpg' to the examples/ folder")
        return
    
    # Initialize detector
    print(f"\nLoading model...")
    detector = ChessPieceDetector(pieces_model_path)
    
    # Load image
    print(f"Loading image: {input_image_path}")
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
    
    # Detect pieces
    print("Detecting chess pieces...")
    detections = detector.detect_pieces(
        image,
        score_threshold=0.3,
        iou_threshold=0.45
    )
    
    # Print results
    print(f"\nDetected {len(detections)} pieces:")
    print("-" * 60)
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['name']} (confidence: {det['score']:.3f})")
    
    # Visualize
    vis_image = detector.visualize_detections(image, detections)
    
    # Save output
    cv2.imwrite(output_image_path, vis_image)
    print(f"\nSaved output to: {output_image_path}")
    
    # Display
    print("\nDisplaying result... (Press any key to close)")
    cv2.imshow('Chess Piece Detection', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Detection complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
