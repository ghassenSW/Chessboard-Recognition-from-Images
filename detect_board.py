"""
Chessboard Corner Detection using CameraChess XCorners Model
Detects the 4 extended corners of a chessboard for perspective correction

Model: 480L_xcorners_float16 (LeYOLO-based)
Usage: python detect_board.py

This script:
1. Detects 4 chessboard corners using ONNX model
2. Draws the board boundaries
3. Can apply perspective transform to get top-down view
4. Saves cropped board region
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Constants
MODEL_WIDTH = 480
MODEL_HEIGHT = 288


class ChessboardDetector:
    """Detect chessboard corners using the CameraChess ONNX model."""
    
    def __init__(self, xcorners_model_path: str):
        """
        Initialize the chessboard corner detector.
        
        Args:
            xcorners_model_path: Path to the xcorners ONNX model
        """
        self.session = self._load_model(xcorners_model_path)
        print(f"[OK] Loaded xcorners model from {xcorners_model_path}")
    
    def _load_model(self, model_path: str) -> ort.InferenceSession:
        """Load ONNX model."""
        if not Path(model_path).exists():
            error_msg = f"\n{'='*60}\nERROR: Model not found: {model_path}\n{'='*60}\n"
            error_msg += "\nTo use this script, you need the XCorners ONNX model.\n"
            error_msg += "\nDownload the xcorners model:\n"
            error_msg += "  https://drive.google.com/file/d/1-2wodbiXag9UQ44e2AYAmoRN6jVpxy83/view?usp=sharing\n"
            error_msg += "  Save as: models/480L_leyolo_xcorners.onnx\n"
            error_msg += "\nThen run:\n"
            error_msg += "  python detect_board.py\n"
            error_msg += "="*60
            raise FileNotFoundError(error_msg)
        
        session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        return session
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Preprocessed image and metadata dictionary
        """
        original_height, original_width = image.shape[:2]
        height, width = original_height, original_width
        
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
        
        # Convert to float16
        input_tensor = input_tensor.astype(np.float16)
        
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'width': width,
            'height': height,
            'padding': [pad_left, pad_right, pad_top, pad_bottom],
            'resize_width': resize_width,
            'resize_height': resize_height
        }
        
        return input_tensor, metadata
    
    def postprocess_predictions(self, preds: np.ndarray, metadata: Dict, conf_threshold: float = 0.3) -> np.ndarray:
        """
        Post-process model predictions to get corner coordinates.
        
        Args:
            preds: Raw model predictions (1, 5, N) - [x, y, w, h, conf]
            metadata: Preprocessing metadata
            conf_threshold: Minimum confidence for corner detection
        
        Returns:
            corners: Array of 4 corner points (4, 2) in format [x, y]
        """
        # Transpose predictions to (1, N, 5)
        preds = np.transpose(preds, (0, 2, 1))
        
        # The xcorners model detects multiple corner points as small boxes
        # We need to find the 4 outermost corners that form the board boundary
        
        confidences = preds[0, :, 4]
        
        # Filter by confidence - get top confident corners
        valid_mask = confidences >= conf_threshold
        valid_preds = preds[0, valid_mask]
        
        if len(valid_preds) < 4:
            # If we don't have enough corners, take top 20 by confidence
            top_indices = np.argsort(confidences)[::-1][:20]
            valid_preds = preds[0, top_indices]
        elif len(valid_preds) > 40:
            # Too many, filter to top 40 by confidence
            indices = np.argsort(valid_preds[:, 4])[::-1][:40]
            valid_preds = valid_preds[indices]
        
        # Extract corner centers (xc, yc) for each detection
        corner_points = []
        pad_left, pad_right, pad_top, pad_bottom = metadata['padding']
        scale_x = metadata['width'] / (MODEL_WIDTH - pad_left - pad_right)
        scale_y = metadata['height'] / (MODEL_HEIGHT - pad_top - pad_bottom)
        
        for pred in valid_preds:
            xc = pred[0]
            yc = pred[1]
            conf = pred[4]
            
            # Remove padding
            xc -= pad_left
            yc -= pad_top
            
            # Scale to original image size
            xc *= scale_x
            yc *= scale_y
            
            corner_points.append([xc, yc])
        
        corner_points = np.array(corner_points, dtype=np.float32)
        
        # Find the 4 outermost corners (convex hull vertices)
        # Calculate the convex hull of all detected corners
        if len(corner_points) >= 4:
            hull = cv2.convexHull(corner_points)
            hull_points = hull.squeeze()
            
            # If hull has more than 4 points, find the 4 corners
            if len(hull_points) > 4:
                # Find the 4 extreme points
                corners = self.find_extreme_corners(hull_points)
            else:
                corners = hull_points
        else:
            corners = corner_points
        
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self.order_corners(corners)
        
        return corners
    
    def find_extreme_corners(self, points: np.ndarray) -> np.ndarray:
        """
        Find the 4 extreme corner points from a set of points.
        
        Args:
            points: Array of points (N, 2)
        
        Returns:
            corners: 4 extreme corners (4, 2)
        """
        # Find extreme points
        min_x_idx = np.argmin(points[:, 0])
        max_x_idx = np.argmax(points[:, 0])
        min_y_idx = np.argmin(points[:, 1])
        max_y_idx = np.argmax(points[:, 1])
        
        # Get unique extreme points
        extreme_indices = list(set([min_x_idx, max_x_idx, min_y_idx, max_y_idx]))
        
        if len(extreme_indices) == 4:
            return points[extreme_indices]
        
        # If we don't have exactly 4, use a different approach
        # Find centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate angles from centroid
        angles = np.arctan2(points[:, 1] - centroid[1], 
                           points[:, 0] - centroid[0])
        
        # Divide into 4 quadrants and find farthest point in each
        corners = []
        for angle_range in [(-np.pi, -np.pi/2), (-np.pi/2, 0), (0, np.pi/2), (np.pi/2, np.pi)]:
            mask = (angles >= angle_range[0]) & (angles < angle_range[1])
            if np.any(mask):
                quadrant_points = points[mask]
                # Find farthest from centroid
                distances = np.linalg.norm(quadrant_points - centroid, axis=1)
                farthest_idx = np.argmax(distances)
                corners.append(quadrant_points[farthest_idx])
        
        return np.array(corners, dtype=np.float32)
    
    def detect_board(self, image: np.ndarray, conf_threshold: float = 0.3) -> Optional[np.ndarray]:
        """
        Detect chessboard corners in an image.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence threshold
        
        Returns:
            corners: 4 corner points (4, 2) or None if not detected
        """
        # Preprocess
        input_tensor, metadata = self.preprocess_image(image)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        preds = self.session.run([output_name], {input_name: input_tensor})[0]
        
        # Postprocess to get 4 corners
        corners = self.postprocess_predictions(preds, metadata, conf_threshold)
        
        return corners
    
    def detect_all_corners(self, image: np.ndarray, conf_threshold: float = 0.3) -> np.ndarray:
        """
        Detect ALL corner points without filtering to just 4.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence threshold
        
        Returns:
            corners: Array of all corner points (N, 3) with [x, y, confidence]
        """
        # Preprocess
        input_tensor, metadata = self.preprocess_image(image)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        preds = self.session.run([output_name], {input_name: input_tensor})[0]
        
        # Transpose predictions to (1, N, 5)
        preds = np.transpose(preds, (0, 2, 1))
        
        confidences = preds[0, :, 4]
        
        # Filter by confidence
        valid_mask = confidences >= conf_threshold
        valid_preds = preds[0, valid_mask]
        
        # Extract corner centers and confidence
        corner_points = []
        pad_left, pad_right, pad_top, pad_bottom = metadata['padding']
        scale_x = metadata['width'] / (MODEL_WIDTH - pad_left - pad_right)
        scale_y = metadata['height'] / (MODEL_HEIGHT - pad_top - pad_bottom)
        
        for pred in valid_preds:
            xc = pred[0]
            yc = pred[1]
            conf = pred[4]
            
            # Remove padding
            xc -= pad_left
            yc -= pad_top
            
            # Scale to original image size
            xc *= scale_x
            yc *= scale_y
            
            corner_points.append([xc, yc, conf])
        
        return np.array(corner_points, dtype=np.float32)
    
    def filter_extreme_4_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Filter corners to get only the 4 extreme corners based on coordinates.
        
        Args:
            corners: Array of corner points (N, 3) with [x, y, confidence]
        
        Returns:
            extreme_corners: 4 corners (4, 3) - [min_x_min_y, max_x_min_y, max_x_max_y, min_x_max_y]
        """
        if len(corners) < 4:
            return corners
        
        # Extract x and y coordinates
        xs = corners[:, 0]
        ys = corners[:, 1]
        
        # Find indices for each extreme corner
        # 1. Min X and Min Y (top-left)
        sum_xy = xs + ys
        min_x_min_y_idx = np.argmin(sum_xy)
        
        # 2. Max X and Max Y (bottom-right)
        max_x_max_y_idx = np.argmax(sum_xy)
        
        # 3. Max X and Min Y (top-right)
        diff_x_minus_y = xs - ys
        max_x_min_y_idx = np.argmax(diff_x_minus_y)
        
        # 4. Min X and Max Y (bottom-left)
        min_x_max_y_idx = np.argmin(diff_x_minus_y)
        
        # Get the 4 extreme corners
        extreme_corners = np.array([
            corners[min_x_min_y_idx],  # Top-left
            corners[max_x_min_y_idx],  # Top-right
            corners[max_x_max_y_idx],  # Bottom-right
            corners[min_x_max_y_idx]   # Bottom-left
        ], dtype=np.float32)
        
        return extreme_corners
    
    def deduplicate_corners(self, corners: np.ndarray, distance_threshold: float = 20.0) -> np.ndarray:
        """
        Remove duplicate corners that are at the same position.
        Groups nearby corners and keeps one representative per group.
        
        Args:
            corners: Array of corner points (N, 3) with [x, y, confidence]
            distance_threshold: Maximum distance to consider corners as duplicates
        
        Returns:
            deduplicated_corners: Array of unique corners (M, 3) where M <= N
        """
        if len(corners) == 0:
            return corners
        
        # Sort corners by confidence (highest first) to prioritize better detections
        sorted_indices = np.argsort(corners[:, 2])[::-1]
        sorted_corners = corners[sorted_indices]
        
        unique_corners = []
        used = np.zeros(len(sorted_corners), dtype=bool)
        
        for i, corner in enumerate(sorted_corners):
            if used[i]:
                continue
            
            # Find all corners within threshold distance
            x, y, conf = corner
            distances = np.sqrt((sorted_corners[:, 0] - x)**2 + (sorted_corners[:, 1] - y)**2)
            nearby_mask = (distances < distance_threshold) & (~used)
            
            # Average the positions of nearby corners (weighted by confidence)
            nearby_corners = sorted_corners[nearby_mask]
            if len(nearby_corners) > 0:
                weights = nearby_corners[:, 2]  # Use confidence as weight
                avg_x = np.average(nearby_corners[:, 0], weights=weights)
                avg_y = np.average(nearby_corners[:, 1], weights=weights)
                avg_conf = np.max(nearby_corners[:, 2])  # Keep highest confidence
                
                unique_corners.append([avg_x, avg_y, avg_conf])
                used[nearby_mask] = True
        
        return np.array(unique_corners, dtype=np.float32)
    
    def draw_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected board boundaries on image.
        
        Args:
            image: Input image (BGR format)
            corners: 4 corner points (4, 2)
        
        Returns:
            Image with drawn board
        """
        vis_image = image.copy()
        
        # Draw board outline
        pts = corners.astype(np.int32)
        cv2.polylines(vis_image, [pts], True, (0, 255, 0), 3)
        
        # Draw corners
        for i, corner in enumerate(corners):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis_image, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(vis_image, f"C{i+1}", (x + 10, y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def draw_all_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw ALL detected corner points on image.
        
        Args:
            image: Input image (BGR format)
            corners: Array of corner points (N, 3) with [x, y, confidence]
        
        Returns:
            Image with all corners drawn
        """
        vis_image = image.copy()
        
        # Draw all corners with color based on confidence
        for i, corner in enumerate(corners):
            x, y, conf = int(corner[0]), int(corner[1]), corner[2]
            
            # Color gradient from red (low conf) to green (high conf)
            color_intensity = int(conf * 255)
            color = (0, color_intensity, 255 - color_intensity)  # BGR format
            
            cv2.circle(vis_image, (x, y), 5, color, -1)
            cv2.circle(vis_image, (x, y), 7, (255, 255, 255), 1)  # White outline
            
            # Show confidence text for high-confidence corners
            if conf > 0.7:
                cv2.putText(vis_image, f"{conf:.2f}", (x + 8, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_image
    
    def get_board_crop(self, image: np.ndarray, corners: np.ndarray, 
                      output_size: int = 800) -> np.ndarray:
        """
        Extract and warp board region to top-down view.
        
        Args:
            image: Input image (BGR format)
            corners: 4 corner points (4, 2)
            output_size: Size of output square image
        
        Returns:
            Warped board image (top-down view)
        """
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self.order_corners(corners)
        
        # Define destination points for perspective transform
        dst_points = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply perspective warp
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))
        
        return warped
    
    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in a consistent way: TL, TR, BR, BL.
        
        Args:
            corners: 4 corner points (4, 2)
        
        Returns:
            Ordered corners (4, 2)
        """
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], 
                           corners[:, 0] - centroid[0])
        
        # Sort by angle (top-right will be first, then going clockwise)
        sorted_indices = np.argsort(angles)
        
        # Reorder: we want TL, TR, BR, BL
        # After sorting by angle from centroid, adjust to get correct order
        ordered = corners[sorted_indices]
        
        # Find top-left (smallest x+y)
        sums = ordered[:, 0] + ordered[:, 1]
        tl_idx = np.argmin(sums)
        
        # Roll array so top-left is first
        ordered = np.roll(ordered, -tl_idx, axis=0)
        
        return ordered


def main():
    """Detect and deduplicate chessboard corners to get 7x7 grid"""
    
    # Paths
    xcorners_model_path = "models/480L_leyolo_xcorners.onnx"
    input_image_path = "examples/test.jpg"
    
    # Initialize detector
    detector = ChessboardDetector(xcorners_model_path)
    
    # Load image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
    
    # Detect all corners
    all_corners = detector.detect_all_corners(image, conf_threshold=0.3)
    
    if len(all_corners) == 0:
        print("No corners detected!")
        return
    
    # Deduplicate corners
    deduplicated_corners = detector.deduplicate_corners(all_corners, distance_threshold=30.0)
    
    # Print results
    print(f"\nDetected {len(deduplicated_corners)} unique corners:\n")
    print("Corner_ID | X_Position | Y_Position")
    print("-" * 40)
    for i, corner in enumerate(deduplicated_corners, 1):
        print(f"   {i:3d}    | {corner[0]:8.1f}   | {corner[1]:8.1f}")
    
    # Draw corners on image
    vis_image = image.copy()
    for i, corner in enumerate(deduplicated_corners, 1):
        x, y = int(corner[0]), int(corner[1])
        # Draw corner point
        cv2.circle(vis_image, (x, y), 6, (0, 255, 0), -1)  # Green filled circle
        cv2.circle(vis_image, (x, y), 8, (255, 255, 255), 1)  # White outline
        # Draw corner number
        cv2.putText(vis_image, str(i), (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save image
    output_path = "output/corners_visualization.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"\n✓ Saved visualization to: {output_path}")


if __name__ == '__main__':
    main()
