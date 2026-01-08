"""
Chess Image to FEN Converter
Detects pieces and board, maps them, and outputs the standard FEN string.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Import detectors
from detect_pieces import ChessPieceDetector
from detect_board import ChessboardDetector

def create_board_grid(corners: np.ndarray) -> np.ndarray:
    if len(corners) < 4: return None
    pts = corners[:, :2].astype(np.float32)
    hull = cv2.convexHull(pts).squeeze()
    if len(hull) < 4: return None
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def get_square_from_position_homography(x: float, y: float, grid: np.ndarray) -> str:
    if grid is None or len(grid) != 4: return "unknown"
    src_pts = grid.reshape(-1, 1, 2)
    dst_pts = np.float32([[1, 1], [7, 1], [7, 7], [1, 7]]).reshape(-1, 1, 2)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    piece_point = np.float32([[[x, y]]])
    transformed_point = cv2.perspectiveTransform(piece_point, M)
    row_idx = max(0, min(7, int(np.floor(transformed_point[0][0][0]))))
    col_idx = max(0, min(7, int(np.floor(transformed_point[0][0][1]))))
    return f"{'abcdefgh'[col_idx]}{8 - row_idx}"

def map_pieces_to_squares(pieces: List[Dict], grid: np.ndarray) -> List[Tuple[str, str]]:
    results = []
    for piece in pieces:
        bbox = piece['bbox']
        # Foot position: center X, 90% down Y
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[1] + (bbox[3] - bbox[1]) * 0.90 
        square = get_square_from_position_homography(cx, cy, grid)
        results.append((piece['class'], square))
    return results

def generate_fen(detected_pieces: List[Tuple[str, str]]) -> str:
    # 8x8 Board (Rank 8 at index 0, Rank 1 at index 7)
    board = [[None for _ in range(8)] for _ in range(8)]
    
    for piece_char, square in detected_pieces:
        if square == "unknown": continue
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1:]) # Rank 8 -> index 0
        if 0 <= row < 8 and 0 <= col < 8:
            board[row][col] = piece_char

    fen_rows = []
    for row in board:
        empty = 0
        row_str = ""
        for cell in row:
            if cell is None:
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += cell
        if empty > 0: row_str += str(empty)
        fen_rows.append(row_str)
    fen_rows.reverse()  # FEN starts from rank 8
    
    # Standard FEN suffix: White to move, no castling/en-passant info known
    return "/".join(fen_rows) + " w - - 0 1"

def main():
    # --- CONFIGURATION ---
    input_image_path = "examples/test.jpg"
    pieces_model = "models/480M_leyolo_pieces.onnx"
    corners_model = "models/480L_leyolo_xcorners.onnx"
    # ---------------------

    if not Path(input_image_path).exists():
        print("Error: Image not found.")
        return

    image = cv2.imread(input_image_path)
    if image is None: return

    # 1. Detect Pieces
    piece_detector = ChessPieceDetector(pieces_model)
    pieces = piece_detector.detect_pieces(image, score_threshold=0.3)

    # 2. Detect Board Corners
    corner_detector = ChessboardDetector(corners_model)
    try:
        input_tensor, metadata = corner_detector.preprocess_image(image)
        outputs = corner_detector.session.run(None, {corner_detector.session.get_inputs()[0].name: input_tensor})
        preds = np.transpose(outputs[0], (0, 2, 1))
        # Filter raw predictions manually to get all points
        valid_preds = preds[0][preds[0, :, 4] >= 0.3]
        
        raw_corners = []
        pad_l, pad_r, pad_t, pad_b = metadata['padding']
        scale_x = metadata['width'] / (480 - pad_l - pad_r)
        scale_y = metadata['height'] / (288 - pad_t - pad_b)
        
        for p in valid_preds:
            xc = (p[0] - pad_l) * scale_x
            yc = (p[1] - pad_t) * scale_y
            raw_corners.append([xc, yc])
        
        all_corners = np.array(raw_corners)
    except:
        print("Error processing board corners.")
        return

    # 3. Build Grid and Map
    grid = create_board_grid(all_corners)
    if grid is None:
        print("Error: Could not determine board grid.")
        return

    mapped_data = map_pieces_to_squares(pieces, grid)
    
    # 4. Output FEN
    fen = generate_fen(mapped_data)
    print("https://lichess.org/analysis/fromPosition/" + fen)

if __name__ == '__main__':
    main()