"""
Chessboard Corner Detection
Detects the 4 corners of a chessboard's playing area (8×8 grid, excluding annotations)
"""

import cv2
import numpy as np

def order_corners(pts):
    """
    Order corner points consistently as: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array of 4 points (x, y)
    
    Returns:
        Ordered array of 4 points
    """
    # Sort by y-coordinate to separate top and bottom
    pts = pts[np.argsort(pts[:, 1])]
    
    # Top two points (smallest y)
    top = pts[:2]
    top = top[np.argsort(top[:, 0])]  # Sort by x: left to right
    
    # Bottom two points (largest y)
    bottom = pts[2:]
    bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x: left to right
    
    # Return in order: top-left, top-right, bottom-right, bottom-left
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

def find_chessboard_corners(image_path, display=True):
    """
    Detect the 4 corners of a chessboard's playing area (8×8 grid only)
    
    Args:
        image_path: Path to the chessboard image
        display: Whether to display the result (default: True)
    
    Returns:
        corners: Array of 4 corner points (x, y)
        img_result: Annotated image with detected corners
    """
    # Step 1: Load the image
    print("Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    print(f"Image loaded: {img.shape[1]}×{img.shape[0]} pixels")
    
    # Step 2: Convert to grayscale
    print("Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Apply Gaussian blur to reduce noise
    print("Applying Gaussian blur...")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 4: Apply Canny edge detection
    print("Detecting edges with Canny...")
    edges = cv2.Canny(blurred, 50, 150)
    
    # Optional: Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Step 5: Find contours
    print("Finding contours...")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    if len(contours) == 0:
        print("Error: No contours found!")
        return None, None
    
    # Step 6: Filter contours - find largest quadrilateral
    print("\nSearching for chessboard quadrilateral...")
    img_area = img.shape[0] * img.shape[1]
    best_contour = None
    best_corners = None
    max_area = 0
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by area (must be at least 20% of image, less than 90%)
        if area < 0.20 * img_area or area > 0.90 * img_area:
            continue
        
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        
        # Try different epsilon values to find a quadrilateral
        for epsilon_mult in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon_mult * peri, True)
            
            # Check if we found exactly 4 corners (quadrilateral)
            if len(approx) == 4:
                # Check aspect ratio (should be roughly square)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Accept if aspect ratio is reasonable (0.5 to 2.0)
                if 0.5 <= aspect_ratio <= 2.0:
                    if area > max_area:
                        max_area = area
                        best_contour = contour
                        best_corners = approx.reshape(4, 2)
                        print(f"  Candidate {i}: Area={area:.0f} ({100*area/img_area:.1f}%), "
                              f"Aspect={aspect_ratio:.2f}, Epsilon={epsilon_mult}")
                break
    
    if best_corners is None:
        print("Error: Could not find a suitable quadrilateral!")
        return None, None
    
    # Step 7: Order the corners consistently
    print("\nOrdering corners...")
    ordered_corners = order_corners(best_corners)
    
    # Step 8: Print corner coordinates
    print("\n" + "="*60)
    print("DETECTED CHESSBOARD CORNERS (Playing Area 8×8)")
    print("="*60)
    labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
    for i, (x, y) in enumerate(ordered_corners):
        print(f"{labels[i]:15s}: ({x:7.1f}, {y:7.1f})")
    print("="*60)
    
    # Step 9: Draw visualization
    print("\nCreating visualization...")
    img_result = img.copy()
    
    # Draw the contour
    cv2.drawContours(img_result, [best_contour], -1, (0, 255, 0), 3)
    
    # Draw corner points and labels
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Cyan
    for i, (corner, label) in enumerate(zip(ordered_corners, labels)):
        x, y = int(corner[0]), int(corner[1])
        
        # Draw circle at corner
        cv2.circle(img_result, (x, y), 15, colors[i], -1)
        cv2.circle(img_result, (x, y), 18, (255, 255, 255), 2)
        
        # Draw label
        cv2.putText(img_result, f"{i+1}", (x - 10, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_result, label, (x - 50, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    
    # Add title
    cv2.putText(img_result, "Chessboard Corner Detection", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Step 10: Display result
    if display:
        print("Displaying result... (Press any key to close)")
        
        # Resize if image is too large
        max_display_width = 1200
        if img_result.shape[1] > max_display_width:
            scale = max_display_width / img_result.shape[1]
            new_width = int(img_result.shape[1] * scale)
            new_height = int(img_result.shape[0] * scale)
            img_display = cv2.resize(img_result, (new_width, new_height))
        else:
            img_display = img_result
        
        cv2.imshow('Chessboard Corners', img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return ordered_corners, img_result, img

def warp_chessboard_to_square(img, corners, output_size=800):
    """
    Apply perspective transformation to obtain a top-down, squared chessboard view
    
    Args:
        img: Original image
        corners: Ordered corners [top-left, top-right, bottom-right, bottom-left]
        output_size: Size of output square image (default: 800x800)
    
    Returns:
        warped: Squared and undistorted chessboard image
        M: Perspective transformation matrix
    """
    print("\n" + "="*60)
    print("PERSPECTIVE TRANSFORMATION")
    print("="*60)
    
    # Define destination points for the square output image
    # These represent where each corner should map to in the output
    dst_points = np.array([
        [0, 0],                              # Top-left
        [output_size - 1, 0],                # Top-right
        [output_size - 1, output_size - 1],  # Bottom-right
        [0, output_size - 1]                 # Bottom-left
    ], dtype=np.float32)
    
    print(f"\nSource corners (from original image):")
    labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
    for i, (label, corner) in enumerate(zip(labels, corners)):
        print(f"  {label:15s}: ({corner[0]:7.1f}, {corner[1]:7.1f})")
    
    print(f"\nDestination points (in {output_size}×{output_size} square):")
    for i, (label, point) in enumerate(zip(labels, dst_points)):
        print(f"  {label:15s}: ({point[0]:7.1f}, {point[1]:7.1f})")
    
    # Compute the perspective transformation matrix (homography)
    print("\nComputing perspective transformation matrix...")
    M = cv2.getPerspectiveTransform(corners, dst_points)
    
    print("Transformation matrix (3×3 homography):")
    print(M)
    
    # Apply the perspective transformation to warp the image
    print(f"\nWarping image to {output_size}×{output_size} square...")
    warped = cv2.warpPerspective(img, M, (output_size, output_size))
    
    print("✓ Perspective transformation complete!")
    print(f"  Output shape: {warped.shape}")
    print("="*60)
    
    return warped, M

def display_before_after(original, warped, wait=True):
    """
    Display the original and warped images side by side
    
    Args:
        original: Original image with corner detection
        warped: Warped square chessboard
        wait: Whether to wait for key press (default: True)
    """
    print("\nDisplaying comparison...")
    
    # Resize original to match warped height for side-by-side display
    target_height = 800
    scale = target_height / original.shape[0]
    resized_original = cv2.resize(original, 
                                  (int(original.shape[1] * scale), target_height))
    
    # Create side-by-side comparison
    comparison = np.hstack([resized_original, warped])
    
    # Add labels
    cv2.putText(comparison, "Original (Corners Detected)", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Warped (Top-Down Square)", 
               (resized_original.shape[1] + 20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('Chessboard: Before & After Transformation', comparison)
    
    if wait:
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return comparison

def main():
    """Main function"""
    # Image path - update this to your chessboard image
    image_path = r"C:\Users\ghass\OneDrive\Desktop\practice projects\chessboard to FEN\chessboard project\train\images\fa3cf2724c1648a8822b59ac0759475f_jpg.rf.8b23c032ccc0fd6484f9d390cb6a2b9c.jpg"
    
    print("="*60)
    print("CHESSBOARD CORNER DETECTION & PERSPECTIVE TRANSFORMATION")
    print("="*60)
    print()
    
    # Step 1: Detect corners
    corners, img_result, img_original = find_chessboard_corners(image_path, display=False)
    
    if corners is None:
        print("\n✗ Detection failed!")
        return
    
    print("\n✓ Corner detection successful!")
    print(f"  Corners shape: {corners.shape}")
    print(f"  Corners dtype: {corners.dtype}")
    
    # Step 2: Apply perspective transformation to get squared chessboard
    warped_board, transform_matrix = warp_chessboard_to_square(img_original, corners, output_size=800)
    
    # Step 3: Display and save comparison
    comparison = display_before_after(img_result, warped_board, wait=True)
    
    # Save only the comparison image
    comparison_path = 'chessboard_comparison.jpg'
    cv2.imwrite(comparison_path, comparison)
    print(f"\n✓ Comparison image saved to: {comparison_path}")
    
    print("\n" + "="*60)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*60)
    print(f"Output file: {comparison_path}")
    print("="*60)

if __name__ == "__main__":
    main()
