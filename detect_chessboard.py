"""
Chessboard Square Detection Pipeline
Detects and segments chessboard into 64 squares and extracts their coordinates
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def step1_read_image(image_path):
    """
    Step 1: Read the image
    
    Args:
        image_path: Path to the chessboard image
    
    Returns:
        Original image in BGR format
    """
    print("Step 1: Reading image...")
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    print(f"Image shape: {img.shape}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    return img


def step2_convert_grayscale(img):
    """
    Step 2: Convert image to grayscale
    
    Args:
        img: BGR image
    
    Returns:
        Grayscale image
    """
    print("\nStep 2: Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Grayscale shape: {gray.shape}")
    
    return gray


def step3_bilateral_filter(gray, d=9, sigma_color=75, sigma_space=75):
    """
    Step 3: Apply bilateral filter to reduce noise while preserving edges
    
    Args:
        gray: Grayscale image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image
    """
    print("\nStep 3: Applying bilateral filter...")
    print(f"Parameters: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
    
    filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    
    print("Bilateral filter applied successfully")
    
    return filtered


def step4_canny_edge_detection(filtered, threshold1=50, threshold2=150):
    """
    Step 4: Apply Canny edge detection to find edges
    
    Args:
        filtered: Filtered grayscale image
        threshold1: Lower threshold for edge detection
        threshold2: Upper threshold for edge detection
    
    Returns:
        Edge image (binary)
    """
    print("\nStep 4: Applying Canny edge detection...")
    print(f"Parameters: threshold1={threshold1}, threshold2={threshold2}")
    
    edges = cv2.Canny(filtered, threshold1, threshold2)
    
    print(f"Edges detected: {np.sum(edges > 0)} edge pixels")
    
    return edges


def step5_hough_line_transform(edges, rho=1, theta=np.pi/180, threshold=100):
    """
    Step 5: Apply Hough Line Transform to find lines from edges
    
    Args:
        edges: Edge image from Canny
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Minimum number of intersections to detect a line
    
    Returns:
        Detected lines in (rho, theta) format
    """
    print("\nStep 5: Applying Hough Line Transform...")
    print(f"Parameters: rho={rho}, theta={theta:.4f} rad, threshold={threshold}")
    
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    
    if lines is not None:
        print(f"Detected {len(lines)} lines")
    else:
        print("No lines detected")
    
    return lines


def visualize_steps(img, gray, filtered, edges, lines=None):
    """
    Visualize all processing steps
    
    Args:
        img: Original BGR image
        gray: Grayscale image
        filtered: Filtered image
        edges: Edge image
        lines: Detected lines (optional)
    """
    print("\nVisualizing results...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("2. Grayscale")
    axes[0, 1].axis('off')
    
    # Filtered
    axes[0, 2].imshow(filtered, cmap='gray')
    axes[0, 2].set_title("3. Bilateral Filter")
    axes[0, 2].axis('off')
    
    # Edges
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title("4. Canny Edges")
    axes[1, 0].axis('off')
    
    # Lines on original image
    if lines is not None:
        img_with_lines = img.copy()
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[1, 1].imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"5. Detected Lines ({len(lines)})")
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, "No lines detected", ha='center', va='center')
        axes[1, 1].set_title("5. Detected Lines")
        axes[1, 1].axis('off')
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete!")


def main():
    """
    Main function to run the complete pipeline
    """
    # Get current directory
    current_dir = Path(__file__).parent
    
    # === CONFIGURE IMAGE PATH ===
    # Update this path to your chessboard image
    image_path = current_dir / "chessboard project" / "test" / "images" / "example.jpg"
    
    print("="*60)
    print("Chessboard Square Detection Pipeline")
    print("="*60)
    print(f"Image path: {image_path}\n")
    
    # Step 1: Read image
    img = step1_read_image(image_path)
    
    # Step 2: Convert to grayscale
    gray = step2_convert_grayscale(img)
    
    # Step 3: Apply bilateral filter
    filtered = step3_bilateral_filter(gray, d=9, sigma_color=75, sigma_space=75)
    
    # Step 4: Apply Canny edge detection
    edges = step4_canny_edge_detection(filtered, threshold1=50, threshold2=150)
    
    # Step 5: Apply Hough Line Transform
    lines = step5_hough_line_transform(edges, rho=1, theta=np.pi/180, threshold=100)
    
    # Visualize all steps
    visualize_steps(img, gray, filtered, edges, lines)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return img, gray, filtered, edges, lines


if __name__ == "__main__":
    main()
