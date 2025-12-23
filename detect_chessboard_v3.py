"""
Chessboard Grid Detection v3
Advanced approach: Detect actual board corners, warp to square, then find grid
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def find_chessboard_corners(gray):
    """
    Find the 4 corners of the chessboard using improved corner detection
    """
    print("Step 1: Finding chessboard corners...")
    
    img_h, img_w = gray.shape
    img_area = img_h * img_w
    
    # Try multiple approaches
    all_candidates = []
    
    # Approach 1: Adaptive thresholding
    print("  Approach 1: Adaptive thresholding...")
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours1, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    Found {len(contours1)} contours")
    
    # Approach 2: Canny edges
    print("  Approach 2: Canny edges...")
    edges = cv2.Canny(gray, 50, 150)
    # Dilate more aggressively to connect annotation boundaries
    dilated = cv2.dilate(edges, kernel, iterations=4)
    contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    Found {len(contours2)} contours")
    
    # Approach 3: Simple thresholding
    print("  Approach 3: Otsu thresholding...")
    _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph3 = cv2.morphologyEx(binary3, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours3, _ = cv2.findContours(morph3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    Found {len(contours3)} contours")
    
    # Combine all contours
    all_contours = list(contours1) + list(contours2) + list(contours3)
    print(f"\n  Total: {len(all_contours)} contours from all approaches")
    
    # Filter candidates
    candidates = []
    debug_count = 0
    
    for contour in all_contours:
        area = cv2.contourArea(contour)
        
        # Debug first few large contours
        if debug_count < 5 and area > 0.01 * img_area:
            print(f"    Debug: Contour area={area:.0f} ({100*area/img_area:.1f}%)", end="")
            debug_count += 1
        
        # More lenient: Must be at least 30% of image area (to catch full board with annotations)
        # but less than 95%
        if area < 0.30 * img_area or area > 0.95 * img_area:
            if debug_count <= 5 and area > 0.01 * img_area:
                print(f" → filtered (size)")
            continue
        
        # Approximate to polygon - try multiple epsilon values
        peri = cv2.arcLength(contour, True)
        approx = None
        
        # Try different approximation factors to find a quadrilateral
        for eps_mult in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
            test_approx = cv2.approxPolyDP(contour, eps_mult * peri, True)
            if len(test_approx) == 4:
                approx = test_approx
                if debug_count <= 5:
                    print(f" → 4 corners (eps={eps_mult})", end="")
                break
        
        # If still not 4 corners, try to extract 4 extreme points
        if approx is None or len(approx) != 4:
            if area > 0.2 * img_area:  # Only for large contours (likely the board)
                # Get convex hull
                hull = cv2.convexHull(contour)
                # Approximate the hull
                peri_hull = cv2.arcLength(hull, True)
                for eps_mult in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
                    approx_hull = cv2.approxPolyDP(hull, eps_mult * peri_hull, True)
                    if len(approx_hull) == 4:
                        approx = approx_hull
                        if debug_count <= 5:
                            print(f" → convex hull 4 corners (eps={eps_mult})", end="")
                        break
                
                # If still not 4, extract 4 extreme points manually
                if approx is None or len(approx) != 4:
                    points = hull.reshape(-1, 2)
                    # Find extreme points
                    top_left = points[np.argmin(points[:, 0] + points[:, 1])]
                    top_right = points[np.argmax(points[:, 0] - points[:, 1])]
                    bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
                    bottom_left = points[np.argmax(-points[:, 0] + points[:, 1])]
                    approx = np.array([[top_left], [top_right], [bottom_right], [bottom_left]])
                    if debug_count <= 5:
                        print(f" → extreme points from hull", end="")
            else:
                if debug_count <= 5:
                    test_approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    print(f" → filtered ({len(test_approx)} corners, not 4)")
                continue
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # More lenient aspect ratio (0.3 to 3.0)
            if 0.3 <= aspect_ratio <= 3.0:
                candidates.append({
                    'contour': contour,
                    'approx': approx,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'bbox': (x, y, w, h)
                })
    
    if len(candidates) == 0:
        print("  ⚠️ Could not find any suitable quadrilaterals!")
        return None
    
    # Remove duplicates (similar corners)
    unique_candidates = []
    for cand in candidates:
        is_duplicate = False
        corners = cand['approx'].reshape(4, 2)
        for existing in unique_candidates:
            existing_corners = existing['approx'].reshape(4, 2)
            # Check if corners are very similar
            avg_dist = np.mean([np.linalg.norm(corners[i] - existing_corners[i]) for i in range(4)])
            if avg_dist < 20:  # Less than 20 pixels difference
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(cand)
    
    candidates = unique_candidates
    
    print(f"\n  Found {len(candidates)} unique candidate quadrilaterals:")
    for i, cand in enumerate(candidates):
        x, y, w, h = cand['bbox']
        print(f"    {i+1}. Area: {cand['area']:.0f} ({100*cand['area']/img_area:.1f}%), " +
              f"Aspect: {cand['aspect_ratio']:.2f}, Size: {w}×{h}")
    
    if len(candidates) == 0:
        print("  ⚠️ No candidates found! Using fallback detection...")
        # Fallback: detect edges and find the bounding box
        edges = cv2.Canny(gray, 30, 100)
        # Find all edge points
        edge_points = np.column_stack(np.where(edges > 0))
        if len(edge_points) > 100:
            # Get convex hull of all edges
            hull = cv2.convexHull(edge_points)
            # Get bounding rectangle
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect).astype(int)
            # Swap x and y because edge_points are in (row, col) format
            box = box[:, [1, 0]]
            return order_corners(box.astype(np.float32))
        return None
    
    # Score: prefer larger area with aspect ratio close to 1.0
    # Also prefer centered positions
    # Prioritize LARGEST area (likely includes annotations)
    for cand in candidates:
        x, y, w, h = cand['bbox']
        center_x = x + w/2
        center_y = y + h/2
        
        # Distance from image center
        dist_from_center = np.sqrt((center_x - img_w/2)**2 + (center_y - img_h/2)**2)
        max_dist = np.sqrt((img_w/2)**2 + (img_h/2)**2)
        center_score = 1.0 - (dist_from_center / max_dist)
        
        # Combined score - heavily favor larger area (includes annotation zone)
        area_score = cand['area'] / img_area
        aspect_score = 2.0 - abs(1.0 - cand['aspect_ratio'])
        
        cand['score'] = area_score * 0.8 + aspect_score * 0.1 + center_score * 0.1
    
    # Select candidate with largest area (most likely includes annotations)
    best_candidate = max(candidates, key=lambda c: c['area'])
    
    corners = best_candidate['approx'].reshape(4, 2)
    x, y, w, h = best_candidate['bbox']
    
    print(f"\n  ✓ Selected best candidate:")
    print(f"    Area: {best_candidate['area']:.0f} ({100*best_candidate['area']/img_area:.1f}%)")
    print(f"    Aspect ratio: {best_candidate['aspect_ratio']:.2f}")
    print(f"    Size: {w}×{h} pixels")
    print(f"    Score: {best_candidate['score']:.3f}")
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners)
    
    print(f"\n  Full board corners (including annotations):")
    for i, (cx, cy) in enumerate(corners):
        labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
        print(f"    {labels[i]}: ({cx:.0f}, {cy:.0f})")
    
    return corners

def order_corners(pts):
    """
    Order points as: top-left, top-right, bottom-right, bottom-left
    """
    # Sort by y-coordinate
    pts = pts[np.argsort(pts[:, 1])]
    
    # Top two points
    top = pts[:2]
    top = top[np.argsort(top[:, 0])]  # left to right
    
    # Bottom two points
    bottom = pts[2:]
    bottom = bottom[np.argsort(bottom[:, 0])]  # left to right
    
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

def shrink_corners_to_playing_area(corners, margin_percent=0.12):
    """
    Shrink the board corners inward to exclude annotation margins (letters/numbers)
    
    Args:
        corners: 4 corners of detected board
        margin_percent: Percentage of board dimension to exclude on each side (default 12%)
    
    Returns:
        Tuple of (playing_area_corners, full_board_corners)
    """
    print(f"\nStep 1.5: Defining playing area (excluding {margin_percent*100:.0f}% margins)...")
    print(f"  Detected corners:")
    for i, (x, y) in enumerate(corners):
        labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
        print(f"    {labels[i]}: ({x:.0f}, {y:.0f})")
    
    # Use detected corners as the full board
    full_board_corners = corners.copy()
    
    # Extract corners
    tl, tr, br, bl = corners
    
    # Calculate board vectors
    top_vec = tr - tl
    bottom_vec = br - bl
    left_vec = bl - tl
    right_vec = br - tr
    
    # Shrink from each edge to get playing area
    tl_new = tl + margin_percent * left_vec + margin_percent * top_vec
    tr_new = tr + margin_percent * right_vec - margin_percent * top_vec
    bl_new = bl - margin_percent * left_vec + margin_percent * bottom_vec
    br_new = br - margin_percent * right_vec - margin_percent * bottom_vec
    
    playing_area_corners = np.array([tl_new, tr_new, br_new, bl_new], dtype=np.float32)
    
    print(f"\n  Full board corners (with annotations):")
    for i, (x, y) in enumerate(full_board_corners):
        labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
        print(f"    {labels[i]}: ({x:.0f}, {y:.0f})")
    
    print(f"\n  Playing area corners (8×8 squares, {margin_percent*100:.0f}% margin excluded):")
    for i, (x, y) in enumerate(playing_area_corners):
        labels = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
        print(f"    {labels[i]}: ({x:.0f}, {y:.0f})")
    print(f"  ✓ Playing area extracted\n")
    
    return playing_area_corners, full_board_corners

def warp_perspective(img, corners, size=800):
    """
    Warp the board to a square perspective
    """
    print(f"\nStep 2: Warping to {size}x{size} square...")
    
    # Destination points (square)
    dst = np.array([
        [0, 0],
        [size-1, 0],
        [size-1, size-1],
        [0, size-1]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst)
    
    # Warp image
    warped = cv2.warpPerspective(img, M, (size, size))
    
    print("  ✓ Perspective correction applied")
    
    return warped, M

def detect_grid_lines_on_square(warped_gray, n_lines=9):
    """
    Detect grid lines on the warped square board
    Since the board is now square and normalized, we can create evenly-spaced lines
    """
    print(f"\nStep 3: Creating {n_lines}×{n_lines} evenly-spaced grid...")
    
    board_size = warped_gray.shape[0]
    spacing = board_size / (n_lines - 1)
    
    print(f"  Board size: {board_size}×{board_size} pixels")
    print(f"  Grid spacing: {spacing:.1f} pixels")
    
    # Create horizontal lines (theta = 90 degrees = pi/2)
    h_lines = []
    for i in range(n_lines):
        rho = i * spacing
        theta = np.pi / 2  # Horizontal
        h_lines.append([rho, theta])
    
    # Create vertical lines (theta = 0 degrees)
    v_lines = []
    for i in range(n_lines):
        rho = i * spacing
        theta = 0  # Vertical
        v_lines.append([rho, theta])
    
    h_lines = np.array(h_lines)
    v_lines = np.array(v_lines)
    
    print(f"  ✓ Created {len(h_lines)}×{len(v_lines)} evenly-spaced grid lines")
    print(f"    Horizontal spacing: {spacing:.1f} pixels (uniform)")
    print(f"    Vertical spacing: {spacing:.1f} pixels (uniform)")
    
    return h_lines, v_lines

def remove_duplicate_lines(lines, eps=10):
    """Remove duplicate lines using DBSCAN"""
    if len(lines) == 0:
        return lines
    
    rhos = lines[:, 0].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(rhos)
    labels = clustering.labels_
    
    unique_lines = []
    for label in set(labels):
        mask = labels == label
        cluster_lines = lines[mask]
        # Take median
        median_idx = len(cluster_lines) // 2
        sorted_idx = np.argsort(cluster_lines[:, 0])
        unique_lines.append(cluster_lines[sorted_idx[median_idx]])
    
    return np.array(unique_lines)

def select_n_evenly_spaced(lines, n, board_size, is_horizontal):
    """
    Select n evenly-spaced lines across the board
    """
    if len(lines) < n:
        print(f"    ⚠️ Only {len(lines)} lines, need {n}")
        # Generate evenly-spaced lines manually
        spacing = board_size / (n - 1)
        ideal_lines = []
        for i in range(n):
            rho = i * spacing
            theta = np.pi/2 if is_horizontal else 0
            ideal_lines.append([rho, theta])
        return np.array(ideal_lines)
    
    # Sort by rho
    lines = lines[np.argsort(lines[:, 0])]
    
    # Calculate ideal spacing
    ideal_spacing = board_size / (n - 1)
    
    # Select lines closest to ideal positions
    selected = []
    available = list(range(len(lines)))
    
    for i in range(n):
        ideal_rho = i * ideal_spacing
        
        # Find closest available line
        best_idx = None
        best_dist = float('inf')
        for idx in available:
            dist = abs(lines[idx, 0] - ideal_rho)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        selected.append(best_idx)
        available.remove(best_idx)
    
    selected_lines = lines[sorted(selected)]
    
    # Print spacing stats
    spacings = np.diff(selected_lines[:, 0])
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    
    direction = "horizontal" if is_horizontal else "vertical"
    print(f"    {direction.capitalize()}: spacing {mean_spacing:.1f}±{std_spacing:.1f} px ({100*std_spacing/mean_spacing:.1f}%)")
    
    return selected_lines

def visualize_grid(img, warped, h_lines, v_lines, full_board_corners, playing_area_corners):
    """
    Visualize the detected grid on both original and warped images
    """
    print("\nStep 4: Visualizing results...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original with corners
    img_corners = img.copy()
    
    # Draw full board (green)
    cv2.polylines(img_corners, [full_board_corners.astype(int)], True, (0, 255, 0), 3)
    for i, corner in enumerate(full_board_corners):
        cv2.circle(img_corners, tuple(corner.astype(int)), 12, (0, 255, 0), -1)
    
    # Draw playing area (blue)
    cv2.polylines(img_corners, [playing_area_corners.astype(int)], True, (255, 0, 0), 3)
    for i, corner in enumerate(playing_area_corners):
        cv2.circle(img_corners, tuple(corner.astype(int)), 8, (255, 0, 0), -1)
        cv2.putText(img_corners, str(i+1), tuple(corner.astype(int) - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    axes[0].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Board Detection (Green=Full, Blue=Playing Area)', fontsize=11)
    axes[0].axis('off')
    
    # Warped square
    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Warped to Square', fontsize=12)
    axes[1].axis('off')
    
    # Warped with grid
    warped_grid = warped.copy()
    
    # Draw horizontal lines
    for rho, theta in h_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(warped_grid, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw vertical lines
    for rho, theta in v_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(warped_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    axes[2].imshow(cv2.cvtColor(warped_grid, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Grid Overlay ({len(h_lines)}×{len(v_lines)})', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('grid_detection_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved to grid_detection_v3.png")

def calculate_square_coordinates(h_lines, v_lines, M_inv, original_shape):
    """
    Calculate the coordinates of all 64 squares in the original image
    Returns: array of shape (64, 4, 2) with corners of each square
    """
    print("\nStep 5: Calculating square coordinates...")
    
    if len(h_lines) != 9 or len(v_lines) != 9:
        print(f"  ⚠️ Need 9×9 grid, got {len(h_lines)}×{len(v_lines)}")
        return None
    
    # Calculate intersection points in warped space
    intersections = []
    for h_rho, h_theta in h_lines:
        for v_rho, v_theta in v_lines:
            # Calculate intersection
            a1 = np.cos(h_theta)
            b1 = np.sin(h_theta)
            a2 = np.cos(v_theta)
            b2 = np.sin(v_theta)
            
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue
            
            x = (b2 * h_rho - b1 * v_rho) / det
            y = (a1 * v_rho - a2 * h_rho) / det
            
            intersections.append([x, y])
    
    intersections = np.array(intersections).reshape(9, 9, 2)
    
    # Transform back to original image coordinates
    intersections_original = []
    for row in intersections:
        row_original = []
        for point in row:
            point_homogeneous = np.array([[point[0], point[1]]], dtype=np.float32)
            point_transformed = cv2.perspectiveTransform(
                point_homogeneous.reshape(1, 1, 2), M_inv
            ).reshape(2)
            row_original.append(point_transformed)
        intersections_original.append(row_original)
    
    intersections_original = np.array(intersections_original)
    
    # Extract 64 squares (each square has 4 corners)
    squares = []
    for row in range(8):
        for col in range(8):
            corners = [
                intersections_original[row, col],      # top-left
                intersections_original[row, col+1],    # top-right
                intersections_original[row+1, col+1],  # bottom-right
                intersections_original[row+1, col]     # bottom-left
            ]
            squares.append(corners)
    
    squares = np.array(squares)
    
    print(f"  ✓ Calculated {len(squares)} square coordinates")
    print(f"  Shape: {squares.shape} (64 squares × 4 corners × 2 coordinates)")
    
    return squares

def main():
    print("="*60)
    print("Chessboard Grid Detection v3")
    print("Strategy: Find corners → Warp → Detect grid → Transform back")
    print("="*60)
    
    img_path = r"C:\Users\ghass\OneDrive\Desktop\practice projects\chessboard to FEN\chessboard project\train\images\fa4e2b9a8cf58f405f69a56c662834f2_jpg.rf.9c9fb84a36bdfc518396a26a21758f9c.jpg"
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Image: {img.shape[1]}×{img.shape[0]} pixels\n")
    
    # Step 1: Find corners (full board including annotations)
    detected_corners = find_chessboard_corners(gray)
    if detected_corners is None:
        print("Failed to find board corners")
        return
    
    # Step 1.5: Shrink to get playing area (exclude annotation margins)
    playing_area_corners, full_board_corners = shrink_corners_to_playing_area(detected_corners, margin_percent=0.12)
    
    # Step 2: Warp perspective (using playing area only)
    warped, M = warp_perspective(img, full_board_corners, size=800)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Detect grid lines on warped image
    h_lines, v_lines = detect_grid_lines_on_square(warped_gray, n_lines=9)
    
    if h_lines is None or v_lines is None:
        print("Failed to detect grid lines")
        return
    
    # Step 4: Visualize
    visualize_grid(img, warped, h_lines, v_lines, full_board_corners, full_board_corners)
    
    # Step 5: Calculate square coordinates
    M_inv = np.linalg.inv(M)
    squares = calculate_square_coordinates(h_lines, v_lines, M_inv, img.shape)
    
    if squares is not None:
        # Save to file
        np.save('square_coordinates.npy', squares)
        print("\n  ✓ Saved square coordinates to square_coordinates.npy")
    
    print("\n" + "="*60)
    print("✓ Processing complete!")
    print("="*60)
    print(f"Grid: {len(h_lines)}×{len(v_lines)} = {(len(h_lines)-1)*(len(v_lines)-1)} squares")
    if squares is not None:
        print(f"Coordinates: {squares.shape} array saved")
    print("="*60)

if __name__ == "__main__":
    main()
