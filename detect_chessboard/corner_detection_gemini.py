import cv2
import numpy as np
from scipy.spatial import KDTree
import os

# ==========================================
# PART 1: FEATURE DETECTION
# ==========================================

def find_potential_quads(img_gray):
    # Standard settings
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 150, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) > 100: 
                quads.append(approx)
    return quads

def get_saddle_points(img_gray):
    dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    return centroids

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

# ==========================================
# PART 2: CENTROID-ANCHORED RECONSTRUCTION
# ==========================================

def find_best_grid_and_reconstruct(img_gray, quads, saddle_points):
    if len(saddle_points) < 10: return None
    if len(quads) == 0: return None
    
    tree = KDTree(saddle_points)
    
    # --- FIX 1: Find the "Central" Quad ---
    # Instead of taking the first quad, we find the one closest to the center of all quads
    # This prevents the grid from "drifting" if the first quad is at the edge.
    
    all_centers = []
    for q in quads:
        M = cv2.moments(q)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            all_centers.append([cX, cY])
            
    if not all_centers: return None
    
    # Calculate average center of the board
    avg_center = np.mean(all_centers, axis=0)
    
    # Find the quad closest to this average center
    best_dist = float('inf')
    central_quad = quads[0]
    
    for i, center in enumerate(all_centers):
        dist = np.linalg.norm(center - avg_center)
        if dist < best_dist:
            best_dist = dist
            central_quad = quads[i]

    # --- Step A: Build Cloud from Central Quad ---
    central_quad = central_quad.reshape(4, 2)
    ordered_quad = order_points(central_quad)
    tl, tr, br, bl = ordered_quad
    
    vec_x = (tr - tl) * 0.5 + (br - bl) * 0.5
    vec_y = (bl - tl) * 0.5 + (br - tr) * 0.5
    step_x = np.linalg.norm(vec_x)
    step_y = np.linalg.norm(vec_y)
    
    best_grid_cloud = []
    
    # Expand outwards from the center. 
    # Since we are in the middle, -6 to +6 should cover the whole board easily.
    for y in range(-8, 9): 
        for x in range(-8, 9):
            pred_pt = tl + (vec_x * x) + (vec_y * y)
            dist, idx = tree.query(pred_pt)
            if dist < (step_x + step_y) / 2.5:
                # We normalize coordinates so (0,0) is our central quad
                best_grid_cloud.append({'pt': saddle_points[idx], 'grid_pos': (x, y)})

    if len(best_grid_cloud) < 10: return None

    # --- Step B: Calculate Homography ---
    src_points = []
    dst_points = []
    # Use the central quad as (0,0) reference
    origin_x, origin_y = best_grid_cloud[0]['grid_pos'] 
    
    for p in best_grid_cloud:
        gx, gy = p['grid_pos']
        src_points.append(p['pt'])
        dst_points.append([(gx - origin_x) * 50, (gy - origin_y) * 50])

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    H_inv = np.linalg.inv(H)
    
    # --- Step C: Density Voting (Same as before) ---
    all_saddles_flat = cv2.perspectiveTransform(np.array([saddle_points], dtype=np.float32), H)
    all_saddles_flat = all_saddles_flat[0]
    grid_coords = all_saddles_flat / 50.0 # Normalize to grid units
    
    max_density = 0
    best_offset = (0, 0)
    
    # Scan window. Since we started in the center, the offset should be small.
    # We check -7 to +7 just to be safe.
    for dy in range(-7, 8):
        for dx in range(-7, 8):
            # Check for 7x7 grid fit
            # Check X bounds
            in_x = (grid_coords[:, 0] >= dx - 0.5) & (grid_coords[:, 0] <= dx + 6.5)
            # Check Y bounds
            in_y = (grid_coords[:, 1] >= dy - 0.5) & (grid_coords[:, 1] <= dy + 6.5)
            
            count = np.sum(in_x & in_y)
            
            if count > max_density:
                max_density = count
                best_offset = (dx, dy)

    # --- Step D: Reconstruct Full Board ---
    # Note on Coordinates:
    # Our "Best Offset" (dx, dy) points to the Top-Left of the *Internal* 7x7 grid.
    # The Full Board extends -1 from there (Top/Left edge) and +8 (Bottom/Right edge).
    
    sx, sy = best_offset
    final_flat_grid = []
    
    # Generate 9x9 grid lines (for 8x8 squares)
    for y in range(-1, 8): 
        for x in range(-1, 8):
            final_flat_grid.append([(x + sx) * 50, (y + sy) * 50])
            
    final_flat_grid = np.array([final_flat_grid], dtype=np.float32)
    final_points = cv2.perspectiveTransform(final_flat_grid, H_inv)[0]
    
    return final_points

# ==========================================
# MAIN EXECUTION
# ==========================================

image_path = r"C:\Users\ghass\Downloads\chessboard.jpg"

if os.path.exists(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Step 1: Feature Detection...")
    quads = find_potential_quads(gray)
    saddle_points = get_saddle_points(gray)
    saddle_points_arr = np.array([pt for pt in saddle_points], dtype=np.float32)

    print(f"Step 2: Centered Reconstruction...")
    final_grid = find_best_grid_and_reconstruct(gray, quads, saddle_points_arr)

    result_img = img.copy()

    if final_grid is not None:
        print(f"Success!")
        grid_shape = final_grid.reshape(9, 9, 2)
        
        # Draw Lines
        for i in range(9):
            cv2.polylines(result_img, [np.int32(grid_shape[i, :])], False, (0, 255, 0), 2)
            cv2.polylines(result_img, [np.int32(grid_shape[:, i])], False, (0, 255, 0), 2)
            
        # Draw Corners
        for pt in final_grid:
            cv2.circle(result_img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    else:
        print("Could not align grid.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "final_centered_fix.jpg")
    cv2.imwrite(output_path, result_img)
    
    cv2.imshow("Final Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found.")