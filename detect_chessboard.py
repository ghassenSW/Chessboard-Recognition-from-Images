"""
Chessboard Square Detection Pipeline
Detects and segments chessboard into 64 squares and extracts their coordinates
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN


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


def step6_find_line_intersections(lines, img_shape):
    """
    Step 6: Find all intersections between detected lines
    
    Args:
        lines: Detected lines in (rho, theta) format
        img_shape: Image shape (height, width)
    
    Returns:
        intersections: List of (x, y) intersection points
        line_pairs: List of (i, j) indices of intersecting line pairs
    """
    print("\nStep 6: Finding all line intersections...")
    
    if lines is None or len(lines) < 2:
        print("Not enough lines to find intersections")
        return [], []
    
    intersections = []
    line_pairs = []
    h, w = img_shape[:2]
    
    # Check all pairs of lines
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            
            # Convert to line equations: ax + by = c
            a1 = np.cos(theta1)
            b1 = np.sin(theta1)
            c1 = rho1
            
            a2 = np.cos(theta2)
            b2 = np.sin(theta2)
            c2 = rho2
            
            # Solve for intersection point
            determinant = a1 * b2 - a2 * b1
            
            # Check if lines are not parallel (determinant != 0)
            if abs(determinant) > 1e-10:
                x = (b2 * c1 - b1 * c2) / determinant
                y = (a1 * c2 - a2 * c1) / determinant
                
                # Keep intersection if it's within reasonable bounds
                # (can be outside image for vanishing points)
                if -w * 2 < x < w * 3 and -h * 2 < y < h * 3:
                    intersections.append((x, y))
                    line_pairs.append((i, j))
    
    print(f"Found {len(intersections)} intersection points")
    
    return intersections, line_pairs


def step7_find_two_perpendicular_line_sets(lines, angle_threshold=10):
    """
    Step 7: Find two perpendicular line sets for chessboard (horizontal and vertical)
    
    Args:
        lines: Detected lines in (rho, theta) format
        angle_threshold: Maximum angle difference (degrees) to consider lines parallel
    
    Returns:
        two_sets: Dictionary with 'set1' and 'set2' containing line indices
    """
    print("\nStep 7: Finding two perpendicular line sets...")
    print(f"Angle threshold: {angle_threshold} degrees")
    
    if lines is None or len(lines) == 0:
        return {}
    
    # Convert angle threshold to radians
    angle_thresh_rad = np.deg2rad(angle_threshold)
    
    # Extract angles and normalize to [0, pi]
    angles = []
    for line in lines:
        theta = line[0][1]
        # Normalize to [0, pi]
        angles.append(theta % np.pi)
    
    angles = np.array(angles)
    
    # Use DBSCAN to cluster by angle
    dbscan = DBSCAN(eps=angle_thresh_rad, min_samples=2)
    labels = dbscan.fit_predict(angles.reshape(-1, 1))
    
    # Group lines by cluster
    angle_clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Noise
            continue
        if label not in angle_clusters:
            angle_clusters[label] = []
        angle_clusters[label].append(i)
    
    print(f"Found {len(angle_clusters)} angle clusters:")
    for label, indices in angle_clusters.items():
        mean_angle = np.mean([angles[i] for i in indices])
        print(f"  Cluster {label}: {len(indices)} lines, mean angle: {np.rad2deg(mean_angle):.1f}°")
    
    # Select two largest clusters (should be perpendicular)
    if len(angle_clusters) < 2:
        print("Warning: Less than 2 angle clusters found!")
        return {}
    
    # Sort by cluster size
    sorted_clusters = sorted(angle_clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # ALWAYS search for the most perpendicular pair (closest to 90°)
    # Pre-filter clusters by merging duplicates to see actual line count
    print("Evaluating clusters after duplicate elimination...")
    cluster_info = []
    
    for i, (label, indices) in enumerate(sorted_clusters[:5]):  # Check top 5 clusters
        # Quick duplicate elimination to count unique lines
        rho_values = np.array([lines[idx][0][0] for idx in indices]).reshape(-1, 1)
        dbscan = DBSCAN(eps=10, min_samples=1)
        labels_dbscan = dbscan.fit_predict(rho_values)
        n_unique = len(set(labels_dbscan))
        
        mean_angle = np.mean([angles[k] for k in indices])
        cluster_info.append({
            'index': i,
            'label': label,
            'indices': indices,
            'raw_count': len(indices),
            'unique_count': n_unique,
            'mean_angle': mean_angle
        })
        print(f"  Cluster {i}: {len(indices)} raw → {n_unique} unique lines at {np.rad2deg(mean_angle):.1f}°")
    
    # Now find best perpendicular pair with at least 9 UNIQUE lines each
    print("\nSearching for the most perpendicular pair (with ≥9 unique lines)...")
    best_pair = None
    best_perpendicularity_score = float('inf')
    
    for i in range(len(cluster_info)):
        for j in range(i + 1, len(cluster_info)):
            cluster_i = cluster_info[i]
            cluster_j = cluster_info[j]
            
            # Require both sets to have at least 9 UNIQUE lines
            if cluster_i['unique_count'] < 9 or cluster_j['unique_count'] < 9:
                continue
            
            angle_i = cluster_i['mean_angle']
            angle_j = cluster_j['mean_angle']
            diff = abs(angle_i - angle_j)
            diff = min(diff, np.pi - diff)
            diff_deg = np.rad2deg(diff)
            
            # Score: how far from 90 degrees (lower is better)
            perp_score = abs(diff_deg - 90)
            
            if 75 < diff_deg < 105 and perp_score < best_perpendicularity_score:
                best_perpendicularity_score = perp_score
                best_pair = (i, j)
                print(f"  ✓ Cluster {i} ({cluster_i['unique_count']} unique, {np.rad2deg(angle_i):.1f}°) + "
                      f"Cluster {j} ({cluster_j['unique_count']} unique, {np.rad2deg(angle_j):.1f}°) → "
                      f"diff={diff_deg:.1f}°, score={perp_score:.2f}")
    
    if best_pair is None:
        print("Warning: Could not find perpendicular sets with at least 9 unique lines each!")
        print("Falling back to two largest clusters by raw count...")
        # Fallback to two largest
        set1_label, set1_indices = sorted_clusters[0]
        set2_label, set2_indices = sorted_clusters[1]
        mean_angle1 = np.mean([angles[i] for i in set1_indices])
        mean_angle2 = np.mean([angles[i] for i in set2_indices])
        angle_diff = abs(mean_angle1 - mean_angle2)
        angle_diff = min(angle_diff, np.pi - angle_diff)
    else:
        print(f"\n✓ Selected best perpendicular pair!")
        i, j = best_pair
        set1_indices = cluster_info[i]['indices']
        set2_indices = cluster_info[j]['indices']
        mean_angle1 = cluster_info[i]['mean_angle']
        mean_angle2 = cluster_info[j]['mean_angle']
        angle_diff = abs(mean_angle1 - mean_angle2)
        angle_diff = min(angle_diff, np.pi - angle_diff)
        set1_label = cluster_info[i]['label']
        set2_label = cluster_info[j]['label']
    
    print(f"\nSelected two largest angle clusters:")
    print(f"  Set 1: {len(set1_indices)} lines, mean angle: {np.rad2deg(mean_angle1):.1f}°")
    print(f"  Set 2: {len(set2_indices)} lines, mean angle: {np.rad2deg(mean_angle2):.1f}°")
    print(f"  Angle difference: {np.rad2deg(angle_diff):.1f}° (should be ~90° for chessboard)")
    
    return {'set1': set1_indices, 'set2': set2_indices}


def step8_eliminate_duplicate_lines_in_set(lines, line_indices, eps=15):
    """
    Step 8: Eliminate duplicate lines within a set using DBSCAN on rho values
    
    Args:
        lines: All detected lines in (rho, theta) format
        line_indices: Indices of lines in this set
        eps: Maximum distance between rho values to be considered duplicates
    
    Returns:
        merged_lines: List of (rho, theta) for unique lines
    """
    print(f"\nStep 8: Eliminating duplicates from {len(line_indices)} lines...")
    
    if len(line_indices) == 0:
        return []
    
    # Extract rho values for lines in this set
    rho_values = []
    for idx in line_indices:
        rho, theta = lines[idx][0]
        rho_values.append(rho)
    
    rho_values = np.array(rho_values).reshape(-1, 1)
    
    # Apply DBSCAN on rho values to find duplicates
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(rho_values)
    
    # Merge duplicates by averaging
    merged_lines = []
    n_clusters = len(set(labels))
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = [line_indices[i] for i, mask in enumerate(cluster_mask) if mask]
        
        # Average rho and theta for this cluster
        rhos = [lines[idx][0][0] for idx in cluster_indices]
        thetas = [lines[idx][0][1] for idx in cluster_indices]
        
        mean_rho = np.mean(rhos)
        mean_theta = np.mean(thetas)
        merged_lines.append((mean_rho, mean_theta))
    
    print(f"  Reduced to {len(merged_lines)} unique lines")
    
    return merged_lines


def step9_select_best_9_lines(merged_lines, img_width, img_height=None):
    """
    Step 9: Select best 9 lines by ensuring equal spacing between them
    
    Strategy:
    1. Filter lines to ensure they span the board area (not clustered at edges)
    2. Remove outlier lines that are too far from neighbors
    3. Find 9 consecutive lines with most uniform spacing and good coverage
    
    Args:
        merged_lines: List of (rho, theta) line parameters
        img_width: Image width for calculating line positions
        img_height: Image height (optional)
    
    Returns:
        best_9_lines: List of 9 selected lines with most uniform spacing
    """
    print(f"\nStep 9: Selecting best 9 lines from {len(merged_lines)} available...")
    
    if len(merged_lines) < 9:
        print(f"  Warning: Only {len(merged_lines)} lines available, need 9")
        return merged_lines
    
    if len(merged_lines) == 9:
        print("  Exactly 9 lines, no selection needed")
        return merged_lines
    
    # Sort lines by rho
    sorted_lines = sorted(merged_lines, key=lambda x: x[0])
    rho_values = [rho for rho, theta in sorted_lines]
    
    n = len(sorted_lines)
    
    # NEW: Filter out lines at extreme edges (likely board boundaries, not grid lines)
    if n > 12:
        print(f"  Step 1: Filtering edge boundary lines...")
        # Calculate the span
        min_rho = min(rho_values)
        max_rho = max(rho_values)
        span = max_rho - min_rho
        
        # Remove lines too close to the min/max (outer 10% on each side)
        margin = span * 0.1
        filtered_lines = [(rho, theta) for rho, theta in sorted_lines 
                         if min_rho + margin < rho < max_rho - margin]
        
        if len(filtered_lines) >= 9:
            print(f"    Removed {n - len(filtered_lines)} edge lines")
            print(f"    Kept {len(filtered_lines)} interior lines")
            sorted_lines = filtered_lines
            rho_values = [rho for rho, theta in sorted_lines]
            n = len(sorted_lines)
        else:
            print(f"    Edge filtering too aggressive, keeping all lines")
    
    # Strategy 1: For 10-15 lines, remove outliers first
    if 10 <= n <= 15:
        print(f"  Step 1: Filtering outliers from {n} lines...")
        
        # Calculate all spacings
        all_spacings = [rho_values[i+1] - rho_values[i] for i in range(n-1)]
        median_spacing = np.median(all_spacings)
        
        print(f"    Median spacing: {median_spacing:.2f} pixels")
        
        # Remove lines that create abnormally large gaps (>2x median)
        outlier_threshold = 2.0 * median_spacing  # More aggressive
        filtered_lines = [sorted_lines[0]]  # Always keep first line
        
        for i in range(1, n):
            gap = rho_values[i] - rho_values[i-1]
            if gap <= outlier_threshold:
                filtered_lines.append(sorted_lines[i])
            else:
                print(f"    Removing outlier: gap={gap:.2f} > threshold={outlier_threshold:.2f}")
        
        # If we removed too many, try again with larger threshold
        if len(filtered_lines) < 9:
            print(f"    Too aggressive! Re-filtering with 2.5x threshold...")
            outlier_threshold = 2.5 * median_spacing
            filtered_lines = [sorted_lines[0]]
            for i in range(1, n):
                gap = rho_values[i] - rho_values[i-1]
                if gap <= outlier_threshold:
                    filtered_lines.append(sorted_lines[i])
        
        print(f"    After filtering: {len(filtered_lines)} lines remain")
        
        # Update for next step
        sorted_lines = filtered_lines
        rho_values = [rho for rho, theta in sorted_lines]
        n = len(sorted_lines)
    
    # Strategy 2: Find best consecutive 9 lines
    if n >= 9:
        from itertools import combinations
        
        best_score = float('inf')
        best_start = 0
        
        if n > 20:
            # Sliding window for many lines
            print(f"  Step 2: Using sliding window on {n} lines...")
            
            # Prioritize windows that span a good range (not clustered)
            best_coverage_score = -1
            
            for start in range(n - 8):
                window_rhos = rho_values[start:start+9]
                spacings = [window_rhos[i+1] - window_rhos[i] for i in range(8)]
                
                # Score based on uniformity AND coverage
                std_score = np.std(spacings)
                coverage = (window_rhos[-1] - window_rhos[0])  # Total span
                
                # Combined score: low std deviation + good coverage
                # Penalize if coverage is too small (clustered lines)
                if coverage < 100:  # Arbitrary threshold, adjust based on image size
                    coverage_penalty = 1000 / (coverage + 1)
                else:
                    coverage_penalty = 0
                
                score = std_score + coverage_penalty
                
                if score < best_score:
                    best_score = score
                    best_start = start
            
            best_9_lines = sorted_lines[best_start:best_start+9]
            
        elif n <= 12:
            # Try all combinations for small n
            print(f"  Step 2: Evaluating all combinations from {n} lines...")
            best_combination = None
            
            for indices in combinations(range(n), 9):
                combo_rhos = sorted([rho_values[i] for i in indices])
                spacings = [combo_rhos[i+1] - combo_rhos[i] for i in range(8)]
                
                # Score: prefer low std deviation AND avoid extreme gaps
                mean_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                max_gap_ratio = max(spacings) / mean_spacing if mean_spacing > 0 else float('inf')
                
                # Combined score: std + penalty for large gaps
                score = std_spacing + (max_gap_ratio - 1) * 10
                
                if score < best_score:
                    best_score = score
                    best_combination = [sorted_lines[i] for i in indices]
            
            best_9_lines = best_combination
            
        else:
            # Prefer consecutive lines for medium n
            print(f"  Step 2: Finding best consecutive 9 from {n} lines...")
            
            for start in range(n - 8):
                window_rhos = rho_values[start:start+9]
                spacings = [window_rhos[i+1] - window_rhos[i] for i in range(8)]
                
                # Score with coverage consideration
                std_score = np.std(spacings)
                coverage = (window_rhos[-1] - window_rhos[0])
                
                # Penalize small coverage (clustered lines)
                if coverage < 100:
                    coverage_penalty = 500 / (coverage + 1)
                else:
                    coverage_penalty = 0
                
                score = std_score + coverage_penalty
                
                if score < best_score:
                    best_score = score
                    best_start = start
            
            best_9_lines = sorted_lines[best_start:best_start+9]
    else:
        print(f"  Warning: Only {n} lines after filtering!")
        best_9_lines = sorted_lines
    
    # Calculate final spacing info only if we have enough lines
    if len(best_9_lines) < 2:
        print(f"  ⚠️  ERROR: Only {len(best_9_lines)} line(s) available!")
        return best_9_lines
    
    final_rhos = sorted([rho for rho, theta in best_9_lines])
    
    if len(final_rhos) >= 2:
        final_spacings = [final_rhos[i+1] - final_rhos[i] for i in range(len(final_rhos)-1)]
        mean_spacing = np.mean(final_spacings)
        std_spacing = np.std(final_spacings)
        max_gap = max(final_spacings)
        min_gap = min(final_spacings)
        coverage = final_rhos[-1] - final_rhos[0]  # Total span
        
        print(f"  ✓ Selected {len(best_9_lines)} lines:")
        print(f"    Coverage: {coverage:.2f} pixels (span of selected lines)")
        print(f"    Mean spacing: {mean_spacing:.2f} pixels")
        print(f"    Std deviation: {std_spacing:.2f} pixels ({std_spacing/mean_spacing*100:.1f}% of mean)")
        print(f"    Spacing range: {min_gap:.2f} - {max_gap:.2f} pixels")
        print(f"    Max/Min ratio: {max_gap/min_gap:.2f}x")
        
        # Quality check
        if len(best_9_lines) < 9:
            print(f"  ⚠️  WARNING: Only {len(best_9_lines)}/9 lines! Grid will be incomplete.")
        elif max_gap / min_gap > 3.0:
            print(f"  ⚠️  WARNING: Spacing is NOT uniform! Grid may be inaccurate.")
            print(f"      Consider adjusting detection parameters or trying a different image.")
        elif std_spacing / mean_spacing > 0.4:
            print(f"  ⚠️  WARNING: High spacing variation. Grid may not be perfectly aligned.")
        else:
            print(f"  ✅ Grid quality: Good!")
    
    return best_9_lines


def process_two_line_sets_to_9_each(lines, line_sets, img_width, img_height):
    """
    Process both line sets to get 9 lines from each
    
    Args:
        lines: All detected lines
        line_sets: Dictionary with 'set1' and 'set2' containing line indices
        img_width: Image width
        img_height: Image height
    
    Returns:
        final_sets: Dictionary with 'set1' and 'set2', each containing 9 lines
    """
    print("\n" + "="*60)
    print("Processing Line Sets to Extract 9 Lines Each")
    print("="*60)
    
    final_sets = {}
    
    for set_name in ['set1', 'set2']:
        if set_name not in line_sets:
            continue
            
        line_indices = line_sets[set_name]
        print(f"\nProcessing {set_name} ({len(line_indices)} lines):")
        
        # Step 8: Eliminate duplicates (tighter eps to keep more lines)
        merged_lines = step8_eliminate_duplicate_lines_in_set(lines, line_indices, eps=8)
        
        # Adaptive: If too few lines, try even tighter eps
        if len(merged_lines) < 9:
            print(f"  ⚠️  Only {len(merged_lines)} unique lines. Trying tighter merging (eps=5)...")
            merged_lines = step8_eliminate_duplicate_lines_in_set(lines, line_indices, eps=5)
            print(f"  → {len(merged_lines)} unique lines after adjustment")
        
        # Step 9: Select best 9 lines
        best_9 = step9_select_best_9_lines(merged_lines, img_width, img_height)
        
        final_sets[set_name] = best_9
        print(f"{set_name} final: {len(best_9)} lines")
        
        # Diagnostic: Report if we couldn't get 9
        if len(best_9) < 9:
            print(f"  ⚠️  WARNING: Could only extract {len(best_9)}/9 lines for {set_name}")
            print(f"      This may indicate the image doesn't have clear grid lines in this direction.")
    
    return final_sets


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


def visualize_clusters(img, lines, intersections, vanishing_points, line_clusters, parallel_clusters):
    """
    Visualize line clusters, intersections, and vanishing points
    
    Args:
        img: Original image
        lines: Detected lines
        intersections: List of intersection points
        vanishing_points: List of vanishing points
        line_clusters: Dictionary of line clusters from vanishing points
        parallel_clusters: List of parallel line clusters
    """
    print("\nVisualizing clusters and vanishing points...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. All intersections
    img_intersections = img.copy()
    for x, y in intersections:
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_intersections, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    axes[0, 0].imshow(cv2.cvtColor(img_intersections, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"6. Intersections ({len(intersections)})")
    axes[0, 0].axis('off')
    
    # 2. Vanishing points
    img_vp = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for idx, (x, y) in enumerate(vanishing_points):
        color = colors[idx % len(colors)]
        cv2.circle(img_vp, (int(x), int(y)), 10, color, -1)
        cv2.putText(img_vp, f"VP{idx}", (int(x) + 15, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    axes[0, 1].imshow(cv2.cvtColor(img_vp, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"8. Vanishing Points ({len(vanishing_points)})")
    axes[0, 1].axis('off')
    
    # 3. Line clusters from vanishing points
    img_line_clusters = img.copy()
    for cluster_id, line_indices in line_clusters.items():
        color = colors[cluster_id % len(colors)]
        for line_idx in line_indices:
            rho, theta = lines[line_idx][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_line_clusters, (x1, y1), (x2, y2), color, 2)
    
    axes[1, 0].imshow(cv2.cvtColor(img_line_clusters, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"9. Line Clusters from VP ({len(line_clusters)})")
    axes[1, 0].axis('off')
    
    # 4. Parallel line clusters
    img_parallel = img.copy()
    for cluster_id, line_indices in enumerate(parallel_clusters):
        color = colors[cluster_id % len(colors)]
        for line_idx in line_indices:
            rho, theta = lines[line_idx][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_parallel, (x1, y1), (x2, y2), color, 2)
    
    axes[1, 1].imshow(cv2.cvtColor(img_parallel, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"7. Parallel Clusters ({len(parallel_clusters)})")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Cluster visualization complete!")


def visualize_final_9x9_grid(img, final_clusters):
    """
    Visualize the final 9x9 line grid forming 64 squares
    
    Args:
        img: Original image
        final_clusters: Dictionary with 2 clusters, each containing 9 lines
    """
    print("\nVisualizing final 9x9 chessboard grid...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Draw both cluster sets
    colors = [(255, 0, 0), (0, 0, 255)]  # Red and Blue
    titles = ["Cluster 0: 9 Lines", "Cluster 1: 9 Lines"]
    
    for idx, (cluster_id, lines_9) in enumerate(final_clusters.items()):
        img_grid = img.copy()
        
        for rho, theta in lines_9:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(img_grid, (x1, y1), (x2, y2), colors[idx], 2)
        
        axes[idx].imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(titles[idx] if idx < len(titles) else f"Cluster {cluster_id}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Combined view
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    img_combined = img.copy()
    
    for idx, (cluster_id, lines_9) in enumerate(final_clusters.items()):
        for rho, theta in lines_9:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(img_combined, (x1, y1), (x2, y2), colors[idx], 2)
    
    ax.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    ax.set_title("Combined: 9x9 Chessboard Grid (64 Squares)")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Final grid visualization complete!")


def main():
    """
    Main function to run the complete pipeline
    """
    # Get current directory
    current_dir = Path(__file__).parent
    # === CONFIGURE IMAGE PATH ===
    # Update this path to your chessboard image
    image_path = current_dir / "chessboard project" / "test" / "images" / "73a38a5c8f8f1b09f093f304660d5326_jpg.rf.2d2fa2f4b419d9f2a57fb82d38d8bc6b.jpg"
    # image_path=r"C:\Users\ghass\Downloads\phpUnsLhe.png"
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
    
    # Step 4: Apply Canny edge detection (with adaptive thresholds)
    # Try different threshold combinations to get better line detection
    edges = step4_canny_edge_detection(filtered, threshold1=30, threshold2=100)
    
    # Step 5: Apply Hough Line Transform (lowered threshold for more sensitivity)
    lines = step5_hough_line_transform(edges, rho=1, theta=np.pi/180, threshold=70)
    
    # Visualize basic steps
    visualize_steps(img, gray, filtered, edges, lines)
    
    if lines is not None and len(lines) > 0:
        # Step 6 is no longer needed with new approach
        # Step 7: Find two perpendicular line sets (horizontal and vertical)
        line_sets = step7_find_two_perpendicular_line_sets(lines, angle_threshold=10)
        
        if len(line_sets) == 2:
            # Step 8 & 9: Process both sets to get 9 lines from each
            final_sets = process_two_line_sets_to_9_each(lines, line_sets, img.shape[1], img.shape[0])
            
            # Visualize final 9x9 grid
            if len(final_sets) == 2:
                visualize_final_9x9_grid(img, final_sets)
            else:
                print("\nWarning: Could not extract 9 lines from both sets.")
                final_sets = None
        else:
            print("\nWarning: Could not find 2 perpendicular line sets. Cannot form chessboard grid.")
            final_sets = None
        
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"Total lines detected: {len(lines)}")
        print(f"Perpendicular line sets found: {len(line_sets)}")
        if final_sets:
            set1_count = len(final_sets.get('set1', []))
            set2_count = len(final_sets.get('set2', []))
            if set1_count == 9 and set2_count == 9:
                print(f"✓ Final chessboard grid: {set1_count}×{set2_count} = 64 squares ✓")
            else:
                print(f"⚠️  Partial grid: {set1_count}×{set2_count} = {(set1_count-1)*(set2_count-1)} squares")
                print(f"    Expected: 9×9 = 64 squares")
                print(f"\n💡 Suggestions:")
                print(f"    • Try a different image with clearer grid lines")
                print(f"    • Adjust Canny thresholds (currently: threshold1=50, threshold2=150)")
                print(f"    • Ensure good lighting and contrast in the image")
        print("="*60)
        
        return {
            'img': img,
            'lines': lines,
            'line_sets': line_sets if 'line_sets' in locals() else {},
            'final_sets': final_sets if final_sets else None
        }
    else:
        print("\nNo lines detected. Cannot proceed with clustering.")
        return None


if __name__ == "__main__":
    main()
