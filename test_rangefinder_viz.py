#!/usr/bin/env python3

import numpy as np
import cv2

def create_rangefinder_visualization(normalized_ranges, strip_height=50, strip_width=None):
    """Create a horizontal strip visualization of rangefinder data.
    
    Args:
        normalized_ranges: Array of normalized rangefinder values (0-1)
        strip_height: Height of the visualization strip in pixels
        strip_width: Width per sensor (if None, auto-calculated)
    
    Returns:
        BGR image array for OpenCV display
    """
    num_sensors = len(normalized_ranges)
    
    if strip_width is None:
        strip_width = max(8, 640 // num_sensors)  # At least 8 pixels per sensor, max 640 total width
    
    total_width = num_sensors * strip_width
    
    # Create grayscale image: 0 = black (far), 255 = white (close)
    grayscale_values = (normalized_ranges * 255).astype(np.uint8)
    
    # Create the strip by repeating each value strip_width times horizontally
    strip = np.repeat(grayscale_values.reshape(1, -1), strip_height, axis=0)
    strip = np.repeat(strip, strip_width, axis=1)
    
    # Convert to BGR for OpenCV (all channels same for grayscale)
    bgr_strip = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    
    # Add sensor index labels every 10th sensor
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (0, 255, 0)  # Green text
    
    for i in range(0, num_sensors, 10):
        x_pos = i * strip_width + 2
        y_pos = strip_height - 5
        cv2.putText(bgr_strip, str(i), (x_pos, y_pos), font, font_scale, text_color, font_thickness)
    
    return bgr_strip

if __name__ == "__main__":
    # Test with 64 sensors (like in avoidance_v1.py)
    print("Testing rangefinder visualization...")
    
    # Create some test data simulating different scenarios
    test_scenarios = [
        ("no_obstacles", np.zeros(64)),  # All far/no obstacles
        ("all_close", np.ones(64)),      # All close obstacles
        ("gradient", np.linspace(0, 1, 64)),  # Gradient from left to right
        ("center_obstacle", np.exp(-((np.arange(64) - 32)**2) / 200)),  # Obstacle in center
        ("side_obstacles", np.concatenate([np.ones(10), np.zeros(44), np.ones(10)])),  # Obstacles on sides
    ]
    
    for name, test_data in test_scenarios:
        print(f"Creating visualization for: {name}")
        viz = create_rangefinder_visualization(test_data)
        filename = f"test_rangefinder_{name}.png"
        cv2.imwrite(filename, viz)
        print(f"  Saved: {filename} (shape: {viz.shape})")
        
        # Print some stats
        print(f"  Data range: {test_data.min():.3f} to {test_data.max():.3f}")
        print(f"  Grayscale range: {(test_data * 255).astype(np.uint8).min()} to {(test_data * 255).astype(np.uint8).max()}")
        print()
    
    print("Test completed! Check the generated PNG files.")