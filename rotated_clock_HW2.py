from matrices_vectors import *
import torch
import numpy as np
import cv2

def convert_tensor_to_cv_point(target_tensor):
    """Converts a 2x1 tensor to an (x, y) integer tuple for OpenCV drawing."""
    return (int(target_tensor[0].item()), int(target_tensor[1].item()))

def draw_clock_system(canvas, center_pos, hour_hand_pos, minute_hand_pos, color, radius, hh_thickness, mh_thickness, circle_thickness):
    """
    Encapsulated drawing function to render the clock face and hands.
    """
    pt_center = convert_tensor_to_cv_point(center_pos)
    pt_hour = convert_tensor_to_cv_point(hour_hand_pos)
    pt_minute = convert_tensor_to_cv_point(minute_hand_pos)

    # Drawing the clock components
    cv2.circle(canvas, pt_center, radius, color, circle_thickness) 
    cv2.line(canvas, pt_center, pt_hour, color, hh_thickness)
    cv2.line(canvas, pt_center, pt_minute, color, mh_thickness)
    return canvas

def normalize_to_image_coordinate(logic_vec, image_scale, canvas_center_vector, canvas_height):
    """
    Transforms a vector from mathematical coordinate system to image coordinate system.
    Note: Mathematical Y points up, but Image Y points down.
    """
    vec_img = logic_vec.clone()
    # TODO: Step 1. Scale the vector length using 'scalar_matrix'
    vec_img = scalar_matrix(image_scale, vec_img) # 縮放成圖像需求大小
    canvas_center_vector = torch.tensor(canvas_center_vector, dtype=torch.float32).reshape(2, 1)
    # TODO: Step 2. Align to center using 'matrix_sum'
    vec_img = matrix_sum(vec_img,canvas_center_vector) # 重中心點出發
    # TODO: Step 3. Flip Y direction using 'matrix_vector_product' 
    #       (Hint: A reflection matrix might be useful here)
    vec_img = matrix_vector_product(torch.tensor([[1.0,0.0],[0.0,-1.0]] , dtype=torch.float32),vec_img) # 沿x軸翻轉
    # TODO: Step 4. Move origin to top-left using 'matrix_sum'
    vec_img = matrix_sum(vec_img,torch.tensor([[0.0],[canvas_height]],dtype = torch.float32)) # 把它推回螢幕
    return vec_img

# --- Physical Simulation Parameters ---
deg2rad = np.pi / 180
clock_radius_math = 1.0  # Radius in abstract mathematical units
unit_vector_up = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(2, 1) # Points to 12 o'clock

hour_hand_length = 0.3 * clock_radius_math
hour_angular_velocity = 0.2
minute_hand_length = 0.8 * clock_radius_math
minute_angular_velocity = hour_angular_velocity * 12 # Minute hand is 12x faster

# Initial vectors at the 12 o'clock position
base_hour_vector = scalar_matrix(hour_hand_length, unit_vector_up)
base_minute_vector = scalar_matrix(minute_hand_length, unit_vector_up)

# Current vectors for the standard clock
current_hour_vector = base_hour_vector.clone()
current_minute_vector = base_minute_vector.clone()

# Current vectors for the mirrored clock
current_hour_vector_mirrored = base_hour_vector.clone()
current_minute_vector_mirrored = base_minute_vector.clone()

# --- Image & Video Rendering Parameters ---
canvas_width, canvas_height = 400, 300
clock_color_rgb = [255, 128, 0] # Orange

# Scaling and layout logic
image_scale = 0.5 * min(canvas_width/2, canvas_height/2) 
clock_panning_offset = 0.5 * (canvas_width/2) 
clock_radius_pixel = int(clock_radius_math * image_scale)
canvas_center_list = [int(canvas_width/2), int(canvas_height/2)]

# Initialize Video Writer
video_writer = cv2.VideoWriter('clock_simulation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (canvas_width, canvas_height))

# --- Main Simulation Loop ---
hour_angle_deg = 0.0
minute_angle_deg = 0.0
while hour_angle_deg <= 360:
    # Create a black background for every frame
    image_canvas = np.zeros([canvas_height, canvas_width, 3], dtype=np.uint8)
    
    canvas_center_tensor = torch.tensor(canvas_center_list, dtype=torch.float32).reshape(2, 1)

    # --- 1. Render Right Clock (Standard) ---
    # TODO: Use 'matrix_sum' to update hour_pos_img, minute_pos_img, and clock_center_img.
    # Hint: Combine panning_vector_right and the normalized hour/minute/center positions.
    panning_vector_right = torch.tensor([clock_panning_offset, 0.0], dtype=torch.float32).reshape(2, 1)
    hour_pos_img = normalize_to_image_coordinate(current_hour_vector, image_scale, canvas_center_list, canvas_height)
    minute_pos_img = normalize_to_image_coordinate(current_minute_vector, image_scale, canvas_center_list, canvas_height)
    clock_center_img = canvas_center_tensor
    #==============================================================
    hour_pos_img = matrix_sum(hour_pos_img, panning_vector_right) # 把指針推到右邊
    minute_pos_img = matrix_sum(minute_pos_img, panning_vector_right) # 把指針推到右邊
    clock_center_img = matrix_sum(clock_center_img, panning_vector_right) # 把指針推到右邊
    #==============================================================
    image_canvas = draw_clock_system(image_canvas, clock_center_img, hour_pos_img, minute_pos_img, clock_color_rgb, clock_radius_pixel, 5, 3, 3)

    # --- 2. Render Left Clock (Mirrored) ---
    # TODO: Use 'matrix_sum' to update hour_pos_img_mirr, minute_pos_img_mirr, and clock_center_img_mirr.
    # Hint: Combine panning_vector_left and the normalized hour/minute/center positions.
    panning_vector_left = torch.tensor([-clock_panning_offset, 0.0], dtype=torch.float32).reshape(2, 1)
    hour_pos_img_mirr = normalize_to_image_coordinate(current_hour_vector_mirrored, image_scale, canvas_center_list, canvas_height)
    minute_pos_img_mirr = normalize_to_image_coordinate(current_minute_vector_mirrored, image_scale, canvas_center_list, canvas_height)
    clock_center_img_mirr = canvas_center_tensor
    #==============================================================
    hour_pos_img_mirr = matrix_sum(hour_pos_img_mirr , panning_vector_left) # 把指針推到左邊
    minute_pos_img_mirr = matrix_sum(minute_pos_img_mirr , panning_vector_left) # 把指針推到左邊
    clock_center_img_mirr = matrix_sum(clock_center_img_mirr , panning_vector_left) # 把指針推到左邊
    #==============================================================
    image_canvas = draw_clock_system(image_canvas, clock_center_img_mirr, hour_pos_img_mirr, minute_pos_img_mirr, clock_color_rgb, clock_radius_pixel, 5, 3, 3)

    video_writer.write(image_canvas)

    # --- 3. Matrix Transformation Updates ---
    hour_angle_rad = hour_angle_deg * deg2rad
    minute_angle_rad = minute_angle_deg * deg2rad 
    
    # TODO: Update 'rotation_matrix_hour' by calling 'compute_rotation_matrix_2d' with (-hour_angle_rad)
    # Note: Clockwise rotation requires a negative angle in a standard Cartesian system.
    
    rotation_matrix_hour = torch.eye(2)
    rotation_matrix_hour = compute_rotation_matrix_2d(torch.tensor(-hour_angle_rad,dtype=torch.float32)) # 旋轉矩陣式往逆時針為正所以要加負號便逆時針
    # TODO: Update 'rotation_matrix_minute' by calling 'compute_rotation_matrix_2d' with (-minute_angle_rad)
    rotation_matrix_minute = torch.eye(2)
    rotation_matrix_minute = compute_rotation_matrix_2d(torch.tensor(-minute_angle_rad,dtype=torch.float32)) # 旋轉矩陣式往逆時針為正所以要加負號便逆時針
    # Update Standard Clock
    # TODO: Use 'matrix_vector_product' to update current_hour_vector and current_minute_vector
    #更新指針
    current_hour_vector = matrix_vector_product(rotation_matrix_hour, base_hour_vector)
    current_minute_vector = matrix_vector_product(rotation_matrix_minute, base_minute_vector)
    # TODO: Update 'mirror_matrix' by calling 'compute_y_mirror_matrix_2d'
    # This matrix will be used to reflect the clock hands across the Y-axis.
    #製造一個沿y軸翻轉的矩陣
    mirror_matrix = torch.eye(2)
    mirror_matrix = compute_y_mirror_matrix_2d()
    # --- Updating Mirrored Clock Vectors via Matrix Composition ---
    # TODO: Compute 'rotation_matrix_hour_mirrored' by multiplying 'mirror_matrix' and 'rotation_matrix_hour'
    # Hint: Use 'matrix_multiplication'.
    #更新時針
    rotation_matrix_hour_mirrored = torch.eye(2)
    rotation_matrix_hour_mirrored = matrix_multiplication( mirror_matrix,rotation_matrix_hour)
    # TODO: Compute 'rotation_matrix_minute_mirrored' by multiplying 'mirror_matrix' and 'rotation_matrix_minute'
    # Hint: Use 'matrix_multiplication'.
    #更新分針
    rotation_matrix_minute_mirrored = torch.eye(2)
    rotation_matrix_minute_mirrored = matrix_multiplication( mirror_matrix,rotation_matrix_minute)
    # Update Mirrored Clock
    # TODO: Use 'matrix_vector_product' to update current_hour_vector_mirrored and current_minute_vector_mirrored
    #更新指針
    current_hour_vector_mirrored = matrix_vector_product(rotation_matrix_hour_mirrored, base_hour_vector)
    current_minute_vector_mirrored = matrix_vector_product(rotation_matrix_minute_mirrored, base_minute_vector)
    hour_angle_deg += hour_angular_velocity
    minute_angle_deg += minute_angular_velocity

video_writer.release()
