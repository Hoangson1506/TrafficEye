import gradio as gr
import cv2
import numpy as np
import json
import os
from core.traffic_system import TrafficSystem
from utils.zones import save_zones, load_zones
from utils.config import save_config
from utils.storage import MinioClient

# Initialize System
# TrafficSystem represents the physical detection system, so a single global instance is appropriate.
system = TrafficSystem()

# Initialize MinIO Client gracefully
try:
    minio_client = MinioClient()
except Exception as e:
    print(f"Warning: Could not initialize MinIO Client: {e}")
    minio_client = None

# --- Dashboard Logic ---
def get_dashboard_stats():
    if not minio_client:
        return "MinIO storage not available."
    try:
        proofs = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'])
        count = proofs.get('KeyCount', 0)
        return f"Total Violations Recorded: {count}"
    except Exception as e:
        return f"Error connecting to MinIO: {e}"

def get_proof_gallery():
    if not minio_client:
        return []
    try:
        objects = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'], MaxKeys=10)
        images = []
        if 'Contents' in objects:
            for obj in objects['Contents']:
                # Note: In a real deployment, use presigned URLs or a proxy.
                # Assuming localhost access for now as per original code.
                images.append((f"http://localhost:9000/{minio_client.buckets['proofs']}/{obj['Key']}", obj['Key']))
        return images
    except:
        return []

# --- Visualization Logic ---
def stream_video():
    if not system.running:
        system.start()
    
    # We iterate over the generator
    for frame, stats in system._process_flow():
        if not system.running:
            break
        yield frame, str(stats)

def stop_system():
    system.stop()
    return None, "System Stopped"

# --- Drawing Logic (State-based) ---

def get_initial_drawing_state():
    return {
        "points": [],
        "image": None,
        "mode": "Polygon"
    }

def capture_frame_for_drawing(state):
    frame = system.capture_first_frame()
    if frame is None:
        return None, "Failed to capture frame. Start system first to get a frame?", state
    
    # Update state
    state["image"] = frame
    state["points"] = [] # Reset points on new image
    
    return frame, "Frame Captured", state

def load_image_for_drawing(image_input, state):
    if image_input is None:
        return None, state
    
    state["image"] = image_input
    state["points"] = []
    
    return image_input, state

def on_select(evt: gr.SelectData, image_input, state, mode_input):
    if state["image"] is None and image_input is not None:
        state["image"] = image_input
        
    if state["image"] is None:
        return image_input, state
    
    x, y = evt.index[0], evt.index[1]
    state["points"].append([x, y])
    points = state["points"]
    mode = state["mode"] if "mode" in state else mode_input # Use state mode or input
    
    # Draw logic
    img_copy = state["image"].copy()
    
    # Draw points
    for pt in points:
        cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)
    
    # Draw lines/polygons/rectangles
    if mode == "Polygon" and len(points) >= 2:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], True if len(points) >= 3 else False, (255, 0, 0), 2)
    
    elif mode in light_zone_map and len(points) >= 2:
        # Draw rectangles for light zones (2 points = 1 rectangle)
        for j in range(0, len(points)-1, 2):
            cv2.rectangle(img_copy, tuple(points[j]), tuple(points[j+1]), (0, 255, 255), 2)
        
    elif mode in line_zone_map and len(points) >= 2:
        # Draw line pairs for line-based modes
        for j in range(0, len(points)-1, 2):
            cv2.line(img_copy, tuple(points[j]), tuple(points[j+1]), (0, 0, 255), 2)

    return img_copy, state

def clear_points(state):
    state["points"] = []
    return state["image"], state

def revert_point(state):
    if state["points"]:
        state["points"].pop()
    
    # Redraw
    if state["image"] is None:
        return None, state
        
    img_copy = state["image"].copy()
    points = state["points"]
    mode = state["mode"]
    
    for pt in points:
        cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)
        
    if mode == "Polygon" and len(points) >= 2:
         pts = np.array(points, np.int32).reshape((-1, 1, 2))
         cv2.polylines(img_copy, [pts], len(points)>=3, (255, 0, 0), 2)
    elif mode in light_zone_map and len(points) >= 2:
         for j in range(0, len(points)-1, 2):
            cv2.rectangle(img_copy, tuple(points[j]), tuple(points[j+1]), (0, 255, 255), 2)
    elif mode in line_zone_map and len(points) >= 2:
         for j in range(0, len(points)-1, 2):
            cv2.line(img_copy, tuple(points[j]), tuple(points[j+1]), (0, 0, 255), 2)
            
    return img_copy, state

line_zone_map = {
    "Violation Lines": "violation_lines", 
    "Special Violation Lines": "special_violation_lines",
    "Left Exception Lines": "left_exception_lines",
    "Right Exception Lines": "right_exception_lines",
    "Other Exception Lines": "other_exception_lines"
}

# Mapping for traffic light detection zones (rectangles)
light_zone_map = {
    "Straight Light Zone": "straight",
    "Left Light Zone": "left",
    "Right Light Zone": "right"
}

def save_drawing(state):
    points = state["points"]
    mode = state["mode"]
    
    if not points:
        return "No points to save"
    
    zones = load_zones()
    
    if mode == "Polygon":
        if len(points) < 3:
            return "Polygon needs at least 3 points"
        zones["polygon"] = points
    elif mode in light_zone_map:
        # Traffic light zone (rectangle mode)
        if len(points) < 2:
            return "Light zone needs at least 2 points (top-left, bottom-right)"
        if len(points) % 2 != 0:
            return "Light zone needs an even number of points (pairs of corners)"
        
        category = light_zone_map[mode]
        # Ensure 'light_zones' exists
        if "light_zones" not in zones:
            zones["light_zones"] = {"straight": [], "left": [], "right": []}
        
        zones["light_zones"][category] = points
    elif mode in line_zone_map:
        # It's a line category
        category = line_zone_map.get(mode, "violation_lines")
        
        # Ensure 'lines_config' exists
        if "lines_config" not in zones:
            zones["lines_config"] = {}
            
        zones["lines_config"][category] = points
        # Backward compatibility
        if category == "violation_lines":
             zones["lines"] = points
        
    save_zones(zones)
    return f"Saved {len(points)} points for {mode}"

def set_drawing_mode(new_mode, state):
    state["mode"] = new_mode
    state["points"] = []
    # Return status, cleared image (or original image cleared of points), and updated state
    return f"Mode switched to {new_mode}. Points cleared.", state["image"], state


# --- Settings Logic ---
def update_settings(data_path, vehicle_model, license_model, tracker, conf, fps):
    new_config = system.config.copy()
    
    if 'system' not in new_config: new_config['system'] = {}
    new_config['system']['data_path'] = data_path
    new_config['system']['vehicle_model'] = vehicle_model
    new_config['system']['license_model'] = license_model
    new_config['system']['tracker'] = tracker
    
    new_config['detections']['conf_threshold'] = conf
    new_config['violation']['fps'] = fps
    
    system.update_config(new_config)
    save_config(new_config, system.config_path)
    return "Settings Saved. Restart system to apply changes."


# --- UI Construction ---
with gr.Blocks(title="Traffic Violation Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Traffic Violation Detection System")
    
    # Global state for the session
    drawing_state_comp = gr.State(get_initial_drawing_state())
    
    with gr.Tabs():
        # --- Tab 1: Dashboard ---
        with gr.Tab("Dashboard"):
            gr.Markdown("### System Statistics")
            stats_output = gr.Textbox(label="Status", value=get_dashboard_stats)
            refresh_btn = gr.Button("Refresh Stats")
            refresh_btn.click(get_dashboard_stats, outputs=stats_output)
            
        # --- Tab 2: Visualization ---
        with gr.Tab("Visualization"):
            with gr.Row():
                start_btn = gr.Button("Start System", variant="primary")
                stop_btn = gr.Button("Stop System", variant="stop")
            
            with gr.Row():
                video_output = gr.Image(label="Live Feed", interactive=False)
                logs_output = gr.Textbox(label="Live Statistics")
            
            start_btn.click(stream_video, outputs=[video_output, logs_output])
            stop_btn.click(stop_system, outputs=[video_output, logs_output])
            
        # --- Tab 3: Zone Drawing ---
        with gr.Tab("Zone Drawing"):
            gr.Markdown("### Interactive Zone Editor")
            with gr.Row():
                with gr.Column(scale=1):
                    drawing_modes = ["Polygon", "Violation Lines", "Special Violation Lines", "Left Exception Lines", "Right Exception Lines", "Other Exception Lines", "Straight Light Zone", "Left Light Zone", "Right Light Zone"]
                    mode_dropdown = gr.Dropdown(drawing_modes, label="Drawing Mode", value="Polygon")
                    
                    capture_btn = gr.Button("Capture Frame from Source")
                    save_poly_btn = gr.Button("Save Configuration")
                    revert_btn = gr.Button("Revert Last Point")
                    clear_btn = gr.Button("Clear All Points")
                    draw_status = gr.Textbox(label="Status")
                
                with gr.Column(scale=3):
                    canvas = gr.Image(label="Drawing Canvas", interactive=True, type="numpy")

            # Update state when mode changes
            mode_dropdown.change(set_drawing_mode, 
                               inputs=[mode_dropdown, drawing_state_comp], 
                               outputs=[draw_status, canvas, drawing_state_comp])
            
            # Event listeners
            capture_btn.click(capture_frame_for_drawing, 
                            inputs=[drawing_state_comp], 
                            outputs=[canvas, draw_status, drawing_state_comp])
            
            canvas.upload(load_image_for_drawing, 
                        inputs=[canvas, drawing_state_comp], 
                        outputs=[canvas, drawing_state_comp])
            
            canvas.select(on_select, 
                        inputs=[canvas, drawing_state_comp, mode_dropdown], 
                        outputs=[canvas, drawing_state_comp])
            
            revert_btn.click(revert_point, 
                           inputs=[drawing_state_comp], 
                           outputs=[canvas, drawing_state_comp])
            
            clear_btn.click(clear_points, 
                          inputs=[drawing_state_comp], 
                          outputs=[canvas, drawing_state_comp])
            
            save_poly_btn.click(save_drawing, 
                              inputs=[drawing_state_comp], 
                              outputs=[draw_status])
            
        # --- Tab 4: Settings ---
        with gr.Tab("Settings"):
            gr.Markdown("### System Configuration")
            with gr.Row():
                data_path_input = gr.Textbox(label="Data Path / RSTP URL", value=system.config.get('system', {}).get('data_path', 'cam_ai'))
                tracker_input = gr.Dropdown(["sort", "bytetrack"], label="Tracker", value=system.config.get('system', {}).get('tracker', 'bytetrack'))
            
            with gr.Row():
                vehicle_model_input = gr.Textbox(label="Vehicle Model Path", value=system.config.get('system', {}).get('vehicle_model', 'detect_gtvn.pt'))
                license_model_input = gr.Textbox(label="License Plate Model Path", value=system.config.get('system', {}).get('license_model', 'lp_yolo11s.pt'))
            
            with gr.Row():
                conf_slider = gr.Slider(0.0, 1.0, value=system.config['detections']['conf_threshold'], label="Confidence Threshold")
                fps_slider = gr.Slider(1, 60, value=system.config['violation']['fps'], label="FPS")
            
            save_settings_btn = gr.Button("Save Settings")
            settings_status = gr.Textbox(label="Status")
            
            save_settings_btn.click(update_settings, 
                                    inputs=[data_path_input, vehicle_model_input, license_model_input, tracker_input, conf_slider, fps_slider], 
                                    outputs=settings_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
