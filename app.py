from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import uuid
from pathlib import Path
import tempfile
import shutil
import json

# Initialize FastAPI application
app = FastAPI()

# Persistent storage directory for processed videos
PROCESSED_STORAGE_DIR = "processed_videos"
os.makedirs(PROCESSED_STORAGE_DIR, exist_ok=True)

def rotate_image_with_padding(image, angle):
    """Rotate image without cropping by adding padding."""
    height, width = image.shape[:2]
    
    # Compute the diagonal length of the image
    diagonal = int(np.sqrt(height**2 + width**2))
    
    # Create a larger canvas to avoid cropping
    padded_image = cv2.copyMakeBorder(image, 
                                      (diagonal - height) // 2, (diagonal - height) // 2,
                                      (diagonal - width) // 2, (diagonal - width) // 2,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))  # Black padding
    
    # Get new center after padding
    padded_h, padded_w = padded_image.shape[:2]
    center = (padded_w // 2, padded_h // 2)
    
    # Apply rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_w, padded_h))
    
    return rotated_image

def extract_corrected_frames(video_path, rotations, corrected_folder):
    # Load JSON data
    # with open(json_path, 'r') as file:
    #     data = json.load(file)
    
    # Create output directory if it doesn't exist
    os.makedirs(corrected_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_counter = 1  # Counter for sequential naming
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return False

    for idx, entry in enumerate(rotations):
        if idx % 20 != 0:  # Skip frames that are not multiples of 20
            continue
        
        timestamp_ms = entry["timestamp"]
        x_tilt = entry["x"]
        
        # Convert milliseconds to seconds
        timestamp = timestamp_ms / 1000.0
        
        # Calculate frame number
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            print(f"Frame at {timestamp_ms} ms not found.")
            continue
        
        # Correct tilt with padding
        corrected_frame = rotate_image_with_padding(frame, -x_tilt)
        
        corrected_frame = cv2.resize(corrected_frame, (1100, 1100))
        
        # Save corrected frame with sequential naming
        corrected_filename = f"{corrected_folder}/frame_{frame_counter}.jpg"
        cv2.imwrite(corrected_filename, corrected_frame)
        
        frame_counter += 1  # Increment counter for next saved frame
    
    cap.release()
    print("Corrected frames saved successfully.")
    return True if frame_counter > 1 else False

# Function to apply a grey color tint with 90% grey and 10% transparency
def apply_black_tint(image, mask):
    tinted_image = image.copy()
    mask = (mask * 255).astype(np.uint8)  
    black_color = np.array([0, 0, 0])  

    for c in range(3):  
        tinted_image[:, :, c] = np.where(
            mask == 255,  
            0.85 * black_color[c] + 0.15 * image[:, :, c],  
            image[:, :, c]  
        )

    return tinted_image

# Function to apply blur to the segmented regions (for plate)
def apply_blur(image, mask):
    blurred_image = image.copy()
    mask = (mask * 255).astype(np.uint8)  
    blurred_region = image.copy()
    blurred_region[mask == 255] = cv2.GaussianBlur(image[mask == 255], (99, 99), 30)
    return blurred_region

# Function to apply edge detection
def apply_edge_detection(image):
    # Step 1: Use a stronger bilateral filter to smoothen while preserving edges
    bilateral_filtered = cv2.bilateralFilter(image, d=7, sigmaColor=70, sigmaSpace=70)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Canny edge detection with higher thresholds (less sensitivity)
    edges = cv2.Canny(gray, threshold1=100, threshold2=150)

    # Step 4: Reduce noise using a milder morphological opening
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 5: Keep only very small noise removal to preserve fine details
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 30:  # Ignore only very tiny spots
            cv2.drawContours(edges, [contour], -1, 0, thickness=cv2.FILLED) 

    # Step 6: Reduce dilation to avoid thickening edges too much
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 7: Apply a mild Gaussian blur
    edges = cv2.GaussianBlur(edges, (5, 5), 2)

    # Convert edges to a 3-channel image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
    
    return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    
def normalize_car_distance(image, mask, background_path="background.png", offset_x=0, offset_y=50, scale=1.2,reference_size=None):
    mask = (mask * 255).astype(np.uint8)
    car_pixels = cv2.bitwise_and(image, image, mask=mask)
    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped_car = car_pixels[y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

        car_height, car_width = cropped_car.shape[:2]
        car_size = max(car_width, car_height)

        if reference_size is None:
            reference_size = car_size

        scale_factor = (reference_size / car_size) * scale
        new_car_width = int(car_width * scale_factor)
        new_car_height = int(car_height * scale_factor)

        cropped_car = cv2.resize(cropped_car, (new_car_width, new_car_height), interpolation=cv2.INTER_CUBIC)
        cropped_mask = cv2.resize(cropped_mask, (new_car_width, new_car_height), interpolation=cv2.INTER_NEAREST)

        background = cv2.imread(background_path)
        if background is None:
            print("Error: Background image not found. Using white canvas instead.")
            background = np.full((1080, 2048, 3), 255, dtype=np.uint8)
        else:
            background = cv2.resize(background, (2048, 1080))

        x_offset = max(0, (2048 - new_car_width) // 2 + offset_x)
        y_offset = max(0, (1080 - new_car_height) // 2 + offset_y)
        x_offset = min(x_offset, 2048 - new_car_width)
        y_offset = min(y_offset, 1080 - new_car_height)

        for c in range(3):
            background[y_offset:y_offset + new_car_height, x_offset:x_offset + new_car_width, c] = np.where(
                cropped_mask > 0,
                cropped_car[:, :, c],
                background[y_offset:y_offset + new_car_height, x_offset:x_offset + new_car_width, c]
            )

        return background, reference_size
    else:
        print("No valid car mask found.")
        return image, reference_size

# Function to process frames with depth and individual tints
def process_frames_with_depth_and_individual_tints(frames_folder, processed_folder, yolo_model_path, sam_model_path):
    yolo_model = YOLO(yolo_model_path)
    
    sam_model = SAM(sam_model_path)

    
    os.makedirs(processed_folder, exist_ok=True)

    frame_files = [f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))]
    frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    if not frame_files:
        print("No frames found for processing.")
        return False

    print("Processing frames with edge detection, car detection, and individual region tinting...")

    region_classes = ["back", "front", "windb", "windf"]

    for idx, frame_file in enumerate(frame_files, start=1):
        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)

        image_with_edges = apply_edge_detection(image)

        results = yolo_model(image_with_edges)

        class_names = results[0].names
        region_class_ids = {region: next((k for k, v in class_names.items() if v == region), None) for region in region_classes}
        plate_class_id = next((k for k, v in class_names.items() if v == "plate"), None)
        car_class_id = next((k for k, v in class_names.items() if v == "car"), None)

        region_boxes = {region: [] for region in region_classes}
        plate_boxes = []
        car_boxes = []

        for idx, cls in enumerate(results[0].boxes.cls):
            for region in region_classes:
                if int(cls) == region_class_ids[region]:
                    region_boxes[region].append(results[0].boxes.xyxy[idx].tolist())
            if int(cls) == plate_class_id:
                plate_boxes.append(results[0].boxes.xyxy[idx].tolist())
            if int(cls) == car_class_id:
                bbox = results[0].boxes.xyxy[idx].tolist()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                car_boxes.append((bbox, area))

        for region in region_classes:
            boxes = region_boxes.get(region, [])
            if boxes:
                sam_results_region = sam_model(image_with_edges, bboxes=np.array(boxes), verbose=False, save=False, device="cpu")
                region_masks = [result.masks.data.cpu().numpy() for result in sam_results_region]

                for masks in region_masks:
                    for mask in masks:
                        image = apply_black_tint(image, mask)

        if plate_boxes:
            sam_results_plate = sam_model(image_with_edges, bboxes=np.array(plate_boxes), verbose=False, save=False, device="cpu")
            plate_masks = [result.masks.data.cpu().numpy() for result in sam_results_plate]

            for masks in plate_masks:
                for mask in masks:
                    image = apply_blur(image, mask)

        if car_boxes:
            largest_car_box = max(car_boxes, key=lambda x: x[1])[0]
            sam_results_car = sam_model(image_with_edges, bboxes=np.array([largest_car_box]), verbose=False, save=False, device="cpu")
            car_masks = [result.masks.data.cpu().numpy() for result in sam_results_car]

            for masks in car_masks:
                for mask in masks:
                    processed_image, _ = normalize_car_distance(image, mask,background_path="background.png" , offset_x=0, offset_y=70, scale=1.5)

                    output_path = os.path.join(processed_folder, f"processed_frame_{frame_file}")
                    cv2.imwrite(output_path, processed_image)
                    print(f"Processed and saved: {output_path}")

        else:
            print(f"No cars detected in frame {frame_file}.")
    print("All frames processed.")
    return True
    

def images_to_video(image_folder, output_video, frame_rate=10):
    """
    Converts images from a folder into a video.
    
    Args:
        image_folder (str): Path to the folder containing images.
        output_video (str): Path to save the output video (e.g., "output.mp4").
        frame_rate (int): Frame rate for the video.
    """
    # Get a list of all images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))]
    
    # Sort images based on numbers in filenames (if filenames contain numbers)
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # Sort by numerical part of filename

    if not images:
        print("No images found in the folder.")
        return False

    # Read the first image to determine the frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")
    return True


# Process video API
@app.post("/process_video")
async def process_video_api(video: UploadFile = File(...), rotations: str = Form(...)):
    """
    Expects:
      - a video file upload,
      - a form field 'rotations' containing a JSON string (list of dicts with keys "timestamp" and "x")
    """

    try:
        # Parse rotations JSON from form field
        rotations_data = json.loads(rotations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rotations JSON: {e}")

    try:
        if not video:
            raise HTTPException(status_code=400, detail="No video file uploaded")

        # Create temporary directories for this request
        with tempfile.TemporaryDirectory() as upload_dir, \
             tempfile.TemporaryDirectory() as frames_dir, \
             tempfile.TemporaryDirectory() as processed_dir:

            # Save the video to a temporary directory
            video_path = os.path.join(upload_dir, "input_video.mp4")
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)

            # Extract frames
            frames_extracted = extract_corrected_frames(video_path, rotations_data, frames_dir)
            if not frames_extracted:
                return JSONResponse(
                    status_code=422,
                    content={"message": "No frames could be extracted from the video"}
                )

            # Process frames
            yolo_model_path = "lastnew2.pt"
            sam_model_path = "sam2.1_l.pt"
            frames_processed = process_frames_with_depth_and_individual_tints(frames_dir, processed_dir, yolo_model_path, sam_model_path)
            if not frames_processed:
                return JSONResponse(
                    status_code=422,
                    content={"message": "No cars detected in video frames"}
                )

            # Combine frames into a video
            temp_video_path = os.path.join(upload_dir, "output_video.mp4")
            video_created = images_to_video(processed_dir, temp_video_path, frame_rate=10)
            if not video_created:
                return JSONResponse(
                    status_code=422,
                    content={"message": "Could not create output video from processed frames"}
                )
            
            # Move the output video to persistent storage with a unique filename
            unique_filename = f"output_{uuid.uuid4()}.mp4"  # Unique filename
            final_video_path = os.path.join(PROCESSED_STORAGE_DIR, unique_filename)
            shutil.move(temp_video_path, final_video_path)

            # Return the filename for downloading
            return JSONResponse({
                "message": "Processing completed successfully",
                "download_url": f"/download_and_delete/{unique_filename}"
            })

    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"message": f"Video processing failed: {str(e)}"}
        )

# Download and delete endpoint
@app.get("/download_and_delete/{filename}")
async def download_and_delete_video(filename: str):
    try:
        # Locate the processed video in the persistent storage directory
        file_path = Path(PROCESSED_STORAGE_DIR) / filename

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Return the file and delete it after download
        response = FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=filename
        )
        # os.remove(file_path)  # Delete the file after download
        return response

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to download or delete file: {str(e)}")

@app.get("/delete_video/{filename}")
async def root(filename: str):
    try:
        # Locate the processed video in the persistent storage directory
        file_path = Path(PROCESSED_STORAGE_DIR) / filename

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        return JSONResponse({
            "message": f"File {filename} deleted successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to delete file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)