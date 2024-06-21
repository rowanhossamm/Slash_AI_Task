import streamlit as st
from PIL import Image
import numpy as np
import torch
from pathlib import Path

# Install YOLOv5 if not already installed
if not (Path("yolov5").exists()):
    st.write("Installing YOLOv5...")
    git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5 repository
    pip install -r yolov5/requirements.txt

st.title("Object Detection using YOLOv5")

# Function to perform object detection using YOLOv5
def detect_objects(image):
    # Load YOLOv5 model
    model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', force_reload=True)

    # Convert PIL image to numpy array
    img = np.array(image)

    # Perform inference
    results = model(img)

    # Return results
    return results.pandas().xyxy[0]

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Button to trigger the analysis
    if st.button('Detect Objects'):
        st.write("Detecting objects...")
        object_results = detect_objects(image)
        
        # Display detected objects
        st.write("Detected objects:")
        for index, row in object_results.iterrows():
            st.write(f"- {row['name']} - confidence: {row['confidence']:.2f}, bounding box: {row['xmin']},{row['ymin']} to {row['xmax']},{row['ymax']}")
