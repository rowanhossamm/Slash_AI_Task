import streamlit as st
from PIL import Image
import numpy as np
import torch
import pandas as pd

def detect_objects(image):
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # Load YOLOv5 model

    results = model(image)

    return results.pandas().xyxy[0]


uploaded_file = st.file_uploader("upload an image")

if uploaded_file is not None:
    
    image = Image.open(uploaded_file) # Display the uploaded image
    st.image(image, use_column_width=True)
    st.write("")
    

    if st.button('Analyse Image'):
        st.write("analysing image...")
        img = np.array(image) # Convert PIL image to numpy array
        
        object_results = detect_objects(img) # Perform object detection

        st.write(f"Total Objects: {len(object_results)}")

        st.write("objects:")
        for index, row in object_results.iterrows():
            st.write(f"- {row['name']}")


