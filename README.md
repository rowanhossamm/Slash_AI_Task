Object Detection with YOLOv5 and Streamlit
This repository hosts a web application built with Streamlit to perform object detection using the YOLOv5 model. Users can upload images to the interface, where the model identifies and labels objects in real-time.

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/rowanhossamm/Slash_AI_streamlit
cd Slash_AI_streamlit

Set up the environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
streamlit run app.py
Usage
Upload an image using the file uploader.
Click the "Analyse Image" button.
View the detected objects and their labels.

Credits
YOLOv5 by Ultralytics: GitHub Repository
Streamlit: Official Website
License
This project is licensed under the MIT License - see the LICENSE file for details.
