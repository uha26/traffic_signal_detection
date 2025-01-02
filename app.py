from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
import io
import os

# Initialize FastAPI
app = FastAPI()

# Load class labels from label.csv
def load_class_labels(csv_file="labels.csv"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert the DataFrame into a dictionary (index -> label)
    class_labels = dict(zip(df['ClassId'], df['Name']))
    return class_labels

# Load the labels into a dictionary
class_labels = load_class_labels("labels.csv")

# Check if model file exists and load it
model_file = "model.keras"
if os.path.exists(model_file):
    model = load_model(model_file)  # Load the model in the .keras format
else:
    model_file = "model.h5"  # Fallback to .h5 format if .keras is not found
    if os.path.exists(model_file):
        model = load_model(model_file)  # Load the model in the .h5 format
    else:
        raise ValueError(f"Model file {model_file} not found.")

# Preprocessing function to match the training data preprocessing
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (only if needed)
    img = cv2.equalizeHist(img)  # Histogram equalization (if used in training)
    img = img / 255.0  # Normalize the image (if used in training)
    return img

# Utility function to read image from upload
def read_image(file: UploadFile):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes))
    return img

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img = read_image(file)

        # Preprocess the image
        img = np.array(img)
        img = preprocess_image(img)

        # Print the shape of the image for debugging
        print("Image Shape after Preprocessing:", img.shape)
        
        # Resize the image to match the input size expected by the model
        img = cv2.resize(img, (32, 32))
        print("Resized Image Shape:", img.shape)
        
        # Reshape the image to (batch_size, height, width, channels)
        img = img.reshape(1, 32, 32, 1)  # For grayscale images
        print("Image Shape after Reshaping:", img.shape)

        # Predict the class of the image
        prediction = model.predict(img)

        # Print the raw prediction output for debugging
        print("Raw Prediction Output:", prediction)

        # Get the predicted class index (highest probability)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        print("Predicted Class Index:", predicted_class_index)

        # Get the description for the predicted class from the CSV
        predicted_class_description = class_labels.get(predicted_class_index, "Unknown Class")
        
        # Get the confidence of the prediction
        confidence = float(np.max(prediction))  # The highest confidence value

        return JSONResponse(content={
            "predicted_class": predicted_class_description,
            "confidence": confidence
        })

    except Exception as e:
        # Catch any errors and return a meaningful message
        print(f"Internal Server Error: {e}")
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)
