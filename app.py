from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run


MODEL_PATH = "C:/Users/hario/Desktop/wheat_dat2"


model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    # Update the prediction mappings
    if preds == 0:
        preds = "The Wheat is healthy"
    elif preds == 1:
        preds = "The Wheat has yellow rust"
    elif preds == 2:
        preds = "The wheat has brown rust"
    else:
        preds = "Unknown disease"

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == '__main__':
    app.run()