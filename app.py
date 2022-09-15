#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:43:30 2022

@author: abhinav
"""


from __future__ import division, print_function

import sys 
import os
import numpy as np 

# Keras libararies
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import tensorflow as tf
# Flask utils 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# defining a flask app 

app = Flask(__name__,template_folder="template")

# Loading the model saved with keras 
model = load_model('Model.h5')


def model_predict(path, model):
    img = image.load_img(path, target_size=(224,224))
    
    # Preprocessing the image
    img = image.img_to_array(img)
    
    img = img/ 255
        
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    
    prediction = model.predict(img)
    
    return prediction

@app.route("/",methods=["GET"])
def home():
    return render_template("/index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method == "POST":
        target = request.files["file"]
        
        basepath = os.path.dirname(__file__)
        
        file_path = os.path.join(basepath,'uploads',secure_filename(target.filename))
        target.save(file_path)

        op = np.array(model_predict(file_path, model))
        
        
        if np.argmax(op)==0:
            return render_template("/index.html",prediction_text = " MODERATE DEMENTED")
        elif np.argmax(op)==1:
            return render_template("/index.html",prediction_text = " NON DEMENTED")
        elif np.argmax(op)==2:
            return render_template("/index.html",prediction_text = " MILD DEMENTED")
        else:
            return render_template("/index.html",prediction_text = " VERY MILD DEMENTED")


if __name__ == "__main__":
    app.run(debug=True)  
