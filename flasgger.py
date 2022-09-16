#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 18:45:35 2022

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
import tensorflow 
# Flask utils 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import flasgger 
from flasgger import Swagger 

# defining a flask app 

app = Flask(__name__,template_folder="template")
Swagger(app)

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
    return "WELCOME TO ALZHEIMER'S DETECTION MODEL";

@app.route("/",methods=["POST"])
def predict():        
        """ MOVIE SENTIMENT ANALYSIS
        
        ---
        parameters:
            - name : target 
              in : formData
              type : file 
              required : true 
        responses:
            200:
                description : The OUTPUT VALUES
        """
        
        target = request.files.get("target");
        
        basepath = os.path.dirname(__file__)
        
        file_path = os.path.join(basepath,'uploads',secure_filename(target.filename))
        target.save(file_path)

        op = np.array(model_predict(file_path, model))
        
        if np.argmax(op)==0:
            return " MODERATE DEMENTED"
        elif np.argmax(op)==1:
            return " NON DEMENTED"
        elif np.argmax(op)==2:
            return " MILD DEMENTED"
        else:
            return " VERY MILD DEMENTED"


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)
        


        
    
    
    
    
    
    
    
    
    