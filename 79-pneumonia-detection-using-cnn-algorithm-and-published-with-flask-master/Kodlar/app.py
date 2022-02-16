from __future__ import division, print_function
# coding=utf-8
from skimage.transform import resize
import skimage.transform as st
import tensorflow as tf
# from scipy.misc import imread, imresize
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
# and after predicting my data i inserted this part of code
# K.clear_session()
# Flask utils


# Resize & Show the image
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/models.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
graph = tf.get_default_graph()
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# graph = tf.get_default_graph() # Change
print('Model loaded. Check http://127.0.0.1:5000/')

# def classify(image, model):
#     #Class names for cifar 10
#     class_names = ['airplane','automobile','bird','cat','deer',
#                'dog','frog','horse','ship','truck']
#     preds = model.predict(image)
#     classification = np.argmax(preds)
#     final = pd.DataFrame({'name' : np.array(class_names),'probability' :preds[0]})
#     return final.sort_values(by = 'probability',ascending=False),class_names[classification]


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64, 3))
    # Preprocessing the image
#     x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(img, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

    preds = model.predict(np.array(x))
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
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

# #         #Get image URL as input
# #         image_url = request.form['image_url']
# #         image = io.imread(image_url)

#         #Apply same preprocessing used while training CNN model
#         image_small = st.resize(file_path, (32,32,3))
#         x = np.expand_dims(image_small.transpose(2, 0, 1), axis=0)

#         #Call classify function to predict the image class using the loaded CNN model
#         final,pred_class = classify(x, model)
#         print(pred_class)
#         print(final)

        # Store model prediction results to pass to the web page
#         message = "Model prediction: {}".format(pred_class)
        # Make prediction
        global graph
        with graph.as_default():

            preds = model_predict(file_path, model)
            print(preds)

            number_to_class = ['Normal', 'Pneumonia']
            index = np.argsort(preds[0, :])
#         for x in range(len(number_to_class)):
#             if number_to_class[x] == 1:
#                 print(preds[0][i])

        # Process your result for human
            pred_class = preds.argmax(axis=-1)            # Simple argmax
#         pred_class = decode_predictions(preds, top=1) # ImageNet Decode
#         result = str(pred_class[0][1]) # Convert to string
            return str(number_to_class[index[1]])+str(pred_class)

        return None


# K.clear_session()

# if _name_ == '_main_':
#     app.run(debug=True, threaded=False)