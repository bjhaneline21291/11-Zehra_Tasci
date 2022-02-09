from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import argparse
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow_docs as tfdocs
#import tensorflow_docs.modeling
#import tensorflow_docs.plots
from tensorflow import keras
from tensorflow.keras import layers
from google.cloud import storage

print(tf.__version__)

#keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name',
            type=str,
            default='gs://',
            help='The bucket where the output has to be stored')

    parser.add_argument('--epochs',
            type=int,
            default=1,
            help='Number of epochs for training the model')

    args = parser.parse_known_args()[0]
    return args

def train(bucket_name, epochs):

    train_file = bucket_name + '/output/train.csv'
    test_file = bucket_name + '/output/test.csv'
    train_labels = bucket_name + '/output/train_label.csv'
    test_labels = bucket_name + '/output/test_label.csv'

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']

    trainDF = pd.read_csv(train_file)
    trainDF = trainDF.drop(trainDF.columns[0], axis = 1) 
    trainLabelDF  = pd.read_csv(train_labels)
    trainLabelDF  = trainLabelDF.drop(trainLabelDF.columns[0], axis = 1) 

    print(trainDF.shape)
    print(trainLabelDF.shape)

    testDF = pd.read_csv(test_file)
    testLabelsDF = pd.read_csv(test_labels)

    checkpoint_path =  bucket_name + '/export/model/'

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    EPOCHS = epochs

    model = build_model(trainDF)
    print(model.summary())

    print(trainDF.keys())
    print(trainDF)
    print(trainLabelDF)

    history = model.fit(trainDF, trainLabelDF, epochs=EPOCHS, validation_split = 0.2, verbose=1,
            #callbacks=cp_callback)
            callbacks=[early_stop, cp_callback])



def build_model(trainDF):
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(trainDF.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

if __name__ == '__main__':

    args = parse_arguments()
    print(args)
    train(args.bucket_name, int(args.epochs))
