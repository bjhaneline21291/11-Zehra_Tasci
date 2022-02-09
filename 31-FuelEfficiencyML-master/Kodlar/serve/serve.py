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
    args = parser.parse_known_args()[0]
    return args


def train(bucket_name):

    train_file = bucket_name + '/output/train.csv'
    test_file = bucket_name + '/output/test.csv'
    test_labels = bucket_name + '/output/test_label.csv'

    trainDF = pd.read_csv(train_file)
    trainDF = trainDF.drop(trainDF.columns[0], axis = 1)

    testDF = pd.read_csv(test_file)
    testDF = testDF.drop(testDF.columns[0], axis = 1) 
    testLabelDF  = pd.read_csv(test_labels)
    testLabelDF  = testLabelDF.drop(testLabelDF.columns[0], axis = 1) 

    print(testLabelDF['MPG'])

    checkpoint_path =  bucket_name + '/export/model/'

    model = build_model(trainDF)
    print(model.summary())

    model.load_weights(checkpoint_path)

    loss, mae, mse = model.evaluate(testDF, testLabelDF, verbose=2)
    print("loss " )
    print(loss)
    print("mae", mae)



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
   train(args.bucket_name)
