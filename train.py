from __future__ import division, absolute_import
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn import lstm, embedding
from tflearn.layers.estimator import regression
import csv
from os.path import isfile
from sets import Set
import sys

min_x = 0
max_x = 1
min_y = 0
max_y = 1

divisions = 5

def load_dataset(x_count, y_count):
  print '[+] Loading data'
  X = []
  Y = []
  places = Set()
  data = np.load('grid/data-{0}-{1}.npy'.format(x_count, y_count))
  for row in data:
    x = map(float, row[1:5])
    time = row[4]
    x.extend([
      (time // 60) % 24 + 1, # Hour
      (time // 1440) % 7 + 1, # Day
      (time // 43200) % 12 + 1, # Month
      (time // 525600) + 1 # Year
    ])
    X.append(x)
    Y.append(row[5])
    places.add(row[5])
  places = list(places)
  Y = [places.index(y) for y in Y]
  Y = to_categorical(Y, len(places))
  print '[+] All data loaded'
  return X, Y

def create_model(input_len, output_len):
  print '[+] Creating NN'
  with tf.Graph().as_default():
    network = input_data(shape = [None, input_len])
    #network = lstm(network, 120, dropout = 0.8)
    #network = dropout(network, 0.5)
    network = fully_connected(network, 400, activation = 'tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 400, activation = 'tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 400, activation = 'tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, output_len, activation = 'softmax')
    network = regression(network,
      optimizer = 'momentum',
      loss = 'categorical_crossentropy')
    model = tflearn.DNN(
      network,
      checkpoint_path = 'save/fb_predicction-{0}-{1}'.format(x_count, y_count),
      max_checkpoints = 1,
      tensorboard_verbose = 3
    )
    print '[+] NN Created'
  return model

for x_count in np.arange(0, divisions, 0.5):
  for y_count in np.arange(0, divisions, 0.5):
    X, Y = load_dataset(x_count, y_count)
    model = create_model(len(X[0]), len(Y[0]))
    print '[+] Training ...'
    model.fit(
      X, Y,
      validation_set = 0.15,
      n_epoch = 50,
      batch_size = 100,
      shuffle = True,
      show_metric = True,
      snapshot_step = 200,
      snapshot_epoch = True
    )
    model.save('save/model-{0}-{1}.tflearn'.format(x_count, y_count))