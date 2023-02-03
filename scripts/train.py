import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
random.seed(42)

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Embedding, Dense, TimeDistributed, Bidirectional, LSTM, Attention, Dropout, Flatten, GlobalMaxPool1D, Reshape

from sklearn.model_selection import train_test_split
from keras import backend as K


import logging
import time
import os
import sys

from matplotlib import pyplot as plt

from preprocess import dataload

timestr = time.strftime("%Y%m%d_%H%M%S")

newpath = timestr 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# making an argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", help = "path to the data", type = str, default = '2022-08-06-pdb-intersect-pisces_pc25_r2.5.csv')
parser.add_argument("--kmer", help = "k-mer size", type = int, default = 3)
parser.add_argument("--sst", help = "sst type", type = str, default = 'sst3')
parser.add_argument("--model_config", help = "model configuration", type = str, default = 'GRU_64_GRU_32')
parser.add_argument("--epochs", help = "number of epochs", type = int, default = 10)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 128)
args = parser.parse_args()

path = args.path
kmer = args.kmer
sst = args.sst
model_config = args.model_config
epochs = args.epochs
batch_size = args.batch_size




train_sequences, valid_sequences, test_sequences, y, maxlen = dataload(path, kmer)

# Build the model for predicting the SSTsequence


def get_model(model_config):
    # split the input string into a list
    input_list = model_config.split("_")
    
    # initialize an empty sequential model
    model = Sequential()
    
    # add an Embedding layer to the model
    model.add(Embedding(input_dim = y[sst]['n_words'], output_dim = 128, input_length = maxlen))
    
    # initialize a counter for looping through the input_list
    i = 0
    
    # loop through the input_list and add Bidirectional layers to the model
    while i < len(input_list):
        if input_list[i] == "GRU":
            # add a Bidirectional GRU layer to the model
            model.add(Bidirectional(GRU(units = int(input_list[i + 1]), return_sequences = True, recurrent_dropout = 0)))
            i += 2
        elif input_list[i] == "LSTM":
            # add a Bidirectional LSTM layer to the model
            model.add(Bidirectional(LSTM(units = int(input_list[i + 1]), return_sequences = True, recurrent_dropout = 0)))
            i += 2
    
    # add a TimeDistributed dense layer with a softmax activation to the model
    model.add(TimeDistributed(Dense(y[sst]['n_ssts'], activation = 'softmax')))
    
    # return the constructed model
    return model


model = get_model(model_config)

# For manual override of model
# model = Sequential([
#     Embedding(input_dim = y[sst]['n_words'], output_dim = 128, input_length = maxlen),
#     Bidirectional(GRU(units = 64, return_sequences = True, recurrent_dropout = 0)),
#     Bidirectional(GRU(units = 32, return_sequences = True, recurrent_dropout = 0)),
#     TimeDistributed(Dense(y[sst]['n_ssts'], activation = 'softmax'))])


model.summary()

# Train the model


# Q3 Accuracy Implementation from https://www.kaggle.com/code/helmehelmuto/secondary-structure-prediction-with-keras/notebook
# "SS prediction is usually evaluated by Q3 or Q8 accuracy, which measures the percent of residues for which 3-state or 8-state 
# secondary structure is correctly predicted"  (doi: 10.1038/srep18962)
def q3_acc(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy", q3_acc])

# training the model
hist = model.fit(train_sequences, 
          y[sst]['train_sequences'], 
          batch_size = batch_size, 
          epochs = epochs, 
          validation_data = (valid_sequences, 
                             y[sst]['valid_sequences']), 
          verbose = 1)


# get test set predictions
test_preds = model.predict(test_sequences)

# compute overall Q3 Accuracy
q3 = 0
total_count = 0
for i, pred in enumerate(test_preds):
    acc = q3_acc(y[sst]['test_sequences'][i], pred)
    q3 += np.sum(acc)
    total_count += len(acc)
    
print('Test Set Q3 Accuracy: ', np.round(q3 / total_count, 2))

print('This case was ', model_config)

# Plot the validation accuracy and training loss
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['val_q3_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val acc', 'val q3 acc'])
plt.savefig(newpath + '/val_acc.png')
plt.close()
plt.plot(hist.history['loss'])
plt.title('training loss for ' + model_config)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(newpath + '/loss.png')