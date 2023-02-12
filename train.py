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

from protein_data import ProteinData

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
parser.add_argument("--maxlen", help="maximum sequence length", type=int, default=300)
args = parser.parse_args()

path = args.path
kmer = args.kmer
sst = args.sst
model_config = args.model_config
epochs = args.epochs
batch_size = args.batch_size
maxlen = args.maxlen

df = pd.read_csv(path)
df = df.query(f'len_x >= {100} & len_x <= {300}')

data = ProteinData(df = df, target = sst, n = kmer, maxlen = maxlen)

# Build the model for predicting the SSTsequence
def get_model(model_config):
    # split the input string into a list
    input_list = model_config.split("_")
    
    # initialize an empty sequential model
    model = Sequential()
    
    # add an Embedding layer to the model
    model.add(Embedding(input_dim = data.n_words, output_dim = 128, input_length = maxlen))
    
    # initialize a counter for looping through the input_list
    i = 0
    
    # loop through the input_list and add Bidirectional layers to the model
    while i < len(input_list):
        if input_list[i] == "GRU":
            # add a Bidirectional GRU layer to the model
            model.add(Bidirectional(GRU(units = int(input_list[i + 1]), return_sequences = True)))
        elif input_list[i] == "LSTM":
            # add a Bidirectional LSTM layer to the model
            model.add(Bidirectional(LSTM(units = int(input_list[i + 1]), return_sequences = True)))
        elif input_list[i] == "Dense":
            # add a TimeDistributed Dense layer to the model 
            model.add(TimeDistributed(Dense(units=int(input_list[i + 1]), activation='relu')))
        elif input_list[i] == "Dropout":
            # add a TimeDistributed Dropout layer to the model
            model.add(TimeDistributed(Dropout(float(input_list[i + 1])/100)))
        i += 2
    
    # add a TimeDistributed dense layer with a softmax activation to the model
    model.add(TimeDistributed(Dense(data.n_ssts, activation = 'softmax')))
    
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

vloss = []
vacc = []
vq3 = []
nume = []

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy", q3_acc])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=newpath, histogram_freq=1)


class valCallback(tf.keras.callbacks.Callback):
    def __init__(self, Model, batch_size):
        self.Model = Model
        self.batch_size = batch_size
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % 1 == 0:
            # print('\nModel: ', self.Model.name)
            val_loss, val_accuracy, val_q3_acc = self.Model.evaluate(data.valid_sequences, data.y_valid_sequences, steps=data.valid_sequences.shape[0]//batch_size, verbose=0)
            print('\nval_loss: ', val_loss)
            print('val_accuracy: ', val_accuracy)
            print('val_q3_acc: ', val_q3_acc)
            nume.append(epoch+1)
            vloss.append(val_loss)
            vacc.append(val_accuracy)
            vq3.append(val_q3_acc)
            print('\n')
            plt.figure()
            plt.plot(nume, vloss, 'r', label='val_loss')
            plt.title('val loss for ' + model_config)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.grid()
            plt.savefig(newpath + '/loss.png')
            plt.close()
            plt.plot(nume, vacc, 'b', label='val_accuracy')
            plt.plot(nume, vq3, 'g', label='val_q3_acc')
            plt.title('model accuracy for ' + model_config)
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.grid()
            plt.legend(['val acc', 'val q3 acc'])
            plt.savefig(newpath + '/val_acc.png')

validation_callback = valCallback(Model=model, batch_size=batch_size)

save_name = str(newpath) + '/cp.ckpt'
save = tf.keras.callbacks.ModelCheckpoint(
    save_name,
    verbose=1,
    save_weights_only=True,
    save_freq= (data.train_sequences.shape[0] // batch_size) * 10
)



# training the model
hist = model.fit(data.train_sequences, 
          data.y_train_sequences, 
          batch_size = batch_size, 
          epochs = epochs, 
        #   validation_data = (data.valid_sequences, 
        #                      data.y_valid_sequences), 
          callbacks = [validation_callback, save],
          verbose = 1)

# get test set predictions
test_preds = model.predict(data.test_sequences)

# compute overall Q3 Accuracy
q3 = 0
total_count = 0
for i, pred in enumerate(test_preds):
    acc = q3_acc(data.y_test_sequences[i], pred)
    q3 += np.sum(acc)
    total_count += len(acc)
    
print('Test Set Q3 Accuracy: ', np.round(q3 / total_count, 2))

print('This case was ', model_config)

# Plot the training loss
plt.close()
plt.plot(hist.history['loss'])
plt.title('training loss for ' + model_config)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(newpath + '/trainloss.png')