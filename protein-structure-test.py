import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Embedding, Dense, TimeDistributed, Bidirectional

from sklearn.model_selection import train_test_split
from keras import backend as K

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

import random
random.seed(42)

# globals

# k-mer word size
kmer_size = 3

# protein/sst3 sequence size limts
minlen = 100
maxlen = 300

# Load the various datasets

# updated data with 25% identity and 2.0 Angstrom cutoffs 
ss_2022_25_20 = pd.read_csv('./2022-08-06-pdb-intersect-pisces_pc25_r2.0.csv')

# updated data with 25% identity and 2.5 Angstrom cutoffs 
ss_2022_25_25 = pd.read_csv('./2022-08-06-pdb-intersect-pisces_pc25_r2.5.csv')

# updated data with 30% identity and 2.5 Angstrom cutoffs 
ss_2022_30_25 = pd.read_csv('./2022-08-06-pdb-intersect-pisces_pc30_r2.5.csv')

# the full, unfiltered dataset
# duplicates are dropped and the first instance kept 
ss_cleaned = pd.read_csv('./2022-08-03-ss.cleaned.csv') #.drop_duplicates(subset = 'seq', keep = 'first')

# what fraction of sequences are within desired ranges?
protein_df = pd.DataFrame({'Dataset': ['ss_2022_25_20', 'ss_2022_25_25', 'ss2022_30_25', 'ss_cleaned'],
              'Total Sequences': [len(ss_2022_25_20), len(ss_2022_25_25), len(ss_2022_30_25), len(ss_cleaned)],
              f'Sequences {minlen} <= length <= {maxlen}': [len(ss_2022_25_20.query(f'len_x >= {minlen} & len_x <= {maxlen}')),
                                                            len(ss_2022_25_25.query(f'len_x >= {minlen} & len_x <= {maxlen}')),
                                                            len(ss_2022_30_25.query(f'len_x >= {minlen} & len_x <= {maxlen}')),
                                                            len(ss_cleaned.query(f'len >= {minlen} & len <= {maxlen}'))],
              f'Fraction Between {minlen} and {maxlen} aa': [len(ss_2022_25_20.query(f'len_x >= {minlen} & len_x <= {maxlen}')) / len(ss_2022_25_20),
                                                             len(ss_2022_25_25.query(f'len_x >= {minlen} & len_x <= {maxlen}')) / len(ss_2022_25_25),
                                                             len(ss_2022_30_25.query(f'len_x >= {minlen} & len_x <= {maxlen}')) / len(ss_2022_30_25),
                                                             len(ss_cleaned.query(f'len >= {minlen} & len <= {maxlen}')) / len(ss_cleaned)]}).round(3)

# data subsetting
ss_2520 = ss_2022_25_20.query(f'len_x >= {minlen} & len_x <= {maxlen}')
ss_2525 = ss_2022_25_25.query(f'len_x >= {minlen} & len_x <= {maxlen}')
ss_2530 = ss_2022_30_25.query(f'len_x >= {minlen} & len_x <= {maxlen}')

# also get data from ss_cleaned.csv - duplicates were dropped above
ss_all = ss_cleaned.query(f'len >= {minlen} & len <= {maxlen} & not has_nonstd_aa')

# use all sequences from large set to test
test_seqs = ss_2530.seq.tolist() 
test_sst3 = ss_2530.sst3.tolist()

# get all sequences from ss_cleaned that are not in the test set
ss_all_seqs = ss_all.seq.tolist()
ss_all_sst3 = ss_all.sst3.tolist()

train_idx = [i for i, seq in enumerate(ss_all_seqs) if seq not in test_seqs]
train_seqs = [ss_all_seqs[i] for i in train_idx]
train_sst3 = [ss_all_sst3[i] for i in train_idx]

# create a validation set without any internal duplicates
# or duplicates with training set
valid_idx = random.sample(range(len(train_seqs)), int(len(train_seqs) * .25))
valid_seqs = [train_seqs[i] for i in valid_idx]
valid_sst3 = [train_sst3[i] for i in valid_idx]

train_seqs = [seq for i, seq in enumerate(train_seqs) if i not in valid_idx]
train_sst3 = [sst3 for i, sst3 in enumerate(train_sst3) if i not in valid_idx]

valid_seqs_nodups = []
valid_sst3_nodups = []
for seq, sst in zip(valid_seqs, valid_sst3):
    if seq not in valid_seqs_nodups:
        valid_seqs_nodups.append(seq)
        valid_sst3_nodups.append(sst)
        
valid_seqs = valid_seqs_nodups
valid_sst3 = valid_sst3_nodups

print('Training set size: ', len(train_seqs))
print('Validation set size: ', len(valid_seqs))
print('Test set size: ', len(test_seqs))

def seq2kmers(seqs, n = 3):
    return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs], dtype = object)

# generate k-mers from protein sequences
train_kmers = seq2kmers(train_seqs, n = kmer_size)
valid_kmers = seq2kmers(valid_seqs, n = kmer_size)

# encode the protein sequences
encoder = Tokenizer()
encoder.fit_on_texts(train_kmers)
X_train = encoder.texts_to_sequences(train_kmers)
X_train = sequence.pad_sequences(X_train, maxlen = maxlen, padding = 'post')

X_valid = encoder.texts_to_sequences(valid_kmers)
X_valid = sequence.pad_sequences(X_valid, maxlen = maxlen, padding = 'post')

# encode the target secondary structures as categorical
decoder = Tokenizer(char_level = True)
decoder.fit_on_texts(train_sst3)
y_train = decoder.texts_to_sequences(train_sst3)
y_train = sequence.pad_sequences(y_train, maxlen = maxlen, padding = 'post')
y_train = to_categorical(y_train)

y_valid = decoder.texts_to_sequences(valid_sst3)
y_valid = sequence.pad_sequences(y_valid, maxlen = maxlen, padding = 'post')
y_valid = to_categorical(y_valid)

# generate k-mers from protein sequences
train_kmers = seq2kmers(train_seqs, n = kmer_size)
valid_kmers = seq2kmers(valid_seqs, n = kmer_size)

# encode the protein sequences
encoder = Tokenizer()
encoder.fit_on_texts(train_kmers)
X_train = encoder.texts_to_sequences(train_kmers)
X_train = sequence.pad_sequences(X_train, maxlen = maxlen, padding = 'post')

X_valid = encoder.texts_to_sequences(valid_kmers)
X_valid = sequence.pad_sequences(X_valid, maxlen = maxlen, padding = 'post')

# encode the target secondary structures as categorical
decoder = Tokenizer(char_level = True)
decoder.fit_on_texts(train_sst3)
y_train = decoder.texts_to_sequences(train_sst3)
y_train = sequence.pad_sequences(y_train, maxlen = maxlen, padding = 'post')
y_train = to_categorical(y_train)

y_valid = decoder.texts_to_sequences(valid_sst3)
y_valid = sequence.pad_sequences(y_valid, maxlen = maxlen, padding = 'post')
y_valid = to_categorical(y_valid)

n_words = len(encoder.word_index) + 1
n_ssts = len(decoder.word_index) + 1

model = Sequential([
    Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen),
    Bidirectional(GRU(units = 64, return_sequences = True, recurrent_dropout = 0.1)),
    TimeDistributed(Dense(n_ssts, activation = 'softmax'))])

print(model.summary())

def q3_acc(y_true, y_pred):
    y = tf.argmax(y_true, axis=-1)
    y_ = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy", q3_acc])
model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_data = (X_valid, y_valid), verbose = 1)

## encode the test set
test_kmers = seq2kmers(test_seqs, n = kmer_size)
X_test = encoder.texts_to_sequences(test_kmers)
X_test = sequence.pad_sequences(X_test, maxlen = maxlen, padding = 'post')

# encode the target secondary structures as categorical
y_test = decoder.texts_to_sequences(test_sst3)
y_test = sequence.pad_sequences(y_test, maxlen = maxlen, padding = 'post')
y_test = to_categorical(y_test)

# reversal functions from https://www.kaggle.com/code/helmehelmuto/secondary-structure-prediction-with-keras
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

reverse_decoder_index = {value:key for key,value in decoder.word_index.items()}

# get test set predictions
test_preds = model.predict(X_test)

# compute overall Q3 Accuracy
q3 = 0
total_count = 0
for i, pred in enumerate(test_preds):
    acc = q3_acc(y_test[i], pred)
    q3 += np.sum(acc)
    total_count += len(acc)
    
print('Test Set Q3 Accuracy: ', np.round(q3 / total_count, 2))

# show some examples of actual vs predicted
for index in [1, 10, 100, 1000]:
    actual = onehot_to_seq(y_test[index], reverse_decoder_index)
    predicted = onehot_to_seq(test_preds[index], reverse_decoder_index)
    match = ''.join(['*' if a == p else '-' for a, p in zip(actual, predicted) ])
    acc = q3_acc(y_test[index], test_preds[index])
    
print('Example #', index)
print('Q3 Accuracy: ', np.round(np.sum(acc) / len(acc), 2))
print('Actual:\t', actual + '\n' + 
        'Match:\t\t', match + '\n' + 
        'Precited:\t', predicted + '\n' +
        '\t\t ' + ''.join(['|' if (n + 1) % 10 == 0 else '-' for n in range(len(actual))]))