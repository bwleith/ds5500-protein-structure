import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
random.seed(42)

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical



def dataload(path, kmer):

    # load data

    # # updated data with 25% identity and 2.0 Angstrom cutoffs 
    # ss_2022_25_20 = pd.read_csv('2022-08-06-pdb-intersect-pisces_pc25_r2.0.csv')

    # # updated data with 25% identity and 2.5 Angstrom cutoffs 
    # ss_2022_25_25 = pd.read_csv('2022-08-06-pdb-intersect-pisces_pc25_r2.5.csv')

    # updated data with 30% identity and 2.5 Angstrom cutoffs 
    ss = pd.read_csv(path)

    ss = ss.query(f'len_x >= {100} & len_x <= {300}')

    # start by making an 80/20 split to obtain the training data
    X_train, X_test, y_train, y_test = train_test_split(ss['seq'], ss[['sst3','sst8']], test_size = 0.2, random_state = 42)
    # then do another 50/50 split on the "test" data obtained in the 
    # previous line to get the 10% test sample and the 10% validation
    # sample
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=383)

    print('There are %s records in X_train' %len(X_train))
    print('There are %s records in X_valid' %len(X_valid))
    print('There are %s records in X_test' %len(X_test))


    # convert each sequence to an array of k-mers (i.e. n-grams)
    # using k-mers allows us to include more information about the context 
    # in which each amino acid appears within the primary protein structure
    def seq2kmers(seqs, n = kmer):
        return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs], dtype = object)

    train_kmers = seq2kmers(X_train, n = kmer)
    valid_kmers = seq2kmers(X_valid, n = kmer)
    test_kmers  = seq2kmers(X_test, n=kmer)

    # find the maximum sequence length across all three folds
    # of the dataset (we need to use this to pad the sequences
    # later on)
    maxlen = 0
    for kmers in [train_kmers, valid_kmers, test_kmers]:
        for seq in train_kmers:
            if len(seq) > maxlen:
                maxlen = len(seq)




    # convert the k-mers to encoded sequences
    encoder = Tokenizer()
    encoder.fit_on_texts(train_kmers)
    train_sequences = encoder.texts_to_sequences(train_kmers)
    train_sequences = sequence.pad_sequences(train_sequences, maxlen = maxlen, padding = 'post')

    test_sequences = encoder.texts_to_sequences(test_kmers)
    test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen, padding='post')

    valid_sequences = encoder.texts_to_sequences(valid_kmers)
    valid_sequences = sequence.pad_sequences(valid_sequences, maxlen=maxlen, padding='post')

    # encode the target sequences 
    y = {}

    # convert training, validation, and test SSTs for both SST3 and SST8 into
    # numeric sequences that can be used by the model.
    for target in ['sst3', 'sst8']:
        y[target] = {}
        decoder = Tokenizer(char_level = True)
        decoder.fit_on_texts(y_train[target])

        # convert texts to sequences 
        y_train_sequences = decoder.texts_to_sequences(y_train[target])
        # pad the sequences so that they are all the same length (we
        # use the length of the longest sequence to avoid truncation)
        y_train_sequences = sequence.pad_sequences(y_train_sequences, maxlen = maxlen, padding = 'post')
        # convert the sequences to categorical tensors
        y_train_sequences = to_categorical(y_train_sequences)

        # repeat this process on the test labels 
        y_test_sequences = decoder.texts_to_sequences(y_test[target])
        y_test_sequences = sequence.pad_sequences(y_test_sequences, maxlen = maxlen, padding = 'post')
        y_test_sequences = to_categorical(y_test_sequences)

        # repeat the process on the validation labels
        y_valid_sequences = decoder.texts_to_sequences(y_valid[target])
        y_valid_sequences = sequence.pad_sequences(y_valid_sequences, maxlen = maxlen, padding = 'post')
        y_valid_sequences = to_categorical(y_valid_sequences)

        # store the sequences and model metadata in a dict
        y[target]['train_sequences'] = y_train_sequences
        y[target]['valid_sequences'] = y_valid_sequences 
        y[target]['test_sequences'] = y_test_sequences 
        y[target]['n_words'] = len(encoder.word_index) + 1 
        y[target]['n_ssts'] = len(decoder.word_index) + 1
        
    return train_sequences, valid_sequences, test_sequences, y, maxlen
