import numpy as np
import pandas as pd 
from typing import Optional
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

class ProteinData:
    '''
        This class is designed to produce and store all of the necessary data 
        for training a sequence model to predict secondary protein structure 
        based on sequences of characters representing primary protein structure.

        The __init__ method takes a dataframe containing primary protein sequences
        in a column called seq and the name of the target column (should be 'sst3'
        or 'sst8' and produces padded sequences up to a given max length that can
        be used to train the model. The list of class attributes is below)
    '''
    # sequences representing primary protein structure
    train_sequences: np.ndarray
    valid_sequences: np.ndarray
    test_sequences: np.ndarray 
    # sequences representing secondary protein structure
    y_train_sequences = np.ndarray
    y_valid_sequences = np.ndarray
    y_test_sequences = np.ndarray
    # useful metadata for setting up the model
    n_words: int 
    n_ssts: int 
    target: str
    maxlen: int
    kmer: int

    # converts a sequence representing primary protein structure into k-mers (aka n-grams)
    # this can be a useful preprocessing step for improving the predictive capability of 
    # the model
    def _seqtokmers(self, 
                    seqs: pd.core.series.Series, 
                    n: int): 

        return np.array([[seq[i:i+n] for i in range(len(seq))] for seq in seqs], dtype = object)

    # splits the data into an 80/10/10 split
    def _split_data(self, 
                    df: pd.DataFrame, 
                    target: str,
                    random_state: Optional[int] = 42,
                    n: Optional[int] = 3, 
                    maxlen: Optional[int] = None):

        # filter the data based on maxlen if one is provided
        if maxlen: 
            df = df[df['seq'].apply(lambda x: len(x) <= maxlen)]
        # split the data 80/20 to get the training set (the 80% subsample)
        X_train, X_test, y_train, y_test = train_test_split(df['seq'], df[[target]], test_size = 0.2, random_state = random_state)
        # split the 20% subsample to get the 10% test sample and the 10% validation sample
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=383)
        return {'X_train': X_train, 
                'X_valid': X_valid, 
                'X_test': X_test, 
                'y_train': y_train, 
                'y_valid': y_valid, 
                'y_test': y_test}



    def __init__(self, 
                 df: pd.DataFrame, 
                 target: str,
                 n: int,
                 random_state: Optional[int] = 42, 
                 maxlen: Optional[int] = None):
        
        # split the data 80/10/10 train/validation/test
        split = self._split_data(df = df, target = target) 
        
        # convert the primary protein sequences into k-mers (i.e. n-grams)
        train_kmers = self._seqtokmers(seqs = split['X_train'], n=n)
        valid_kmers = self._seqtokmers(seqs = split['X_valid'], n=n)
        test_kmers  = self._seqtokmers(seqs = split['X_test'],  n=n)

        # find the longest sequence length across all of the 
        # data if we are not filtering to sequences of a given
        # length
        if not maxlen:
            maxlen = 0
            for kmers in [train_kmers, valid_kmers, test_kmers]:
                for seq in train_kmers:
                    if len(seq) > maxlen:
                        maxlen = len(seq)

        # convert kmers to sequences that can be used in the model
        encoder = Tokenizer()
        encoder.fit_on_texts(train_kmers)
        train_sequences = encoder.texts_to_sequences(train_kmers)
        train_sequences = sequence.pad_sequences(train_sequences, maxlen = maxlen, padding = 'post')

        test_sequences = encoder.texts_to_sequences(test_kmers)
        test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen, padding='post')

        valid_sequences = encoder.texts_to_sequences(valid_kmers)
        valid_sequences = sequence.pad_sequences(valid_sequences, maxlen=maxlen, padding='post')

        # instantiate the ProteinData object
        decoder = Tokenizer(char_level = True)

        decoder.fit_on_texts(split['y_train'][target])

        # convert texts to sequences 
        y_train_sequences = decoder.texts_to_sequences(split['y_train'][target])
        # pad the sequences so that they are all the same length (we
        # use the length of the longest sequence to avoid truncation)
        y_train_sequences = sequence.pad_sequences(y_train_sequences, maxlen = maxlen, padding = 'post')
        # convert the sequences to categorical tensors
        y_train_sequences = to_categorical(y_train_sequences)

        # repeat this process on the test labels 
        y_test_sequences = decoder.texts_to_sequences(split['y_test'][target])
        y_test_sequences = sequence.pad_sequences(y_test_sequences, maxlen = maxlen, padding = 'post')
        y_test_sequences = to_categorical(y_test_sequences)

        # repeat the process on the validation labels
        y_valid_sequences = decoder.texts_to_sequences(split['y_valid'][target])
        y_valid_sequences = sequence.pad_sequences(y_valid_sequences, maxlen = maxlen, padding = 'post')
        y_valid_sequences = to_categorical(y_valid_sequences)

        # store the sequences and model metadata in a dict
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences 
        self.test_sequences = test_sequences 
        self.y_train_sequences = y_train_sequences
        self.y_valid_sequences = y_valid_sequences
        self.y_test_sequences = y_test_sequences
        self.n_words = len(encoder.word_index) + 1 
        self.n_ssts = len(decoder.word_index) + 1
        self.target = target
        self.maxlen = maxlen