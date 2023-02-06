# Protein Secondary Structure Classification
*Project for Northeastern DS5500: Capstone*

## Approach

Protein Secondary Structures can be predicted using many different approaches, usually using NLP and ML models. Our approach is centered around LSTMs, GNNs and GRUs (Gated Recurrent Unit) networks. Below is an explanation on how each on works as well as how we used them in this specific problem. We use a combination of GRU layers and LSTM layers to improve accuracy.

## LSTM

LSTMs or Long Short Term Memory networks are a type of RNN (recurrent neural networks) that make them ideal for solving problems that involve sequences of data such as videos, speech, and in our case, strands of amino acids. We are using the Tensorflow Keras LSTM which follows Sepp Hochreiter's 1991 paper titled *Long Short-Term Memory*. His approach is to use a gradient based RNN. The architecture involves input and output gates as well as a more complex gate called a memory or forget gate (forget or memory due to the fact that they remember the previous state in order to decide whether or not to forget or remember data from each iteration to the next). These gates help the main body of the LSTM, the cell, which remembers the information, to regulate the information that flows in and out of it. This allows for the layer to utilize both long and short term memory and helps to deal with the vanishing gradient problem by allowing gradients to flow unchanged so that they do not have to reduce with every iteration and therefore vanish. 

## GRU

GRUs or Gated Recurrent Unit Networks are also a type of RNN that make them ideal for these tasks. We are also using the Tensorflow Keras GRU which follows Kyunghyun Cho's 2014 paper titled *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches*. This approach is similar to an LSTM and uses the same mechanisms except it lacks an output gate, which means it uses fewer parameters. It is very similar to an LSTM and according to the paper, there is not any clear model which is better than the other, it depends on the problem.

## GNN

A GNN is a graph neural network that processes graphical data. While secondary protein structures are estimated using amino acid chains which are normally represented as a sequence in a string. In order to enable our ability to convert the chains to graphs, we first split the amino acid sequences and secondary structures into an array of character arrays. We then go through every individual character array and for each character in that array we get the primary sequence as an adjacency graph. An example is that for the character 'A', we return the array [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] and for the character 'C', we return [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]. We also assign codes of sequences for the output secondary structures such as for a $\alpha$-helix we have code 0 and for $\beta$-strands we have code 2. After splitting the graph into train and test splits, we use svc but are planning to switch to a neural network to train and we test using accuracy and calculate the Mean Absolute Error(MAE), Mean Squared Error(MSE), Root Mean Squared Error(RMSE), Relative Absolute Error(RAE), and Root Relative Squared Error(RRSE). 

## How are they used on Protein Secondary Structure problems

Our implementation takes advantage of the fact that GRU and LSTM are very similar and is able to use a combination of both in order to create a complete RNN. We start by creating our data in the form of a ProteinData class found in the protein_data.py file that takes a dataframe, a target string, a random_state (default is 42), the number of engrams n (default is 3), and a maxlen for string cutoffs to make things more manageable (default is None). It breaks the data into an 80-10-10 split of train, test and validation sets along with storing some metadata like n_words and n_ssts which store the word_index numbers for the decoders and encoders. Then in train.py, we have a parser that allows the user to build the structure of the neural network, providing the path to the data, the k-mer size, the sst type, the model configuration, the number of epochs, the batch size, the maximum sequence length and the order of layers including LSTM, GRU, Dense, Dropout layers and at the end, we use softmax dense activation layer.

## How we evaluate

We use a Q3 Accuracy Implementation from [Kaggle](https://www.kaggle.com/code/helmehelmuto/secondary-structure-prediction-with-keras/notebook). It utilizes categorical cross-entropy accuracy which usually measures the percent of residues for which 3-state or 8-state secondary structure is correctly predicted. We then plot the val_accuracy and val_q3_accuracy vs epochs and then save the graphs along with a graph of the training loss.