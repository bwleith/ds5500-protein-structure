import numpy as np                                     # linear algebra
import pandas as pd                                    # data processing, CSV file I/O (e.g. pd.read_csv)
import copy                                            #to copy list
from sklearn.model_selection import train_test_split   #to split dataset into train and test set
from sklearn.svm import SVC                            #to create svc instance
from sklearn.metrics import classification_report      #to create report for precision,recall,f1-score,accuracy
from sklearn import metrics                            #to get accuracy
from sklearn.model_selection import GridSearchCV       #to optimise the hyper-parameter

df = pd.read_csv('/home/sharma.kin/capstone/2022-08-06-pdb-intersect-pisces_pc25_r2.0.csv')
print(df.head())

maxlen_seq = 128
input_seqs, target_seqs = df[['seq', 'sst8']][(~df.has_nonstd_aa)].values.T
#input_grams = seq2ngrams(input_seqs)
print(input_seqs[0:5])

print(target_seqs[0:5])
print(target_seqs.size)

for row in range(len(target_seqs)):
    secondary_lenth = len(target_seqs[row])
    primary_lenth = len(input_seqs[row])
    
    if(secondary_lenth != primary_lenth):
        print("(",row,") Secondary_Structure ->", target_seqs[row]," Primary_Structure -> ",input_seqs[row])
        
secondary_count = 0
primary_count = 0
for row in range(len(target_seqs)):
    secondary_lenth = len(target_seqs[row])
    primary_lenth = len(input_seqs[row])
    secondary_count = secondary_count + secondary_lenth
    primary_count = primary_count + primary_lenth
    if(secondary_lenth != primary_lenth):
        print("(",row,") Secondary_Structure ->", target_seqs[row]," Primary_Structure -> ",input_seqs[row])
        
print("count of secondary structure : ",secondary_count)
print("count of primary structure : ",primary_count)

def split(sequence): 
    return [char for char in sequence]

primary_split = []
secondary_split = []
for row in range(int(len(target_seqs)/40)):
    primary_split.append(split(input_seqs[row]))
    secondary_split.append(split(target_seqs[row]))
    
def orthogonal_primary(arg):
    switch = {
        'A' : np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),  # 20 amino acids
        'C' : np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'E' : np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'D' : np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'G' : np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'F' : np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'I' : np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        'H' : np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
        'K' : np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
        'M' : np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
        'L' : np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
        'N' : np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        'Q' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
        'P' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
        'S' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
        'R' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        'T' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
        'W' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
        'V' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
        'Y' : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    }
    
    return switch.get(arg)

def orthogonal_secondary(arg):
    switch = {
        'H' : 0,                    # H= α-helix
        'C' : 1,                    # C= Loops and irregular elements
        'E' : 2,                    # E= β-strand
        'B' : 3,                    # B= β-bridge
        'G' : 4,                    # G= 3-helix
        'I' : 5,                    # I= π-helix
        'T' : 6,                    # T= Turn
        'S' : 7                     # S= Bend
    }
    
    return switch.get(arg)

for row in range(len(primary_split)):  
    sequence = primary_split[row]
    for col in range(len(sequence)):
        #print(sequence[col])
        sequence[col] = orthogonal_primary(sequence[col])
        
for row in range(len(secondary_split)):  
    sequenceS = secondary_split[row]
    for col in range(len(sequenceS)):
        sequenceS[col] = orthogonal_secondary(sequenceS[col])
        
def graph_sum2(seq1,seq2):
    result = [None]*len(seq1)
    for col in range(len(seq1)):
        result[col] =  seq1[col]+seq2[col]
    return result


def graph_sum3(seq1,seq2,seq3):
    result = [None]*len(seq1)
    for col in range(len(seq1)):
        result[col] =  seq1[col]+seq2[col]+seq3[col]
    return result

graph_input = copy.deepcopy(primary_split)
for row in range(len(primary_split)):
    sequence = primary_split[row]
    graph_input[row][0]=graph_sum2(sequence[0],sequence[1])
    graph_input[row][len(sequence)-1]=graph_sum2(sequence[len(sequence)-1],sequence[len(sequence)-2])
    for col in range(1,len(sequence)-1):
        graph_input[row][col] = graph_sum3(sequence[col-1],sequence[col],sequence[col+1])
        
graph_input[0:5]

def targetY(data_list):
    Y = []
    for i in range(len(data_list)):
        for j  in range(len(data_list[i])):
            Y.append(data_list[i][j])
    return Y

y_label = targetY(secondary_split)

print(len(y_label))
print(y_label[0:5])

def window_padding_data(size, sequence):
    num = int(size/2)
    #print("initial :",sequence[0])
    #print("")
    zeros = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(len(sequence)):
        for j in range(num):
            sequence[i].append(zeros)
            sequence[i].insert(0, zeros)
            #print(sequence[i])
            #print("")
            
    X = []
    temp = []

    for k in range(len(sequence)):
        #print(sequence[k])
        for l in range(len(sequence[k])-(size-1)):
            temp = sequence[k][l:l+size]
           # print(temp)
            X.append(temp)
            temp = []

    return X

X = window_padding_data(11,graph_input)
len(X)
X[0:5]

np.set_printoptions(threshold=np.inf)
X = np.array(X)
y_label = np.array(y_label)
X = X.reshape(len(X),11*20)
print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size = 0.20,random_state=54)

# for i in range(1,101):
#    X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size = 0.20,random_state=i)
svc = SVC(kernel='rbf', gamma = 0.1, C=1.5)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_true = y_test
#    print("i = ",i,"acc = ",metrics.accuracy_score(y_test, y_pred))
print("Accuracy = ",metrics.accuracy_score(y_test, y_pred)*100)
print(classification_report(y_true,y_pred))

EPSILON = 1e-10
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Relative Absolute Error(RAE):', np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test - np.mean(y_test))) + EPSILON))
print('Root Relative Squared Error(RRSE):', np.sqrt(np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))))

def evaluate_model(cv):
    scores = cross_val_score(svc, X, y_label, scoring='accuracy', cv=cv, n_jobs=-1)
    return np.mean(scores), scores.min(), scores.max()

inputvalue = ['SLLKKLLLA']
actualoutput = ['CHHHHHHHC']

input_split = []
output_split = []
input_split.append(split(inputvalue[0]))
output_split.append(split(actualoutput[0]))

chain = input_split[0]
for col in range(len(chain)):
    chain[col] = orthogonal_primary(chain[col])
    

chainS = output_split[0]
for col in range(len(chainS)):
    chainS[col] = orthogonal_secondary(chainS[col])
    
g_input = copy.deepcopy(input_split)
for row in range(len(input_split)):
    chain = input_split[row]
    g_input[row][0]=graph_sum2(chain[0],chain[1])
    g_input[row][len(chain)-1]=graph_sum2(chain[len(chain)-1],chain[len(chain)-2])
    for col in range(1,len(chain)-1):
        g_input[row][col] = graph_sum3(chain[col-1],chain[col],chain[col+1])
        
actualY = targetY(output_split)

input_X = window_padding_data(11,g_input)
input_X = np.array(input_X)
actualY = np.array(actualY)
input_X = input_X.reshape(len(input_X),11*20)

pred_output = svc.predict(input_X)
print("Actual     :",actualY)
print("Prediction :",pred_output)