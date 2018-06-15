
# coding: utf-8

# ## This is the code for our proposed neural framework that combines Lexicon-based features, an Emoji detecting cnn's pre-trained activations, fully connected layers, and a CNN/LSTM in 'Parallel' to give state-of-the-art results for emotion intensity detection in Twitter tweets.

# ### The code is run separately for each emotion. This single code can be used to produce results for all the emotions for the full network, as well as our ablation tests results (by removing certain components to see the effect on results).

# In[ ]:

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM,Bidirectional,GRU,SimpleRNN
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPooling1D, AveragePooling1D
from keras.layers import Input, merge, Dropout
from keras.models import Model
import tensorflow as tf
#tf.python.control_flow_ops = tf
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
import timeit


# ### Set the emotion variable below

# In[ ]:

emotion = 'anger'
# emotion = 'fear'
# emotion = 'joy'
# emotion = 'sadness'


# ## Load the pre-trained word2vec based train, dev, and test set tweets representations
# ### Please run the corresponding code in ../Supporting_Codes/ to produce these vector representations.
# #### Note that these will be vectors of the form (n, l, d) where n is the number of tweets in the set, l is the chosen maximum length (zero padded to have same sequence length = 50 for all samples), and d is the dimensionality of the word embedding (400, since we are using the Twitter word2vec model)

# In[ ]:

x1_train = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/train/'
                   +emotion+'.npy')
x1_dev = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/dev/'
                   +emotion+'.npy')

'''
we combine the train, dev to serve as the training vector. We have already determined the
best performing hyperparamter on the dev set, and just need to see results on test set now.
'''

x1_train = np.concatenate((x1_train,x1_dev),axis=0)

x1_test = np.load('../../intermediate_files/word2vec_based_concatenated_vectors/test/'
                   +emotion+'.npy')

print('x1_train shape:', x1_train.shape)    # (n, 50, 400)
print('x1_test shape:', x1_test.shape)      # (n, 50, 400)


# ### With reference to Figure 1 in our paper, the above is the first of the parallely connected components. It is the concatenated word2vec representation which can be fed to a CNN/LSTM. 
# 
# ### Below, we form the average embedding by simply taking the mean across the words of the sentence (tweet). This can then serve as input to fully connected layers (component 2 in Figure 1)

# In[ ]:

x2_train = np.mean(x1_train, axis=1)
x2_test = np.mean(x1_test, axis=1)

print('x2_train shape:', x2_train.shape)    # (n, 400)
print('x2_test shape:', x2_test.shape) # (n, 400)


# #### We get the gold labels for our train (=train+dev) and test sets. Note that labels here means the annotated emotion intesities

# In[ ]:

y_train = np.concatenate((np.load('../../intermediate_files/gold_label_vectors/train/'
                                 +emotion+'.npy'),
                          np.load('../../intermediate_files/gold_label_vectors/dev/'
                                 +emotion+'.npy')),axis=0)

y_test = np.load('../../intermediate_files/gold_label_vectors/test/'
                                 +emotion+'.npy')
print(y_train.shape)    #(n,)
print(y_test.shape)     #(n,)


# ### Now, for the third of the parallely connected components - we combine the deepmoji based pre-trained cnn activations (2304 dim. vector) and our lexicon based features (43 dim. vector). These can be produced by using the corresponding code in ../Supporting_Codes/. 

# In[ ]:

x3_train = np.concatenate((np.load('../../intermediate_files/deepmoji_vectors/train/'
                                  +emotion+'.npy'),
                           np.load('../../intermediate_files/lexicon_vectors/train/'
                                  +emotion+'.npy')), axis=1)
x3_dev = np.concatenate((np.load('../../intermediate_files/deepmoji_vectors/dev/'
                                  +emotion+'.npy'),
                           np.load('../../intermediate_files/lexicon_vectors/dev/'
                                  +emotion+'.npy')), axis=1)
x3_train = np.concatenate((x3_train,x3_dev),axis=0)

x3_test = np.concatenate((np.load('../../intermediate_files/deepmoji_vectors/test/'
                                  +emotion+'.npy'),
                           np.load('../../intermediate_files/lexicon_vectors/test/'
                                  +emotion+'.npy')), axis=1)
print('x3_train shape:', x3_train.shape)   #(n, 2347)
print('x3_test shape:', x3_test.shape)    #(n, 2347)


# ### Now we have the input vectors for each of our three parallely connected components. We can go ahead and train our neural network (per Figure 1). We can output the predictions on the test set, and check the pearson correlation with the actual intensities (done in the cell after).
# 
# #### Note that we are already using optimized hyperparameter settings. User may tune dropout rate, neurons in dense layers, number of dense layers, etc. on their own, but must do that on validation set (so above code will have training and dev sets and will not make use of test set in that case).

# In[ ]:

y_pred = np.zeros(y_test.shape[0])    #will ultimately contain test set predictions

#The results when neural nets like CNNs are involved can vary by a bit on every run.
#To account for that, we train and predict 7 times, and take mean to get our final predictions
for i in range(7) : 
    #start = timeit.default_timer() ---> To time the time taken
    print('Iteration ',i+1)
    
    # Same architecture is used for all emotions
    
    #Component 1 - CNN
    
    input1 = Input(shape=(50,400,))
    l1 = Conv1D(128,3,activation='relu')(input1)
    #l2 = Conv1D(64,3,activation='relu')(l2)
    l1 = GlobalMaxPooling1D()(l1)
    l1 = Dropout(p=0.2)(l1)
    l1 = Dense(128,activation='relu')(l1)
    #l2 = Dense(50,activation='relu')(l2)
    #l2 = Dense(25,activation='relu')(l2)
    #o2 = Dense(1)(l2)
    
    #Component 2 - Fully connected layers on average embedding
    
    input2 = Input(shape=(400,))
    l2 = Dense(256, init='normal', activation='relu')(input2)
    l2 = Dropout(p=0.2)(l2)
    l2 = Dense(128, init='normal', activation='relu')(l2)
    #l1 = Dense(50, init='normal', activation='relu')(l1)
    #l1 = Dropout(p=0.2)(l1)
    #l1 = Dense(25, init='normal', activation='relu')(l1)
    #o1 = Dense(1, init='normal')(l1)
    
    #Component 3 - ('LE') Lexicon and deepmoji (Emoji) based feature vector

    input3 = Input(shape=(2347,))
    
    
    # Combining the 3 components - 'Parallely Connected' DNN
    
    merged_output = merge([l1, l2, input3], mode='concat', concat_axis=-1)
    
    #sequentially connected fully-connected layers on top (component 4)
    
    merged_output = Dense(128, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    merged_output = Dense(64, activation='relu')(merged_output)
    merged_output = Dense(32, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    
    predictions = Dense(1, activation='sigmoid')(merged_output)

    model = Model(input=[input1, input2, input3], output=predictions)
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])# metrics=[pearson])
    
    model.fit([x1_train, x2_train, x3_train], y_train, 
              nb_epoch=25, batch_size=8, verbose=0) #set the verbose value according to your needs
    
    tmp = model.predict([x1_test, x2_test, x3_test]) #temporary predictions for each iteration
    #print(pearsonr(tmp[:,0], label_test))
    y_pred += tmp[:,0]    #Add to the final prediction vector
    
    print('Iteration ',i+1,' Done')
    
    #stop = timeit.default_timer()
    #print('Time taken to train LE-PC-DNN: ', stop-start)
    '''
    Note that the time durations in our paper were when experiments were run on a CPU with 4 GB
    RAM and 1.7 GHz Intel core i5 processor. Also, the times for each of the emotion will be added,
    since we have to run this architecture separately for each emotion.
    '''
    
y_pred = y_pred/7.0    #Final predictions
print('Training DONE')

'''
To see the number of trainable parameters in LE-PC-DNN architecture, print the model
summary as in the comment below (just uncomment it). Add the parameters for each of the 
individual architecture (for all 4 emotions).
'''
#print(model.summary())


# In[ ]:

pearson_correlation_score = pearsonr(y_pred, y_test)[0]

print('Pearson Correlation for full model on Test set for emotion - '+emotion)
print(pearson_correlation_score)


# In[ ]:

#you may save the predictions to do error analysis, etc.

#np.save('../just_preds/full_model_pred'+emotion+'.npy',ypred)


# ### Ablation Test - Removing FC layer component (so we have components 1+3+4 as per Figure 1)

# In[ ]:

y_pred = np.zeros(y_test.shape[0])    #will ultimately contain test set predictions

#The results when neural nets like CNNs are involved can vary by a bit on every run.
#To account for that, we train and predict 7 times, and take mean to get our final predictions
for i in range(7) : 
    print('Iteration ',i+1)
    
    # Same architecture is used for all emotions
    
    #Component 1 - CNN
    
    input1 = Input(shape=(50,400,))
    l1 = Conv1D(128,3,activation='relu')(input1)
    #l2 = Conv1D(64,3,activation='relu')(l2)
    l1 = GlobalMaxPooling1D()(l1)
    l1 = Dropout(p=0.2)(l1)
    l1 = Dense(128,activation='relu')(l1)
    #l2 = Dense(50,activation='relu')(l2)
    #l2 = Dense(25,activation='relu')(l2)
    #o2 = Dense(1)(l2)
    
    #Component 2 - Fully connected layers on average embedding
    
#     input2 = Input(shape=(400,))
#     l2 = Dense(256, init='normal', activation='relu')(input2)
#     l2 = Dropout(p=0.2)(l2)
#     l2 = Dense(128, init='normal', activation='relu')(l2)
    
    #Component 3 - ('LE') Lexicon and deepmoji (Emoji) based feature vector

    input3 = Input(shape=(2347,))
    
    
    # Combining the components 1 and 3
    
    merged_output = merge([l1, input3], mode='concat', concat_axis=-1)
    
    #sequentially connected fully-connected layers on top (component 4)
    
    merged_output = Dense(128, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    merged_output = Dense(64, activation='relu')(merged_output)
    merged_output = Dense(32, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    
    predictions = Dense(1, activation='sigmoid')(merged_output)

    model = Model(input=[input1, input3], output=predictions)
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])# metrics=[pearson])
    
    model.fit([x1_train, x3_train], y_train, 
              nb_epoch=25, batch_size=8, verbose=0) #set the verbose value according to your needs
    
    tmp = model.predict([x1_test, x3_test]) #temporary predictions for each iteration
    #print(pearsonr(tmp[:,0], label_test))
    y_pred += tmp[:,0]    #Add to the final prediction vector
    
    print('Iteration ',i+1,' Done')
    
y_pred = y_pred/7.0    #Final predictions
print('Training DONE')


# In[ ]:

pearson_correlation_score = pearsonr(y_pred, y_test)

print('Pearson Correlation for partial model (components 1+3+4) on Test set for emotion - '+emotion)
print(pearson_correlation_score)


# ### Ablation Test - Removing CNN component (so we have components 2+3+4 as per Figure 1)

# In[ ]:

y_pred = np.zeros(y_test.shape[0])    #will ultimately contain test set predictions

#The results when neural nets like CNNs are involved can vary by a bit on every run.
#To account for that, we train and predict 7 times, and take mean to get our final predictions
for i in range(7) : 
    print('Iteration ',i+1)
    
    # Same architecture is used for all emotions
    
    #Component 1 - CNN
    
#     input1 = Input(shape=(50,400,))
#     l1 = Conv1D(128,3,activation='relu')(input1)
#     #l2 = Conv1D(64,3,activation='relu')(l2)
#     l1 = GlobalMaxPooling1D()(l1)
#     l1 = Dropout(p=0.2)(l1)
#     l1 = Dense(128,activation='relu')(l1)
    
    #Component 2 - Fully connected layers on average embedding
    
    input2 = Input(shape=(400,))
    l2 = Dense(256, init='normal', activation='relu')(input2)
    l2 = Dropout(p=0.2)(l2)
    l2 = Dense(128, init='normal', activation='relu')(l2)
    #l1 = Dense(50, init='normal', activation='relu')(l1)
    #l1 = Dropout(p=0.2)(l1)
    #l1 = Dense(25, init='normal', activation='relu')(l1)
    #o1 = Dense(1, init='normal')(l1)
    
    #Component 3 - ('LE') Lexicon and deepmoji (Emoji) based feature vector

    input3 = Input(shape=(2347,))
    
    
    # Combining the components 2 and 3
    
    merged_output = merge([l2, input3], mode='concat', concat_axis=-1)
    
    #sequentially connected fully-connected layers on top (component 4)
    
    merged_output = Dense(128, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    merged_output = Dense(64, activation='relu')(merged_output)
    merged_output = Dense(32, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    
    predictions = Dense(1, activation='sigmoid')(merged_output)

    model = Model(input=[input2, input3], output=predictions)
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])# metrics=[pearson])
    
    model.fit([x2_train, x3_train], y_train, 
              nb_epoch=25, batch_size=8, verbose=0) #set the verbose value according to your needs
    
    tmp = model.predict([x2_test, x3_test]) #temporary predictions for each iteration
    #print(pearsonr(tmp[:,0], label_test))
    y_pred += tmp[:,0]    #Add to the final prediction vector
    
    print('Iteration ',i+1,' Done')
    
y_pred = y_pred/7.0    #Final predictions
print('Training DONE')


# In[ ]:

pearson_correlation_score = pearsonr(y_pred, y_test)

print('Pearson Correlation for partial model (components 2+3+4) on Test set for emotion - '+emotion)
print(pearson_correlation_score)


# ### Ablation Test - Removing Lexicon+DeepMoji component (so we have components 1+2+4 as per Figure 1)

# In[ ]:

y_pred = np.zeros(y_test.shape[0])    #will ultimately contain test set predictions

#The results when neural nets like CNNs are involved can vary by a bit on every run.
#To account for that, we train and predict 7 times, and take mean to get our final predictions
for i in range(7) : 
    print('Iteration ',i+1)
    
    # Same architecture is used for all emotions
    
    #Component 1 - CNN
    
    input1 = Input(shape=(50,400,))
    l1 = Conv1D(128,3,activation='relu')(input1)
    #l2 = Conv1D(64,3,activation='relu')(l2)
    l1 = GlobalMaxPooling1D()(l1)
    l1 = Dropout(p=0.2)(l1)
    l1 = Dense(128,activation='relu')(l1)
    #l2 = Dense(50,activation='relu')(l2)
    #l2 = Dense(25,activation='relu')(l2)
    #o2 = Dense(1)(l2)
    
    #Component 2 - Fully connected layers on average embedding
    
    input2 = Input(shape=(400,))
    l2 = Dense(256, init='normal', activation='relu')(input2)
    l2 = Dropout(p=0.2)(l2)
    l2 = Dense(128, init='normal', activation='relu')(l2)
    #l1 = Dense(50, init='normal', activation='relu')(l1)
    #l1 = Dropout(p=0.2)(l1)
    #l1 = Dense(25, init='normal', activation='relu')(l1)
    #o1 = Dense(1, init='normal')(l1)
    
    #Component 3 - ('LE') Lexicon and deepmoji (Emoji) based feature vector

    #input3 = Input(shape=(2347,))
    
    
    # Combining the components 1 and 2
    
    merged_output = merge([l1, l2], mode='concat', concat_axis=-1)
    
    #sequentially connected fully-connected layers on top (component 4)
    
    merged_output = Dense(128, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    merged_output = Dense(64, activation='relu')(merged_output)
    merged_output = Dense(32, activation='relu')(merged_output)
    merged_output = Dropout(p=0.2)(merged_output)
    
    predictions = Dense(1, activation='sigmoid')(merged_output)

    model = Model(input=[input1, input2], output=predictions)
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])# metrics=[pearson])
    
    model.fit([x1_train, x2_train], y_train, 
              nb_epoch=25, batch_size=8, verbose=0) #set the verbose value according to your needs
    
    tmp = model.predict([x1_test, x2_test]) #temporary predictions for each iteration
    #print(pearsonr(tmp[:,0], label_test))
    y_pred += tmp[:,0]    #Add to the final prediction vector
    
    print('Iteration ',i+1,' Done')
    
y_pred = y_pred/7.0    #Final predictions
print('Training DONE')


# In[ ]:

pearson_correlation_score = pearsonr(y_pred, y_test)

print('Pearson Correlation for partial model (components 1+2+4) on Test set for emotion - '+emotion)
print(pearson_correlation_score)


# ### Further Ablation tests may be performed easily by basic manipulation of above code.
