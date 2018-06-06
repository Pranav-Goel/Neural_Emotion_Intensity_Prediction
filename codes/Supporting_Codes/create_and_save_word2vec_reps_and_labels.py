
# coding: utf-8

# In[ ]:

import pandas as pd
import nltk
import os
import re
from gensim.models import Word2Vec          #gensim version == 1.0.1 -> ensure this
import gensim.models
import numpy as np
import wordsegment         #version - 0.8.0
from keras.preprocessing import sequence
import itertools


# In[ ]:

train_anger = pd.read_csv('../../data/train/anger.tsv',delimiter='\t',header=None)
train_fear = pd.read_csv('../../data/train/fear.tsv',delimiter='\t',header=None)
train_joy = pd.read_csv('../../data/train/joy.tsv',delimiter='\t',header=None)
train_sad = pd.read_csv('../../data/train/sadness.tsv',delimiter='\t',header=None)


# In[ ]:

dev_anger = pd.read_csv('../../data/dev/anger.tsv',delimiter='\t',header=None)
dev_fear = pd.read_csv('../../data/dev/fear.tsv',delimiter='\t',header=None)
dev_joy = pd.read_csv('../../data/dev/joy.tsv',delimiter='\t',header=None)
dev_sad = pd.read_csv('../../data/dev/sadness.tsv',delimiter='\t',header=None)


# In[ ]:

test_anger = pd.read_csv('../../data/test/anger.tsv',delimiter='\t',header=None)
test_fear = pd.read_csv('../../data/test/fear.tsv',delimiter='\t',header=None)
test_joy = pd.read_csv('../../data/test/joy.tsv',delimiter='\t',header=None)
test_sad = pd.read_csv('../../data/test/sadness.tsv',delimiter='\t',header=None)


# In[ ]:

train_anger_tweets = list(train_anger[1])
train_anger_intensities = list(train_anger[3])
train_fear_tweets = list(train_fear[1])
train_fear_intensities = list(train_fear[3])
train_sad_tweets = list(train_sad[1])
train_sad_intensities = list(train_sad[3])
train_joy_tweets = list(train_joy[1])
train_joy_intensities = list(train_joy[3])


# In[ ]:

dev_anger_tweets = list(dev_anger[1])
dev_anger_intensities = list(dev_anger[3])
dev_fear_tweets = list(dev_fear[1])
dev_fear_intensities = list(dev_fear[3])
dev_sad_tweets = list(dev_sad[1])
dev_sad_intensities = list(dev_sad[3])
dev_joy_tweets = list(dev_joy[1])
dev_joy_intensities = list(dev_joy[3])


# In[ ]:

test_anger_tweets = list(test_anger[1])
test_anger_intensities = list(test_anger[3])
test_fear_tweets = list(test_fear[1])
test_fear_intensities = list(test_fear[3])
test_sad_tweets = list(test_sad[1])
test_sad_intensities = list(test_sad[3])
test_joy_tweets = list(test_joy[1])
test_joy_intensities = list(test_joy[3])


# In[ ]:

#load the pre-trained Twitter word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('../../intermediate_files/wordvec_model/word2vec_twitter_model.bin',binary=True,unicode_errors='ignore')


# In[ ]:

'''
we set the length to which each tweet vector will be zero padded to.
#this is based on the maximum length we got on the training set - we do not want to remove
#any words.
'''
maxlen=50


# In[ ]:

#some of the preprocessing steps are inspired from Akhtar et al. - "IITP at EmoInt-2017:
#Measuring Intensity of Emotions using Sentence Embeddings and Optimized Features"

#this function takes a string (the tweet), and gives the processed list of words
def tweet_to_wordlist(s):
    
    #remove characters outside the ascii code 128
    s = ''.join([i if ord(i)<128 else ' ' for i in s])

    #remove any newline characters
    s = s.replace('\n',' ')
    
    #tweets mentions user using '@' followed by username. Replace all those with @user
    s = re.sub('@[^ ]+','@user',s)
    
    #remove URLs
    s = re.sub(r"http\S+", "", s)
    
    #words can have punctuation at their boundaries - have spaces in between punctuation and words
    l = s.split('...')
    for i in range(len(l)):
        l[i] = l[i].replace('/',' / ').replace('\\',' \ ').replace(',',' , ').replace('.',' . ').replace('?',' ? ').replace('!',' ! ').replace("'"," ' ").replace(':',' : ').replace(';',' ; ').replace('-',' - ').replace('(',' ( ').replace(')',' ) ').replace('[',' [ ').replace(']',' ] ').replace('&',' & ').replace('*',' * ').replace('{',' { ').replace('}',' } ').replace('-',' - ').replace('`',' ` ').replace('"',' " ')
    s2 = ' ... '.join(l)
    l2 = s2.split(' ')
    
    #our pre-trained Twitter word2vec model has embeddings of '#' symbols coming together - upto 5 coming together consecutively
    for j in range(len(l2)):
        x=l2[j].count('#')
        y=len(l2[j])
        if(x==y and x>5):
            l2[j]='#####'
        
        #wordsegment will convert the hashtag based joined words, for example, it will segment #iamthebest to ['i','am','the','best']
        if l2[j] not in model:
            l2[j] = ' '.join(wordsegment.segment(l2[j]))
            
    s3 = ' '.join(l2)
     
    #convert words with lots of vowels to their normal form, for example, 'loooooveee' to 'love'    
    l3 = s3.split()        
    for k in range(len(l3)):
        if l3[k] not in model:
            l3[k] = ''.join(''.join(s)[:2] for _, s in itertools.groupby(l3[k]))
            if l3[k] not in model:
                l3[k] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(l3[k]))
    for e in range(len(l3)):
        if l3[e] not in model:
            l3[e] = l3[e].lower()
    
    #'goin' to 'going', etc. - completing the present continuous verbs
    for q in range(len(l3)):
        if l3[q] not in model:
            if len(l3[q])>3:
                if l3[q][-3:]!='ing':
                    if l3[q][-2:]=='ng' or l3[q][-2:]=='in':
                        w2 = l3[q][:-2]+'ing'
                        if nltk.pos_tag([w2])[0][1]=='VBG':
                            l3[q] = w2
    for r in range(len(l3)):
        if l3[r] not in model:
            l3[r] = l3[r].lower()
    
    s4=' '.join(l3)
    
    #remove any extra spaces
    s4=re.sub(' +',' ',s4)
    
    return s4.split()    


# In[ ]:

#concatenate the word2vec embedding of all the words of input list of words
def makeFeatureVecConcat(words):
    featureVec = []

    for word in words:
        if word in model: 
            featureVec.append(model[word])
        else:
            if '#' in word:
                word=word.replace('#','')
                if word in model: 
                    featureVec.append(model[word])

    return featureVec


# ## Finally, we form the concatenated Twitter word2vec based vector representations of the tweets in train, dev and test sets separately for all the emotions
# ### Then we save them to the intermediate_files folder, where they can then be used in the codes for the various neural models

# In[ ]:

train_anger_vecs = []
train_fear_vecs = []
train_joy_vecs = []
train_sadness_vecs = []


# In[ ]:

for s in train_anger_tweets:
    train_anger_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(train_anger_vecs)==len(train_anger_intensities)

for s in train_fear_tweets:
    train_fear_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(train_fear_vecs)==len(train_fear_intensities)

for s in train_sad_tweets:
    train_sadness_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(train_sadness_vecs)==len(train_sad_intensities)

for s in train_joy_tweets:
    train_joy_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(train_joy_vecs)==len(train_joy_intensities)

print('Done forming Training vectors')


# In[ ]:

train_anger_vecs = sequence.pad_sequences(train_anger_vecs, maxlen=maxlen, dtype='float64')
train_fear_vecs = sequence.pad_sequences(train_fear_vecs, maxlen=maxlen, dtype='float64')
train_joy_vecs = sequence.pad_sequences(train_joy_vecs, maxlen=maxlen, dtype='float64')
train_sadness_vecs = sequence.pad_sequences(train_sadness_vecs, maxlen=maxlen, dtype='float64')


# In[ ]:

print(train_anger_vecs.shape)
print(train_fear_vecs.shape)
print(train_joy_vecs.shape)
print(train_sadness_vecs.shape)


# In[ ]:

np.save('../../intermediate_files/word2vec_based_concatenated_vectors/train/anger.npy',
        train_anger_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/train/fear.npy',
        train_fear_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/train/joy.npy',
        train_joy_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/train/sadness.npy',
        train_sadness_vecs)
print('Train vectors saved')


# In[ ]:

dev_anger_vecs = []
dev_fear_vecs = []
dev_joy_vecs = []
dev_sadness_vecs = []

for s in dev_anger_tweets:
    dev_anger_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(dev_anger_vecs)==len(dev_anger_intensities)

for s in dev_fear_tweets:
    dev_fear_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(dev_fear_vecs)==len(dev_fear_intensities)

for s in dev_sad_tweets:
    dev_sadness_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(dev_sadness_vecs)==len(dev_sad_intensities)

for s in dev_joy_tweets:
    dev_joy_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(dev_joy_vecs)==len(dev_joy_intensities)

print('Done forming Dev vectors')

dev_anger_vecs = sequence.pad_sequences(dev_anger_vecs, maxlen=maxlen, dtype='float64')
dev_fear_vecs = sequence.pad_sequences(dev_fear_vecs, maxlen=maxlen, dtype='float64')
dev_joy_vecs = sequence.pad_sequences(dev_joy_vecs, maxlen=maxlen, dtype='float64')
dev_sadness_vecs = sequence.pad_sequences(dev_sadness_vecs, maxlen=maxlen, dtype='float64')

print(dev_anger_vecs.shape)
print(dev_fear_vecs.shape)
print(dev_joy_vecs.shape)
print(dev_sadness_vecs.shape)


# In[ ]:

np.save('../../intermediate_files/word2vec_based_concatenated_vectors/dev/anger.npy',
        dev_anger_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/dev/fear.npy',
        dev_fear_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/dev/joy.npy',
        dev_joy_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/dev/sadness.npy',
        dev_sadness_vecs)
print('Dev vectors saved')


# In[ ]:

test_anger_vecs = []
test_fear_vecs = []
test_joy_vecs = []
test_sadness_vecs = []

for s in test_anger_tweets:
    test_anger_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(test_anger_vecs)==len(test_anger_intensities)

for s in test_fear_tweets:
    test_fear_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(test_fear_vecs)==len(test_fear_intensities)

for s in test_sad_tweets:
    test_sadness_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(test_sadness_vecs)==len(test_sad_intensities)

for s in test_joy_tweets:
    test_joy_vecs.append(makeFeatureVecConcat(tweet_to_wordlist(s)))
assert len(test_joy_vecs)==len(test_joy_intensities)

print('Done forming Test vectors')

test_anger_vecs = sequence.pad_sequences(test_anger_vecs, maxlen=maxlen, dtype='float64')
test_fear_vecs = sequence.pad_sequences(test_fear_vecs, maxlen=maxlen, dtype='float64')
test_joy_vecs = sequence.pad_sequences(test_joy_vecs, maxlen=maxlen, dtype='float64')
test_sadness_vecs = sequence.pad_sequences(test_sadness_vecs, maxlen=maxlen, dtype='float64')

print(test_anger_vecs.shape)
print(test_fear_vecs.shape)
print(test_joy_vecs.shape)
print(test_sadness_vecs.shape)


# In[ ]:

np.save('../../intermediate_files/word2vec_based_concatenated_vectors/test/anger.npy',
        test_anger_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/test/fear.npy',
        test_fear_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/test/joy.npy',
        test_joy_vecs)
np.save('../../intermediate_files/word2vec_based_concatenated_vectors/test/sadness.npy',
        test_sadness_vecs)
print('Test vectors saved')


# ### Finally, we save the gold annotated intensities for each set (labels)

# In[ ]:

np.save('../../intermediate_files/gold_label_vectors/train/anger.npy',np.array(train_anger_intensities))
np.save('../../intermediate_files/gold_label_vectors/train/fear.npy',np.array(train_fear_intensities))
np.save('../../intermediate_files/gold_label_vectors/train/joy.npy',np.array(train_joy_intensities))
np.save('../../intermediate_files/gold_label_vectors/train/sadness.npy',np.array(train_sad_intensities))        


# In[ ]:

np.save('../../intermediate_files/gold_label_vectors/dev/anger.npy',np.array(dev_anger_intensities))
np.save('../../intermediate_files/gold_label_vectors/dev/fear.npy',np.array(dev_fear_intensities))
np.save('../../intermediate_files/gold_label_vectors/dev/joy.npy',np.array(dev_joy_intensities))
np.save('../../intermediate_files/gold_label_vectors/dev/sadness.npy',np.array(dev_sad_intensities))        


# In[ ]:

np.save('../../intermediate_files/gold_label_vectors/test/anger.npy',np.array(test_anger_intensities))
np.save('../../intermediate_files/gold_label_vectors/test/fear.npy',np.array(test_fear_intensities))
np.save('../../intermediate_files/gold_label_vectors/test/joy.npy',np.array(test_joy_intensities))
np.save('../../intermediate_files/gold_label_vectors/test/sadness.npy',np.array(test_sad_intensities))        


# In[ ]:



