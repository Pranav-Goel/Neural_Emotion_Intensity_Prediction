# -*- coding: utf-8 -*-

""" Use DeepMoji to encode texts into emotional feature vectors.
"""
from __future__ import print_function, division
import example_helper
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import pandas as pd
import sys

TEST_SENTENCES = list(pd.read_csv(sys.argv[1],delimiter='\t',header=None, encoding='utf-8')[1])	#change path to cover all 4 emotions for all 3 sets

maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
model.summary()

print('Encoding texts..')
encoding = model.predict(tokenized)

#print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[0]))
print(encoding.shape)

np.save(sys.argv[2], encoding)
print('SAVED')

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.
