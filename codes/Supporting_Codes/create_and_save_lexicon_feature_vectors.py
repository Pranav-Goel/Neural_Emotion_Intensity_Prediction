
# coding: utf-8

# ### This is a small script that takes the csv files containing the 43 dimensional lexicon-based features and just converts them into numpy vectors and stores them, for ease of use in the actual codes. Most of the work is done by following the instruction in intermediate_files/lexicons/README.md. It is necessary to successfully follow those instructions first, before executing this script.

# In[ ]:

import numpy as np
import pandas as pd


# In[ ]:

train_anger = pd.read_csv('../../intermediate_files/lexicons/train/anger.csv').values
train_fear = pd.read_csv('../../intermediate_files/lexicons/train/fear.csv').values
train_joy = pd.read_csv('../../intermediate_files/lexicons/train/joy.csv').values
train_sadness = pd.read_csv('../../intermediate_files/lexicons/train/sadness.csv').values


# In[ ]:

dev_anger = pd.read_csv('../../intermediate_files/lexicons/dev/anger.csv').values
dev_fear = pd.read_csv('../../intermediate_files/lexicons/dev/fear.csv').values
dev_joy = pd.read_csv('../../intermediate_files/lexicons/dev/joy.csv').values
dev_sadness = pd.read_csv('../../intermediate_files/lexicons/dev/sadness.csv').values


# In[ ]:

test_anger = pd.read_csv('../../intermediate_files/lexicons/test/anger.csv').values
test_fear = pd.read_csv('../../intermediate_files/lexicons/test/fear.csv').values
test_joy = pd.read_csv('../../intermediate_files/lexicons/test/joy.csv').values
test_sadness = pd.read_csv('../../intermediate_files/lexicons/test/sadness.csv').values


# In[ ]:

np.save('../../intermediate_files/lexicon_vectors/train/anger.npy',train_anger)
np.save('../../intermediate_files/lexicon_vectors/train/fear.npy',train_fear)
np.save('../../intermediate_files/lexicon_vectors/train/joy.npy',train_joy)
np.save('../../intermediate_files/lexicon_vectors/train/sadness.npy',train_sadness)        


# In[ ]:

np.save('../../intermediate_files/lexicon_vectors/dev/anger.npy',dev_anger)
np.save('../../intermediate_files/lexicon_vectors/dev/fear.npy',dev_fear)
np.save('../../intermediate_files/lexicon_vectors/dev/joy.npy',dev_joy)
np.save('../../intermediate_files/lexicon_vectors/dev/sadness.npy',dev_sadness)        


# In[ ]:

np.save('../../intermediate_files/lexicon_vectors/test/anger.npy',test_anger)
np.save('../../intermediate_files/lexicon_vectors/test/fear.npy',test_fear)
np.save('../../intermediate_files/lexicon_vectors/test/joy.npy',test_joy)
np.save('../../intermediate_files/lexicon_vectors/test/sadness.npy',test_sadness)        

