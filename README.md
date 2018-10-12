## **How Emotional Are You?** Neural Architectures for Emotion Intensity Prediction in Microblogs
by *Devang Kulshreshtha^, Pranav Goel^ and Anil Kumar Singh* (^ = equal contribution from both authors in the paper)

This repository contains all the documented scripts and files (or links for downloading some resources) to run our experiments which give state-of-the-art performance for Emotion Intensity Detection in Twitter tweets (in the EmoInt shared task setting described [here](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)). The codes will allow replicating our results and should also help people looking to get started with using Keras (a deep learning library) to code neural architectures.

The neural models include a proposed architecture that combines *L*exicon-based features, pre-trained activations of a CNN trained for *E*moji detection in tweets, a CNN and fully connected layers in parallel (hence, it is a *P*arallely *C*onnected *D*eep *N*eural *N*etwork) called LE-PC-DNN by us (Section 2 in the paper). There are also 2 proposed Deep Multi-Task Learning architectures (Section 3). We also include code for our ablation tests to test the importance of the various components of the neural model, and provide scripts for running our pairwise correlation tests (Section 6).

*Details can be found in our paper (with the title above) accepted for publication at COLING 2018 (full main conference paper). The PDF is available [here](http://www.aclweb.org/anthology/C18-1247).*

### If you use the code here, or the experiments described help your own work, please cite our paper: _Kulshreshtha, Devang, Pranav Goel, and Anil Kumar Singh. "How emotional are you? Neural Architectures for Emotion Intensity Prediction in Microblogs." Proceedings of the 27th International Conference on Computational Linguistics. 2018._ [[Bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:c_IRDVy4AP4J:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAW8DaS_KpHT29e84HbQvYPRfyR50Ye56U&scisf=4&ct=citation&cd=-1&hl=en)]

### To successfully and smoothly run our experiments, please follow the steps below - 

1. Download the Twitter Word2Vec model and store it in the subdirectory intermediate_files/wordvec_model/ (instructions in the README there).
2. Head to the data/ directory and download the raw data files as instructed in the README.
3. Head to intermediate_files/lexicons/ and follow the instructions in that README to properly populate that directory.
4. Head to intermdiate_files/deepmoji and follow the instructions in that README properly.
5. Run the 2 codes in codes/Supporting_Codes/
6. Now you may run any of the other codes in codes/ subdirectory.

Note - Please make sure to get all the citations in place if you use the data files and other resources that our code uses (provided in the various READMEs in the subdirectories - please go through them all).

### Subdirectories - 

codes/			- 		This contains all the Python Scripts. Python 3 is used, and both Jupyter Notebook files and .py files for each script are included.
data/				- 		This contains the data files for the training, dev and test sets for all the emotions (WITH the annotated emotional intensities). Please check out the README to know more.
intermediate_files/	- 		This contains all the files that are usually produced from our own scripts - processed versions over the raw data, pre-trained word2vec model, etc. Directories wordvec_model, deepmoji and lexicons all need to be populated properly by following instructions in the corresponding README file. They are output of some code/bash scripts, and serve as input to some other codes.

*The deep learning architectures were coded using the Keras library with Tensorflow backend.*

### Following are the Python libraries used in our codes - 

pandas (0.20.1)

numpy (1.14.1)

keras (2.1.4)

tensorflow (1.5.0)

sklearn (0.19.0)

scipy (1.0.0)

nltk (3.2.3)

gensim (1.0.1)

wordsegment (0.8.0) ([https://github.com/grantjenks/python-wordsegment])

itertools

os

re

timeit

Note - If user faces problems in successfully creating and storing the intermediate files required, please email at "pranavgoel403@gmail.com" and we can try and give the numpy vectors directly (although they will be of large sizes).
