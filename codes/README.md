This directory contains the codes for the experiments explained in the paper. 

Supporting_Codes/	-	This contains codes that use the raw data files to get various intermediate files like word2vec representations of tweets, and other feature vectors required as input for our neural architectures. Hence, these codes MUST be run before the codes in directories mentioned below.

LE_PC_DNN/		-		This contains the code for the main proposed neural network (section 2 in the paper), which combines various deep networks, lexicon-based and transfer-learning-based features in parallel. This model is run separately for each emotion.

Multi_task/		-		This contains two codes following the explanations given in Section 3.1 and 3.2 of the paper. These two architectures are deep multi-task learning models that effectively handle all four emotions in a single architecture. 

Correlation_Tests/	-	This contains the codes for testing pairwise correlations of different pairs of emotions (as explained in Section 6 of the paper).

All codes were run on Python 3 using Jupyter Notebook (both the ipython notebooks and .py files are given).