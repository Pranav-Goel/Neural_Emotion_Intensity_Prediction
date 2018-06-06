This directory stores the intermediate files that are required to run all the codes in the ../codes/ directory. For some of these, the files have not been provided directly by us and *MUST be downloaded* before proceeding with actual execution of codes. 

Subdirectories - 

deepmoji/							- 	This contains the files required for creating feature vectors from the pre-trained activations of a CNN trained for emoji detection task (transfer learning for our task of emotion intensity detection). The files here are required to run the code ../codes/Supporting_Codes/create_and_save_deepmoji_and_lexicon_feature_vectors.ipynb.

deepmoji_vectors/					- 	(Initially empty) This contains the numpy vectors (for train, dev and test sets for all the emotions) which are basically 2304 dimensional vector representations which help us in our task of emotional intensity detection. Must be created and stored here by running the code ../codes/Supporting_Codes/create_and_save_deepmoji_and_lexicon_feature_vectors.ipynb. 

gold_label_vectors/					- 	(Initially empty) This contains the numpy vectors for the annotated emotional intensities for all the train, dev and test tweets for all emotions. Used in all the codes for training and evaluation purposes. Must be created and stored here by running the code ../codes/Supporting_Codes/create_and_save_word2vec_reps_and_labels.ipynb.

lexicons/							- 	This contains the files required for creating feature vectors from various external lexicons . The files here are required to run the code ../codes/Supporting_Codes/create_and_save_deepmoji_and_lexicon_feature_vectors.ipynb.

lexicon_vectors/					- 	(Initially empty) This contains the numpy vectors (for train, dev and test sets for all the emotions) which are basically 43 dimensional vector representations which help us in our task of emotional intensity detection. Must be created and stored here by running the code ../codes/Supporting_Codes/create_and_save_deepmoji_and_lexicon_feature_vectors.ipynb. 

word2vec_based_concatenated_vectors/	- 	(Initially empty) This contains the numpy vectors (for train, dev and test sets for all the emotions) which are basically (n, 50, 400) dimensional Twitter word2vec based vector representations which help us in our task of emotional intensity detection. Must be created and stored here by running the code ../codes/Supporting_Codes/create_and_save_word2vec_reps_and_labels.ipynb. 

word2vec_model/					- 	(Initially empty) This contains the pre-trained Twitter word2vec model, which must be downloaded and stored here (link in the README inside).

