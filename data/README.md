The data consists of train, dev and test sets for each of the four emotions. Explanation for the data, and the data files can be downloaded from [here](https://competitions.codalab.org/competitions/16380#learn_the_details-datasets)

Note that under the latest rules given by Twitter, we cannot directly provide the tweets. Hence, please go to the above site, and - 
1. Download the text files provided there. Please store the corresponding files (WITH intensity labels) in the train/, dev/ and test/ directories.
2. Rename the files within these subdirectories as anger.txt, fear.txt, joy.txt and sadness.txt corresponding to the emotions.

Finally, each of the train, dev and test subdirectories contain four files - anger.txt, fear.txt, joy.txt and sadness.txt, containing the tweets marked for corresponding emotion WITH the gold annotated emotion intensities (for purpose of running the codes and replicating our experiments, the labelled intensities are already stored in the directory intermediate_files/gold_label_vectors/ in the form of numpy vectors). The files would be actually tab separated, with an ID, tweet text, emotion, and intensity of that emotion.
