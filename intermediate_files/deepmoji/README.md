### Please follow the instructions below to create the deepmoji based feature vectors (2304 dimensional) and to store them as numpy vectors in required directory. These pre-trained encodings are very important to replicate our experiments (in codes/)

1. Clone or download the DeepMoji code from the Github repository - https://github.com/bfelbo/DeepMoji. Store it in this directory (unzip to get the folder named DeepMoji-master/).

2. Run the command (while being in DeepMoji-master directory) - "sudo pip2 install -e .".

3. Replace DeepMoji-master/examples/encode_texts.py with the encode_texts.py provided by us in this directory.

4. Run the following command from DeepMoji-master/ : python2 examples/encode_texts.py '../../../../data/train/EMOTION.txt' '../../../deepmoji_vectors/SET/EMOTION.npy' where SET can be 'train', 'dev' or 'test' and EMOTION can be 'anger', 'fear', 'joy', 'sadness' (must replace). Hence, this command (variations) will be run 12 times. The last line of the output printed to the terminal at each running of the command should be 'SAVED'.


For using DeepMoji, please cite - 

@inproceedings{felbo2017,
  title={Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm},
  author={Felbo, Bjarke and Mislove, Alan and S{\o}gaard, Anders and Rahwan, Iyad and Lehmann, Sune},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2017}
}

Link to the code/resources used here - [https://github.com/bfelbo/DeepMoji]