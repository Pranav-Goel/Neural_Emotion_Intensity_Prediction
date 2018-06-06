## The files in this directory, ONCE POPULATED, will serve as INPUT to the code ../../codes/Supporting_Codes/create_and_save_deepmoji_and_lexicon_feature_vectors.ipynb, which in turn will produce lexicon based feature vectors (numpy files) that are required to run all the other codes of our actual experiments.

### Please follow the steps to produce the necessary files for lexicon-based features -

1. Installing Afective Tweets - The AffectiveTweets package allows calculating multiple features from a tweet. Installation instructions for AffectiveTweets1.0.1 are given in the project's [webpage](https://affectivetweets.cms.waikato.ac.nz/install/). Please install weka and the package to proceed.

2. Using the script provided - tweets_to_arff.py (downloaded from [here](https://github.com/felipebravom/EmoInt#2-weka-baseline-system)) - convert all the original .txt data files into the .arff format. Basically run the following command for all the emotion files for train, dev and test (assuming you are in this current directory in the command line) -
python2 tweets_to_arff.py ../../data/train/anger.txt train/anger.arff
Now we have a total of 12 .arff files (4 each in train, dev and test subdirectories, named after the name of emotion).

3. Open the Weka GUI - cd weka-3-8-2/ followed by java -jar weka.jar.

4. Click on Explorer.

5. In the Preprocess tab (wherein everything will be carried out), click on Open file. Choose the .arff file (for example, train/anger.arff).

6. In the Attributes sub-window, 4 attributes will be shown - id, tweet, emotion and score. Choose (Select the checkbox for) tweet.

7. In the Filter sub-window, click on Choose. Then go to filters -> unsupervised -> attribute and select TweetToLexiconFeatureVector.

8. In the Filter sub-window, right click on the text dialog box and select "Enter Configuration". In the next window, enter - "weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 2 -A -D -F -H -J -L -N -P -Q -R -T -U". Click on OK.

9. Click on Apply.

10. Now see that the Attribute sub-window will have 47 attributes. Select the checkbox for the first four - id, tweet, emotion and score, and click on Remove. This leaves 43 attributes (can be seen in the Current relation sub-window).

11. Click on Save. Change the File Name (for example - anger.arff to anger.csv). Also change the File Type to below to the CSV format before saving.

12. Repeat steps 5 to 11 for all the four emotions, for all the three sets.

Finally we will have anger.csv, fear.csv, joy.csv and sadness.csv files in train, dev and test subdirectories for a total of 12 .csv files. You may delete the .arff files at this point. These csv files will serve as input to one of the supporting codes.

To download java jdk - [http://www.oracle.com/technetwork/java/javase/downloads/jdk10-downloads-4416644.html].


### CITATIONS and details -

If you use these files, please cite - Emotion Intensities in Tweets. Saif M. Mohammad and Felipe Bravo-Marquez. In Proceedings of the Joint Conference on Lexical and Computational Semantics (*Sem), August 2017, Vancouver, Canada. Also cite the papers for specific resources as mentioned below.

Link to the code/resources used here - [https://github.com/felipebravom/AffectiveTweets]. PLEASE visit this link to know more, and also get the Bibtex citations for the overall resource, as well as citations for the individual resources specified below.

*TweetToLexiconFeatureVector* is a filter in the AffectiveTweets (Mohammad and Bravo-Marquez, 2017) package for converting tweets into numeric 43-dimensional vectors that can be used directly as features in our machine learning system. The filter calculates the features from the tweet using several lexicons:
(a) MPQA Subjectivity Lexicon: Calculates the number of positive and negative words from the lexicon (Theresa Wilson, Janyce Wiebe, and Paul Hoffmann. 2005. Recognizing contextual polarity in phraselevel sentiment analysis. In Proceedings of the conference on human language technology and empirical methods in natural language processing. Association for Computational Linguistics, pages 347?354.)
(b) Bing-Lui: Calculates the number of positive and negative words from the lexicon (Konstantin Bauman, Bing Liu, and Alexander Tuzhilin. 2017. Aspect based recommendations: Recommending items with the most valuable aspects based on user reviews.)
(c) AFINN: Wordlist-based approach for calculating positive and negative sentiment scores from the lexicon (Finn Arup Nielsen. 2011. A new anew: Evaluation of a ? word list for sentiment analysis in microblogs. arXiv preprint arXiv:1103.2903.)
(d) Sentiment140: Calculates positive and negative sentiment score provided by the lexicon in which tweets are annotated by lexicons (Saif M. Mohammad and Peter D. Turney. 2013. Crowdsourcing a word-emotion association lexicon 29(3):436?465.)
(e) NRC Hashtag Sentiment lexicon: Uses same lexicon as Sentiment 140 but here tweets with only emotional hashtags are considered during training.
(f) NRC-10 Expanded: Emotional associations of words matching the Twitter specific expansion of the lexicon (Felipe Bravo-Marquez, Eibe Frank, Saif M Mohammad, and Bernhard Pfahringer. 2016. Determining word-emotion associations from tweets by multilabel classification. In Web Intelligence (WI), 2016 IEEE/WIC/ACM International Conference on. IEEE, pages 536?539.) are added to give the vale of this feature.
(g) NRC Hashtag Emotion Association Lexicon: Emotional associations of words of the lexicon (Saif M Mohammad and Svetlana Kiritchenko. 2015. Using hashtags to capture fine emotion categories from tweets. Computational Intelligence 31(2):301?326.) are added to give the vale of this feature.
(h) SentiWordNet: Calculates positive and negative sentiment score using SentiWordNet (Stefano Baccianella, Andrea Esuli, and Fabrizio Sebastiani. 2010. Sentiwordnet 3.0: An enhanced lexical resource for sentiment analysis and opinion mining. In LREC. volume 10, pages 2200?2204.)
(i) Emoticons: Calculates sentiment scores using word associations provided by emoticons from the lexicon (Finn Arup Nielsen. 2011. A new anew: Evaluation of a ? word list for sentiment analysis in microblogs. arXiv preprint arXiv:1103.2903.)
(j) Negations: This feature simply count the number of negating words in the tweet.