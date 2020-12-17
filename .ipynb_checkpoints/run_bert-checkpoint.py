# -*- coding: utf-8 -*-
"""run.py

The following script uses the Keras wrapper library ktrain to build a BERT classifier
on the twitter dataset which can be downloaded here : 
https://www.aicrowd.com/challenges/epfl-ml-text-classification

ktrain : 
@article{maiya2020ktrain,
         title={ktrain: A Low-Code Library for Augmented Machine Learning},
         author={Arun S. Maiya},
         journal={arXiv},
         year={2020},
         volume={arXiv:2004.10703 [cs.LG]}
}

BERT : https://arxiv.org/abs/1810.04805

Keras : 
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}"""


#to import ktrain use pip install ktrain
import ktrain
import pandas as pd
from ktrain import text


# Returns a dataframe with data in twitter-datasets
def load_tweets(full = False):
    if full == False:
        train_neg_r = open("twitter-datasets/train_neg.txt").readlines()
        train_pos_r = open("twitter-datasets/train_pos.txt").readlines()
        
    elif full == True:
        train_neg_r = open("twitter-datasets/train_neg_full.txt").readlines()
        train_pos_r = open("twitter-datasets/train_pos_full.txt").readlines()

    # create dataframe with positive tweets and "1" label
    pos_tr = pd.DataFrame()
    pos_tr['text'] = train_pos_r
    pos_tr['label'] = 1

    # create dataframe with negative tweets and "0" label
    neg_tr = pd.DataFrame()
    neg_tr['text'] = train_neg_r
    neg_tr['label'] = 0

    # concatenate dataframes
    tweets_df = pd.concat([pos_tr, neg_tr], ignore_index=True)
    tweets_df.index.name = 'id'

    print('loaded', len(tweets_df), 'tweets in dataframe with columns:', tweets_df.columns)
    return tweets_df  # dataframe


tweets=load_tweets(full=True)

#store tweets in dataframe for bert preprocessing
df_tweets = pd.DataFrame()
df_tweets['label'] = tweets['label'].replace(0,'neg')
df_tweets['label'] = df_tweets['label'].replace(1,'pos')
df_tweets['text'] = tweets.text

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(df_tweets, 'text', 'label', 
                                                                  maxlen=30, preprocess_mode='bert')

model1 = ktrain.load_predictor('bert_predictor2').model
predictor1 = ktrain.get_predictor(model1, preproc)

tweets= open("twitter-datasets/test_data.txt").readlines()

# create dataframe with tweets
tweets_df = pd.DataFrame()
tweets_df['text'] = tweets

#function to clean test set which contain indices in strings
def strip_tweet(tweet):
    comma_idx = tweet.find(',')
    return tweet[comma_idx+1:]
tweets_df = pd.DataFrame()
tweets_df['text'] = [strip_tweet(t) for t in tweets]

tweets_df['y_pred'] = tweets_df.text.apply(predictor1.predict)

Y_pred = tweets_df['y_pred']
Y_pred = Y_pred.replace('pos',1)
Y_pred = Y_pred.replace('neg',-1)
Y_pred.index+=1

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

create_csv_submission(Y_pred.index, Y_pred.values,"output.csv")
