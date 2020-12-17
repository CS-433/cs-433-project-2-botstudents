import numpy as np
import pandas as pd


# Returns a numpy array
def load_glove_embedding(embeddings_npy):
    # Opening embedding.npy file
    with open(embeddings_npy, 'rb') as f:
        word_vect = np.load(f)
        print('loaded glove embedding with shape', word_vect.shape)
        return word_vect  # numpy array


# Returns a dictionary
def load_vocabulary(voc_pkl):
    # Opening vocabulary pickle
    with open(voc_pkl, 'rb') as f:
        vocab = np.load(f, allow_pickle=True)
        print('loaded vocabulary containing', len(vocab), 'words')
        return vocab  # dictionary


# Returns a dataframe
def load_tweets(full = False):
    if full == False:
        train_neg_r = open("../twitter-datasets/train_neg.txt").readlines()
        train_pos_r = open("../twitter-datasets/train_pos.txt").readlines()
        
    elif full == True:
        train_neg_r = open("../twitter-datasets/train_neg_full.txt").readlines()
        train_pos_r = open("../twitter-datasets/train_pos_full.txt").readlines()

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
