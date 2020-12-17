import numpy as np
import pandas as pd
import pickle as pkl


# Returns a numpy array
def load_glove_embedding_pp(pp):
    
    path = 'embeddings/embeddings_pp' + str(pp) + '.npy'
    
    # Opening embedding.npy file
    with open(path, 'rb') as f:
        word_vect = np.load(f)
        print('loaded glove embedding with shape', word_vect.shape)
        return word_vect  # numpy array


# Returns a dictionary
def load_vocabulary_pp(pp):
    
    path = 'vocab/vocab_pp' + str(pp) + '.pkl'
    
    # Opening vocabulary pickle
    with open(path, 'rb') as f:
        vocab = pkl.load(f)
        print('loaded vocabulary containing', len(vocab), 'words')
        return vocab  # dictionary


# Returns a dataframe
def load_tweets_pp(pp):
    
    path = 'pptweets/tweets_pp' + str(pp) + '.txt'
    
    pptweets = open(path).readlines()

    # create dataframe with positive tweets and "1" label
    pos_tr = pd.DataFrame()
    pos_tr['text'] = pptweets[:100000]
    pos_tr['label'] = 1

    # create dataframe with negative tweets and "0" label
    neg_tr = pd.DataFrame()
    neg_tr['text'] = pptweets[100000:200000]
    neg_tr['label'] = 0

    # concatenate dataframes
    tweets_df = pd.concat([pos_tr, neg_tr], ignore_index=True)
    tweets_df.index.name = 'id'

    print('loaded', len(tweets_df), 'tweets in dataframe with columns:', tweets_df.columns)
    return tweets_df  # dataframe