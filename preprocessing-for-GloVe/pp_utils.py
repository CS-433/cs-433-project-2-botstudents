import pandas as pd 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt

from scipy.sparse import *
from sklearn.feature_extraction.text import CountVectorizer


# Loading tweets

def load_tweets(full = False):
    
    if full == False:
        train_neg_r = open("twitter-datasets/train_neg.txt", encoding='utf8').readlines()
        train_pos_r = open("twitter-datasets/train_pos.txt", encoding='utf8').readlines()
        
    elif full == True:
        train_neg_r = open("twitter-datasets/train_neg_full.txt", encoding='utf8').readlines()
        train_pos_r = open("twitter-datasets/train_pos_full.txt", encoding='utf8').readlines()

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



# Create DataFrames to host tweets 

def create_df(nb_pos_tweets, nb_neg_tweets): 

    # create dataframe with positive tweets and "1" label
    pos_tr = pd.DataFrame()
    pos_tr['text'] = ['x'] * nb_pos_tweets
    pos_tr['label'] = 1

    # create dataframe with negative tweets and "0" label
    neg_tr = pd.DataFrame()
    neg_tr['text'] = ['x'] * nb_neg_tweets
    neg_tr['label'] = 0

    # concatenate dataframes
    tweets_df = pd.concat([pos_tr, neg_tr], ignore_index=True)
    tweets_df.index.name = 'id'

    return tweets_df


# Create a vocabulary with the words who appear more than 100 times

def create_topwords(tweets, min_df) : 
    
    # Create a new vocabulary from tokenized tweets 
    cv = CountVectorizer(token_pattern=r'[^\s]+', min_df = min_df)
    cv.fit(tweets.text)
    cv_fit = cv.fit_transform(tweets.text)
    
    # Extracting top words that are neither hashtags or tokens 
    d = {'Words': cv.get_feature_names(), 'Count': cv_fit.toarray().sum(axis=0)}
    res = pd.DataFrame(data=d)
    res.sort_values(by='Count', inplace=True, ascending=False)
    
    term_idx = ~res['Words'].str.startswith('#') & ~res['Words'].str.contains('<')
    top_terms = res[term_idx]
    
    top = np.array(top_terms['Words'])
    
    # Saving top words as a txt file 
    with open("PreprocessingFiles/topwords.txt", "w") as txt_file:
        for term in top:
            txt_file.write(term + "\n")
        txt_file.write('<number>' + "\n")






# GloVe Embedding 


def cooc(pp):
    
    print("creating cooccurrence matrix ...")
    
    name = 'vocab/vocab_pp' + str(pp) + '.pkl'
    with open(name , 'rb') as f:
        vocab = pkl.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    
    with open('pptweets/tweets_pp'+ str(pp)+'.txt') as f:
        for line in f :
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            tokens = [t for t in tokens if t >= 0]
            for t in tokens:
                for t2 in tokens:
                    data.append(1)
                    row.append(t)
                    col.append(t2)
            
    cooc = coo_matrix((data, (row, col)))
    cooc.sum_duplicates()
    
    with open('cooc/cooc_pp' + str(pp) + '.pkl', 'wb') as f:
        pkl.dump(cooc, f, pkl.HIGHEST_PROTOCOL)


# Utils

# Load Stopwords list 

def load_sw() :
    content = None 
    with open('PreprocessingFiles/stopwords.txt') as f: # A file containing common english words
        content = f.readlines()
    return [word.rstrip('\n') for word in content]


# Create Vocab pickles 

def pickle_vocab(pp) :
    vocab = dict()

    path = 'vocab/vocab_cut_pp' + str(pp) + '.txt'
    with open(path) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    name = 'vocab/vocab_pp' + str(pp) + '.pkl'
    with open(name, 'wb') as f:
        pkl.dump(vocab, f, pkl.DEFAULT_PROTOCOL)



# Create vocabularies from existing one 

def create_vocab_pp3() : 

    # Open Vocabulary
    pp3 = pd.DataFrame()
    pp3['text'] = open("vocab/vocab_cut_pp1.txt").readlines()
    pp3.index.name = 'id'
    
    stopwords = load_sw()
    
    # Dropping stopwords 
    idx = [False] * len(pp3) 
    for i in range(len(stopwords)) :
        idx = idx | pp3.text.apply(lambda x : x==stopwords[i]+ '\n')
        
    with open("vocab/vocab_cut_pp3.txt", "w") as txt_file:
        txt_file.write('<stopword>\n')
        for word in np.array(pp3['text']) :
            txt_file.write(word)
            
    # Create a pickle version        
    pickle_vocab(3)
               
def create_vocab_pp2() :  

    # Open Vocabulary 
    pp2 = pd.DataFrame()
    pp2['text'] = open("vocab/vocab_cut_pp1.txt").readlines()
    pp2.index.name = 'id'
        
    with open("vocab/vocab_cut_pp2.txt", "w") as txt_file:
        txt_file.write('<hashtag>\n')
        for word in np.array(pp2['text']) :
            txt_file.write(word)
   
    # Create a pickle version 
    pickle_vocab(2)
            
          

# GloVe Embedding 
# - ML Course Repository - 


def cooc(pp):
    
    print("creating cooccurrence matrix ...")
    
    name = 'vocab/vocab_pp' + str(pp) + '.pkl'
    with open(name , 'rb') as f:
        vocab = pkl.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    
    with open('pptweets/tweets_pp'+ str(pp)+'.txt') as f:
        for line in f :
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            tokens = [t for t in tokens if t >= 0]
            for t in tokens:
                for t2 in tokens:
                    data.append(1)
                    row.append(t)
                    col.append(t2)


def glove(pp):
    
    print("loading cooccurrence matrix ...")
    
    with open('cooc/cooc_pp' + str(pp) + '.pkl', 'rb') as f:
        cooc = pkl.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4
    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
            
    np.save('embeddings/embeddings_pp' + str(pp), xs)
    

   