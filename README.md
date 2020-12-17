# Sentiment Analysis with Natural Language Processing 

### Machine Learning Course - 2020/2021 

## Abstract 

This project aims to use Natural Language Processing to perform sentiment analysis on a twitter dataset. The task is to predict the presence of a positive :) or negative :( smiley within a tweet. The datset is composed of 2.500.000 tweets from which the smileys have been removed. 

## Twitter Dataset 

Link - https://www.aicrowd.com/challenges/epfl-ml-text-classification  

- *train_pos.txt* and *train_neg.txt* : a small set of training tweets for each of the two classes.     
- *train_pos_full.txt* and *train_neg_full.txt* : a complete set of training tweets for each of the two classes, about 1M tweets per class.      
- *sampleSubmission.csv* : a sample submission file in the correct format - each test tweet is numbered. (submission of predictions: -1 = negative prediction, 1 = positive prediction)    

## Overview 

We tried various approcaches on this project highligting the field improvements in the last few years. From classical ML methods to recent breakthroughs we explored different paths to achieve the same task. 
 
The best result, is achieved with the Bidirectional Encoder Representations for Transforrmers :  
We reached a 89.98% accuracy on the validation set.

## Summary 

1 - Classical Nachine Learning Methods    
2 - Preprocessing for GloVe Embeddings   
3 - Glove Pretrained    
4 - Neural Network Tunning   
5 - Bidirectional Encoder Representations from Transformers   


### 1 - Classical Machine Learning Methods 

In this part, we used various classical Natural Language Processing methods. 

### 2 - Preprocessing for GloVe Embedding 

The project suggested to train our own GloVe embeddings directly from the tweets. We implemented three preprocessing tasks to see if we were able to get better word-vector representations and to quantify the use of preprocessing. We tested our preprocessing options using LSTM Neural Network. 

### 3 - GloVe Pretrained 

In this part, we used pretrained GloVe embeddings to classify the tweets using LSTM Neural Networks. 

### 4 - Neural Network Tuning 

In this part we used the keras embedding layer to build our own word vectors with supervised learning. 
We used both LSTM and CNN architectures and compared the results. 

### 5 - Bidirectional Encoder Representations from Transformers 

In this part we train a BERT classifier using the ktrain library. 


## Downloads 

The following files need to be downloaded and added to the project architecture : 
- twitter-datasets 
- 


## Project Structure 

├── run.py                            
├── README.md        
├── report      
├── output.csv                          
|    
├── twitter-datasets               - This folder needs to be downloaded     
│   ├── train-   
│   ├── train-   
│    
├── classical-ml  
│   ├── classical_nlp_tests.ipynb 
│   ├── classical_ml_utils.py  
│   ├── load_utils.py
|
├── preprocessing-for-GloVe        
│   ├── vocab 
│   │   ├── vocab_pp0.pkl
│   │   ├── vocab_pp1.pkl
│   │   ├── vocab_pp2.pkl
│   │   ├── vocab_pp3.pkl
│   │
│   ├── pptweets 
│   │   ├── tweets_pp0.pkl
│   │   ├── tweets_pp1.pkl
│   │   ├── tweets_pp2.pkl
│   │   ├── tweets_pp3.pkl
│   │
│   ├── embeddings 
│   │   ├── embeddings_pp0.pkl
│   │   ├── embeddings_pp1.pkl
│   │   ├── embeddings_pp2.pkl
│   │   ├── embeddings_pp3.pkl
│
├── glove-pre-trained
│   ├── glove_pre_trained.ipynb
│   ├── load_utils.py
│   ├── stanford_preprocessing.py 
│   ├── glove.twitter.27B.25d.txt     - Pre-trained word vectors of twitter dataset by Stanford NLP group.
|
├── neura-networks-tuning
│   ├── neural_networks.ipynb
│   ├── FeaturesBuilder.py
│   ├── neural_net_utils.py
│   ├── load.py
│   ├── glove_embeddings.npy
│   ├── vocab.pkl
|
├── bert-ktrain
│   ├── bert_ktrain.ipynb
│   ├── load_utils.py
│   ├── proj1_helpers.py

## Citations 

##### Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation
##### @misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{\i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}


#####
#####
#####


