# Sentiment Analysis with Natural Language Processing 

### Machine Learning Course - 2020/2021 

## Abstract 

This project aims to use Natural Language Processing to perform sentiment analysis on a twitter dataset. The task is to predict the presence of a positive :) or negative :( smiley within a tweet. The datset is composed of 2.500.000 tweets from which the smileys have been removed. 

## Overview 

We tried various approcaches on this project highligting the field improvements in the last few years. From classical ML methods to recent breakthroughs we explored different paths to achieve the same task. 
 
The best result, is achieved with the Bidirectional Encoder Representations for Transforrmers :
We achieved a 89.98% accuracy on the validation set.

## Summary 

1 - Classical Nachine Learning Methods  
2 - Preprocessing for GloVe Embeddings
3 - Glove Pretrained  
4 - Neural Network Tunning 
5 - Bidirectional Encoder Representations from Transformers 


### 1 - Classical Machine Learning Methods 

In this part, we used various classical Natural Language Processing mMthods. 

### 2 - Preprocessing for GloVe Embedding 

The project suggested to train our own GloVe embeddings directly from the tweets. We implemented three preprocessing tasks to see if we were able to get better word-vector representations and to quantify the use of preprocessing. We tested our preprocessing options using LSTM Neural Network. 

### 3 - GloVe Pretrained 

In this part, we used pretrained GloVe embeddings to classify the tweets using LSTM Neural Networks. 

##### Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation

### 4 - Neural Network Tuning 

In this part we used the keras embedding layer to build our own word vectors with supervised learning. We used both LSTM and CNN architectures and compared the results. 



### 5 - Bidirectional Encoder Representations from Transformers 

In this part we train a BERT classifier using the ktrain library. 

## Citations 

##### Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation
#####
#####
#####
#####


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
│   ├── negative-words.txt         : Dataset of negative words in English.
│   ├── positive-words.txt         : Dataset of positive words in English.
│   ├── README.md                  : Brief explanation of the opinion-lexicon datasets.
|
├── preprocessing-for-GloVe        
│   ├── 
│   │
│   │
|   │
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


