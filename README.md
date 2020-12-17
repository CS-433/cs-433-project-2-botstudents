# Sentiment Analysis with Natural Language Processing 

### Machine Learning Course - 2020/2021 

## Abstract 

This project aims to use Natural Language Processing to perform sentiment analysis on a twitter dataset. The task is to predict the presence of a positive :) or negative :( smiley within a tweet. Those have been removed. The dataset is composed of 2.500.000 tweets from which the smileys have been removed. 

## Twitter Dataset 

Link - https://www.aicrowd.com/challenges/epfl-ml-text-classification  

- *train_pos.txt* and *train_neg.txt* : a small set of training tweets for each of the two classes.     
- *train_pos_full.txt* and *train_neg_full.txt* : a complete set of training tweets for each of the two classes, about 1M tweets per class.      
- *sampleSubmission.csv* : a sample submission file in the correct format - each test tweet is numbered. (submission of predictions: -1 = negative prediction, 1 = positive prediction)    

## Overview 

We tried various approcaches on this project highligting the field improvements in the last few years. From classical ML methods to recent breakthroughs we explored different paths to achieve the same task. 
 
The best result, is achieved with the Bidirectional Encoder Representations for Transformers :  
We reached a 89.98% accuracy on the validation set.

## How to create submission 

- Download twitter-datasets 
- Download BERT weights 
- pip install tensorflow 
- pip install ktrain 
- run_bert.py 


## Summary 

**1 - Classical Machine Learning Methods**       
**2 - Preprocessing for GloVe Embeddings**     
**3 - Glove Pre-trained**       
**4 - Neural Network Tuning**     
**5 - Bidirectional Encoder Representations from Transformers**     

#### 1 - Classical Machine Learning Methods   

In this part, we used various classical Natural Language Processing methods.

#### 2 - Preprocessing for GloVe Embedding 

The project suggested to train our own GloVe embeddings directly from the tweets. We implemented three preprocessing tasks to see if we were able to get better word-vector representations and to quantify the use of preprocessing. We tested our preprocessing options using LSTM Neural Networks.

**GloVeEmbedding.ipynb** is the notebook used to process the tweets and produce the embeddings as well as the corresponding vocabularies. No need to run it as we already placed the preprocessed tweets, embeddings and vocabularies in the folder for convenience. Every step of preprocessing is explained in the notebook. 

**GloVeTraining.ipynb** is the notebook used to compare preprocessing options. The results from preprocessing testing were not satisfying which encouraged us to quickly move to other options such as pretrained embeddings or using the keras embedding layers.    

#### 3 - GloVe Pretrained 

In this part, we used pretrained GloVe embeddings to classify the tweets using a CNN Neural Network. 

#### 4 - Neural Network Tuning 

In this part we used the Keras embedding layer to build our own word vectors with supervised learning. 
We used both LSTM and CNN architectures and compared the results. 

#### 5 - Bidirectional Encoder Representations from Transformers 

In this part we train a BERT classifier using the ktrain library.  

To run the BERT model please download the bert-predictor2 folder (containing the model weights) on the following link and place it in the root of the project.   

## Downloads 

The following files need to be downloaded and added to the project structure: 
- twitter-datasets 
- bert_predictors2
https://drive.google.com/drive/folders/1cQ1fte2ILDfBD4zeXmAuM8f3rL12pCKx?fbclid=IwAR0HLpsom2lGMZX7SwebI0peFoqHdg3_s_lANzcCL_bOESZeBHueUQf6FZg
- glove.twitter.27B.25d.txt
https://nlp.stanford.edu/projects/glove/



## Project Structure 

├── run_bert.py                            
├── README.md             
├── output.csv                          
|      
├── twitter-datasets               - This folder needs to be downloaded       
│   ├── train_pos.txt     
│   ├── train_neg.txt    
│   ├── train_pos_full.txt   
│   ├── train_neg_full.txt  
│   ├── sampleSubmission.csv      
│  
├── classical-ml      
│   ├── classical_nlp_tests.ipynb   
│   ├── classical_ml_utils.py    
│   ├── load_utils.py  
|  
├── preprocessing-for-GloVe   
│   ├── GloVeEmbedding.ipynb    - Notebook used for preprocessing / No need to rub it 
│   ├── GloVeTraining.ipynb   
│   ├── preprocessing.py 
│   ├── FeatureBuilder.py   
│   ├── load_utils_pp.py   
│   │ 
│   ├── PreprocessingFiles   - Files needed for preprocessing 
│   │
│   ├── vocab        - Vocab folder for preprocessing options    
│   │   
│   ├── vocab        - Vocab folder for preprocessing options    
│   ├── pptweets     - Preprocessed tweets folder for each preprocessing options    
│   ├── embeddings   - Embedding folder for preprocessing options        
│   ├── models       - Models folder for preprocessing options    
│   │  
│   ├──  build_vocab.sh  
│   ├──  cut_vocab.sh  
│   ├──  build_vocab_pp.sh  
│   ├──  cut_vocab_pp.sh    
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


## Hoy to create submission 

- Download twitter-datasets 
- Download BERT weights 
- pip install tensorflow if needed
- pip install ktrain 
- run_bert.py 

  
## Citations   
  
##### Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation  

##### TensorFlow
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.

#####  Ktrain   
ktrain: A Low-Code Library for Augmented Machine Learning  
author: Arun S. Maiya  
https://github.com/amaiya/ktrain?fbclid=IwAR1C_hYsDbOPQdvYGu9K0twrPhrxMb_dnp2FRjyDFX-SBDulvZc4MZH7x-k  

#####  Scikit-learn   
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.  
 

