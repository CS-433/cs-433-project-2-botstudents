import sys
import re

# PREPROCESSING FUNCTION FOR A SINGLE TWEET

def preprocess_tweet(tweet, tokenize, split_hashtags, remove_stopwords) :
    
    '''
    Inputs 
    string : tweet 
    bool   : tokenize / split_hashtags / remove_stopwords
    
    Output 
    string : processed tweet 
    '''
    
    if tokenize :   
        tweet = make_tokens(tweet)
        
    if split_hashtags :
        dico = load_dico()
        tweet = split_htags(tweet, dico)
        
    if remove_stopwords :
        stopwords = load_stopwords()
        tweet = rm_stopwords(tweet, stopwords)
        
    return tweet 
        


# TOKENS

FLAGS = re.MULTILINE | re.DOTALL

# Load emoticons
def load_emoticons(name) :
    
    path = 'PreprocessingFiles/Emoticons/' + name + '.txt'
    content = None 
    
    with open(path, encoding='utf8') as f: 
        content = f.readlines()
        
    return [word.rstrip('\n') for word in content]

EMOTIONS = ['smiling','kissing','laughing','heart','skeptical','surprise','tongue','angrysad','brokenheart','heart']


def make_tokens(text):
    
    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    for emotion in EMOTIONS :
        emoticon_list = load_emoticons(emotion)
        for emoticon in emoticon_list[1:] :
            text = text.replace(emoticon.lower(), emoticon_list[0])
 
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    
    text = text.replace('.', ' . ')        
    text = re_sub(r"/"," / ")
    text = re_sub(r"  ", r" ")
   
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    
    return text.lower()


# HASHTAGS 

# Loading words from the top words txt file 
def load_dico() :
    content = None 
    with open('PreprocessingFiles/topwords.txt') as f: 
        content = f.readlines()
    return [word.rstrip('\n') for word in content]


# Function to parse word
def parse(sequence, dico):
    
    words = []
    # Remove hashtag, split by dash
    tags = sequence[1:].split('-')
      
    for tag in tags:
        word = find_word(tag, dico)    
        while word != None and len(tag) > 0:
            words.append(word)            
            if len(tag) == len(word): 
                break
            tag = tag[len(word):]
            word = find_word(tag, dico)
            
    #Testing meaning of  the output
    nb_words = len(words)
    tot_letters = len(sequence)-1
    if nb_words>0 :
        if tot_letters/nb_words < 2.0 :
            return ''
            
    return " ".join(words)


# Function to find word by removing the last letters one by one 
def find_word(letters, dico):
    i = len(letters) + 1
    while i > 1:
        i -= 1
        if letters[:i] in dico:
            return letters[:i]
    return None 


# Function to parse hashtags within the tweets
def split_htags(tweet, dico) : 
    
    new_tweet = "" 
    terms = tweet.split(' ')
    
    for term in terms :
        if len(term)>0 :
            if term[0] == '#': 
                new_tweet += ('<hashtag> ' + term + ' ')
                new_tweet += parse(term, dico)
            else: 
                new_tweet += term
            new_tweet += " " 
    
        
    return new_tweet



# REMOVING STOPWORDS

# Loading words from the top words txt file 
def load_stopwords() :
    content = None 
    with open('PreprocessingFiles/stopwords.txt') as f: 
        content = f.readlines()
    return [word.rstrip('\n') for word in content]


def rm_stopwords(tweet, stopwords) :
    
    new_tweet = ""
    terms = tweet.split(' ')
    
    for term in terms :
        new_term = term
        for stopword in stopwords :
            if term==stopword :
                new_term = '<stopword> '
        new_tweet += new_term
        new_tweet += " " 
        
    return new_tweet
    
    
    
    
    
    