import numpy as np


# manage building temporal word sequences for training
class FeaturesBuilder:
    def __init__(self, tweets_df, vocab, word_vect=None, target_length=30):
        self.tweets_df = tweets_df
        self.vocab = vocab
        self.target_length = target_length

        if word_vect is not None:
            # catalog of word embeddings
            self.word_vect = word_vect
            self.embedding_dim = word_vect.shape[1]

    # _____ BUILD AVERAGE TWEET EMBEDDINGS
    # compute aggregated tweet vectors
    def build_avg_tweet_embedding(self):
        def tweet_vector(tweet):
            words = tweet.split()
            return np.mean(self.word_vect[[self.vocab[word] for word in words if word in self.vocab]], axis=0)

        features = np.stack(self.tweets_df.text.apply(tweet_vector))
        labels = np.array(self.tweets_df.label)
        # filter out features with nan values
        validFeatures = np.sum(np.isnan(features), axis=1) == 0
        print('built features with shape', features[validFeatures].shape)
        return features[validFeatures], labels[validFeatures]

    # _____ BUILD SEQUENCES OF WORD EMBEDDINGS
    # trim or pad with null embedding to match target_length
    def _pad_with_null_embedding(self, tweet):
        words = tweet.split()
        sequence = self.word_vect[[self.vocab[word] for word in words if word in self.vocab]]
        length = len(sequence)
        if length >= self.target_length:
            return sequence[length - self.target_length:]  # todo: try trimming start instead ?
        else:
            return np.concatenate([np.zeros((self.target_length - length, self.embedding_dim)), sequence])

    # build dataset
    def build_word_embedding_sequences(self):
        sequences = np.stack(self.tweets_df.text.apply(self._pad_with_null_embedding)).astype('float32')
        labels = np.array(self.tweets_df.label).astype('int')
        print('built features with shape', sequences.shape)
        return sequences, labels

    # _____ BUILD SEQUENCES OF VOCABULARY INDEXES
    # trim or pad with null index to match target_length
    def _pad_with_zeros(self, tweet):
        words = tweet.split()
        sequence = np.array([self.vocab[word] for word in words if word in self.vocab])
        length = len(sequence)
        if length >= self.target_length:
            return sequence[length - self.target_length:]  # todo: try trimming start instead
        else:
            return np.concatenate([np.zeros((self.target_length - length)), sequence])

    # build dataset
    def build_vocab_idx_sequences(self):
        sequences = np.stack(self.tweets_df.text.apply(self._pad_with_zeros)).astype('int')
        labels = np.array(self.tweets_df.label).astype('int')
        print('built features with shape', sequences.shape)
        return sequences, labels
