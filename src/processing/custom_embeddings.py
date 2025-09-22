import multiprocessing
import numpy as np
from gensim.models import Word2Vec

class Word2VecHelper:
    """
    A helper class to train a Word2Vec model and create sentence embeddings
    from a text corpus.
    """
    def __init__(self, vector_size, window, min_count, sg, epochs, alpha, negative):
        """
        Initializes the Word2VecHelper with hyperparameters for model training.

        Parameters:
        -----------
        vector_size : int, default 300
            Dimensionality of the word vectors.
        window : int, default 5
            Maximum distance between the current and predicted word within a sentence.
        min_count : int, default 5
            Ignores all words with a total frequency lower than this.
        sg : int, default 1
            Training algorithm: 1 for skip-gram, 0 for CBOW.
        epochs : int, default 30
            Number of training iterations.
        alpha : float, default 0.025
            The initial learning rate.
        negative : int, default 20
            Number of negative samples to use for negative sampling.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = multiprocessing.cpu_count()
        self.sg = sg
        self.epochs = epochs
        self.alpha = alpha
        self.negative = negative
        self.model = None

    def train_model(self, sentences):
        """
        Trains the Word2Vec model using the provided sentences.

        Parameters:
        -----------
        sentences : list of list of str
            A list of sentences, where each sentence is a list of words.
        """
        print("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            alpha=self.alpha,
            negative=self.negative
        )
        print("Training complete.")
        # return self.model, self.vector_size 

    def create_sentence_embeddings(self, sentences, text_column):
        """
        Creates sentence embeddings by averaging the word vectors.

        Parameters:
        -----------
        sentences : pandas.DataFrame
            DataFrame containing the sentences to embed.
        text_column : str, default 'translated_text'
            Name of the column containing the text to embed.
        embedding_column : str, default 'custom_word2vec_embedding'
            Name of the column to store the embeddings.

        Returns:
        --------
        tuple
            A tuple containing:
            - pandas.DataFrame: The original DataFrame with an added embedding column.
            - list: A list of words that were not found in the model's vocabulary.
        """


        if self.model is None:
            raise RuntimeError("Model has not been trained. Please call train_model() first.")
        
        custom_vectors = []
        custom_missing_words = []
        
        for sentence in sentences[text_column]:
            words = sentence.lower().split()
            word_vectors = []
            
            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
                else:
                    custom_missing_words.append(word)
            
            if word_vectors:
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                sentence_vector = np.zeros(self.vector_size)
            
            custom_vectors.append(sentence_vector)
            
        sentences["custom_word2vec_embedding"] = custom_vectors
        
        return sentences, list(set(custom_missing_words))