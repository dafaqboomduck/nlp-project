import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

class BertEmbeddingsHelper:
    """
    A helper class to generate BERT embeddings for text data.
    
    This class loads the BERT model and tokenizer once and provides methods
    to create sentence embeddings using different pooling strategies.
    """
    
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initializes the BERT tokenizer and model.
        
        Parameters:
        model_name (str): The name of the pre-trained BERT model to use.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_embeddings_batch(self, sentences, pooling_method='cls', max_length=512):
        """
        Generates BERT embeddings for a list of sentences in a single batch.
        
        Parameters:
        sentences (list): A list of strings to be embedded.
        pooling_method (str): 'cls' for CLS token or 'mean' for mean pooling.
        max_length (int): Maximum sequence length for BERT.
        
        Returns:
        np.ndarray: A NumPy array of embeddings.
        """
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        if pooling_method == 'cls':
            # Use the CLS token embedding (the first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:  # mean pooling
            # Calculate mean of all token embeddings, masked by attention mask
            attention_mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        return embeddings

    def create_bert_embeddings(self, df: pd.DataFrame, text_column: str, pooling_method='cls', max_length=512, batch_size=8):
        """
        Creates and adds BERT embeddings as a new column to a DataFrame.
        
        This method processes the DataFrame in batches for efficiency.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing sentences.
        pooling_method (str): 'cls' for CLS token or 'mean' for mean pooling.
        max_length (int): Maximum sequence length for BERT.
        batch_size (int): Batch size for processing.
        
        Returns:
        pd.DataFrame: The original DataFrame with a new 'bert_embedding' column.
        """
        sentences_list = df[text_column].tolist()
        all_embeddings = []
        
        for i in range(0, len(sentences_list), batch_size):
            batch_sentences = sentences_list[i:i+batch_size]
            batch_embeddings = self._get_embeddings_batch(
                batch_sentences,
                pooling_method=pooling_method,
                max_length=max_length
            )
            all_embeddings.extend(batch_embeddings)
            
        df['bert_embedding'] = [emb for emb in all_embeddings]
        
        return df