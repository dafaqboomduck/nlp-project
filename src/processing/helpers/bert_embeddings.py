import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BertEmbeddingsHelper:
    """
    Helper to generate BERT embeddings for text data with error handling.
    """
    def __init__(self, model_name='bert-base-uncased', device=None):
        # Validate model_name
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")

        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        try:
            # Use AutoTokenizer/AutoModel for broader compatibility
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            logger.exception("Failed to load tokenizer/model '%s'.", model_name)
            raise RuntimeError(f"Failed to load model/tokenizer for '{model_name}'. Ensure the name is correct and internet is available.") from e

    def _get_embeddings_batch(self, sentences, pooling_method='cls', max_length=512):
        """
        Get embeddings for a small batch. Returns numpy array.
        """
        if not isinstance(sentences, (list, tuple)):
            raise TypeError("sentences must be a list or tuple of strings.")

        if pooling_method not in ('cls', 'mean'):
            raise ValueError("pooling_method must be either 'cls' or 'mean'.")

        try:
            encoded = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            # Move tensors to device safely
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            if pooling_method == 'cls':
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                attention_mask = encoded.get('attention_mask')
                if attention_mask is None:
                    # fallback to simple mean
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    # mean pooling with mask
                    mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            return embeddings

        except RuntimeError as e:
            # Typical PyTorch OOM or CUDA errors
            logger.exception("Runtime error while generating BERT embeddings (possible OOM).")
            raise RuntimeError("Error generating BERT embeddings. This may be due to GPU/CPU memory limits.") from e
        except Exception as e:
            logger.exception("Unexpected error during BERT embedding generation.")
            raise

    def create_bert_embeddings(self, df: pd.DataFrame, text_column: str, pooling_method='cls', max_length=512, batch_size=8, embedding_column='bert_embedding'):
        """
        Processes DataFrame in batches and appends embeddings.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        if text_column not in df.columns:
            raise KeyError(f"text_column '{text_column}' not present in DataFrame. Available columns: {list(df.columns)}")

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        sentences_list = df[text_column].fillna("").astype(str).tolist()
        all_embeddings = []

        try:
            for i in range(0, len(sentences_list), batch_size):
                batch_sentences = sentences_list[i:i + batch_size]
                batch_embeddings = self._get_embeddings_batch(
                    batch_sentences,
                    pooling_method=pooling_method,
                    max_length=max_length
                )
                all_embeddings.extend(batch_embeddings)
        except Exception:
            logger.exception("BERT embedding generation failed during batch processing.")
            raise

        # sanity check length
        if len(all_embeddings) != len(df):
            raise RuntimeError("Number of embeddings generated does not match number of input rows.")

        df[embedding_column] = [emb for emb in all_embeddings]
        return df
