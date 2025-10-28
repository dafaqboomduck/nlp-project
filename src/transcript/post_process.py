from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import logging

logger = logging.getLogger(__name__)

class TranscriptionPostProcessor:
    def __init__(self, punctuation_model_name):
        """
        Initializes the post-processor with a pre-trained model for punctuation and capitalization.
        
        Args:
            punctuation_model_name (str): The name of the model to use from Hugging Face.
        """
        try:
            self.punc_model = pipeline("text-generation", model=punctuation_model_name, device_map='auto')
        except Exception as e:
            print(f"Error loading model: {e}")
            self.punc_model = None
            print("Punctuation model not loaded. Post-processing will be limited.")

    def _add_punctuation_and_capitalization(self, text):
        """
        Uses a pre-trained model to add punctuation and capitalization to the raw text.
        This is a much more sophisticated approach than a rule-based system.
        """
        if not self.punc_model:
            return text
        
        messages = [
            {"role": "user", "content": f"Add punctuation and capitalization to the following text sequence without changing its content. Decide where a sentence should end and where another should begin based on the flow of speech. The text is: '{text}'"},
        ]
        
        outputs = self.punc_model(
            messages,
            max_new_tokens=len(text) + len(text) / 5,
        )
        
        processed_text = outputs[0]["generated_text"][-1]
        
        return processed_text

    def process(self, raw_text):
        """
        The main method to run the post-processing pipeline.
        
        Args:
            raw_text (str): The raw text output from a speech-to-text model.
            
        Returns:
            str: The final, processed text.
        """
        # A simple normalization step can still be useful before the model.
        normalized_text = raw_text.lower().strip()

        # Use the model for the heavy lifting of punctuation and capitalization.
        final_text = self._add_punctuation_and_capitalization(normalized_text)
        
        return final_text

class TranscriptionGrammarPostProcessor:
    def __init__(self, device=None):
        """ 
        Initializes the post-processor with a specialized grammar correction model.

        Args:
            device (str): Device to run on ('cuda', 'cpu', or None for auto)
        """
        try:
            logger.info('Loading grammar correction model...')

            # Auto-detect device if not specified
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.device = device
            
            # Use a more reliable grammar correction model
            model_name = "pszemraj/flan-t5-large-grammar-synthesis"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f'Model loaded successfully on {self.device}')

        except Exception as e:
            logger.error(f'Error loading model: {e}')
            self.model = None
            self.tokenizer = None
            raise

    def _correct_text(self, text):
        """
        Corrects grammar, spelling, punctuation, and capitalization.

        Args:
            text (str): Raw text to correct

        Returns:
            str: Corrected text
        """
        if not self.model or not self.tokenizer:
            return text
        
        try:
            # Prepare input
            input_text = f"grammar: {text}"
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate correction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove "grammar:" prefix if it appears in the output
            if corrected_text.lower().startswith('grammar:'):
                corrected_text = corrected_text[8:].strip()
            
            return corrected_text
            
        except Exception as e:
            logger.error(f'Error correcting text: {e}')
            return text
    
    def _split_into_sentences(self, text):
        """
        Split corrected text into sentences.

        Args:
            text (str): Corrected text with proper punctuation

        Returns:
            list: List of sentences
        """
        # Split on sentence-ending punctuation followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

        # Clean and filter
        cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

        return cleaned_sentences
    
    def _process_in_chunks(self, text, chunk_size=300):
        """
        Process long text in chunks to avoid token limits.
        Uses sentence-aware chunking to maintain context.

        Args:
            text (str): Long text to process
            chunk_size (int): Number of words per chunk
        
        Returns:
            str: Fully corrected text
        """
        # First, try to split into sentences (even rough ones)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If no sentence boundaries, fall back to word chunking
        if len(sentences) == 1:
            words = text.split()
            if len(words) <= chunk_size:
                return self._correct_text(text)
            
            logger.info(f'Processing {len(words)} words in chunks of {chunk_size}...')
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                corrected_chunk = self._correct_text(chunk)
                chunks.append(corrected_chunk)
                
                if i > 0 and i % (chunk_size * 5) == 0:
                    logger.info(f'Processed {i}/{len(words)} words...')
            
            return ' '.join(chunks)
        
        # Sentence-aware chunking
        logger.info(f'Processing {len(sentences)} sentences in chunks...')
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, process current chunk
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                corrected = self._correct_text(chunk_text)
                chunks.append(corrected)
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Process remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            corrected = self._correct_text(chunk_text)
            chunks.append(corrected)
        
        logger.info(f'Processed {len(chunks)} chunks')
        return ' '.join(chunks)
    
    def process(self, raw_text):
        """
        Main method to correct transcription text for:
        - Grammar errors
        - Spelling mistakes
        - Punctuation
        - Capitalization

        Then splits into clean sentences.

        Args:
            raw_text (str): Raw transcription text

        Returns:
            list: List of corrected sentences
        """
        if not raw_text or not raw_text.strip():
            return []
        
        logger.info('Correcting grammar, spelling, punctuation, and capitalization...')

        # Process text (with chunking for long texts)
        words = raw_text.split()
        if len(words) > 300:
            corrected_text = self._process_in_chunks(raw_text)
        else:
            corrected_text = self._correct_text(raw_text)

        logger.info(f'Original: {raw_text[:100]}...')
        logger.info(f'Corrected: {corrected_text[:100]}...')

        # Split into sentences
        sentences = self._split_into_sentences(corrected_text)

        logger.info(f'Processed into {len(sentences)} sentences')

        return sentences
    
    def __del__(self):
        """Cleanup: release resources"""
        if hasattr(self, 'model') and self.model is not None:
            self.model.cpu()
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()