from transformers import pipeline

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
            {"role": "user", "content": f"Add punctuation and capitalization to the following sentence without changing its content: '{text}'"},
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


