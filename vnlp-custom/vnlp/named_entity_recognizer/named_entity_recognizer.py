from typing import List, Tuple
import logging

from .charner import CharNER
from .spu_context_ner import SPUContextNER

# --- Singleton Cache and Factory ---
_MODEL_INSTANCE_CACHE = {}

def get_ner_instance(model_name: str, evaluate: bool):
    """
    Singleton factory for all NamedEntityRecognizer models.

    Ensures that any given model configuration is initialized only ONCE.
    """
    cache_key = (model_name, evaluate)

    if cache_key not in _MODEL_INSTANCE_CACHE:
        logging.info(f"Initializing and caching new NER model instance: {cache_key}. This may take a moment...")
        
        if model_name == 'SPUContextNER':
            instance = SPUContextNER(evaluate)
        elif model_name == 'CharNER':
            instance = CharNER(evaluate)
        else:
            raise ValueError(f'{model_name} is not a valid model.')
        
        _MODEL_INSTANCE_CACHE[cache_key] = instance
        logging.info(f"Model instance {cache_key} cached successfully.")
        
    return _MODEL_INSTANCE_CACHE[cache_key]


class NamedEntityRecognizer:
    """
    Main API class for Named Entity Recognizer implementations.

    This class uses a Singleton pattern to ensure that heavy models are loaded
    into memory only once, making subsequent initializations instantaneous.

    Available models: ['SPUContextNER', 'CharNER']

    In order to evaluate, initialize the class with "evaluate = True" argument.
    This will load the model weights that are not trained on test sets.
    """

    def __init__(self, model: str = 'SPUContextNER', evaluate: bool = False):
        self.models = ['SPUContextNER', 'CharNER']
        self.evaluate = evaluate

        if model not in self.models:
            raise ValueError(f'{model} is not a valid model. Try one of {self.models}')

        # Call the factory function to get the cached instance.
        self.instance = get_ner_instance(model_name=model, evaluate=evaluate)

    def predict(self, sentence: str, displacy_format: bool = False) -> List[Tuple[str, str]]:
        """
        High level user API for Named Entity Recognition.

        Args:
            sentence: Input sentence/text.
            displacy_format: If True, returns result in spacy.displacy format.

        Returns:
            NER result as pairs of (token, entity).

        Example::

            from vnlp import NamedEntityRecognizer
            ner = NamedEntityRecognizer() # Slow on first call
            ner2 = NamedEntityRecognizer() # Instant on second call
            ner.predict("Benim adım Melikşah, İstanbul'da yaşıyorum.")

            [('Benim', 'O'), ('adım', 'O'), ('Melikşah', 'PER'),
            (',', 'O'), ("İstanbul'da", 'LOC'), ('yaşıyorum', 'O'), ('.', 'O')]
        """
        return self.instance.predict(sentence, displacy_format)