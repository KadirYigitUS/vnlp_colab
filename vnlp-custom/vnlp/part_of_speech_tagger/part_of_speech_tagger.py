from typing import List, Tuple
import logging

from .spu_context_pos import SPUContextPoS
from .treestack_pos import TreeStackPoS

# --- Singleton Cache and Factory ---
# This global dictionary will act as a cache to store the heavy model instances
# once they are loaded. This is the core of the Singleton pattern.
_MODEL_INSTANCE_CACHE = {}


def get_tagger_instance(model_name: str, evaluate: bool, stemmer_analyzer=None):
    """
    Singleton factory function for all PoSTagger models.

    Ensures that any given model configuration is initialized only ONCE.
    Subsequent requests for the same configuration return the cached instance
    instantly, avoiding slow reloading of weights and TF graph compilation.

    Args:
        model_name (str): The name of the model ('SPUContextPoS' or 'TreeStackPoS').
        evaluate (bool): Determines which weights to load (prod vs eval).
        stemmer_analyzer: An optional dependency for the TreeStackPoS model.

    Returns:
        The loaded and ready-to-use model instance.
    """
    # Create a unique key for the cache based on the model and its mode.
    # Note: We don't include stemmer_analyzer in the key, assuming it's a stateless utility.
    cache_key = (model_name, evaluate)

    if cache_key not in _MODEL_INSTANCE_CACHE:
        logging.info(f"Initializing and caching new PoS model instance: {cache_key}. This may take a moment...")

        # Instantiate the correct underlying model class.
        if model_name == 'SPUContextPoS':
            instance = SPUContextPoS(evaluate)
        elif model_name == 'TreeStackPoS':
            instance = TreeStackPoS(evaluate, stemmer_analyzer)
        else:
            raise ValueError(f'{model_name} is not a valid model.')

        # Store the newly created, expensive-to-load instance in our cache.
        _MODEL_INSTANCE_CACHE[cache_key] = instance
        logging.info(f"Model instance {cache_key} cached successfully.")

    return _MODEL_INSTANCE_CACHE[cache_key]


class PoSTagger:
    """
    Main API class for Part of Speech Tagger implementations.

    This class uses a Singleton pattern to ensure that heavy models are loaded
    into memory only once, making subsequent initializations instantaneous.

    Available models: ['SPUContextPoS', 'TreeStackPoS']

    In order to evaluate, initialize the class with "evaluate = True" argument.
    This will load the model weights that are not trained on test sets.
    """

    def __init__(self, model: str = 'SPUContextPoS', evaluate: bool = False, *args):
        self.models = ['SPUContextPoS', 'TreeStackPoS']
        self.evaluate = evaluate

        if model not in self.models:
            raise ValueError(f'{model} is not a valid model. Try one of {self.models}')

        # Handle the optional stemmer_analyzer argument for TreeStackPoS
        stemmer_analyzer = args[0] if model == 'TreeStackPoS' and args else None

        # The key change: Call the factory function to get the cached instance.
        self.instance = get_tagger_instance(model, evaluate, stemmer_analyzer)

    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        """
        High level user API for Part of Speech Tagging.

        Args:
            sentence: Input text(sentence).

        Returns:
             List of (token, pos_label).

        Example::

            from vnlp import PoSTagger

            # First initialization is slow.
            pos = PoSTagger()
            # Second initialization is instant.
            pos2 = PoSTagger()

            pos.predict("Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım.")

            [('Vapurla', 'NOUN'), ("Beşiktaş'a", 'PROPN'), ('geçip', 'VERB'),
             ('yürüyerek', 'ADV'), ('Maçka', 'PROPN'), ("Parkı'na", 'PROPN'),
             ('ulaştım', 'VERB'), ('.', 'PUNCT')]
        """
        return self.instance.predict(sentence)

    def __getattr__(self, name):
        """
        Delegates attribute access to the underlying model instance.
        """
        return getattr(self.instance, name)