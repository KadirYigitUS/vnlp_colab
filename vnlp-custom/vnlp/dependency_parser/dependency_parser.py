# File: /home/ben/miniconda3/envs/bookanalysis/lib/python3.12/site-packages/vnlp/dependency_parser/dependency_parser.py

from typing import List, Tuple
import logging

from .spu_context_dp import SPUContextDP
from .treestack_dp import TreeStackDP

# --- Singleton Cache and Factory for Performance ---
_MODEL_INSTANCE_CACHE = {}

def get_parser_instance(model_name: str, evaluate: bool):
    """
    Singleton factory that ensures a heavy model is loaded only ONCE.
    """
    cache_key = (model_name, evaluate)

    if cache_key not in _MODEL_INSTANCE_CACHE:
        logging.info(f"Initializing and caching new model instance: {cache_key}. This may take a moment...")
        
        if model_name == 'SPUContextDP':
            instance = SPUContextDP(evaluate)
        elif model_name == 'TreeStackDP':
            instance = TreeStackDP(evaluate)
        else:
            raise ValueError(f'{model_name} is not a valid model.')

        _MODEL_INSTANCE_CACHE[cache_key] = instance
        logging.info(f"Model instance {cache_key} cached successfully.")

    return _MODEL_INSTANCE_CACHE[cache_key]


class DependencyParser:
    """
    Main API class using a Singleton pattern for instantaneous re-initialization.
    Available models: ['SPUContextDP', 'TreeStackDP']
    """

    def __init__(self, model: str = 'SPUContextDP', evaluate: bool = False):
        self.models = ['SPUContextDP', 'TreeStackDP']
        
        if model not in self.models:
            raise ValueError(f'{model} is not a valid model. Try one of {self.models}')

        # This call is now extremely fast after the first time.
        self.instance = get_parser_instance(model_name=model, evaluate=evaluate)

    def predict(self, sentence: str, displacy_format: bool = False, pos_result: List[Tuple[str, str]] = None) -> List[Tuple[int, str, int, str]]:
        """High level user API for Dependency Parsing."""
        return self.instance.predict(sentence, displacy_format, pos_result)

    def __getattr__(self, name):
        """Delegates attribute access to the underlying model instance."""
        return getattr(self.instance, name)