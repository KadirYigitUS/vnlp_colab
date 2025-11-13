# coding=utf-8
#
# Copyright 2025 VNLP Project Authors.
#
# Licensed under the GNU Affero General Public License, Version 3.0 (the "License");
# ... (license text) ...
"""
Tokenization utilities for VNLP Colab, providing Treebank and WordPunct tokenizers.
"""
import re
from typing import List

def WordPunctTokenize(text: str) -> List[str]:
    """
    Splits text into a sequence of alphabetic and non-alphabetic characters.
    A simplified version of NLTK's WordPunctTokenizer.
    """
    pattern = r"\w+|[^\w\s]+"
    return re.findall(pattern, text, flags=re.UNICODE | re.MULTILINE | re.DOTALL)

def TreebankWordTokenize(text: str) -> List[str]:
    """
    Tokenizes text with rules that preserve hyphens and apostrophes inside words.
    """
    # This regex handles words with internal hyphens/apostrophes, plain words, and punctuation.
    tokens = re.findall(r"\b\w+(?:[-’/']\w+)+\b|\w+|[^\w\s]", text)
    # Normalize unicode apostrophe to ASCII for consistency
    return [t.replace("’", "'") for t in tokens]