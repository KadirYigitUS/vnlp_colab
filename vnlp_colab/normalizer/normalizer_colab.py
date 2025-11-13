# coding=utf-8
#
# Copyright 2025 VNLP Project Authors.
#
# Licensed under the GNU Affero General Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/agpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Full-featured, stateful Normalizer for VNLP Colab.

This module provides a comprehensive Normalizer class with lazy-loading for heavy
resources (dictionaries, lexicons) and dependency injection for the StemmerAnalyzer.
It is designed for high-performance, repeated use within a pipeline.
"""
import logging
import re
from typing import List, Optional, Dict
from pathlib import Path

# Try to import optional dependency spylls for typo correction
try:
    from spylls.hunspell import Dictionary
except ImportError:
    Dictionary = None

# Updated imports for package structure
from vnlp_colab.normalizer._deasciifier import Deasciifier
from vnlp_colab.utils_colab import download_resource
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer

# Forward-declare StemmerAnalyzer for type hinting to avoid circular import
class StemmerAnalyzer:
    pass

logger = logging.getLogger(__name__)

# --- Resource URLs ---
LEXICON_URL = "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/resources/turkish_known_words_lexicon.txt"
HUNSPELL_AFF_URL = "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/resources/tdd-hunspell-tr-1.1.0/tr_TR.aff"
HUNSPELL_DIC_URL = "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/resources/tdd-hunspell-tr-1.1.0/tr_TR.dic"


class Normalizer:
    """
    Stateful Normalizer class with lazy-loading and dependency injection.
    """
    _LOWERCASE_MAP = { "İ": "i", "I": "ı", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ş": "ş", "Ç": "ç" }
    _ACCENT_MAP = { 'â':'a', 'ô':'o', 'î':'i', 'ê':'e', 'û':'u', 'Â':'A', 'Ô':'O', 'Î':'İ', 'Ê':'E', 'Û': 'U' }
    _PUNCTUATION_CHARS = ".,!?;:'\""

    def __init__(self, stemmer_analyzer_instance: Optional[StemmerAnalyzer] = None):
        self._words_lexicon: Optional[Dict[str, None]] = None
        self._stemmer_analyzer: Optional[StemmerAnalyzer] = None
        self._dictionary: Optional['Dictionary'] = None
        self._shared_stemmer_analyzer = stemmer_analyzer_instance

    # --- Private Lazy Loaders ---
    def _load_lexicon(self):
        if self._words_lexicon is None:
            logger.info("Lazy-loading Turkish known words lexicon...")
            lexicon_path = download_resource("turkish_known_words_lexicon.txt", LEXICON_URL)
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                self._words_lexicon = dict.fromkeys(line.strip() for line in f)

    def _load_dictionary(self):
        if self._dictionary is None:
            if Dictionary is None:
                logger.warning("`spylls` not installed. Typo correction is disabled. Run 'pip install spylls'.")
                return
            logger.info("Lazy-loading Hunspell dictionary...")
            aff_path = download_resource("tr_TR.aff", HUNSPELL_AFF_URL)
            dic_path = download_resource("tr_TR.dic", HUNSPELL_DIC_URL)
            self._dictionary = Dictionary.from_files(str(dic_path.parent / 'tr_TR'))

    def _load_stemmer_analyzer(self):
        if self._stemmer_analyzer is None:
            logger.info("Lazy-loading StemmerAnalyzer for typo correction...")
            if self._shared_stemmer_analyzer:
                self._stemmer_analyzer = self._shared_stemmer_analyzer
            else:
                # Late import to prevent circular dependency
                from stemmer_colab import get_stemmer_analyzer
                self._stemmer_analyzer = get_stemmer_analyzer()

    # --- Static Methods ---
    @staticmethod
    def lower_case(text: str) -> str:
        if not isinstance(text, str): return ""
        for k, v in Normalizer._LOWERCASE_MAP.items():
            text = text.replace(k, v)
        return text.lower()

    @staticmethod
    def remove_accent_marks(text: str) -> str:
        """Removes accent marks. This version is corrected to ensure î -> i."""
        if not isinstance(text, str): return ""
        return "".join(Normalizer._ACCENT_MAP.get(char, char) for char in text)

    @staticmethod
    def remove_punctuations(text: str) -> str:
        if not isinstance(text, str): return ""
        return ''.join([t for t in text if (t.isalnum() or t == " ")])

    @staticmethod
    def deasciify(tokens: List[str]) -> List[str]:
        if not tokens: return []
        return [Deasciifier(token).convert_to_turkish() for token in tokens]

    # --- Stateful Methods ---
    def correct_typos(self, tokens: List[str]) -> List[str]:
        """
        Note: This function was historically removed from VNLP due to packaging
        issues, as noted in GitHub issue #18. This version re-implements it
        using `spylls` and integrates it with other VNLP components.
        """
        self._load_dictionary()
        self._load_stemmer_analyzer()
        self._load_lexicon()

        if not self._dictionary or not self._stemmer_analyzer or not self._words_lexicon:
            logger.warning("Typo correction dependencies not fully loaded. Skipping correction.")
            return tokens

        corrected_tokens = []
        for token in tokens:
            l_punct, core_word, t_punct = self._strip_punctuation(token)
            if not core_word:
                corrected_tokens.append(token)
                continue

            analysis = self._stemmer_analyzer.candidate_generator.get_analysis_candidates(core_word)
            is_valid_by_stemmer = (analysis and analysis[0] and 'Unknown' not in analysis[0][2])
            is_known = (core_word.isnumeric() or
                        self._dictionary.lookup(core_word) or
                        self._dictionary.lookup(core_word.lower()) or
                        core_word in self._words_lexicon or
                        core_word.lower() in self._words_lexicon or
                        is_valid_by_stemmer)

            if is_known:
                corrected_tokens.append(token)
            else:
                suggestions = list(self._dictionary.suggest(core_word))
                corrected_word = suggestions[0] if suggestions else core_word
                corrected_tokens.append(l_punct + corrected_word + t_punct)
                
        return corrected_tokens

    def _strip_punctuation(self, token: str) -> tuple[str, str, str]:
        """Separates leading/trailing punctuation from a core word."""
        leading_punct = ""
        while token and token[0] in self._PUNCTUATION_CHARS:
            leading_punct += token[0]
            token = token[1:]
        trailing_punct = ""
        while token and token[-1] in self._PUNCTUATION_CHARS:
            trailing_punct = token[-1] + trailing_punct
            token = token[:-1]
        return leading_punct, token, trailing_punct

    def convert_numbers_to_words(self, tokens: List[str], num_dec_digits: int = 6, decimal_separator: str = ',') -> List[str]:
        converted_tokens = []
        for token in tokens:
            # Prepare token for float conversion
            processed_token = token
            if any(char.isnumeric() for char in token):
                if decimal_separator == ',':
                    processed_token = processed_token.replace('.', '').replace(',', '.')
                elif decimal_separator == '.':
                    processed_token = processed_token.replace(',', '')
                else:
                    raise ValueError(f"'{decimal_separator}' is not a valid decimal separator. Use '.' or ','.")
            
            # Attempt conversion
            try:
                num = float(processed_token)
                converted_tokens.extend(self._num_to_words(num, num_dec_digits).split())
            except (ValueError, TypeError):
                converted_tokens.append(token) # Append original token if not a number
        return converted_tokens

    def _int_to_words(self, main_num: int) -> str:
        """Converts an integer to its Turkish word representation, with improvements."""
        if main_num == 0:
            return "sıfır"
        if main_num < 0:
            return f"eksi {self._int_to_words(abs(main_num))}"

        tp = ["", " bin", " milyon", " milyar", " trilyon", " katrilyon"]
        dec = ["", " bir", " iki", " üç", " dört", " beş", " altı", " yedi", " sekiz", " dokuz"]
        ten = ["", " on", " yirmi", " otuz", " kırk", " elli", " altmış", " yetmiş", " seksen", " doksan"]
        
        num_str = str(main_num)
        num_len = len(num_str)
        groups = (num_len + 2) // 3
        num_str = num_str.zfill(groups * 3)
        
        text_parts = []
        for i in range(groups):
            group_val = int(num_str[i*3 : (i+1)*3])
            if group_val == 0:
                continue
            
            h = group_val // 100
            t = (group_val % 100) // 10
            u = group_val % 10
            
            group_text = ""
            if h > 0:
                group_text += " yüz" if h == 1 else f"{dec[h]} yüz"
            group_text += f" {ten[t]}" if t > 0 else ""
            group_text += f" {dec[u]}" if u > 0 else ""
            
            # Handle "bir bin" -> "bin"
            if group_text.strip() == "bir" and i == groups - 2:
                 text_parts.append(tp[groups - 1 - i])
            else:
                 text_parts.append(group_text + tp[groups - 1 - i])

        return " ".join(part.strip() for part in text_parts if part).strip()

    def _num_to_words(self, num: float, num_dec_digits: int) -> str:
        """Converts a float to its Turkish word representation."""
        integer_part = int(num)
        
        if abs(num - integer_part) < (10**-num_dec_digits):
            return self._int_to_words(integer_part)

        str_decimal = f"{num:.{num_dec_digits}f}".split('.')[1].rstrip('0')
        if not str_decimal:
            return self._int_to_words(integer_part)

        decimal_part = int(str_decimal)
        return f"{self._int_to_words(integer_part)} virgül {self._int_to_words(decimal_part)}"