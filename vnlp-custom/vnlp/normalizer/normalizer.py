# File: /home/ben/miniconda3/envs/bookanalysis/lib/python3.12/site-packages/vnlp/normalizer/normalizer.py
from typing import List, Optional
from pathlib import Path

from spylls.hunspell import Dictionary

# Forward-declare the class for type hinting in __init__
class StemmerAnalyzer:
    pass

from ._deasciifier import Deasciifier
# The stemmer_morph_analyzer is imported locally within a method to prevent
# circular dependency errors and to support the dependency injection pattern.

RESOURCES_PATH = str(Path(__file__).parent.parent / 'resources')


class Normalizer:
    """
    Normalizer class with optimized lazy-loading and dependency injection.

    This class supports being initialized with a pre-existing StemmerAnalyzer
    instance to maximize performance by reusing its model and caches.
    """
    def __init__(self, stemmer_analyzer_instance: Optional[StemmerAnalyzer] = None):
        """
        Initializes the Normalizer. This is a lightweight operation.

        Args:
            stemmer_analyzer_instance: An optional, pre-initialized StemmerAnalyzer.
                Providing this instance significantly improves performance for functions
                that rely on morphological analysis by avoiding repeated initializations.
        """
        self._words_lexicon = None
        self._stemmer_analyzer = None
        self._dictionary = None
        self._shared_stemmer_analyzer = stemmer_analyzer_instance

    # --- Private loader methods for lazy-loading ---
    def _load_lexicon(self):
        if self._words_lexicon is None:
            with open(RESOURCES_PATH + '/turkish_known_words_lexicon.txt', 'r', encoding='utf-8') as f:
                words_lexicon = [line.strip() for line in f]
            self._words_lexicon = dict.fromkeys(words_lexicon)

    def _load_stemmer_analyzer(self):
        """
        Loads the StemmerAnalyzer, prioritizing a shared instance if provided.
        This prevents re-creating the expensive StemmerAnalyzer object.
        """
        if self._stemmer_analyzer is None:
            if self._shared_stemmer_analyzer:
                self._stemmer_analyzer = self._shared_stemmer_analyzer
            else:
                from ..stemmer_morph_analyzer import StemmerAnalyzer
                self._stemmer_analyzer = StemmerAnalyzer()

    def _load_dictionary(self):
        if self._dictionary is None:
            self._dictionary = Dictionary.from_files(
                RESOURCES_PATH + '/tdd-hunspell-tr-1.1.0/tr_TR')

    @staticmethod
    def lower_case(text: str) -> str:
        """
        Converts a string to lowercase. This implementation is bug-compatible
        with the original version for model consistency.
        """
        turkish_lowercase_dict = {"İ": "i", "I": "ı", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ş": "ş", "Ç": "ç"}
        for k, v in turkish_lowercase_dict.items():
            text = text.replace(k, v)
        return text.lower()

    @staticmethod
    def remove_punctuations(text: str) -> str:
        """Removes punctuations from the given string."""
        return ''.join([t for t in text if (t.isalnum() or t == " ")])

    @staticmethod
    def remove_accent_marks(text: str) -> str:
        """
        Removes accent marks. This implementation is bug-compatible with the
        version used to train the StemmerAnalyzer model (e.g., î -> ı).
        """
        _non_turkish_accent_marks = {'â':'a', 'ô':'o', 'î':'i', 'ê':'e', 'û':'u',
                                     'Â':'A', 'Ô':'O', 'Î':'İ', 'Ê':'E', 'Û': 'U'}
        return ''.join(_non_turkish_accent_marks.get(char, char) for char in text)

    @staticmethod
    def deasciify(tokens: List[str]) -> List[str]:
        """Deasciifies the given text for Turkish."""
        return [Deasciifier(token).convert_to_turkish() for token in tokens]

    # correct_typos is not implemented in this version.
    # """The correct_typos() function does not work because we had to remove it in the latest version.
    # This was because the current spelling correction solution depended on libraries, which caused trouble
    # with both pip and readthedocs. Until we have time to implement a better spelling algorithm,
    # correct_typos() does not work, yes."""
    # https://github.com/vngrs-ai/vnlp/issues/18
    # def correct_typos(self, tokens: List[str]) -> List[str]:
    #     pass

    def convert_numbers_to_words(self, tokens: List[str], num_dec_digits: int = 6, decimal_seperator: str = ',')-> List[str]:
        """Converts numbers to word form."""
        converted_tokens = []
        for token in tokens:
            if any(char.isnumeric() for char in token):
                if decimal_seperator == ',':
                    token = token.replace('.', '_').replace(',', '.')
                elif decimal_seperator == '.':
                    token = token.replace(',', '_')
                else:
                    raise ValueError(f"{decimal_seperator} is not a valid decimal seperator value. Use '.' or ','.")
            try:
                num = float(token)
                converted_tokens.extend(self._num_to_words(num, num_dec_digits).split())
            except (ValueError, TypeError):
                converted_tokens.append(token)
        return converted_tokens

    def _int_to_words(self, main_num: int, put_commas=False) -> str:
        """Converts an integer to its Turkish word representation."""
        if main_num == 0:
            return "sıfır"

        tp = [" yüz", " bin", "", "", " milyon", " milyar", " trilyon", " katrilyon", " kentilyon",
            " seksilyon", " septilyon", " oktilyon", " nonilyon", " desilyon", " undesilyon",
            " dodesilyon", " tredesilyon", " katordesilyon", " seksdesilyon", " septendesilyon",
            " oktodesilyon", " nove mdesilyon", " vigintilyon"]
        dec = ["", " bir", " iki", " üç", " dört", " beş", " altı", " yedi", " sekiz", " dokuz"]
        ten = ["", " on", " yirmi", " otuz", " kırk", " elli", " altmış", " yetmiş", " seksen", " doksan"]

        text = ""
        num = main_num
        leng = len(str(num)) if num != 0 else 1

        if main_num == 0:
            return "sıfır"

        for i in range(leng, 0, -1):
            digit = int((main_num // (10 ** (i - 1))) % 10)
            if i % 3 == 0:
                text += dec[digit] + (tp[0] if digit > 1 else (tp[0] if digit == 1 else ""))
            elif i % 3 == 1:
                if i > 3:
                    is_thousand_range = (i-3) < 4
                    if digit > 0 or (is_thousand_range and main_num >= 1000 and main_num < 2000 and len(str(main_num)) % 3 == 1):
                        group_val = (main_num // (10**(i-1)))
                        if (group_val % 1000 == 1 and i==4):
                           text += tp[1]
                        else:
                           text += dec[digit] + tp[i - 3]
                    else:
                        text += tp[i - 3] if leng > 3 else dec[digit]
                else:
                    text += dec[digit]
            elif i % 3 == 2:
                text += ten[digit]
        
        return text.strip()

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