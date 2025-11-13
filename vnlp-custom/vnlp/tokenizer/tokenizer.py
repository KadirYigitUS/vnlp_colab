import re
from typing import List

def WordPunctTokenize(text: str) -> List[str]:
    """
    This is a simplified version of NLTK's WordPunctTokenizer that can be found in
    https://github.com/nltk/nltk/blob/90fa546ea600194f2799ee51eaf1b729c128711e/nltk/tokenize/regexp.py
    """

    pattern = r"\w+|[^\w\s]+"
    pattern = getattr(pattern, "pattern", pattern)
    flags = re.UNICODE | re.MULTILINE | re.DOTALL
    regexp = re.compile(pattern, flags)

    return regexp.findall(text)

# def TreebankWordTokenize(text: str) -> List[str]:
#     """Tokenizes text with VNLP’s Treebank rules,
#        but keeps apostrophes (‘ ’ or ') intact when they
#        sit between two alphanumeric characters (e.g. 2’de, John's)."""

#     # 1) Surround all double-quotes (ASCII " or curly “ ”) with spaces
#     text = re.sub(r'([“”"])', r' \1 ', text)

#     # 2) Surround any single-quote (ASCII ' or curly ‘ ’) with spaces
#     #    only if it is NOT between two alphanumeric characters.
#     #    This preserves 2'de, 2’de, John's, John’s.
#     text = re.sub(
#         r"(?<![A-Za-z0-9])(['‘’])|(['‘’])(?![A-Za-z0-9])",
#         lambda m: f" {m.group(0)} ",
#         text
#     )

#     # now apply the rest of VNLP’s TreebankWordTokenizer rules:
#     STARTING_QUOTES = [
#         (re.compile(r"^\""), r"``"),
#         (re.compile(r"(``)"), r" \1 "),
#         (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
#     ]
#     PUNCTUATION = [
#         (re.compile(r"([:,])([^\d])"), r" \1 \2"),
#         (re.compile(r"([:,])$"), r" \1 "),
#         (re.compile(r"\.\.\."), r" ... "),
#         (re.compile(r"[;@#$%&]"), r" \g<0> "),
#         (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r"\1 \2\3 "),
#         (re.compile(r"[?!]"), r" \g<0> "),
#         (re.compile(r"([^'])' "), r"\1 ' "),
#     ]
#     PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")
#     DOUBLE_DASHES = (re.compile(r"--"), r" -- ")
#     ENDING_QUOTES = [
#         (re.compile(r"''"), " '' "),
#         (re.compile(r'"'), " '' "),
#         (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
#         (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
#     ]

#     # apply them in order
#     for regexp, sub in STARTING_QUOTES:
#         text = regexp.sub(sub, text)
#     for regexp, sub in PUNCTUATION:
#         text = regexp.sub(sub, text)
#     regexp, sub = PARENS_BRACKETS
#     text = regexp.sub(sub, text)
#     regexp, sub = DOUBLE_DASHES
#     text = regexp.sub(sub, text)

#     text = " " + text + " "
#     for regexp, sub in ENDING_QUOTES:
#         text = regexp.sub(sub, text)

#     # collapse any multiple spaces and split
#     tokens = re.sub(r"\s+", " ", text).strip().split()
#     # normalize curly ’ → straight '
#     return [t if t == "’" else t.replace("’", "'") for t in tokens]

def TreebankWordTokenize(text: str) -> List[str]:
    """
    Tokenize input text into words/punctuation, preserving hyphens and apostrophes inside words.
    """
    # Regex explanation:
    #   [\w-]+   = one or more word characters or hyphens
    #   (?:[’']\w+)* = optionally followed by segments that start with an apostrophe (’ or ') and more word chars
    #   |\S      = or match any non-whitespace character (punctuation) as a separate token
    tokens = re.findall(r"\b\w+(?:[-’/']\w+)+\b|\w+|[^\w\s]", text)
    # Replace unicode apostrophe with ASCII apostrophe for consistency (if any)
    tokens = [t if t == "’" else t.replace("’", "'") for t in tokens]
    # IMPORTANT:
    # the following filtering is to remove tokens that are not useful for counting ngrams
    # when tokenization is for parsing or analysis, we want to keep all tokens
    # It removes tokens that have the length of 1 and are punctuation
    # tokens = [t for t in tokens if not (len(t) == 1 and t.isalnum()==False)]
    return tokens