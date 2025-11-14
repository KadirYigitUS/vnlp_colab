# vnlp_colab/stemmer/_yildiz_analyzer.py
# ---
# This file has been refactored to use importlib.resources for robust
# access to package data, resolving the FileNotFoundError.

# -*- coding: utf-8 -*-
import re
import logging
from collections import namedtuple

# --- MODIFIED: Use the robust resource locator from utils_colab ---
from vnlp_colab.utils_colab import get_resource_path

# --- REMOVED: The os-based path is no longer needed ---
# import os
# resources_path = os.path.join(os.path.dirname(__file__), "resources/")

_GENERATOR_INSTANCE_CACHE = {}

def get_candidate_generator_instance(case_sensitive=True, asciification=False, suffix_normalization=False):
    """Singleton factory for TurkishStemSuffixCandidateGenerator."""
    cache_key = (case_sensitive, asciification, suffix_normalization)
    if cache_key not in _GENERATOR_INSTANCE_CACHE:
        instance = TurkishStemSuffixCandidateGenerator(case_sensitive, asciification, suffix_normalization)
        _GENERATOR_INSTANCE_CACHE[cache_key] = instance
    return _GENERATOR_INSTANCE_CACHE[cache_key]

class TurkishStemSuffixCandidateGenerator(object):
    """
    Generates morphological analysis candidates for Turkish words.
    This version uses a Singleton pattern for initialization and memoization
    for the analysis function to significantly improve performance.
    """
    ROOT_TRANSFORMATION_MAP = {"tıp": "tıb", "prof.": "profesör", "dr.": "doktor", "yi": "ye", "ed": "et", "di": "de"}
    TAG_FLAG_MAP = {0: "Adj", 1: "Adverb", 2: "Conj", 3: "Det", 4: "Dup", 5: "Interj", 6: "Noun", 7: "Postp", 8: "Pron", 9: "Ques", 10: "Verb", 11: "Num", 12: "Noun+Prop"}
    
    # --- MODIFIED: Use get_resource_path to find files within the package ---
    _RESOURCE_PKG_PATH = "vnlp_colab.stemmer.resources"
    SUFFIX_DICT_FILE_PATH = get_resource_path(_RESOURCE_PKG_PATH, "Suffixes&Tags.txt")
    STEM_LIST_FILE_PATH = get_resource_path(_RESOURCE_PKG_PATH, "StemListWithFlags_v2.txt")
    EXACT_LOOKUP_TABLE_FILE_PATH = get_resource_path(_RESOURCE_PKG_PATH, "ExactLookup.txt")
    
    CONSONANT_STR = "[bcdfgğhjklmnprsştvyzxwqBCDFGĞHJKLMNPRSŞTVYZXWQ]"
    VOWEL_STR = "[aeıioöuüAEIİOÖUÜ]"
    NARROW_VOWELS_STR = "[uüıiUÜIİ]"
    STARTS_WITH_UPPER = re.compile(r"^[ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜQVYXZ].*$")
    ENDS_WITH_SOFT_CONSONANTS_REGEX = re.compile(r"^.*[bcdğBCDĞgG]$")
    SUFFIX_TRANSFORMATION_REGEX1 = re.compile(r"[ae]")
    SUFFIX_TRANSFORMATION_REGEX2 = re.compile(r"[ıiuü]")
    ENDS_TWO_CONSONANT_REG = re.compile(r"^.*{}{}$".format(CONSONANT_STR, CONSONANT_STR))
    STARTS_VOWEL_REGEX = re.compile(r"^{}.*$".format(VOWEL_STR))
    ENDS_NARROW_REGEX = re.compile(r"^.*{}$".format(NARROW_VOWELS_STR))
    TAG_SEPARATOR_REGEX = re.compile(r"[\+\^]")
    NON_WORD_REGEX = re.compile(r"^[^A-Za-zışğüçöÜĞİŞÇÖ]+$")
    CONTAINS_NUMBER_REGEX = re.compile(r"^.*[0-9].*$")

    def __init__(self, case_sensitive=True, asciification=False, suffix_normalization=False):
        self.case_sensitive = case_sensitive
        self.asciification = asciification
        self.suffix_normalization = suffix_normalization
        self.suffix_dic = self._read_suffix_dic()
        self.stem_dic = self._read_stem_list()
        self.exact_lookup_table = self._read_exact_lookup_table()
        self.analysis_cache = {}

    def _read_exact_lookup_table(self):
        table = {}
        with open(self.EXACT_LOOKUP_TABLE_FILE_PATH, "r", encoding="UTF-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                table[splits[0]] = splits[1].split(" ")
        return table

    def _read_suffix_dic(self):
        dic = {}
        with open(self.SUFFIX_DICT_FILE_PATH, "r", encoding="UTF-8") as f:
            for line in f:
                suffix, tag = line.strip().split("\t")
                if suffix not in dic:
                    dic[suffix] = []
                dic[suffix].append(tag)
        return dic

    def _read_stem_list(self):
        dic = {}
        with open(self.STEM_LIST_FILE_PATH, "r", encoding="UTF-8") as f:
            for line in f:
                stem, flag_str = line.strip().split("\t")
                if not self.case_sensitive:
                    stem = to_lower(stem)
                postags = self._parse_flag(int(flag_str.strip()))
                if stem in dic:
                    dic[stem].extend(p for p in postags if p not in dic[stem])
                else:
                    dic[stem] = postags
        return dic

    @staticmethod
    def _parse_flag(flag):
        res = []
        for i in range(12, -1, -1):
            power_of_2 = 1 << i
            if flag >= power_of_2:
                res.append(TurkishStemSuffixCandidateGenerator.TAG_FLAG_MAP[i])
                flag -= power_of_2
        if flag != 0:
            raise IOError("Error: problems in stem flags!")
        return res

    @staticmethod
    def _transform_soft_consonants(text):
        replacements = [
            (r"^(.*)b$", r"\1p"), (r"^(.*)B$", r"\1P"), (r"^(.*)c$", r"\1ç"),
            (r"^(.*)C$", r"\1Ç"), (r"^(.*)d$", r"\1t"),
            (r"^(.*)D$", r"\1T"), (r"^(.*)ğ$", r"\1k"),
            (r"^(.*)Ğ$", r"\1K"), (r"^(.*)g$", r"\1k"),
            (r"^(.*)G$", r"\1K")
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        return text

    @staticmethod
    def _root_transform(candidate_roots):
        for i, root in enumerate(candidate_roots):
            if root in TurkishStemSuffixCandidateGenerator.ROOT_TRANSFORMATION_MAP:
                candidate_roots[i] = TurkishStemSuffixCandidateGenerator.ROOT_TRANSFORMATION_MAP[root]

    @classmethod
    def suffix_transform(cls, candidate_suffixes):
        for i, suffix in enumerate(candidate_suffixes):
            candidate_suffixes[i] = cls.suffix_transform_single(suffix)

    @classmethod
    def suffix_transform_single(cls, candidate_suffix):
        candidate_suffix = to_lower(candidate_suffix)
        candidate_suffix = cls.SUFFIX_TRANSFORMATION_REGEX1.sub("A", candidate_suffix)
        candidate_suffix = cls.SUFFIX_TRANSFORMATION_REGEX2.sub("H", candidate_suffix)
        return candidate_suffix

    @staticmethod
    def _add_candidate_stem_suffix(stem_candidate, suffix_candidate, candidate_roots, candidate_suffixes):
        # This is a complex rule-based method that remains unchanged from the original.
        if "'" in suffix_candidate: candidate_roots.append(stem_candidate); candidate_suffixes.append(suffix_candidate); return
        if stem_candidate == "ban" and suffix_candidate == "a": candidate_roots.append("ben"); candidate_suffixes.append("a")
        elif stem_candidate == "Ban" and suffix_candidate == "a": candidate_roots.append("Ben"); candidate_suffixes.append("a")
        elif stem_candidate == "san" and suffix_candidate == "a": candidate_roots.append("sen"); candidate_suffixes.append("a")
        elif stem_candidate == "San" and suffix_candidate == "a": candidate_roots.append("Sen"); candidate_suffixes.append("a")
        else:
            candidate_roots.append(stem_candidate); candidate_suffixes.append(suffix_candidate)
            if len(stem_candidate) > 2 and len(suffix_candidate) > 0 and stem_candidate[-1] == suffix_candidate[0] and stem_candidate[-1] in TurkishStemSuffixCandidateGenerator.CONSONANT_STR: candidate_roots.append(stem_candidate); candidate_suffixes.append(suffix_candidate[1:])
            elif len(stem_candidate) > 1 and TurkishStemSuffixCandidateGenerator.ENDS_NARROW_REGEX.match(stem_candidate) and "yor" in suffix_candidate:
                if stem_candidate.endswith("i") or stem_candidate.endswith("ü"): candidate_roots.append(stem_candidate[:-1] + "e"); candidate_suffixes.append(suffix_candidate)
                elif stem_candidate.endswith("ı") or stem_candidate.endswith("u"): candidate_roots.append(stem_candidate[:-1] + "a"); candidate_suffixes.append(suffix_candidate)
            if len(stem_candidate) > 2 and TurkishStemSuffixCandidateGenerator.ENDS_TWO_CONSONANT_REG.match(stem_candidate) and TurkishStemSuffixCandidateGenerator.STARTS_VOWEL_REGEX.match(suffix_candidate):
                suffix_start_letter = to_lower(suffix_candidate[0])
                if suffix_start_letter in ["u", "ü", "ı", "i"]: candidate_roots.append(stem_candidate[:-1] + suffix_start_letter + stem_candidate[-1]); candidate_suffixes.append(suffix_candidate)
                elif suffix_start_letter == "e": candidate_roots.append(stem_candidate[:-1] + "i" + stem_candidate[-1]); candidate_suffixes.append(suffix_candidate); candidate_roots.append(stem_candidate[:-1] + "ü" + stem_candidate[-1]); candidate_suffixes.append(suffix_candidate)
                elif suffix_start_letter == "a": candidate_roots.append(stem_candidate[:-1] + "ı" + stem_candidate[-1]); candidate_suffixes.append(suffix_candidate); candidate_roots.append(stem_candidate[:-1] + "u" + stem_candidate[-1]); candidate_suffixes.append(suffix_candidate)
            if len(stem_candidate) > 2 and TurkishStemSuffixCandidateGenerator.ENDS_WITH_SOFT_CONSONANTS_REGEX.match(stem_candidate): candidate_roots.append(TurkishStemSuffixCandidateGenerator._transform_soft_consonants(stem_candidate)); candidate_suffixes.append(suffix_candidate)

    def get_stem_suffix_candidates(self, surface_word):
        candidate_roots, candidate_suffixes = [], []
        for i in range(1, len(surface_word)):
            candidate_root, candidate_suffix = surface_word[:i], surface_word[i:]
            if not self.case_sensitive:
                self._add_candidate_stem_suffix(to_lower(candidate_root), to_lower(candidate_suffix), candidate_roots, candidate_suffixes)
            else:
                lower_suffix = to_lower(candidate_suffix)
                self._add_candidate_stem_suffix(to_lower(candidate_root), lower_suffix, candidate_roots, candidate_suffixes)
                if self.STARTS_WITH_UPPER.match(candidate_root): self._add_candidate_stem_suffix(capitalize(candidate_root), lower_suffix, candidate_roots, candidate_suffixes)
        candidate_suffixes.append(""), candidate_roots.append(to_lower(surface_word))
        if self.case_sensitive and self.STARTS_WITH_UPPER.match(surface_word): candidate_suffixes.append(""), candidate_roots.append(capitalize(surface_word))
        self._root_transform(candidate_roots)
        if self.asciification: candidate_roots, candidate_suffixes = [asciify(r) for r in candidate_roots], [asciify(s) for s in candidate_suffixes]
        if self.suffix_normalization: self.suffix_transform(candidate_suffixes)
        return candidate_roots, candidate_suffixes

    def get_tags(self, suffix, stem_tags=None):
        tags = self.suffix_dic.get(suffix)
        if not tags:
            if suffix.startswith("'") and suffix[1:] in self.suffix_dic:
                tags = self.suffix_dic[suffix[1:]]
            elif not suffix: # Handles "null" suffix case
                tags = self.suffix_dic["null"]
            else:
                return []
        res = []
        for tag in set(tags):
            tag_sequences = self.TAG_SEPARATOR_REGEX.split(tag)
            first_tag = "+".join(tag_sequences[0:2]) if len(tag_sequences) > 1 and tag_sequences[1] in ["Prop", "Time"] else tag_sequences[0]
            if not stem_tags or first_tag in stem_tags:
                res.append(tag_sequences)
        return res

    def get_analysis_candidates(self, surface_word):
        if surface_word in self.analysis_cache:
            return self.analysis_cache[surface_word]

        surface_word_lower = to_lower(surface_word)
        if surface_word_lower in self.exact_lookup_table:
            analyses = []
            for analysis_str in self.exact_lookup_table[surface_word_lower]:
                root, tags_str = analysis_str.split("/")
                root_part = self.TAG_SEPARATOR_REGEX.split(tags_str)[0]
                tags = self.TAG_SEPARATOR_REGEX.split(tags_str)[1:]
                analyses.append((root_part, root, tags))
            self.analysis_cache[surface_word] = analyses
            return analyses

        candidate_analyzes, candidate_analyzes_str = [], set()
        candidate_roots, candidate_suffixes = self.get_stem_suffix_candidates(surface_word)

        for c_root, c_suffix in zip(candidate_roots, candidate_suffixes):
            if self.NON_WORD_REGEX.match(c_root):
                stem_tags = ["Num", "Noun+Time"] if self.CONTAINS_NUMBER_REGEX.match(c_root) else ["Punc"]
            elif c_root not in self.stem_dic:
                if len(c_suffix) == 0: continue
                stem_tags = ["Noun+Prop"] if "'" in c_suffix and c_suffix in self.suffix_dic else []
                if not stem_tags: continue
            else:
                stem_tags = list(self.stem_dic[c_root])
                is_upper = self.STARTS_WITH_UPPER.match(c_root)
                has_prop = "Noun+Prop" in stem_tags
                if not is_upper and has_prop: stem_tags.remove("Noun+Prop")
                elif is_upper and has_prop: stem_tags = ["Noun+Prop"]
                elif c_suffix.startswith("'") and has_prop: stem_tags = ["Noun+Prop"]
                elif is_upper and not has_prop: continue

            for candidate_tag in self.get_tags(c_suffix, stem_tags):
                analysis_str = to_lower(c_root) + "+" + "+".join(candidate_tag).replace("+DB", "^DB")
                if analysis_str not in candidate_analyzes_str:
                    candidate_analyzes.append((to_lower(c_root), c_suffix, candidate_tag))
                    candidate_analyzes_str.add(analysis_str)
        
        if not candidate_analyzes:
            candidate_analyzes.append((surface_word_lower, "", ["Unknown"]))
        
        self.analysis_cache[surface_word] = candidate_analyzes
        return candidate_analyzes

def to_lower(text):
    text = text.replace("İ", "i")
    text = text.replace("I", "ı")
    text = text.replace("Ğ", "ğ")
    text = text.replace("Ü", "ü")
    text = text.replace("Ö", "ö")
    text = text.replace("Ş", "ş")
    text = text.replace("Ç", "ç")
    return text.lower()

def capitalize(text):
    if len(text) > 1:
        return asciify(text[0]).upper() + to_lower(text[1:])
    else:
        return text

def asciify(text):
    text = text.replace("İ", "I")
    text = text.replace("Ç", "C")
    text = text.replace("Ğ", "G")
    text = text.replace("Ü", "U")
    text = text.replace("Ş", "S")
    text = text.replace("Ö", "O")
    text = text.replace("ı", "i")
    text = text.replace("ç", "c")
    text = text.replace("ğ", "g")
    text = text.replace("ü", "u")
    text = text.replace("ş", "s")
    text = text.replace("ö", "o")
    return text