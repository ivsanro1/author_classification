import re
from typing import Tuple
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
STOPWORDS_EN = set(stopwords.words('english'))

# For lemmatization
nlp = spacy.load("en_core_web_sm", disable=['parser','ner'])

# Dictionary of English Contractions
DICT_CONTRACTIONS = {"ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}


# Regular expression for finding contractions
REGEX_CONTRACTIONS = re.compile('(%s)' % '|'.join(DICT_CONTRACTIONS.keys()))

CHARS_QUOTE = '''«»‹›„“‟”’"❝❞❮❯⹂〝〞〟＂'''
REGEX_ALL_QUOTES = '|'.join([re.escape(c) for c in CHARS_QUOTE])

CHARS_APOSTROPHE = '''’‘‛❛❜❟'''
REGEX_ALL_APOSTROPHES = '|'.join([re.escape(c) for c in CHARS_APOSTROPHE])

CHARS_PARENTHESIS = '''()[]'''
CHARS_HYPHEN = '-–—−‧'
CHARS_EXCLAM_QUESTI = '?!'
CHARS_PUNCT_STOP = '.'
CHARS_PUNCT_PAUSES = ',;:'
CHARS_UNDERSCORE = '_'
CHARS_SPECIAL = '£&{}$*'

CHARS_ALL = CHARS_QUOTE + CHARS_APOSTROPHE + CHARS_PARENTHESIS + CHARS_HYPHEN + CHARS_EXCLAM_QUESTI + CHARS_PUNCT_STOP + CHARS_PUNCT_PAUSES + CHARS_UNDERSCORE + CHARS_SPECIAL
REGEX_ALL_CHARS_SUB = '|'.join([re.escape(c) for c in CHARS_ALL])

# Function for expanding contractions (borrowed)
def expand_contractions(txt:str) -> Tuple(str, int):
    '''
    Expands the different contractions defined by `DICT_CONTRACTIONS` in the text. For example, "mustn't've" in the `txt` would become "must not have"

    Returns:
        Tuple(str, int): A `Tuple`. The first element of the tuple is a `str` containing the text with replaced
        contractions and the second element of the tuple is an `int` indicating the number of replacements made.
    '''
    num_replacements = 0
    def replace(match):
        nonlocal num_replacements
        num_replacements += 1
        return DICT_CONTRACTIONS[match.group(0)]
    return (REGEX_CONTRACTIONS.sub(replace, txt), num_replacements)


def normalize_spaces(txt:str) -> str:
    '''Returns the text with the different kinds of spaces (newline, tab, space,
       nonbreaking space, etc.) converted to one space'''
    return re.sub('(\s|\t|\n)+', ' ', txt)


def normalize_quotes(txt:str) -> str:
    '''Returns the text with the different kinds of quotes
    (`« » ‹ › “ ‟ ” " ❝ ❞ ❮ ❯ ⹂ 〝 〞 〟 ＂`) converted to normal quote (`"`)'''
    return re.sub(REGEX_ALL_QUOTES, '"', txt)


def normalize_apostrophes(txt:str) -> str:
    '''Returns the text with the different kinds of apostrophes
    (`’ ‘ ‛ ❛ ❜ ❟`) converted to normal apostrophe (`'`)'''
    return re.sub(REGEX_ALL_APOSTROPHES, "'", txt)


def remove_punct(txt:str) -> str:
    '''Returns the text with the characters defined by `REGEX_ALL_CHARS_SUB` substituted by one space'''
    return re.sub(REGEX_ALL_CHARS_SUB, ' ', txt)


def remove_single_quotes(txt:str) -> str:
    '''
    Remove single quote in a not generic way, to avoid removing contractions (e.g. they've)
        https://regex101.com/r/oGSErP/1
    '''
    return re.sub('''(?<!\w)'|'(?!\w)''', '', txt)
    

def remove_stopwords(txt:str) -> str:
    return ' '.join([w for w in word_tokenize(txt) if w.lower() not in STOPWORDS_EN])


def preprocess_text(txt:str, lemmatize:bool=False) -> str:
    '''
    Preprocesses the text applying a composed function that does:
    - Quote normalization
    - Apostrophe normalization
    - Contraction expansion
    - Lemmatize (if `lemmatize==True`)
    - 
    '''
    for fn in [
        normalize_quotes,
        normalize_apostrophes,
        lambda txt: expand_contractions(txt)[0],
        (lambda txt: txt) if not lemmatize else lemmatize_text, # if lemmatize is False, use identity function, otherwise lemmatize at this point
        remove_punct,
        normalize_spaces,
        remove_single_quotes,
        lambda txt: txt.lower(),
        remove_stopwords,
        lambda txt: re.sub(" ' ", '', txt) # For certain texts, after processing there might be some residual apostrophes that should be removed
    ]:
        txt = fn(txt)
    return txt




def lemmatize_text(txt:str) -> str:
    return ' '.join([tok.lemma_ for tok in nlp(txt)])