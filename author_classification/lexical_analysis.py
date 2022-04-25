from collections import Counter
from nltk.tokenize import word_tokenize

NUM_TOP_WORDS = 50

def most_common_words(txt, n=NUM_TOP_WORDS):
    return Counter(word_tokenize(txt)).most_common(n)

