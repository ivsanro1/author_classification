import pandas as pd
from functools import partial

import textstat
from textblob import TextBlob
from tqdm import tqdm
from typing import Dict
import numpy as np

from .preprocessing import *

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown', quiet=True)

# Text features
def char_freq(text:str, charset:str) -> int:
    '''Returns the number of characters in text that are in the specified `charset`'''
    return len([c for c in text if c in charset])


def char_freq_ratio(text:str, charset:str) -> float:
    '''Returns the ratio `n_chars_charset / len_text_in_chars`.
    This is the ratio of characters in text that are in the specified `charset`'''
    return char_freq(text, charset) / len(text)


def get_text_punctuation_stat_features(txt:str) -> Dict:
    '''Returns a dictionary whose keys are all the extracted features' names and the values being the feature values'''
    list_funcs = []
    list_feature_names = []
    
    for fname, fn in [
            ('char_freq', char_freq),
            ('char_freq_ratio', char_freq_ratio),
        ]:
        for charset_name, charset in [
            ('quote', CHARS_QUOTE),
            ('apostrophe', CHARS_APOSTROPHE),
            ('parenthesis', CHARS_PARENTHESIS),
            ('hyphen', CHARS_HYPHEN),
            ('exclam_questi', CHARS_EXCLAM_QUESTI),
            ('punct_stop', CHARS_PUNCT_STOP),
            ('punct_pauses', CHARS_PUNCT_PAUSES),
            ('underscore', CHARS_UNDERSCORE),
            ('special', CHARS_SPECIAL)
        ]:
            list_funcs.append(partial(fn, charset=charset))
            list_feature_names.append(f'{fname}_{charset_name}')
            
            
    list_funcs.append(lambda txt: expand_contractions(txt)[1]) # Count contractions
    list_feature_names.append('num_expanded_contractions')

    return {list_feature_names[i]:list_funcs[i](txt) for i in range(len(list_feature_names))}


def set_text_polarity_subjectivity(df:pd.DataFrame, colname_text:str='text') -> pd.DataFrame:
    '''
    Sets the columns in df of the sentiment analysis statistics. This should always be computed on unprocessed text.
    '''
    df['sentiment'] = df[colname_text].progress_apply(lambda txt: TextBlob(txt).sentiment)
    df['polarity'] = df['sentiment'].apply(lambda sentiment: sentiment.polarity)
    df['subjectivity'] = df['sentiment'].apply(lambda sentiment: sentiment.polarity)
    return df.drop('sentiment', axis=1)


LIST_TUPLES_READABILITY_STATISTICS = [
        ('flesch_reading_ease', textstat.flesch_reading_ease),
        ('flesch_kincaid_grade', textstat.flesch_kincaid_grade),
        ('gunning_fog', textstat.gunning_fog),
        ('automated_readability_index', textstat.automated_readability_index),
        ('coleman_liau_index', textstat.coleman_liau_index),
        ('linsear_write_formula', textstat.linsear_write_formula),
        ('dale_chall_readability_score', textstat.dale_chall_readability_score),
        ('mcalpine_eflaw', textstat.mcalpine_eflaw),
        ('reading_time', textstat.reading_time),
        ('syllable_count', textstat.syllable_count),
        ('lexicon_count', textstat.lexicon_count),
        ('char_count', textstat.char_count),
        ('letter_count', textstat.letter_count),
        ('polysyllabcount', textstat.polysyllabcount),
        ('monosyllabcount', textstat.monosyllabcount)
    ]

def set_text_readability_statistics(df:pd.DataFrame, colname_text:str='text') -> pd.DataFrame:
    '''
    Sets the columns in df of the readability statistics. This should always be computed on unprocessed text.
    '''
    for new_colname, fn in tqdm(LIST_TUPLES_READABILITY_STATISTICS):
        df[new_colname] = df[colname_text].apply(fn)
    return df



def get_feature_matrix_from_dataframe_with_feature_columns(df:pd.DataFrame) -> np.ndarray:
    '''
    Gets the matrix of features `num_samples x num_features` given a dataframe that contains all the expected extracted features for this use case.
    '''
    return df[['char_freq_quote',
       'char_freq_apostrophe', 'char_freq_parenthesis', 'char_freq_hyphen',
       'char_freq_exclam_questi', 'char_freq_punct_stop',
       'char_freq_punct_pauses', 'char_freq_underscore', 'char_freq_special',
       'char_freq_ratio_quote', 'char_freq_ratio_apostrophe',
       'char_freq_ratio_parenthesis', 'char_freq_ratio_hyphen',
       'char_freq_ratio_exclam_questi', 'char_freq_ratio_punct_stop',
       'char_freq_ratio_punct_pauses', 'char_freq_ratio_underscore',
       'char_freq_ratio_special', 'num_expanded_contractions',
       'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
       'automated_readability_index', 'coleman_liau_index',
       'linsear_write_formula', 'dale_chall_readability_score',
       'mcalpine_eflaw', 'reading_time', 'syllable_count', 'lexicon_count',
       'char_count', 'letter_count', 'polysyllabcount', 'monosyllabcount',
       'polarity', 'subjectivity']].to_numpy()