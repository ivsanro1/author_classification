import argparse
import pickle
import zipfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tqdm import tqdm

from author_classification.feature_extraction import (
    LIST_TUPLES_READABILITY_STATISTICS,
    get_feature_matrix_from_dataframe_with_feature_columns,
    get_text_punctuation_stat_features, set_text_polarity_subjectivity,
    set_text_readability_statistics)
from author_classification.modeling import scale_fn  # Needed to load the model
from author_classification.preprocessing import preprocess_text
from author_classification.utils import expand_col_dicts

tqdm.pandas() # Enable progress_apply

def pickleload(f):
    with open(f, 'rb') as fp:
        return pickle.load(fp)


# Parse user args
parser = argparse.ArgumentParser()
parser.add_argument("input", nargs=1)
parser.add_argument("output", nargs=1)
args = parser.parse_args()
args_input = args.input[0]
args_output = args.output[0]

# Validate user args
try:
    path_input = Path(args_input)
except:
    raise ValueError(f'Error: Could not parse input path "{args_input}"')

try:
    path_output = Path(args_output)
except:
    raise ValueError(f'Error: Could not parse output path "{args_output}"')

if not path_input.exists:
    raise ValueError(f'Error: Input path "{args_input}" does not exist')


# Load input file
df = pd.read_csv(args_input)[['text']]

# Check if all needed models are in the models folder
dir_models = Path('models')
list_expected_model_files = [
    'clf.h5',
    'le.pkl',
    'scaler.pkl',
    'tfidf_char.pkl',
    'tfidf_word.pkl',
    'tsvd_char.pkl',
    'tsvd_word.pkl',
]

# If not, download them
if not all([(dir_models / f).exists() for f in list_expected_model_files]):
    print('Downloading Author Classification models...')

    url_models = 'https://drive.google.com/uc?id=1_qIZv8BkPavoTiEFoukAea9bjyToeiDk'
    zip_models = dir_models / 'models.zip'
    gdown.download(url_models, str(zip_models), quiet=False)
    print('Unpacking Author Classification models...')
    
    with zipfile.ZipFile(zip_models, 'r') as zip_ref:
        zip_ref.extractall(dir_models)


# Load the model and the assets for feature extraction and label encoding/decoding
tfidf_char = pickleload(dir_models / 'tfidf_char.pkl')
tsvd_char = pickleload(dir_models / 'tsvd_char.pkl')
tfidf_word = pickleload(dir_models / 'tfidf_word.pkl')
tsvd_word = pickleload(dir_models / 'tsvd_word.pkl')
scaler = pickleload(dir_models / 'scaler.pkl')
le = pickleload(dir_models / 'le.pkl')
clf_best = load_model(dir_models / 'clf.h5', custom_objects={'scale_fn': scale_fn})

# Extract features from original texts
df['dict_punct_features'] = df['text'].progress_apply(get_text_punctuation_stat_features)

# Put each punctuation feature in a column in the dataframe
df = expand_col_dicts(df, colname_dict='dict_punct_features')

# Extract readability and polarity features
df = set_text_readability_statistics(df, colname_text='text')
df = set_text_polarity_subjectivity(df, colname_text='text')

# Preprocess text for other features
df['text_processed'] = df['text'].progress_apply(lambda txt: preprocess_text(txt, lemmatize=True))

# Get statistical features
X_stats = get_feature_matrix_from_dataframe_with_feature_columns(df)
# Scale statistical features
X_stats_scaled = scaler.transform(X_stats)

# Get semantic embedding features
model_sent_embs = SentenceTransformer('all-MiniLM-L6-v2')
X_sentembs = model_sent_embs.encode(df['text'].values, show_progress_bar=True)

# Get lexical (word and char) features
X_tfidf_char_tsvd = tsvd_char.transform(tfidf_char.transform(df['text']).astype(np.float32))
X_tfidf_word_tsvd = tsvd_word.transform(tfidf_word.transform(df['text']).astype(np.float32))

# Horizontally concatenate all features
X = np.hstack([X_stats_scaled, X_tfidf_char_tsvd, X_tfidf_word_tsvd, X_sentembs])

# Predict with the model
Y_hat = le.inverse_transform(clf_best.predict(X))

# Load original data to dismiss columns with features, index, etc. and append new column with prediction results
# Load input file
df = pd.read_csv(args_input)
df['author'] = Y_hat

# Write output prediction to file
df[['text', 'author']].to_csv(args_output)
print(f'Results have been written in "{args_output}"')
