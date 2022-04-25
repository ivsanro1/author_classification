from nltk.tokenize import word_tokenize

# Generate
from gensim.models import LdaMulticore
from gensim import corpora
# Visualize
import pyLDAvis.gensim
import pyLDAvis

def generate_topic_models(df, colname_text, num_topics=10, bigrams=True):
    if bigrams:
        fn_tokenize = lambda txt: [' '.join(tuple_bigram) for tuple_bigram in list(ngrams(word_tokenize(txt), 2))]
    else:
        fn_tokenize = word_tokenize
    tokenized_texts = df[colname_text].apply(fn_tokenize)
    id2word = corpora.Dictionary(tokenized_texts)

    corpus = [id2word.doc2bow(text) for text in tokenized_texts]
    
    lda = LdaMulticore(corpus=corpus,
                   id2word=id2word,
                   num_topics=num_topics,
                   random_state=288,
                   iterations=200)
    return {
        'lda_model': lda,
        'corpus': corpus,
        'id2word': id2word,
        'str_topics': lda.print_topics()
    }






def plot_topic_models(d_ret_topic_modeling):
    # Visualize the topics
    pyLDAvis.enable_notebook()

    results_LDAavis = pyLDAvis.gensim.prepare(
        d_ret_topic_modeling['lda_model'],
        d_ret_topic_modeling['corpus'],
        d_ret_topic_modeling['id2word']
    )

    return pyLDAvis.display(results_LDAavis)