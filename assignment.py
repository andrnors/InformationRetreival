"""
Author: Andreas Norstein
"""
import random
import codecs
import string
import gensim

RANDOM_NUMBER = random.seed(123)
F = codecs.open("book.txt", 'r')


def partitions(input_file, partion_factor):
    """partition file into paragraphs"""
    return input_file.read().split(partion_factor)

PARTITIONS = partitions(F, "\n")
STOPWORDS = partitions(codecs.open("common_words.txt", "r"), ",")


def filter_partitions(partitions_list):
    """filter partitions by removing the word Gutenberg"""
    return [p for p in partitions_list if "Gutenberg" not in p]

FILTERD_PARTITION = filter_partitions(PARTITIONS)

def tokenize_paragraphs(filter_partitions_list):
    """
    tokenize paragraphs by splitting them into words.
    Returns a list of paragraphs where each paragraph is a list of words
    """
    exclude = set(string.punctuation)
    temp_tokens = [p.split(" ") for p in filter_partitions_list]
    tokens = []
    for token in temp_tokens:
        stem_tokens = []
        if token[0]:
            for word in token:
                if word:
                    word = ''.join(ch for ch in word if ch not in exclude)
                    stem_tokens.append(word)
            tokens.append(stem_tokens)
    return tokens

TOKENIZE_PARAGAPHS = tokenize_paragraphs(FILTERD_PARTITION)

def generate_dictionary(tokenized_paragraphs):
    """Generate Dictionary based on tokenized paragraphs"""
    return gensim.corpora.Dictionary(tokenized_paragraphs)

DICTONARY = generate_dictionary(TOKENIZE_PARAGAPHS)

def filter_stopwords(dictionary):
    """Filter out all stopwords"""
    stop_ids = []
    for word in STOPWORDS:
        if word == STOPWORDS[-1].strip("\n"):
            word = word.strip("\n")
            if word in dictionary.token2id:
                stopword_id = dictionary.token2id[word]
                stop_ids.append(stopword_id)
        else:
            if word in dictionary.token2id:
                stopword_id = dictionary.token2id[word]
                stop_ids.append(stopword_id)
    dictionary.filter_tokens(stop_ids)
    dictionary.compactify()
    return dictionary

DICTIONARY_FILTERD = filter_stopwords(DICTONARY)

def map_to_bow(dictionary, tokenized_paragraphs):
    """ Map paragraphs into Bags-of-Words using a dictionary"""
    corpus = []
    for document in tokenized_paragraphs:
        corpus.append(dictionary.doc2bow(document))
    return corpus

CORPUS = map_to_bow(DICTIONARY_FILTERD, TOKENIZE_PARAGAPHS)

def build_tf_idf_model(corpus):
    """  Build TF-IDF model using corpus (list of paragraphs) from the previous part"""
    return gensim.models.TfidfModel(corpus)

TFIDF_MODEL = build_tf_idf_model(CORPUS)

def map_bow_to_tfidf_weights(tfidf_model, bow):
    """ Map Bags-of-Words into TF-IDF weights"""
    return [tfidf_model[b] for b in bow]

TFIDF_CORPUS = map_bow_to_tfidf_weights(TFIDF_MODEL, CORPUS)

def construct_matrix_similarity(corpus, tfidf_model):
    """
    Construct MatrixSimilarity object that calculate similarities between paragraphs and queries:
    """
    return gensim.similarities.MatrixSimilarity(tfidf_model[corpus])

MATRIX_SIMILARITY = construct_matrix_similarity(TFIDF_CORPUS, TFIDF_MODEL)
print MATRIX_SIMILARITY


"""
3.4 Repeat the above procedure for LSI model using as an input the corpus with TF-IDF weights. Set
number of topics to 100. In the end, each paragraph should be represented with a list of 100 pairs
(topic-index, LSI-topic-weight) ). Some useful code:
$ ... = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary,
num_topics=100)
$ lsi_corpus = lsi_model[...]
$ ... = gensim.similarities.MatrixSimilarity(...)

"""
def build_lsi_model(dictionary, corpus, num_topics):
    """ lol """
    return gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

print build_lsi_model(DICTONARY, TFIDF_CORPUS, 100)