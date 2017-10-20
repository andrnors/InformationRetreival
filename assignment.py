"""
Author: Andreas Norstein
"""
import random
import codecs
import string
import gensim
from nltk.stem.porter import PorterStemmer

RANDOM_NUMBER = random.seed(123)
F = codecs.open("book.txt", 'r', 'utf-8')


def partitions(input_file, partion_factor):
    """partition file into paragraphs"""
    print "Partitions file to paragraphs"
    return input_file.read().split(partion_factor)

PARTITIONS = partitions(F, "\n\n")
STOPWORDS = partitions(codecs.open("common_words.txt", "r"), ",")


def filter_partitions(partitions_list):
    """filter partitions by removing the word Gutenberg"""
    print "Removes the word Gutenberg"
    return [p for p in partitions_list if "Gutenberg".lower() not in p.lower()]

FILTERD_PARTITION = filter_partitions(PARTITIONS)


def tokenize_paragraphs(filter_partitions_list):
    """
    tokenize paragraphs by splitting them into words.
    Returns a list of paragraphs where each paragraph is a list of words
    """
    print "Remove punctuations"
    exclude = set(string.punctuation + "\n\r\t")
    temp_tokens = [p.split(" ") for p in filter_partitions_list]
    tokens = []
    for token in temp_tokens:
        stem_tokens = []
        if token[0]:
            for word in token:
                if word:
                    word = ''.join(ch for ch in word if ch not in exclude)
                    stem_tokens.append(word.lower())
            tokens.append(stem_tokens)
    return tokens

TOKENIZE_PARAGAPHS = tokenize_paragraphs(FILTERD_PARTITION)


def stem(paragraphs):
    print "Stemming"
    stemmer = PorterStemmer()
    temp_paragraphs = []
    for paragraph in paragraphs:
        temp_paragraph = []
        for word in paragraph:
            temp_paragraph.append(stemmer.stem(word.lower()))
        temp_paragraphs.append(temp_paragraph)
    return temp_paragraphs

STEMMED_AND_TOKENIZED_PARAGRAPHS = stem(TOKENIZE_PARAGAPHS)


def generate_dictionary(tokenized_and_stemmed_paragraphs):
    """Generate Dictionary based on tokenized paragraphs"""
    print "Generate Dictionary"
    return gensim.corpora.Dictionary(tokenized_and_stemmed_paragraphs)

DICTONARY = generate_dictionary(STEMMED_AND_TOKENIZED_PARAGRAPHS)


def filter_stopwords(dictionary):
    """Filter out all stopwords"""
    print "Filters Stopwords"
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

DICTIONARY_FILTERED = filter_stopwords(DICTONARY)


def map_to_bow(dictionary, tokenized_and_stemmed_paragraphs):
    """ Map paragraphs into Bags-of-Words using a dictionary"""
    print "Generates bag of words"
    return [dictionary.doc2bow(document) for document in tokenized_and_stemmed_paragraphs]

CORPUS = map_to_bow(DICTIONARY_FILTERED, STEMMED_AND_TOKENIZED_PARAGRAPHS)
print "corpus"
print CORPUS
print

def build_tf_idf_model(corpus):
    """  Build TF-IDF model using corpus (list of paragraphs) from the previous part"""
    print "Builds td-idf model"
    return gensim.models.TfidfModel(corpus)

TFIDF_MODEL = build_tf_idf_model(CORPUS)


def map_bow_to_tfidf_weights(tfidf_model, bow):
    """ Map Bags-of-Words into TF-IDF weights"""
    print "Create tf-idf weights"
    return tfidf_model[bow]


TFIDF_CORPUS = map_bow_to_tfidf_weights(TFIDF_MODEL, CORPUS)


def construct_matrix_similarity(corpus):
    """
    Construct MatrixSimilarity object that calculate similarities between paragraphs and queries:
    """
    print "Constructs matrix similarity"
    return gensim.similarities.MatrixSimilarity(corpus)

TFIDF_MATRIX_SIMILARITY = construct_matrix_similarity(TFIDF_CORPUS)


def build_lsi_model(dictionary, corpus, num_topics):
    """ LSI model using as an input the corpus with TF-IDF weights. Set
    number of topics to 100. In the end, each paragraph should be represented
    with a list of 100 pairs (topic-index, LSI-topic-weight) ). """
    print "Build lsi model"
    return gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

LSI_MODEL = build_lsi_model(DICTIONARY_FILTERED, TFIDF_CORPUS, 100)


def create_lsi_corpus(model, corpus):
    """ creates lsi_corpus"""
    print "Create lsi corpus"
    return model[corpus]

LSI_CORPUS = create_lsi_corpus(LSI_MODEL, CORPUS)
LSI_MATRIX_SIMILARITY = construct_matrix_similarity(LSI_CORPUS)


"""
Part 4 of assignment
"""
queries = [["What is the function of money?"], ["How taxes influence Economics?"]]
def preprocessing(query):
    """preprocessing
    :rtype: list
    :returns list with filtered_query, tokenized_query, dictionary, filtered_dictionary and bag_of_words
    """
    print "preprocessing \n"
    filter_query = filter_partitions(query)
    tokenized_query = tokenize_paragraphs(filter_query)
    stemmed_query = stem(tokenized_query)
    bow = map_to_bow(DICTIONARY_FILTERED, stemmed_query)[0]
    return [filter_query, tokenized_query, stemmed_query, bow]

ex2 = preprocessing(queries[1])

filtered_query, tokenized_query, stemmed_query, query_corpus_bow = ex2[0], ex2[1], ex2[2], ex2[3]

# 4.2 Convert BOW to TF-IDF representation. Report TF-IDF weights. For example, for the query "How
# taxes influence Economics?" TF-IDF weights are:
# (tax: 0.26, econom: 0.82, influenc: 0.52)
# Some useful code:
# $ ... = tfidf_model[...]

tfidf_weights = map_bow_to_tfidf_weights(TFIDF_MODEL, query_corpus_bow)
print "TF-IDF weight:", tfidf_weights

# 4.3 Report top 3 the most relevant paragraphs for the query "What is the function of
# money?" according to TF-IDF model (displayed paragraphs should be in the original
# form before
# processing, but truncated up to first 5 lines).


doc2similarity = enumerate(TFIDF_MATRIX_SIMILARITY[tfidf_weights])

query_paragraphs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
for p in query_paragraphs:
    print
    print "Paragraph: " + str(p[0]) + ", similarity: " + str(p[1])
    # print FILTERD_PARTITION[p[0]]


# 4.4 Convert query TF-IDF representation
# (for the query "What is the function of money?")
# into LSI-topics representation (weights).
# Report top 3. topics with the most significant (with the largest
# absolute values) weights and top 3. the most relevant
# paragraphs according to LSI model. Compare
# retrieved paragraphs with the paragraphs found for TF-IDF model.

ex1 = preprocessing(queries[0])
filtered_query_0, tokenized_query_0, stemmed_query_0, query_corpus_bow_0 = ex1[0], ex1[1], ex1[2], ex1[3]

lsi_query = create_lsi_corpus(LSI_MODEL, tfidf_weights)
lsi_topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
doc2similarity_lsi = enumerate(LSI_MATRIX_SIMILARITY[lsi_query])
top_paragraphs = sorted(doc2similarity_lsi, key=lambda kv: -kv[1])[:3]
for topic in lsi_topics:
    print
    print "Topic: " + str(topic[0]) + ", value: " + str(topic[1])
    print LSI_MODEL.show_topic(topic[0])

for pg in top_paragraphs:
    print
    print "Topic: " + str(pg[0]) + ", value: " + str(pg[1])
    print FILTERD_PARTITION[pg[0]]
