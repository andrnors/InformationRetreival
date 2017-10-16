"""
Part 4 of assignment
"""
from assignment import *

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
    print unicode(LSI_MODEL.show_topic(topic[0]))

for pg in top_paragraphs:
    print
    print "Topic: " + str(pg[0]) + ", value: " + str(pg[1])
    print FILTERD_PARTITION[pg[0]]
