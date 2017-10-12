"""
Part 4 of assignment
"""
from assignment import *


def preprocessing(query):
    """preprocessing 
    :rtype: list
    :returns list with filtered_query, tokenized_query, dictionary, filtered_dictionary and bag_of_words
    """
    filter_query = filter_partitions(query)
    tokenized_query = tokenize_paragraphs(filter_query)
    stemmed_query = stem(tokenized_query)
    dictionary = generate_dictionary(stemmed_query)
    filtered_dictionary = filter_stopwords(dictionary)
    bow = map_to_bow(filtered_dictionary, stemmed_query)
    
    return [filter_query, tokenized_query, dictionary, filtered_dictionary, bow]

preprocessed = preprocessing(["What is the function of money?"])

# ex2 = preprocessing(["How taxes influence Economics?"])

# filtered_query, tokenized_query, dictionary, filtered_dictionary, corpus_bow = ex2[0], ex2[1], ex2[2], ex2[3], ex2[4]
filtered_query, tokenized_query, dictionary, filtered_dictionary, corpus_bow = preprocessed[0], preprocessed[1], preprocessed[2], preprocessed[3], preprocessed[4]
print corpus_bow
print dictionary

# 4.2 Convert BOW to TF-IDF representation. Report TF-IDF weights. For example, for the query "How
# taxes influence Economics?" TF-IDF weights are:
# (tax: 0.26, econom: 0.82, influenc: 0.52)
# Some useful code:
# $ ... = tfidf_model[...]

# tfidf_model = build_tf_idf_model(corpus_bow)
#
# tfidf_weights = map_bow_to_tfidf_weights(tfidf_model, corpus_bow)

# print tfidf_weights
