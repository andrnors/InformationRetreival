"""
Part 4 of assignment
"""
from assignment import *


def preprocessing(query):
    """preprocessing 
    :rtype: list
    :returns list with filtered_query, tokenized_query, dictionary, filtered_dictionary and bag_of_words
    """
    filter_query = filter_partitions([query])
    tokenized_query = tokenize_paragraphs(filter_query)
    dictionary = generate_dictionary(tokenized_query)
    filtered_dictionary = filter_stopwords(dictionary)
    bow = map_to_bow(filtered_dictionary, tokenized_query)
    
    return [filter_query, tokenized_query, dictionary, filtered_dictionary, bow]

preprocessed = preprocessing("What is the function of money?")

filtered_query, tokenized_query, dictionary, filtered_dictionary, bow = preprocessed[0], preprocessed[1], preprocessed[2], preprocessed[3], preprocessed[4]


# 4.2 Convert BOW to TF-IDF representation. Report TF-IDF weights. For example, for the query "How
# taxes influence Economics?" TF-IDF weights are:
# (tax: 0.26, econom: 0.82, influenc: 0.52)
# Some useful code:
# $ ... = tfidf_model[...]


