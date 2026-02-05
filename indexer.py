from collections import Counter, defaultdict

def build_inverted_index(documents):
    inverted_index = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        tokens = doc["tokens"]

        term_freq = Counter(tokens)

        for term, freq in term_freq.items():
            if term not in inverted_index:
                inverted_index[term] = {}

            inverted_index[term][doc_id] = freq

    return inverted_index

def build_inverted_index_bm25(documents):
    index = defaultdict(dict)
    doc_lengths = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        tokens = doc["tokens"]
        doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            index[token][doc_id] = index[token].get(doc_id, 0) + 1

    return index, doc_lengths
