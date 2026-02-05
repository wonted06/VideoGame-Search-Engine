import math
from collections import defaultdict

# Precompute document frequency
def compute_idf(index, num_docs, smooth=False):

# Empty dictionary to store IDF values
    idf = {}

# Loops over every term in index
    for term, postings in index.items():

        # Number of documents containing this term
        df = len(postings)

        if smooth:
            # Smoothed IDF
            idf[term] = math.log((num_docs + 1) / (df + 1)) + 1
        else:
            # Original IDF
            idf[term] = math.log(num_docs / df)

    return idf

# Ranks documents by TF-IDF relevance to a query
def rank_documents(query_tokens, index, idf):

    # Makes every document start with score = 0
    scores = defaultdict(float)

    # Only score documents that contain query terms
    for term in query_tokens:
        if term not in index:
            continue

        # Looks up the postings list, iterates only over relevant documents
        for doc_id, tf in index[term].items():

            # Scoring Logic - Each matching term contributes to the document's relevance score
            scores[doc_id] += tf * idf[term]

    # sorts by score and highlights score first
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Evaluates ranking quality - how accurate the top results are
def precision_at_k(results, relevant_docs, k):

    # Avoids division by zero
    if k == 0:
        return 0.0

    # Takes top k results and extracts only document IDs
    retrieved = [doc_id for doc_id, _ in results[:k]]

    # Finds documents that were both retrieved and relevant
    relevant_retrieved = set(retrieved) & relevant_docs

    # Precision formula - number of relevant retrieved docs / total number of retrieved docs
    return len(relevant_retrieved) / k

# How many relevant docs you find
def recall_at_k(results, relevant_docs, k):

    # If there are no relevant documents, recall is undefined -> return 0
    if not relevant_docs:
        return 0.0

    # Takes top k results and extracts only document IDs
    retrieved = [doc_id for doc_id, _ in results[:k]]

    # Finds documents that were both retrieved and relevant
    relevant_retrieved = set(retrieved) & relevant_docs

    # Recall formula
    return len(relevant_retrieved) / len(relevant_docs)


# Computes average document length
def compute_avg_doc_length(doc_lengths):
    return sum(doc_lengths.values()) / len(doc_lengths)

# Implementing BM25
def rank_documents_bm25(
        query_tokens,
        index,
        idf,
        doc_lengths,
        avg_doc_length,
        k1=1.5,
        b=0.75
):
    scores = defaultdict(float)

    for term in query_tokens:
        if term not in index:
            continue

        for doc_id, tf in index[term].items():
            dl = doc_lengths[doc_id]

            # Limits the benefit of repeated terms
            numerator = tf * (k1 +1)
            denominator = tf + k1 * (1- b + b * (dl / avg_doc_length))

            scores[doc_id] += idf[term] * (numerator / denominator)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def _to_score_dict(results):
    return {doc_id: score for doc_id, score in results}

def combine_weighted_rankings(results_a, results_b, w_a=1.0, w_b=1.0):
    scores = defaultdict(float)

    a = _to_score_dict(results_a)
    b = _to_score_dict(results_b)

    for doc_id, score in a.items():
        scores[doc_id] += w_a * score

    for doc_id, score in b.items():
        scores[doc_id] += w_b * score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def rank_documents_tfidf_field_weighted(query_tokens, title_index, title_idf, body_index, body_idf,
                                        w_title=2.0, w_body=1.0):
    title_results = rank_documents(query_tokens, title_index, title_idf)
    body_results  = rank_documents(query_tokens, body_index,  body_idf)
    return combine_weighted_rankings(title_results, body_results, w_title, w_body)

def rank_documents_bm25_field_weighted(query_tokens, title_index, title_idf, title_lengths, title_avg_dl,
                                       body_index, body_idf, body_lengths, body_avg_dl,
                                       w_title=2.0, w_body=1.0, k1=1.5, b=0.75):

    title_results = rank_documents_bm25(query_tokens, title_index, title_idf, title_lengths, title_avg_dl, k1=k1, b=b)
    body_results  = rank_documents_bm25(query_tokens, body_index,  body_idf,  body_lengths,  body_avg_dl,  k1=k1, b=b)

    return combine_weighted_rankings(title_results, body_results, w_title, w_body)