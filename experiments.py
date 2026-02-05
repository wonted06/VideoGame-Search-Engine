import pandas as pd
from pathlib import Path
from parser import parse_collection
from tokeniser import process_text
from indexer import build_inverted_index_bm25
from ranker import (
    compute_idf,
    rank_documents,
    rank_documents_bm25,
    precision_at_k,
    recall_at_k,
    compute_avg_doc_length,
    rank_documents_tfidf_field_weighted,
    rank_documents_bm25_field_weighted
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "Videogames"
CSV_PATH = BASE_DIR / "data" / "videogame.csv"

PREPROCESSING = {
    "use_stopwords": True,
    "use_stemming": True,
    "use_lemmatization": False
}

# Loads and preprocesses documents
def load_documents():
    docs = parse_collection(DATA_DIR)
    for doc in docs:
        doc["title_tokens"] = process_text(
            doc["title"],
            use_stopwords=PREPROCESSING["use_stopwords"],
            use_stemming=PREPROCESSING["use_stemming"]
        )
        doc["body_tokens"] = process_text(
            doc["body"],
            use_stopwords=PREPROCESSING["use_stopwords"],
            use_stemming=PREPROCESSING["use_stemming"]
        )

        # Keep this for backward compatibility (your current system expects doc["tokens"])
        doc["tokens"] = doc["body_tokens"]

    return docs


# Loads CSV metadata and maps it to doc_ids
def load_metadata():
    df = pd.read_csv(CSV_PATH)

    # Extract filename from url
    df["doc_id"] = df["url"].astype(str).str.split("/").str[-1]

    meta = {}
    for _, row in df.iterrows():
        meta[row["doc_id"]] = {
            "publisher": str(row.get("STRING : publisher", "")),
            "genre": str(row.get("STRING : genre", "")),
            "esrb": str(row.get("STRING : esrb", "")),
        }
    return meta


def relevant_by_publisher(meta, publisher_name="Atari"):
    publisher_name = publisher_name.lower()
    return {doc_id for doc_id, m in meta.items() if m["publisher"].lower() == publisher_name}

def relevant_by_genre_contains(meta, term="Arcade"):
    term = term.lower()
    return {doc_id for doc_id, m in meta.items() if term in m["genre"].lower()}

def relevant_by_title_contains(documents, phrase):
    phrase = phrase.lower()
    return {doc["doc_id"] for doc in documents if phrase in doc["title"].lower()}


def model_and_eval(query, relevant_docs, index, idf, doc_lengths, avg_dl,
                   use_stopwords=True, use_stemming=True, use_lemmatization=False):
    query_tokens = process_text(
        query,
        use_stopwords=use_stopwords,
        use_stemming=use_stemming,
        use_lemmatization=use_lemmatization
    )

    tfidf_results = rank_documents(query_tokens, index, idf)
    bm25_results  = rank_documents_bm25(query_tokens, index, idf, doc_lengths, avg_dl)

    tfidf_p10 = precision_at_k(tfidf_results, relevant_docs, 10)
    tfidf_r10 = recall_at_k(tfidf_results, relevant_docs, 10)

    bm25_p10 = precision_at_k(bm25_results, relevant_docs, 10)
    bm25_r10 = recall_at_k(bm25_results, relevant_docs, 10)

    return (tfidf_p10, tfidf_r10, bm25_p10, bm25_r10, tfidf_results, bm25_results)

# The 6 required queries
QUERIES = [
  "Pokémon Trozei",
  "Tony Hawk’s Downhill Jam",
  "Arcade type games",
  "London Taxi: Rush Hour",
  "Game published by Atari",
  "The Sims 2 Apartment Pets"
]


def build_relevance_sets(documents, meta):
    rel = {}

    rel["Game published by Atari"] = relevant_by_publisher(meta, "Atari")
    rel["Arcade type games"] = relevant_by_genre_contains(meta, "Arcade")

    # Title-based relevance for the specific game queries
    rel["Pokémon Trozei"] = relevant_by_title_contains(documents, "trozei")
    rel["Tony Hawk’s Downhill Jam"] = relevant_by_title_contains(documents, "tony hawk")
    rel["London Taxi: Rush Hour"] = relevant_by_title_contains(documents, "london taxi")
    rel["The Sims 2 Apartment Pets"] = relevant_by_title_contains(documents, "apartment pets")

    return rel


def print_top10(label, results, relevant_docs, doc_titles):
    print(f"\n{label} Top 10 Results:\n")

    for rank, (doc_id, score) in enumerate(results[:10], start=1):
        tag = "REL" if doc_id in relevant_docs else "   "
        title = doc_titles.get(doc_id, "UNKNOWN TITLE")

        print(f"{rank:2d}. [{tag}] {doc_id}")
        print(f"    {title}")
        print(f"    score = {score:.4f}\n")

# Experiment - Stopwords & Stemming
def config_stopword_stem(use_stopwords, use_stemming):
    documents = parse_collection(DATA_DIR)

    for doc in documents:
        doc["tokens"] = process_text(
            doc["body"],
            use_stopwords=use_stopwords,
            use_stemming=use_stemming
        )

    index, doc_lengths = build_inverted_index_bm25(documents)
    avg_dl = compute_avg_doc_length(doc_lengths)
    idf = compute_idf(index, num_docs, smooth=True)

    return documents, index, doc_lengths, avg_dl, idf

# Reuses my existing indexer without rewriting
def make_field_docs(documents, tokens_key):
    return [{"doc_id": d["doc_id"], "tokens": d.get(tokens_key, [])} for d in documents]

if __name__ == "__main__":

    # -------------------------------
    # Load & preprocess
    # -------------------------------
    documents = load_documents()

    # -------------------------------
    # Build BM25-compatible index ONCE
    # -------------------------------
    N = len(documents)

    title_docs = make_field_docs(documents, "title_tokens")
    body_docs = make_field_docs(documents, "body_tokens")

    title_index, title_lengths = build_inverted_index_bm25(title_docs)
    body_index, body_lengths = build_inverted_index_bm25(body_docs)

    title_avg_dl = compute_avg_doc_length(title_lengths)
    body_avg_dl = compute_avg_doc_length(body_lengths)

    title_idf = compute_idf(title_index, N, smooth=True)
    body_idf = compute_idf(body_index, N, smooth=True)

    index, doc_lengths = build_inverted_index_bm25(documents)
    avg_dl = compute_avg_doc_length(doc_lengths)

    num_docs = len(documents)
    idf = compute_idf(index, num_docs, smooth=True)

    # -------------------------------
    # Evaluation setup (example query)
    # -------------------------------
    query = "guitar"

    query_tokens = process_text(
        query,
        use_stopwords=PREPROCESSING["use_stopwords"],
        use_stemming=PREPROCESSING["use_stemming"],
        use_lemmatization=PREPROCESSING.get("use_lemmatization", False)
    )

    documents = load_documents()
    meta = load_metadata()

    # Build title lookup ONCE
    doc_titles = {doc["doc_id"]: doc["title"] for doc in documents}

    # Build one index for BOTH rankers
    index, doc_lengths = build_inverted_index_bm25(documents)
    avg_dl = compute_avg_doc_length(doc_lengths)
    idf = compute_idf(index, len(documents), smooth=True)

    relevance = build_relevance_sets(documents, meta)

    print("Query,TFIDF_P10,TFIDF_R10,BM25_P10,BM25_R10,NumRelDocs")

    for q in QUERIES:
        relevant_docs = relevance.get(q, set())
        if not relevant_docs:
            print(f"# WARNING: no relevant docs found for query: {q}")

        tfp10, tfr10, bmp10, bmr10, tf_res, bm_res = model_and_eval(
            q, relevant_docs, index, idf, doc_lengths, avg_dl
        )

        print(f"{q},{tfp10:.3f},{tfr10:.3f},{bmp10:.3f},{bmr10:.3f},{len(relevant_docs)}")

        # Print top 10 for Pokémon Trozei only (so output isn't huge)
        if q == "Pokémon Trozei":
            print_top10("TF-IDF (Pokémon Trozei)", tf_res, relevant_docs, doc_titles)
            print_top10("BM25 (Pokémon Trozei)", bm_res, relevant_docs, doc_titles)


    # -------------------------------
    # Stopwords & Stemming - Configuration test
    # -------------------------------

    CONFIGS = [
        ("SW_ON_STEM_ON", True, True),
        ("SW_OFF_STEM_ON", False, True),
        ("SW_ON_STEM_OFF", True, False),
        ("SW_OFF_STEM_OFF", False, False),
    ]

    for name, sw, stem in CONFIGS:
        documents, index, doc_lengths, avg_dl, idf = config_stopword_stem(sw, stem)

        print("\n==============================")
        print("CONFIG:", name)
        print("==============================")

        for q in QUERIES:
            query_tokens = process_text(q, use_stopwords=sw, use_stemming=stem)

            tfidf_results = rank_documents(query_tokens, index, idf)
            bm25_results = rank_documents_bm25(query_tokens, index, idf, doc_lengths, avg_dl)

            # Use your existing relevance sets here
            relevant_docs = relevance[q]

            print(f"\nQuery: {q}")
            print("TF-IDF P@10:", precision_at_k(tfidf_results, relevant_docs, 10))
            print("BM25  P@10:", precision_at_k(bm25_results, relevant_docs, 10))

            print("\nTF-IDF R@10:", recall_at_k(tfidf_results, relevant_docs, 10))
            print("BM25  R@10:", recall_at_k(bm25_results, relevant_docs, 10))

    # -------------------------------
    # Lemmatization - Configuration test
    # -------------------------------
    LEMM_CONFIGS = [
        ("BASE_STEM", True, True, False),  # stopwords on, stemming on
        ("LEMMA_ONLY", True, False, True),  # stopwords on, lemmatization on
        ("NO_MORPH", True, False, False),  # stopwords on, no stemming/lemm
    ]

    for name, sw, stem, lem in LEMM_CONFIGS:
        documents = parse_collection(DATA_DIR)
        for doc in documents:
            doc["tokens"] = process_text(
                doc["body"],
                use_stopwords=sw,
                use_stemming=stem,
                use_lemmatization=lem
            )

        index, doc_lengths = build_inverted_index_bm25(documents)
        avg_dl = compute_avg_doc_length(doc_lengths)
        idf = compute_idf(index, len(documents), smooth=True)

        print("\n==============================")
        print("LEMMA CONFIG:", name)
        print("==============================")

        for q in QUERIES:
            query_tokens = process_text(
                q,
                use_stopwords=sw,
                use_stemming=stem,
                use_lemmatization=lem
            )

            tfidf_results = rank_documents(query_tokens, index, idf)
            bm25_results = rank_documents_bm25(query_tokens, index, idf, doc_lengths, avg_dl)

            relevant_docs = relevance[q]

            print(f"\nQuery: {q}")
            print("TF-IDF P@10:", precision_at_k(tfidf_results, relevant_docs, 10))
            print("BM25  P@10:", precision_at_k(bm25_results, relevant_docs, 10))
            print("TF-IDF R@10:", recall_at_k(tfidf_results, relevant_docs, 10))
            print("BM25  R@10:", recall_at_k(bm25_results, relevant_docs, 10))

            # -------------------------------
            # Field-weighted TF-IDF and BM25
            # -------------------------------
            w_title = 2.0
            w_body = 1.0

            query_tokens = process_text(
                q,
                use_stopwords=PREPROCESSING["use_stopwords"],
                use_stemming=PREPROCESSING["use_stemming"],
                use_lemmatization=PREPROCESSING.get("use_lemmatization", False)
            )

            tfidf_fw = rank_documents_tfidf_field_weighted(
                query_tokens,
                title_index, title_idf,
                body_index, body_idf,
                w_title=w_title, w_body=w_body
            )

            bm25_fw = rank_documents_bm25_field_weighted(
                query_tokens,
                title_index, title_idf, title_lengths, title_avg_dl,
                body_index, body_idf, body_lengths, body_avg_dl,
                w_title=w_title, w_body=w_body
            )

            print(f"TFIDF_FW_P10,{precision_at_k(tfidf_fw, relevant_docs, 10):.3f}")
            print(f"BM25_FW_P10,{precision_at_k(bm25_fw, relevant_docs, 10):.3f}")
            print(f"TFIDF_FW_R10,{recall_at_k(tfidf_fw, relevant_docs, 10):.3f}")
            print(f"BM25_FW_R10,{recall_at_k(bm25_fw, relevant_docs, 10):.3f}")