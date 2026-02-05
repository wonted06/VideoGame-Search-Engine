from pathlib import Path
from parser import parse_collection
from tokeniser import process_text
from indexer import build_inverted_index, build_inverted_index_bm25
from ranker import compute_idf, rank_documents, precision_at_k, recall_at_k
from ranker import rank_documents_bm25,compute_avg_doc_length
from datetime import datetime
from experiments import print_top10

# -------------------------------
# PREPROCESSING CONFIGURATION
# -------------------------------

PREPROCESSING = {
    "use_stopwords":True,
    "use_stemming":True
}

# -------------------------------
# 1. PATH SETUP
# -------------------------------

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
# Path to data/Videogames
DATA_DIR = BASE_DIR / "data" / "Videogames"

# -------------------------------
# 2. DATA LOADING & PREPROCESSING
# -------------------------------

def load_and_process_documents():
    docs = parse_collection(DATA_DIR)
    print(f"Number of documents: {len(docs)}")

    # Tokenisation step
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

        # Backward compatibility
        doc["tokens"] = doc["body_tokens"]
    return docs

# -------------------------------
# 3. INDEX CONSTRUCTION
# -------------------------------

def build_index(docs):

    # Indexing step
    index = build_inverted_index(docs)
    print(f"Number of unique tokens: {len(index)}")

    return index

# -------------------------------
# 4. RETRIEVAL EXPERIMENTS
# -------------------------------

def run_retrieval(query, index, idf):
    query_tokens = process_text(
        query,
        use_stopwords=PREPROCESSING["use_stopwords"],
        use_stemming=PREPROCESSING["use_stemming"]
    )
    return rank_documents(query_tokens, index, idf)

# -------------------------------
# 5. EVALUATION
# -------------------------------

def evaluate_query(query, relevant_docs, index, idf):
    results = run_retrieval(query, index, idf)

    print(f"\nQuery: {query}")
    print("Precision@5:", precision_at_k(results, relevant_docs, 5))
    print("Precision@10:", precision_at_k(results, relevant_docs, 10))
    print("Recall@10:", recall_at_k(results, relevant_docs, 10))

    return results

# -------------------------------
# 6. USER QUERY INPUT
# -------------------------------

def save_results_to_file(query, results, doc_titles, k=10):
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = results_dir / f"results_{timestamp}.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Top {k} results: \n\n")
        for rank, (doc_id, score) in enumerate(results[:k], start=1):
            title = doc_titles.get(doc_id, "UNKNOWN TITLE")
            f.write(f"{rank}. {doc_id}\n")
            f.write(f"  {title}\n")
            f.write(f"  score={score:.4f}\n\n")

    print(f"\nSaved results to: {filepath}")

# -------------------------------
# MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":

    # -------------------------------
    # Load & preprocess
    # -------------------------------
    documents = load_and_process_documents()

    # -------------------------------
    # Build BM25-compatible index ONCE
    # -------------------------------
    index, doc_lengths = build_inverted_index_bm25(documents)
    avg_dl = compute_avg_doc_length(doc_lengths)

    num_docs = len(documents)
    idf = compute_idf(index, num_docs, smooth=True)

    # -------------------------------
    # Print user query results & save query results to files
    # -------------------------------

    doc_titles = {doc["doc_id"]: doc["title"] for doc in documents}
    while True:

        query = input("\nEnter query (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        query_tokens = process_text(query)
        results = rank_documents_bm25(query_tokens, index, idf, doc_lengths, avg_dl)
        print_top10("BM25", results, set(), doc_titles)

        save_results_to_file(query, results, doc_titles)
