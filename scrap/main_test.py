from pathlib import Path
from parser import parse_collection
from tokeniser import process_text
from indexer import build_inverted_index, build_inverted_index_bm25
from ranker import compute_idf, rank_documents, precision_at_k, recall_at_k
from ranker import rank_documents_bm25,compute_avg_doc_length

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
# Path to data/Videogames
DATA_DIR = BASE_DIR / "data" / "Videogames"

docs = parse_collection(DATA_DIR)
print(f"Number of documents: {len(docs)}")
# Debug: shows one title
print(docs[0]["title"])

# Tokenisation step
for doc in docs:
    doc ["tokens"] = process_text(doc["body"])

# Debug: shows tokens
print(docs[0]["doc_id"])
print(docs[0]["tokens"][:20])

# Indexing step
index = build_inverted_index(docs)

print(f"Number of unique tokens: {len(index)}")
print(index.get("trozei", {}))

# # -------------------------------
# # HELPER FUNCTION
# # -------------------------------
#
# def make_field_docs(documents, field_key):
#     return [{"doc_id": d["docid"], "tokens": d.get(field_key, [])} for d in documents]

# query = guitar
#     # This relevance set MUST correspond to "mario"
#     relevant_docs_guitar = {
#         '_NxsX-PTCZ3CxQpElmh5apcU45uTecQ3.html',
#         'o3iFz6UbHZ-HI2PfHtDYnaSvKD-GHJDE.html',
#         '3yWXFoPbkP-QkXKm8N9FWTXPl-upaM1b.html',
#         '8jhxB5TKfyFbJd1UJlLArgZ0cFMVjkcA.html',
#         'Ep-h7hTrTJWj2G6C2JmbjtGiMkGVlS1F.html',
#         'sqm17ALfAtze7XHL-5RFV4HmJYC8aDxn.html',
#         'Xb25ZZr4LbNE5U9n2bvoKXrw7RY6GI8-.html',
#         'sIOtJJwBgXDeSZ2PyeB1w5zKqmSKJN-Y.html',
#         'XnJyJu3K7u5y5xfj5fBHGZtQoPYfuOCF.html',
#         '5_vxRWOPWV7YlvxvyGr-b9W0zM8Gwv-6.html',
#         'dTswyLWb2UYehf802rmDe-D9dj4WRkqs.html',
#         'UCbUz-LZjM5ILUrHs5dkoSZR1c-7GwMw.html',
#         'Fe0_TFVoa6RbkoZq_GoIDaRTgOzVAOID.html',
#         'Yb2E_blJcGnZlnw48eV4EdmPUlEu0MeJ.html'
#     }
#
#     # -------------------------------
#     # TF-IDF vs BM25 comparison
#     # -------------------------------
#     tfidf_results = rank_documents(query_tokens, index, idf)
#     bm25_results = rank_documents_bm25(query_tokens, index, idf, doc_lengths, avg_dl)
#
#     print("\nBM25 vs TF-IDF:")
#     print("TF-IDF P@10:", precision_at_k(tfidf_results, relevant_docs_guitar, 10))
#     print("BM25 P@10:", precision_at_k(bm25_results, relevant_docs_guitar, 10))
#
#     print("TF-IDF R@10:", recall_at_k(tfidf_results, relevant_docs_guitar, 10))
#     print("BM25 R@10:", recall_at_k(bm25_results, relevant_docs_guitar, 10))
#
#     print("\nTF-IDF Top 10:")
#     for doc_id, score in tfidf_results[:10]:
#         print(doc_id, "REL" if doc_id in relevant_docs_guitar else "")
#
#     print("\nBM25 Top 10:")
#     for doc_id, score in bm25_results[:10]:
#         print(doc_id, "REL" if doc_id in relevant_docs_guitar else "")
#
#     print("Total relevant docs:", len(relevant_docs_guitar))