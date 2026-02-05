from collections import defaultdict
from ranker import precision_at_k
from ranker import recall_at_k

def test_precision_at_k():
    print ("\nPrecision@k Tests\n")

    results = [
        ("doc1", 3.0),
        ("doc2", 2.0),
        ("doc3", 1.0),
        ("doc4", 0.5)
    ]

    relevant_docs = {"doc1", "doc3"}

    print("Test 1: Precision@2 (expect 0.5)", precision_at_k(results, relevant_docs, 2))

    print("Test 2: Precision@3 (expect 0.666...)", precision_at_k(results, relevant_docs, 3))

    print("Test 3: Precision@4 (expect 0.5)", precision_at_k(results, relevant_docs, 4))

    print("Test 4: Precision@0 (expect 0.0)", precision_at_k(results, relevant_docs, 0))

def test_recall_at_k():
    print("\nRecall@k Tests\n")

    results = [
        ("doc1", 3.0),
        ("doc2", 2.0),
        ("doc3", 1.0),
        ("doc4", 0.5),
    ]

    relevant_docs = {"doc1", "doc3", "doc5", "doc6"}

    print("Test 1: Recall@2 (expect 0.25):", recall_at_k(results, relevant_docs, 2))

    print("Test 2: Recall@3 (expect 0.5):", recall_at_k(results, relevant_docs, 3))

    print("Test 3: Recall@4 (expect 0.5):", recall_at_k(results, relevant_docs, 4))

if __name__ == "__main__":
    test_precision_at_k()
    test_recall_at_k()