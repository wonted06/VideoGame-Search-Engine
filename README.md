# Videogame Search Engine (Information Retrieval Coursework)

This project implements a ranked Information Retrieval (IR) search engine over a collection of videogame HTML pages. It supports:
- HTML parsing and noise removal (script/style/nav/header/footer)
- Configurable preprocessing (tokenisation, lowercasing, stop-word removal, stemming)
- Inverted indexing
- Ranking using TF-IDF and BM25
- Evaluation using Precision@k and Recall@k
- A CLI demo for entering queries and generating ranked outputs
- An experiments script for running controlled evaluations across required queries

---

## Dataset

The system expects the following dataset layout:

data/
Videogames/          # folder containing 727 HTML files

videogame.csv        # metadata CSV (publisher/genre etc.)

The search engine reads the HTML files directly from `data/Videogames/`.  
The CSV file is used by the experiments pipeline for building relevance sets (e.g., publisher/genre queries).

---

## Project Structure

src/
main.py              # CLI search demo (enter queries, view results, save outputs)

experiments.py       # experiment runner (TF-IDF vs BM25, preprocessing experiments, required queries)

parser.py            # parse HTML -> {doc_id, title, body}

tokeniser.py         # preprocessing pipeline (tokenise, stopwords, stemming)

indexer.py           # inverted index builders (TF-IDF + BM25-compatible)

ranker.py            # TF-IDF ranking, BM25 ranking, evaluation metrics

evaluation_tests.py  # simple unit-style tests for Precision@k and Recall@k

main_test.py         # optional debug script (not required to run the system)

results/
(auto-generated)     # query result logs written by main.py

---

## Requirements

### Python Version
- Python 3.9+ recommended

### Install Dependencies
From the project root:

```bash
pip install beautifulsoup4 nltk pandas
```
### Download NLTK Resources

This project uses NLTK tokenisation + stopwords. Run once:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

If your version includes lemmatization/POS tagging experiments, also run:

```python
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
```
---

## How to Run

### 1) Run the Search Engine Demo (CLI)

This is what you show in your presentation demo.

From the src/ directory:

```bash
python main.py
```

You will be prompted:

- Type a query (e.g., Game published by Atari)
- The system prints the Top 10 ranked results
- Results are also saved to a timestamped file in the results/ folder
- Type exit to quit

Example queries:

- Pokémon Trozei
- Tony Hawk’s Downhill Jam
- Arcade type games
- London Taxi: Rush Hour
- Game published by Atari
- The Sims 2 Apartment Pets

### 2) Run Experiments (Evaluation + Required Queries)

This script runs your controlled evaluations (e.g., TF-IDF vs BM25 on required queries).

```bash
python experiments.py
```
Typical outputs include:
- Precision@10 / Recall@10 per query
- parisons across preprocessing configurations (stop-words/stemming, etc.)
- Printed Top-10 results (optionally for inspection)

### 3) Run Evaluation Metric Tests (Optional)

This is a small test harness used to validate evaluation functions.

```bash
python evaluation_tests.py
```
It prints expected vs observed values for Precision@k and Recall@k.

#### Notes on Output FIles

When running main.py, results are automatically saved to:
```bash
results/results_YYYYMMDD-HHMMSS.txt
```
Each file contains:
- the query string
- top-k doc IDs
- titles
- relevance scores

#### Notes on Optional Files

```bash
python main_tests.py
```
This file was used during development for debugging parsing/tokenisation/indexing.
It is not required for running the final system.

For submission cleanliness, it can be removed or moved into a scratch/ folder.

---

## Reproducibility

To ensure fair model comparisons, experiments are run under controlled conditions:
- same document collection
- same preprocessing configuration
- same index/IDF setup
- same relevance sets
- evaluation at the same cutoff (k=10)

This ensures any performance differences are attributable to the ranking method or experimental variable.

---

## Author

Kurt Canillas
