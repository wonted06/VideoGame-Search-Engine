# Videogame Search Engine  
*A Python-Based Information Retrieval System*

![Python](https://img.shields.io/badge/Python-IR-yellow)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Focus](https://img.shields.io/badge/Focus-Information%20Retrieval-purple)

---

## Overview

This project is a **ranked Information Retrieval (IR) search engine** implemented in Python over a collection of videogame HTML pages.  
The system indexes semi-structured web documents, processes free-text queries, and returns **ranked search results** using established IR models.

The project was developed as an **experimental IR system**, with a strong emphasis on **evaluation, comparison, and analysis** of retrieval techniques, rather than simply building a single search pipeline.

---

## Key Features

- HTML parsing with noise removal (scripts, styles, navigation, headers, footers)
- Configurable text preprocessing pipeline:
  - Tokenisation
  - Lowercasing
  - Stop-word removal
  - Stemming
- Inverted index construction
- Ranked retrieval using:
  - **TF-IDF**
  - **BM25**
- Query evaluation using:
  - Precision@k
  - Recall@k
- Command-line interface for interactive querying
- Experimental framework for controlled evaluations

---

## Core Functionality

The search engine performs the following steps:

1. Parse raw HTML videogame pages into clean textual representations
2. Apply configurable NLP preprocessing techniques
3. Build an inverted index over the document collection
4. Rank documents for a given query using TF-IDF or BM25
5. Evaluate retrieval effectiveness using labelled relevance data
6. Compare retrieval models and preprocessing strategies under controlled conditions

The system is designed to be **domain-independent**, allowing it to operate on other HTML-based datasets with minimal modification.

---

## Information Retrieval Techniques

- **Indexing**
  - Inverted index with document-level statistics
- **Ranking Models**
  - TF-IDF (vector space model)
  - BM25 (probabilistic retrieval model)
- **Preprocessing Experiments**
  - Stop-word removal
  - Stemming
  - Token normalisation
- **Evaluation**
  - Precision@10
  - Recall@10
  - Ranked result inspection and comparison

These techniques are analysed empirically through controlled experiments rather than assumed to be optimal.

---

## Dataset

The system uses a videogame dataset consisting of:

- **727 HTML documents** (individual videogame pages)
- A labelled CSV file containing metadata (e.g. publisher, genre)

| Data | Description |
|------------|------------|
| `Videogames/` | HTML documents |
| `videogame.csv` | Metadata and relevance labels |

The HTML collection is used for indexing and retrieval, while the CSV file supports **evaluation and relevance set construction**.

---

## Tech Stack

- **Python 3.9+**
- **BeautifulSoup** – HTML parsing
- **NLTK** – tokenisation, stop-words, stemming
- **Pandas** – dataset handling and evaluation support
- **Standard Python libraries** – data structures and I/O

---

## Project Structure
| Path / File | Description |
|------------|------------|
| `src/main.py` | Command-line search interface for interactive querying |
| `src/experiments.py` | Experimental pipeline for TF-IDF vs BM25 evaluations |
| `src/parser.py` | HTML parsing and noise removal |
| `src/tokeniser.py` | Text preprocessing pipeline (tokenisation, stop-words, stemming) |
| `src/indexer.py` | Inverted index construction |
| `src/ranker.py` | TF-IDF ranking, BM25 ranking, evaluation metrics |
| `src/evaluation_tests.py` | Validation tests for Precision@k and Recall@k |
| `src/main_test.py` | Optional development/debug script |
| `data/Videogames/` | HTML document collection (727 pages) |
| `data/videogame.csv` | Metadata and relevance labels |
| `results/` | Auto-generated ranked result outputs |
| `README.md` | Project documentation |

---

## How to Run
1. Install Dependencies
    ```bash
    pip install beautifulsoup4 nltk pandas
2. Download Required NLTK Resources
    ```python
    import nltk
    nltk.download("punkt")
    nltk(download("stopwords")
3. Optional, for extended experiments
    ```python
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")

---

## Run the Search Engine (CLI Demo)
From the src/ directory:
```bash
python main.py
```
	•	Enter a free-text query
	•	The system prints the Top-10 ranked results
	•	Results are saved to a timestamped file in results/
	•	Type exit to quit

Example queries:
	•	Pokémon Trozei
	•	Tony Hawk’s Downhill Jam
	•	Arcade type games
	•	Game published by Atari
	•	The Sims 2 Apartment Pets

⸻

## Run Experiments
```bash
python experiments.py
```
This executes controlled evaluations comparing:
	•	TF-IDF vs BM25
	•	Preprocessing configurations
	•	Retrieval effectiveness across required queries

Outputs include Precision@10, Recall@10, and ranked result logs.

⸻

## Reproducibility

To ensure fair comparisons, experiments are run under controlled conditions:
	•	Same document collection
	•	Same relevance sets
	•	Same preprocessing configuration
	•	Same evaluation cutoff (k = 10)

This ensures performance differences are attributable solely to the retrieval technique or experimental variable.

⸻

## Skills Demonstrated
	•	Information Retrieval system design
	•	Inverted index construction
	•	Ranking models (TF-IDF, BM25)
	•	NLP preprocessing techniques
	•	Experimental evaluation and analysis
	•	Python-based system design
	•	Reproducible experimentation

⸻

## Academic Context

This project was completed as part of CMP-5036A Information Retrieval, focusing on the design, evaluation, and critical analysis of search engines.
While academic in origin, the system reflects techniques used in real-world web search and ranking systems.

⸻

## License

This project is shared for educational and portfolio purposes only.
Please do not submit this work (or derivatives) as your own for academic assessment.

⸻

## Author

Kurt Canillas
Computer Science Undergraduate
