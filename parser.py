from bs4 import BeautifulSoup
from pathlib import Path
import re

# parse raw HTML into structured documents
def parse_html_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

# tag.decompose() removes specified boilerplate elements from the HTML code
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

# extracted title and body fields
    title = soup.title.string if soup.title else ""
    body = soup.body.get_text(" ") if soup.body else soup.get_text(" ")

# cleans whitespace to produce readable text
    def clean_text(text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    return {
        "doc_id": Path(filepath).name,
        "title": clean_text(title),
        "body": clean_text(body)
    }

def parse_collection(directory):
    directory = Path(directory)
    documents = []

    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        return documents

# Matches any .html file regardless of name
    for file in directory.glob("*.html"):
        documents.append(parse_html_file(file))

    return documents
