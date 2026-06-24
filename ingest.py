import requests
from bs4 import BeautifulSoup
import re
import os

# --- The four papers: arXiv ID-based URL + a short output name ---
PAPERS = [
    {"name": "attention", "url": "https://ar5iv.labs.arxiv.org/html/1706.03762"},
    {"name": "bert",      "url": "https://ar5iv.labs.arxiv.org/html/1810.04805"},
    {"name": "rag",       "url": "https://ar5iv.labs.arxiv.org/html/2005.11401"},
    {"name": "gpt3",      "url": "https://ar5iv.labs.arxiv.org/html/2005.14165"},
]


def extract_text(url):
    """Download an ar5iv page and return its readable paper text."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    article = soup.find("article") or soup.body
    return article.get_text(separator="\n", strip=True)


def clean_text(text):
    """Apply our two lossless cleaning rules."""
    # Rule 1: tidy broken citation brackets
    def tidy_citation(match):
        inside = re.sub(r"\s+", " ", match.group(1)).strip()
        return f"[{inside}]"
    text = re.sub(r"\[([\s\d,;]+?)\]", tidy_citation, text)

    # Rule 2: collapse runs of blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text


# --- Main loop: extract -> clean -> save, for each paper ---
os.makedirs("data", exist_ok=True)

for paper in PAPERS:
    name, url = paper["name"], paper["url"]
    print(f"\nProcessing {name} ...")

    raw = extract_text(url)
    cleaned = clean_text(raw)

    out_path = f"data/{name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"  {len(raw)} chars extracted -> {len(cleaned)} chars cleaned -> {out_path}")

print("\nDone. All four papers processed.")