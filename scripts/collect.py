"""Download and clean central bank documents for the RAG corpus and the tone classifier.

Core banks (for now): Fed (FOMC statements and minutes) and ECB (monetary policy
statements and accounts).

Outputs, in --out (default: data/):
    raw/<bank>/<doc_id>.html   cached downloads (ignored by git)
    documents.jsonl            one line per document, with numbered paragraphs
    sentences.csv              one line per sentence (input for labelling)
    test_docs.txt              documents held out for testing (never used for training)

Usage:
    python scripts/collect.py                       # Fed + ECB, 2022-2026
    python scripts/collect.py --banks ecb --start 2024 --end 2025
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = "central-bank-assistant student project (AIDAMS course)"
SLEEP_SECONDS = 1.0  # be polite: one request per second

FED_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FED_BASE = "https://www.federalreserve.gov"
ECB_BASE = "https://www.ecb.europa.eu"
ECB_LISTS = {
    "statement": ECB_BASE + "/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html",
    "account": ECB_BASE + "/press/accounts/{year}/html/index_include.en.html",
}

MIN_PARAGRAPH_CHARS = 40
MIN_SENTENCE_WORDS = 6


@dataclass
class Document:
    doc_id: str
    bank: str
    doc_type: str
    date: str  # YYYY-MM-DD (Fed: meeting/decision date; ECB: publication date)
    url: str
    title: str = ""
    paragraphs: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- download

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def fetch(url: str, cache: Path | None = None) -> str:
    """Download a page (or read it from the cache)."""
    if cache is not None and cache.exists():
        return cache.read_text(encoding="utf-8")
    time.sleep(SLEEP_SECONDS)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    html = response.text
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(html, encoding="utf-8")
    return html


def iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


# --------------------------------------------------------------------------- list documents

def list_fed(start: int, end: int) -> list[Document]:
    html = fetch(FED_CALENDAR)
    docs: dict[str, Document] = {}
    patterns = {
        "statement": r"/newsevents/pressreleases/monetary(\d{8})a\.htm",
        "minutes": r"/monetarypolicy/fomcminutes(\d{8})\.htm",
    }
    for doc_type, pattern in patterns.items():
        for match in re.finditer(pattern, html):
            date = match.group(1)
            if start <= int(date[:4]) <= end:
                doc_id = f"fed_{doc_type}_{date}"
                docs[doc_id] = Document(doc_id, "fed", doc_type, iso(date), FED_BASE + match.group(0))
    return sorted(docs.values(), key=lambda d: d.doc_id)


def list_ecb(start: int, end: int) -> list[Document]:
    docs: dict[str, Document] = {}
    prefixes = {"statement": "is", "account": "mg"}
    for doc_type, list_url in ECB_LISTS.items():
        prefix = prefixes[doc_type]
        for year in range(start, end + 1):
            try:
                html = fetch(list_url.format(year=year))
            except requests.HTTPError as error:
                print(f"  skip ECB {doc_type} list {year}: {error}")
                continue
            pattern = rf"/press/[\w\-/]+/{year}/html/ecb\.{prefix}(\d{{6}})~\w+\.en\.html"
            for match in re.finditer(pattern, html):
                date = "20" + match.group(1)
                doc_id = f"ecb_{doc_type}_{date}"
                docs[doc_id] = Document(doc_id, "ecb", doc_type, iso(date), ECB_BASE + match.group(0))
    return sorted(docs.values(), key=lambda d: d.doc_id)


# --------------------------------------------------------------------------- parse

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(html: str, bank: str, doc_type: str) -> tuple[str, list[str]]:
    """Return (title, paragraphs) from a document page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    title = clean(soup.title.get_text()) if soup.title else ""
    if bank == "fed":
        root = soup.select_one("#article") or soup.select_one("#content") or soup.body
    else:
        root = soup.select_one("main .section") or soup.select_one("main") or soup.body
    if root is None:
        return title, []

    paragraphs: list[str] = []
    for p in root.find_all(["p", "li"]):
        if p.find_parent("li") is not None and p.name == "p":
            continue  # avoid counting the same text twice
        text = clean(p.get_text(" "))
        if len(text) < MIN_PARAGRAPH_CHARS:
            continue
        if bank == "fed" and text.startswith(("For media inquiries", "Implementation Note", "Voting for")):
            continue
        paragraphs.append(text)
        # ECB statements: keep the statement, stop before the journalists' Q&A.
        if bank == "ecb" and doc_type == "statement" and "ready to take your questions" in text.lower():
            break
    return title, paragraphs


def split_sentences(text: str) -> list[str]:
    try:
        import pysbd

        segmenter = split_sentences.__dict__.setdefault("seg", pysbd.Segmenter(language="en", clean=False))
        sentences = segmenter.segment(text)
    except ImportError:  # simple fallback
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [clean(s) for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]


# --------------------------------------------------------------------------- test split

def choose_test_docs(docs: list[Document], per_bank: int, seed: int) -> list[str]:
    """Hold out whole documents for testing: `per_bank` per bank and document type."""
    rng = random.Random(seed)
    chosen: list[str] = []
    groups: dict[tuple[str, str], list[str]] = {}
    for doc in docs:
        groups.setdefault((doc.bank, doc.doc_type), []).append(doc.doc_id)
    for key in sorted(groups):
        ids = sorted(groups[key])
        chosen += rng.sample(ids, min(per_bank, len(ids)))
    return sorted(chosen)


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--banks", nargs="+", default=["fed", "ecb"], choices=["fed", "ecb"])
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--test-per-type", type=int, default=1,
                        help="documents held out per bank and document type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    listers = {"fed": list_fed, "ecb": list_ecb}
    docs: list[Document] = []
    for bank in args.banks:
        found = listers[bank](args.start, args.end)
        print(f"{bank}: {len(found)} documents found")
        docs += found

    kept: list[Document] = []
    for i, doc in enumerate(docs, 1):
        cache = args.out / "raw" / doc.bank / f"{doc.doc_id}.html"
        try:
            html = fetch(doc.url, cache)
        except requests.RequestException as error:
            print(f"  [{i}/{len(docs)}] FAILED {doc.doc_id}: {error}")
            continue
        doc.title, paragraphs = extract_paragraphs(html, doc.bank, doc.doc_type)
        doc.paragraphs = [{"n": n, "text": text} for n, text in enumerate(paragraphs, 1)]
        if not doc.paragraphs:
            print(f"  [{i}/{len(docs)}] EMPTY {doc.doc_id} (check the parser)")
            continue
        kept.append(doc)
        print(f"  [{i}/{len(docs)}] {doc.doc_id}: {len(doc.paragraphs)} paragraphs")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "documents.jsonl", "w", encoding="utf-8") as f:
        for doc in kept:
            f.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")

    n_sentences = 0
    with open(args.out / "sentences.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sent_id", "doc_id", "bank", "doc_type", "date", "paragraph", "text"])
        for doc in kept:
            for para in doc.paragraphs:
                for k, sentence in enumerate(split_sentences(para["text"]), 1):
                    writer.writerow([f"{doc.doc_id}_p{para['n']}_s{k}", doc.doc_id, doc.bank,
                                     doc.doc_type, doc.date, para["n"], sentence])
                    n_sentences += 1

    test_file = args.out / "test_docs.txt"
    if test_file.exists():
        print(f"{test_file} already exists, keeping it (delete it to draw a new split)")
    else:
        test_docs = choose_test_docs(kept, args.test_per_type, args.seed)
        test_file.write_text("\n".join(test_docs) + "\n", encoding="utf-8")
        print(f"held out {len(test_docs)} test documents -> {test_file}")

    print(f"done: {len(kept)} documents, {n_sentences} sentences -> {args.out}")


if __name__ == "__main__":
    main()
