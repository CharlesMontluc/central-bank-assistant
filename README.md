# Central Bank Communications Assistant

Cited answers and policy tone for central bank communications, beyond the Fed.
Group project, Business Data Science (AIDAMS).

Team: Lucas Magret Delord, Erdem Sanver, Charles Montluc, Adrien Labergere.

## What it does

1. **RAG**: ask a question about central bank statements and minutes, get a short answer that cites
   the bank, date and paragraph, or a refusal when the answer is not in the documents.
2. **Fine-tuning**: a small encoder model tags the tone of each cited sentence as hawkish, neutral or dovish.

Demo: Gradio app on a Hugging Face Space (link to come).

## Data

- **Core:** Fed FOMC statements and minutes; ECB monetary policy statements and accounts, 2022-2026.
- **If time allows:** BoE Monetary Policy Summaries and minutes; BoJ statements and Summaries of Opinions (English).
- **Training labels:** sentences labelled by an LLM with the annotation guide of Shah et al. (ACL 2023).
- **Gold test set:** ~120 sentences from held-out documents, labelled independently by two team members.
  All classifier results are reported on this set.

## Repository layout

```
data/          documents.jsonl, sentences.csv, test_docs.txt (raw downloads in data/raw/)
scripts/       download and cleaning scripts
rag/           chunking, embeddings, retrieval, cited answers, RAG evaluation
classifier/    LLM labelling, gold set, fine-tuning, classifier evaluation
notebooks/     Colab experiments
report/        final report (PDF)
app.py         Gradio app for the Hugging Face Space
```

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Full step-by-step instructions will be added as each part is built.

## Working rules

- Do not push directly to `main`: work on a branch and open a pull request.
- Someone else reviews the pull request before it is merged.
- Never commit API keys. Put them in a local `.env` file (ignored by git) or in Hugging Face Space secrets.
