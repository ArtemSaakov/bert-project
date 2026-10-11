::: IEEEkeywords
Metadata Classification, Digital Curation, Transformer Models, BERT,
Text Classification, Archival Metadata, Natural Language Processing
:::

# Introduction

Metadata underpins discovery, provenance, and preservation in digital
archives. Yet many institutions face backlogs: donated items arrive
faster than they can be cataloged, and donor-provided metadata---often
stored in spreadsheets, text files, or embedded tags---lacks structure
or consistency [@NARA_AI]. Manually mapping each snippet to standardized
fields (e.g., Title, Date, Creator) is labor-intensive.

## Project Goal

We investigate fine-tuning Google's BERT-base-uncased model to
automatically classify free-form metadata snippets into a fixed set of
archival fields. By leveraging BERT's bidirectional contextual
embeddings, we aim to reduce manual mapping effort and improve
consistency.

## Related Work

The National Archives have explored AI for metadata tagging to improve
public access [@NARA_AI]. Carnegie Mellon's CAMPI project used computer
vision to cluster and tag photo collections in bulk [@CMU_CAMPI].
MetaEnhance applied transformer models to correct ETD metadata errors
with F1 $>$ 0.85 on key fields [@MetaEnhance]. Embedding-based entity
resolution has harmonized heterogeneous schemas across datasets
[@Sawarkar2020]. These studies demonstrate AI's potential but leave open
the challenge of mapping arbitrary donor text to discrete fields.

# Method

## Problem Formulation

We cast metadata field mapping as single-label text classification:

- **Input:** free-form snippet $x$ (string).

- **Output:** field label $y \in \{f_1, \dots, f_K\}$, each $f_i$ a
  target schema field.

## Dataset Preparation

We begin with an aggregated JSON document keyed by Dublin Core field
names. A Python script (`harvest_aggregate.ipynb`) flattens this into
one record per metadata entry:

    {"text":"Acquired on 12/31/2024","label":"Date"}

Synthetic expansion to 200 examples across ten fields ensures coverage
of varied formats.

## Model Fine-Tuning

- **Model:** `bert-base-uncased` with $K=10$ labels.

- **Tokenizer:** WordPiece, padding/truncation to 128 tokens.

- **Training:** 80/20 split, cross-entropy loss, LR=2e-5, batch size=8,
  5 epochs via Hugging Face `Trainer` [@Wolf2020].

- **Evaluation:** Accuracy, weighted and macro F1, precision, and recall
  using the `evaluate` library.

## Inference Pipeline

We package our inference logic in `bertley.py`. It loads the fine-tuned
model, tokenizes input (text or file), and handles documents longer than
512 tokens by chunking with overlap (stride=50). Pseudocode excerpt:

    # Load model & tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    classifier = pipeline("text-classification", 
                          model=model, 
                          tokenizer=tokenizer, 
                          return_all_scores=True)

    # For long texts, split into overlapping chunks
    def chunk_and_classify(text):
      tokens = tokenizer(text)['input_ids'][0]
      for i in range(0, len(tokens), max_len - stride):
        chunk = tokenizer.decode(tokens[i:i+max_len])
        scores = classifier(chunk)
        accumulate(scores)
      return average_scores()

This script achieves robust, batch-ready inference for entire documents.

# Results

## Evaluation Metrics

After fine-tuning for 5 epochs, we evaluated on the test set.
Table [1](#tab:eval_metrics){reference-type="ref"
reference="tab:eval_metrics"} summarizes the results:

  **Metric**              **Value**
  ---------------------- -----------
  Loss                     0.1338
  Accuracy                 0.9665
  F1 (weighted)            0.9628
  Precision (weighted)     0.9650
  Recall (weighted)        0.9665
  F1 (macro)               0.8283
  Precision (macro)        0.8551
  Recall (macro)           0.8225
  Runtime (s)               35.83
  Samples/sec              518.70
  Steps/sec                 16.22

  : Test Set Evaluation Metrics {#tab:eval_metrics}

## Interpretation

Overall accuracy of 96.65% and weighted F1 of 96.28% demonstrate
reliable field mapping. The macro F1 (82.83%) suggests room for
improvement on rarer or more ambiguous classes. Inference speed ( 100
snippets/s on GPU) is sufficient for large-scale backlog processing.

# Conclusion

Fine-tuning BERT-base-uncased for metadata classification yields an
overall accuracy of 0.92, confirming the viability of transformer-based
automation in digital curation. Future work will integrate real EAD
finding aids, implement multi-label classification for ambiguous
entries, and incorporate human-in-the-loop validation.

# Acknowledgment {#acknowledgment .unnumbered}

The author thanks the University of Michigan School of Information and
participating archival staff for insights into donor metadata workflows.

::: thebibliography
1 U.S. National Archives and Records Administration, "Artificial
intelligence at the National Archives." \[Online\]. Available:
<https://www.archives.gov/ai>, accessed Apr. 4, 2025.

Carnegie Mellon Univ. Libraries, "Computer vision archive helps
streamline metadata tagging," Oct. 2020. \[Online\]. Available:
<https://www.cmu.edu/news/stories/archives/2020/october/computer-vision-archive.html>.

M. H. Choudhury *et al.*, "MetaEnhance: Metadata Quality Improvement for
Electronic Theses and Dissertations," *arXiv*, Mar. 2023.

K. Sawarkar and M. Kodati, "Automated metadata harmonization using
entity resolution & contextual embedding," *arXiv*, Oct. 2020.

T. Wolf *et al.*, "HuggingFace Transformers: State-of-the-art natural
language processing," in *Proc. EMNLP: Findings*, 2020, pp. 8201--8210.
:::
