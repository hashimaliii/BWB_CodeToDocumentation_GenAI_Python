# Code-to-Doc: BPE + Word2Vec + BiLSTM Documentation Generation System

This repository implements an end-to-end documentation generation system for Python functions:

- Byte-Pair Encoding (BPE) tokenizer and evaluation
- Word2Vec embeddings with similarity analysis and visualizations
- BiLSTM language model for docstring generation with training/evaluation
- Unified documentation generator combining BPE + (optional) Word2Vec + BiLSTM
- Streamlit UI for single and batch generation

## Directory Structure

- `bpe/`: trained BPE pickle models (`bpe_code_model.pkl`, `bpe_doc_model.pkl`, `bpe_combined_model.pkl`)
- `adv_bpe/`: advanced BPE pickle models (if produced)
- `bpe_implementation.py`: BPE training/encode/decode
- `bpe/bpe_evaluation_results.json`: quantitative BPE evaluation report
- `word2vec_implementation.py`, `word2vec_analysis.py`: Word2Vec training, evaluation, visualization
- `bilstm/`: BiLSTM training outputs (checkpoint, metrics, plots)
- `train_bilstm.py`: BiLSTM training/evaluation script
- `docgen_system.py`: Unified documentation generation pipeline
- `app.py`: Streamlit UI
- `python_functions_and_documentation_dataset.csv`: dataset (code + docstrings)

## Setup

```bash
pip install -r requirements.txt
```

If you plan to use Word2Vec via gensim and Streamlit UI, both are already listed in `requirements.txt`.

## 1) BPE Tokenizer

Train/Use BPE (example usage inside the script):

```bash
python run_bpe.py
```

Artifacts:
- Models: `bpe/*.pkl`
- Evaluation JSON: `bpe/bpe_evaluation_results.json` (vocabulary overlap, compression ratio, boundary accuracy, consistency, OOV).

## 2) Word2Vec Embeddings

Train and analyze Word2Vec (gensim if available; custom fallback exists):

```bash
python word2vec_analysis.py --samples 5000
```

Outputs (in `word2vec/`):
- `word2vec_results.json` (semantic similarity, code completion proxy, doc relevance)
- `word2vec_pca_*.png`, `word2vec_tsne_*.png`

## 3) BiLSTM Language Model (Docstring Generation)

Train with default 10 epochs, outputs saved under `bilstm/` by default:

```bash
python train_bilstm.py --samples 5000
```

Artifacts (in `bilstm/`):
- `bilstm_best.pt` (checkpoint)
- `metrics.json` (loss, perplexity, accuracy, BLEU)
- `convergence.png`, `accuracy.png`

Notes:
- BLEU defaults to 0.0 if `nltk` is missing or no valid comparisons; increase coverage with `--bleu_batches 50`.

## 4) Unified Documentation Generator

Generate function summaries and docstrings by integrating BPE + optional Word2Vec + BiLSTM:

```bash
# Generate for 5k samples from the CSV
python docgen_system.py --samples 5000

# Single function
python docgen_system.py --single_code "def add(a,b): return a+b" --single_name add
```

Output: `docgen/generated_docs.json`.

## 5) Streamlit UI

Interactive single/batch generation with downloads:

```bash
streamlit run app.py
```

Sidebar lets you load the BiLSTM checkpoint (`bilstm/bilstm_best.pt`), optionally a Word2Vec model, and set `max_tokens`.

## Deliverables Mapping

- Deliverable 1 (BPE Implementation): `bpe_implementation.py`, models in `bpe/`
- Deliverable 2 (BPE Evaluation Report): `bpe/bpe_evaluation_results.json`
- Deliverable 3 (Word2Vec Implementation): `word2vec_implementation.py`, `word2vec_analysis.py`
- Deliverable 4 (Word2Vec Evaluation Report): results and plots in `word2vec/`
- Deliverable 5 (Language Model Implementation): `train_bilstm.py`, `bilstm/bilstm_best.pt`
- Deliverable 6 (Language Model Performance Analysis): `bilstm/metrics.json`, `convergence.png`, `accuracy.png`
- Deliverable 7 (Integrated System + UI): `docgen_system.py`, `app.py`

## Tips & Notes

- GPU is used automatically if available (PyTorch CUDA); otherwise CPU.
- Paths assume components live in the repository root; BPE models are expected under `bpe/`, BiLSTM under `bilstm/`.
- For Word2Vec with gensim, ensure `gensim` is installed and provide a path if you have a saved model.

## License

MIT
