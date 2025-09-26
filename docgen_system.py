"""
Unified Documentation Generation System (Tasks 8 and 9)
- Integrates BPE tokenizer, Word2Vec embeddings (optional), and BiLSTM LM
- Generates function summaries and detailed docstrings with context-aware decoding
- CLI supports running over CSV dataset or a single function/code input
- Outputs saved under docgen/
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import pandas as pd
import torch
import torch.nn as nn

# Import BiLSTM and tokenizer components from train_bilstm
from train_bilstm import WhitespaceTokenizer, BiLSTMLanguageModel

# Optional Word2Vec (avoid hard dependency during import)
try:
    from gensim.models import Word2Vec  # type: ignore
    GENSIM_AVAILABLE = True
except Exception:
    Word2Vec = Any  # type: ignore
    GENSIM_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# BPE Loader (fallback to whitespace)
# -----------------------------
import pickle

def load_bpe_models() -> Dict[str, Any]:
    paths = {
        "code": os.path.join("bpe", "bpe_code_model.pkl"),
        "doc": os.path.join("bpe", "bpe_doc_model.pkl"),
        "combined": os.path.join("bpe", "bpe_combined_model.pkl"),
    }
    models: Dict[str, Any] = {}
    for k, p in paths.items():
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    models[k] = pickle.load(f)
            except Exception:
                pass
    return models

# -----------------------------
# Word2Vec Loader (optional)
# -----------------------------

def load_word2vec_model(path: Optional[str]) -> Optional[Any]:
    if not (GENSIM_AVAILABLE and path and os.path.exists(path)):
        return None
    try:
        return Word2Vec.load(path)  # type: ignore[attr-defined]
    except Exception:
        return None

# -----------------------------
# BiLSTM Loader
# -----------------------------

def load_bilstm(checkpoint_path: str) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    config = state.get("config", {})
    vocab = state.get("vocab")
    if vocab is None:
        raise ValueError("Checkpoint missing vocabulary")
    tokenizer = WhitespaceTokenizer()
    tokenizer.token_to_id = vocab
    tokenizer.id_to_token = {i: t for t, i in vocab.items()}

    model = BiLSTMLanguageModel(
        vocab_size=len(vocab),
        embed_dim=config.get("embed_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        tie_weights=False,
    ).to(DEVICE)
    model.load_state_dict(state["model_state"])  # type: ignore[index]
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

# -----------------------------
# Generation utilities
# -----------------------------
@torch.no_grad()
def greedy_generate(model: BiLSTMLanguageModel, tokenizer: WhitespaceTokenizer, prompt: str, 
                   max_new_tokens: int = 80) -> str:
    ids = tokenizer.encode(prompt)
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    # iterative greedy next-token generation
    for _ in range(max_new_tokens):
        lengths = torch.tensor([inp.size(1)], dtype=torch.long, device=DEVICE)
        logits = model(inp, lengths)
        next_id = int(torch.argmax(logits[0, -1]))
        inp = torch.cat([inp, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)], dim=1)
        if tokenizer.id_to_token.get(next_id) == tokenizer.eos_token:
            break
    return tokenizer.decode(inp[0].tolist())

# Optional context features using Word2Vec similarity

def build_context_hint(code_text: str, w2v: Optional[Any], top_k: int = 5) -> str:
    if w2v is None:
        return ""
    tokens = code_text.split()
    hints: List[str] = []
    for t in tokens[:10]:
        if hasattr(w2v, 'wv') and (t in w2v.wv):  # type: ignore[operator]
            sims = w2v.wv.most_similar(t, topn=2)  # type: ignore[call-arg]
            hints.extend([s for s, _ in sims])
    if not hints:
        return ""
    return " Context: " + " ".join(hints[:top_k])

# BPE tokenize (fallback to whitespace)

def bpe_tokenize(text: str, bpe_models: Dict[str, Any], kind: str = "combined") -> List[str]:
    model = bpe_models.get(kind)
    if model is None:
        return text.strip().split()
    # Support both fast_bpe model and potential dict; try attribute detection
    if hasattr(model, 'fast_tokenize'):
        return model.fast_tokenize(text)
    if hasattr(model, 'tokenize'):
        return model.tokenize(text)  # type: ignore[attr-defined]
    return text.strip().split()

# -----------------------------
# Main System
# -----------------------------

def generate_for_function(code: str, name: Optional[str], bilstm: Dict[str, Any], bpe_models: Dict[str, Any], 
                           w2v: Optional[Any], max_tokens: int, bpe_kind: str = "combined") -> Dict[str, str]:
    model = bilstm["model"]
    tokenizer = bilstm["tokenizer"]

    # Build prompt
    code_tokens = bpe_tokenize(code, bpe_models, kind=bpe_kind)
    base_prompt = f"Summarize function {name or ''}: " + " ".join(code_tokens[:100])
    # Word2Vec context hint (optional)
    context_hint = build_context_hint(" ".join(code_tokens), w2v)
    prompt = base_prompt + context_hint

    summary = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_tokens//2)
    doc_prompt = f"Detailed docstring for {name or 'function'}: " + summary
    docstring = greedy_generate(model, tokenizer, doc_prompt, max_new_tokens=max_tokens)

    return {"summary": summary, "docstring": docstring}


def main():
    parser = argparse.ArgumentParser(description="Unified Documentation Generation System")
    parser.add_argument("--csv", type=str, default="python_functions_and_documentation_dataset.csv")
    parser.add_argument("--code_col", type=str, default="code")
    parser.add_argument("--name_col", type=str, default="function_name")
    parser.add_argument("--output", type=str, default="docgen")
    parser.add_argument("--bilstm_ckpt", type=str, default=os.path.join("bilstm", "bilstm_best.pt"))
    parser.add_argument("--word2vec_path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--single_code", type=str, default=None, help="Provide single code snippet to generate doc")
    parser.add_argument("--single_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load components
    bpe_models = load_bpe_models()
    w2v = load_word2vec_model(args.word2vec_path)
    bilstm = load_bilstm(args.bilstm_ckpt)

    outputs: List[Dict[str, Any]] = []

    if args.single_code is not None:
        res = generate_for_function(args.single_code, args.single_name, bilstm, bpe_models, w2v, args.max_tokens)
        outputs.append({
            "name": args.single_name,
            "summary": res["summary"],
            "docstring": res["docstring"]
        })
    else:
        df = pd.read_csv(args.csv)
        if args.samples and len(df) > args.samples:
            df = df.head(args.samples)
        for _, row in df.iterrows():
            code = str(row.get(args.code_col, ""))
            name = row.get(args.name_col)
            res = generate_for_function(code, name, bilstm, bpe_models, w2v, args.max_tokens)
            outputs.append({
                "name": name,
                "summary": res["summary"],
                "docstring": res["docstring"]
            })

    # Save outputs
    out_path = os.path.join(args.output, "generated_docs.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"Saved generated documentation to {out_path}")

if __name__ == "__main__":
    main()
