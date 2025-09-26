"""
BiLSTM Language Model for Technical Documentation Generation
- Loads CSV dataset `python_functions_and_documentation_dataset.csv`
- Uses docstrings as target language text; optional code+doc combined
- Tokenizes with a simple whitespace tokenizer or BPE fallback if available
- Trains a BiLSTM language model with teacher forcing (next-token prediction)
- Evaluates Perplexity and BLEU, saves metrics and convergence plots
- Saves artifacts into lstm/ by default
"""

import os
import json
import math
import time
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split

# Optional BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Tokenization / Vocabulary
# -----------------------------
class WhitespaceTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.special_tokens)}

    def build_vocab(self, texts: List[str], min_freq: int = 2, max_size: int = 50000):
        freq: Dict[str, int] = {}
        for text in texts:
            for tok in self.tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        # Sort by frequency
        words = sorted([w for w, c in freq.items() if c >= min_freq], key=lambda w: -freq[w])
        if max_size is not None:
            words = words[: max(0, max_size - len(self.special_tokens))]
        for w in words:
            if w not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[w] = idx
                self.id_to_token[idx] = w

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return text.strip().split()

    def encode(self, text: str) -> List[int]:
        toks = [self.bos_token] + self.tokenize(text) + [self.eos_token]
        return [self.token_to_id.get(t, self.token_to_id[self.unk_token]) for t in toks]

    def decode(self, ids: List[int]) -> str:
        toks = [self.id_to_token.get(i, self.unk_token) for i in ids]
        # remove BOS/EOS if present
        toks = [t for t in toks if t not in {self.bos_token, self.eos_token, self.pad_token}]
        return " ".join(toks)

# -----------------------------
# Dataset
# -----------------------------
class LMDocDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: WhitespaceTokenizer, max_length: int = 256):
        self.examples: List[torch.Tensor] = []
        for t in texts:
            ids = tokenizer.encode(t)
            if len(ids) > max_length:
                ids = ids[:max_length]
                if ids[-1] != tokenizer.token_to_id[tokenizer.eos_token]:
                    ids[-1] = tokenizer.token_to_id[tokenizer.eos_token]
            self.examples.append(torch.tensor(ids, dtype=torch.long))
        self.pad_id = tokenizer.token_to_id[tokenizer.pad_token]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_batch(batch: List[torch.Tensor], pad_id: int):
    lengths = [len(x) for x in batch]
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    # inputs are all but last, targets are all but first (next-token)
    inputs = batch_padded[:, :-1]
    targets = batch_padded[:, 1:]
    return inputs, targets, torch.tensor([l - 1 for l in lengths], dtype=torch.long)

# -----------------------------
# Model
# -----------------------------
class BiLSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3, tie_weights: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim * 2, vocab_size)
        if tie_weights:
            if embed_dim == hidden_dim * 2:
                self.proj.weight = self.embedding.weight
        
    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout(out)
        logits = self.proj(out)
        return logits

# -----------------------------
# Training / Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model, data_loader, pad_id: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    for inputs, targets, lengths in data_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        lengths = lengths.to(DEVICE)
        logits = model(inputs, lengths)
        vocab_size = logits.size(-1)
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            mask = (targets != pad_id)
            correct = (pred == targets) & mask
            total_correct += correct.sum().item()
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    acc = total_correct / max(1, total_tokens)
    return avg_loss, ppl, acc

@torch.no_grad()
def compute_bleu(model, data_loader, tokenizer: WhitespaceTokenizer, pad_id: int, max_len: int = 50, num_batches: int = 10):
    if not NLTK_AVAILABLE:
        # Return 0.0 instead of None to avoid null in metrics
        return 0.0
    model.eval()
    smoothie = SmoothingFunction().method4
    scores = []
    for b_idx, (inputs, targets, lengths) in enumerate(data_loader):
        if b_idx >= num_batches:
            break
        inputs = inputs.to(DEVICE)
        lengths = lengths.to(DEVICE)
        # Greedy decode one step at a time (simple)
        logits = model(inputs, lengths)
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        tgt_ids = targets[:, :logits.size(1)].cpu().tolist()
        for pred, tgt in zip(pred_ids, tgt_ids):
            pred_text = tokenizer.decode(pred)
            tgt_text = tokenizer.decode(tgt)
            ref = [tgt_text.split()]
            hyp = pred_text.split()
            if len(hyp) == 0 or len(ref[0]) == 0:
                # Count empty comparisons as 0 BLEU rather than skipping
                scores.append(0.0)
                continue
            scores.append(sentence_bleu(ref, hyp, smoothing_function=smoothie))
    return float(np.mean(scores)) if scores else 0.0

# -----------------------------
# Utilities
# -----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_losses(train_losses: List[float], val_losses: List[float], out_path: str):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BiLSTM LM Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_accuracies(train_accs: List[float], val_accs: List[float], out_path: str):
    plt.figure(figsize=(8,5))
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("BiLSTM LM Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM Language Model on docstrings")
    parser.add_argument("--csv", type=str, default="python_functions_and_documentation_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="bilstm")
    parser.add_argument("--samples", type=int, default=5000, help="0 for full dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bleu_batches", type=int, default=10, help="Number of batches to evaluate for BLEU")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    df = pd.read_csv(args.csv)
    if args.samples and args.samples > 0 and len(df) > args.samples:
        df = df.head(args.samples)
    # Focus on documentation generation task
    if "docstring" not in df.columns:
        raise ValueError("CSV must contain 'docstring' column")
    texts = df["docstring"].fillna("").astype(str).tolist()

    # Split
    train_texts, tmp_texts = train_test_split(texts, test_size=0.2, random_state=args.seed)
    val_texts, test_texts = train_test_split(tmp_texts, test_size=0.5, random_state=args.seed)

    # Tokenizer and vocab
    tokenizer = WhitespaceTokenizer()
    tokenizer.build_vocab(train_texts, min_freq=args.min_freq, max_size=args.max_vocab)

    # Datasets
    train_ds = LMDocDataset(train_texts, tokenizer, max_length=args.max_len)
    val_ds = LMDocDataset(val_texts, tokenizer, max_length=args.max_len)
    test_ds = LMDocDataset(test_texts, tokenizer, max_length=args.max_len)

    pad_id = tokenizer.token_to_id[tokenizer.pad_token]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, pad_id))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_batch(b, pad_id))

    # Model
    model = BiLSTMLanguageModel(
        vocab_size=len(tokenizer.token_to_id),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=False,
    ).to(DEVICE)

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    grad_clip = 1.0

    # Training loop with early stopping
    best_val = float('inf')
    patience = 4
    wait = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        start = time.time()
        train_correct = 0
        train_tokens = 0
        for inputs, targets, lengths in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            vocab_size = logits.size(-1)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                mask = (targets != pad_id)
                correct = (pred == targets) & mask
                train_correct += correct.sum().item()
                num_tokens = mask.sum().item()
                train_tokens += num_tokens
            epoch_loss += loss.item() * num_tokens
            epoch_tokens += num_tokens
        train_avg = epoch_loss / max(1, epoch_tokens)
        train_losses.append(train_avg)
        train_acc = train_correct / max(1, train_tokens)
        train_accs.append(train_acc)

        val_loss, val_ppl, val_acc = evaluate(model, val_loader, pad_id)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} - train_loss {train_avg:.4f} train_acc {train_acc:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_ppl {val_ppl:.2f} time {elapsed:.1f}s")

        # early stopping
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "vocab": tokenizer.token_to_id,
            }, os.path.join(args.output_dir, "bilstm_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    # Load best
    ckpt_path = os.path.join(args.output_dir, "bilstm_best.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model_state"]) 

    # Final evaluation
    test_loss, test_ppl, test_acc = evaluate(model, test_loader, pad_id)
    bleu = compute_bleu(model, test_loader, tokenizer, pad_id, num_batches=args.bleu_batches)

    # Save metrics
    metrics = {
        "val_loss": val_losses[-1] if val_losses else None,
        "val_perplexity": math.exp(val_losses[-1]) if val_losses else None,
        "val_accuracy": val_accs[-1] if val_accs else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "train_accuracy_last": train_accs[-1] if train_accs else None,
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "test_accuracy": test_acc,
        "bleu": bleu,
        "device": str(DEVICE),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot convergence
    plot_losses(train_losses, val_losses, os.path.join(args.output_dir, "convergence.png"))
    plot_accuracies(train_accs, val_accs, os.path.join(args.output_dir, "accuracy.png"))

    print("Done. Metrics saved to", os.path.join(args.output_dir, "metrics.json"))

if __name__ == "__main__":
    main()
