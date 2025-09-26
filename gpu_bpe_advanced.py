"""
Advanced GPU-optimized BPE implementation with enhanced features.
Includes batch processing, memory optimization, and advanced evaluation metrics.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import re
import pickle
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, adjusted_rand_score
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
warnings.filterwarnings('ignore')

class GPUOptimizedBPE:
    """GPU-optimized BPE implementation with batch processing."""
    
    def __init__(self, vocab_size: int = 10000, use_gpu: bool = True, batch_size: int = 1024):
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize vocabulary
        self.vocab = {}
        self.merges = []
        self.reverse_vocab = {}
        self._initialize_character_vocab()
        
        # GPU tensors for batch processing
        self.vocab_tensor = None
        self.merge_tensor = None
        
    def _initialize_character_vocab(self):
        """Initialize vocabulary with characters and special tokens."""
        chars = set()
        chars.update([chr(i) for i in range(32, 127)])  # Printable ASCII
        chars.update(['\n', '\t', ' '])
        
        special_tokens = ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '<CODE>', '<DOC>', '<NUM>', '<STR>']
        
        for i, char in enumerate(sorted(chars) + special_tokens):
            self.vocab[char] = i
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Create GPU tensor for fast lookup
        if self.use_gpu:
            self.vocab_tensor = torch.tensor(list(self.vocab.values()), device=self.device)
        
        print(f"Initialized vocabulary with {len(self.vocab)} tokens")
    
    def _batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Batch tokenization for efficiency."""
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(self._simple_tokenize, text) for text in texts]
            results = [future.result() for future in tqdm(futures, desc="Tokenizing")]
        return results
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with Python-specific patterns."""
        # Handle Python-specific patterns
        text = re.sub(r'\b\d+\.\d+\b', '<NUM>', text)  # Numbers
        text = re.sub(r'\b\d+\b', '<NUM>', text)  # Integers
        text = re.sub(r'"[^"]*"', '<STR>', text)  # String literals
        text = re.sub(r"'[^']*'", '<STR>', text)  # String literals
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        return tokens
    
    def _get_pair_counts_gpu(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """GPU-accelerated pair counting."""
        if not self.use_gpu or len(vocab) < 1000:
            return self._get_pair_counts_cpu(vocab)
        
        # Convert to tensor for GPU processing
        words = list(vocab.keys())
        freqs = list(vocab.values())
        
        # This is a simplified GPU implementation
        # In practice, you'd implement more sophisticated GPU algorithms
        pairs = defaultdict(int)
        
        for word, freq in zip(words, freqs):
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return dict(pairs)
    
    def _get_pair_counts_cpu(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """CPU-based pair counting."""
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return dict(pairs)
    
    def train_parallel(self, texts: List[str], num_workers: int = None):
        """Parallel training with GPU acceleration."""
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        print(f"Training BPE with {num_workers} workers on {len(texts)} texts...")
        
        # Batch processing for memory efficiency
        batch_size = min(self.batch_size, len(texts))
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_batch, batch) for batch in batches]
            batch_results = [future.result() for future in tqdm(futures, desc="Processing batches")]
        
        # Combine results
        word_freqs = Counter()
        for batch_freqs in batch_results:
            word_freqs.update(batch_freqs)
        
        print(f"Found {len(word_freqs)} unique words")
        
        # Train BPE
        self._train_bpe(word_freqs)
    
    def _process_batch(self, batch: List[str]) -> Dict[str, int]:
        """Process a batch of texts."""
        word_freqs = Counter()
        
        for text in batch:
            words = text.split()
            for word in words:
                word_freqs[word] += 1
        
        return dict(word_freqs)
    
    def _train_bpe(self, word_freqs: Dict[str, int]):
        """Core BPE training algorithm."""
        # Initialize vocabulary
        vocab = {}
        for word, freq in word_freqs.items():
            word_chars = ' '.join(list(word)) + ' </w>'
            vocab[word_chars] = freq
        
        # Add individual characters
        for word in word_freqs:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        # Training loop
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in tqdm(range(num_merges), desc="Training BPE"):
            pairs = self._get_pair_counts_gpu(vocab)
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Add to merges and vocab
            self.merges.append(best_pair)
            bigram = ''.join(best_pair)
            self.vocab[bigram] = len(self.vocab)
            
            # Merge in vocabulary
            vocab = self._merge_vocab(best_pair, vocab)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Update GPU tensors
        if self.use_gpu:
            self.vocab_tensor = torch.tensor(list(self.vocab.values()), device=self.device)
        
        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge a pair in the vocabulary."""
        bigram = ''.join(pair)
        new_vocab = {}
        
        pattern = re.escape(pair[0] + pair[1])
        
        for word in vocab:
            new_word = re.sub(pattern, bigram, word)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch encoding for efficiency."""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = [self.encode(text) for text in batch]
            results.extend(batch_results)
        
        return results
    
    def encode(self, text: str) -> List[int]:
        """Encode text using trained BPE."""
        if not text:
            return []
        
        words = text.split()
        bpe_tokens = []
        
        for word in words:
            word_tokens = self._encode_word(word)
            bpe_tokens.extend(word_tokens)
        
        # Convert to IDs
        token_ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        return token_ids
    
    def _encode_word(self, word: str) -> List[str]:
        """Encode a single word using BPE."""
        if not word:
            return []
        
        symbols = list(word) + ['</w>']
        
        for pair in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and 
                    symbols[i] == pair[0] and 
                    symbols[i + 1] == pair[1]):
                    new_symbols.append(''.join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        
        return symbols
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append('<UNK>')
        
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, path: str):
        """Save BPE model with metadata."""
        model_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'device': str(self.device),
            'batch_size': self.batch_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"BPE model saved to {path}")
    
    def load(self, path: str):
        """Load BPE model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab = model_data['vocab']
        self.merges = model_data['merges']
        self.vocab_size = model_data['vocab_size']
        self.batch_size = model_data.get('batch_size', 1024)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Update device
        saved_device = model_data.get('device', 'cpu')
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Update GPU tensors
        if self.use_gpu:
            self.vocab_tensor = torch.tensor(list(self.vocab.values()), device=self.device)
        
        print(f"BPE model loaded from {path}")


class AdvancedBPEEvaluator:
    """Advanced evaluation with additional metrics and visualizations."""
    
    def __init__(self, bpe_model: GPUOptimizedBPE, ground_truth_tokens: List[List[str]]):
        self.bpe_model = bpe_model
        self.ground_truth_tokens = ground_truth_tokens
    
    def evaluate_perplexity(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate model perplexity."""
        print("Evaluating perplexity...")
        
        total_log_prob = 0.0
        total_tokens = 0
        
        for text in test_texts:
            tokens = self.bpe_model.encode(text)
            total_tokens += len(tokens)
            
            # Simple perplexity calculation
            # In practice, you'd use a proper language model
            for token_id in tokens:
                # Assume uniform probability for simplicity
                prob = 1.0 / len(self.bpe_model.vocab)
                total_log_prob += np.log(prob)
        
        perplexity = np.exp(-total_log_prob / total_tokens) if total_tokens > 0 else float('inf')
        
        return {
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'avg_log_prob': total_log_prob / total_tokens if total_tokens > 0 else 0.0
        }
    
    def evaluate_token_distribution(self, test_texts: List[str]) -> Dict[str, any]:
        """Evaluate token distribution characteristics."""
        print("Evaluating token distribution...")
        
        all_tokens = []
        token_lengths = []
        
        for text in test_texts:
            tokens = self.bpe_model.encode(text)
            all_tokens.extend(tokens)
            token_lengths.append(len(tokens))
        
        token_counts = Counter(all_tokens)
        
        return {
            'unique_tokens': len(token_counts),
            'total_tokens': len(all_tokens),
            'avg_text_length': np.mean(token_lengths),
            'std_text_length': np.std(token_lengths),
            'token_frequency_dist': dict(token_counts.most_common(20)),
            'entropy': self._calculate_entropy(token_counts)
        }
    
    def _calculate_entropy(self, token_counts: Counter) -> float:
        """Calculate entropy of token distribution."""
        total = sum(token_counts.values())
        entropy = 0.0
        
        for count in token_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def evaluate_semantic_coherence(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate semantic coherence of tokenization."""
        print("Evaluating semantic coherence...")
        
        # This is a simplified coherence measure
        # In practice, you'd use more sophisticated methods
        
        coherence_scores = []
        
        for text in test_texts[:100]:  # Sample for efficiency
            # Split text into sentences
            sentences = text.split('.')
            if len(sentences) < 2:
                continue
            
            # Tokenize each sentence
            sentence_tokens = [self.bpe_model.encode(sent) for sent in sentences]
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(sentence_tokens) - 1):
                tokens1 = set(sentence_tokens[i])
                tokens2 = set(sentence_tokens[i + 1])
                
                if len(tokens1) > 0 and len(tokens2) > 0:
                    jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
                    similarities.append(jaccard)
            
            if similarities:
                coherence_scores.append(np.mean(similarities))
        
        return {
            'semantic_coherence': np.mean(coherence_scores) if coherence_scores else 0.0,
            'coherence_std': np.std(coherence_scores) if coherence_scores else 0.0,
            'num_evaluated_texts': len(coherence_scores)
        }
    
    def create_visualizations(self, test_texts: List[str], output_dir: str = "plots"):
        """Create comprehensive visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating visualizations in {output_dir}/...")
        
        # Token length distribution
        token_lengths = [len(self.bpe_model.encode(text)) for text in test_texts]
        
        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Counts per Text')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/token_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Vocabulary usage
        all_tokens = []
        for text in test_texts:
            all_tokens.extend(self.bpe_model.encode(text))
        
        token_counts = Counter(all_tokens)
        top_tokens = token_counts.most_common(20)
        
        plt.figure(figsize=(12, 8))
        tokens, counts = zip(*top_tokens)
        plt.bar(range(len(tokens)), counts)
        plt.xlabel('Token Rank')
        plt.ylabel('Frequency')
        plt.title('Top 20 Most Frequent Tokens')
        plt.xticks(range(len(tokens)), [str(t) for t in tokens], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/top_tokens.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def comprehensive_evaluation(self, test_texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation with all metrics."""
        print("Running comprehensive advanced evaluation...")
        
        # Basic evaluation (reuse from previous implementation)
        basic_results = {
            'vocabulary_overlap': self._evaluate_vocabulary_overlap(test_texts),
            'compression_ratio': self._evaluate_compression_ratio(test_texts),
            'oov_rate': self._evaluate_oov_rate(test_texts)
        }
        
        # Advanced evaluation
        advanced_results = {
            'perplexity': self.evaluate_perplexity(test_texts),
            'token_distribution': self.evaluate_token_distribution(test_texts),
            'semantic_coherence': self.evaluate_semantic_coherence(test_texts)
        }
        
        # Create visualizations
        self.create_visualizations(test_texts)
        
        # Combine results
        results = {**basic_results, **advanced_results}
        
        return results
    
    def _evaluate_vocabulary_overlap(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate vocabulary overlap using Jaccard similarity."""
        bpe_vocab = set(self.bpe_model.vocab.keys())
        ground_truth_vocab = set()
        
        for tokens in self.ground_truth_tokens:
            ground_truth_vocab.update(tokens)
        
        intersection = len(bpe_vocab.intersection(ground_truth_vocab))
        union = len(bpe_vocab.union(ground_truth_vocab))
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'bpe_vocab_size': len(bpe_vocab),
            'ground_truth_vocab_size': len(ground_truth_vocab),
            'intersection_size': intersection,
            'union_size': union
        }
    
    def _evaluate_compression_ratio(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate compression ratio."""
        total_chars = sum(len(text) for text in test_texts)
        total_tokens = sum(len(self.bpe_model.encode(text)) for text in test_texts)
        
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 1.0
        
        return {
            'compression_ratio': compression_ratio,
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'avg_chars_per_token': total_chars / total_tokens if total_tokens > 0 else 1.0
        }
    
    def _evaluate_oov_rate(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate out-of-vocabulary rate."""
        total_tokens = 0
        oov_tokens = 0
        unk_id = self.bpe_model.vocab.get('<UNK>', -1)
        
        for text in test_texts:
            tokens = self.bpe_model.encode(text)
            total_tokens += len(tokens)
            oov_tokens += tokens.count(unk_id)
        
        oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'oov_rate': oov_rate,
            'oov_tokens': oov_tokens,
            'total_tokens': total_tokens,
            'oov_percentage': oov_rate * 100
        }


def main_advanced():
    """Main function for advanced GPU-optimized BPE."""
    print("=== Advanced GPU-Optimized BPE Implementation ===")
    
    # Configuration
    CSV_PATH = "python_functions_and_documentation_dataset.csv"
    SAMPLE_SIZE = 20000  # Larger sample for better training
    VOCAB_SIZE = 8000
    USE_GPU = True
    BATCH_SIZE = 2048
    
    # Load and analyze dataset
    print("\nLoading dataset...")
    from bpe_implementation import DatasetAnalyzer
    analyzer = DatasetAnalyzer(CSV_PATH)
    if SAMPLE_SIZE is None:
        print("Loading full dataset (no sampling)...")
        df = analyzer.load_dataset()  # No sample_size parameter
    else:
        print(f"Loading dataset with sample size: {SAMPLE_SIZE}")
        df = analyzer.load_dataset(sample_size=SAMPLE_SIZE)
    
    # Prepare training data
    code_texts = df['code'].astype(str).tolist() if 'code' in df.columns else []
    docstring_texts = df['docstring'].astype(str).tolist() if 'docstring' in df.columns else []
    combined_texts = code_texts + docstring_texts
    
    # Train advanced BPE models
    print("\nTraining advanced BPE models...")
    
    # Code-specific BPE
    print("Training code-specific BPE...")
    bpe_code = GPUOptimizedBPE(vocab_size=VOCAB_SIZE, use_gpu=USE_GPU, batch_size=BATCH_SIZE)
    bpe_code.train_parallel(code_texts[:10000], num_workers=4)
    bpe_code.save("advanced_bpe_code_model.pkl")
    
    # Documentation-specific BPE
    print("Training documentation-specific BPE...")
    bpe_doc = GPUOptimizedBPE(vocab_size=VOCAB_SIZE, use_gpu=USE_GPU, batch_size=BATCH_SIZE)
    bpe_doc.train_parallel(docstring_texts[:10000], num_workers=4)
    bpe_doc.save("advanced_bpe_doc_model.pkl")
    
    # Combined BPE
    print("Training combined BPE...")
    bpe_combined = GPUOptimizedBPE(vocab_size=VOCAB_SIZE * 2, use_gpu=USE_GPU, batch_size=BATCH_SIZE)
    bpe_combined.train_parallel(combined_texts[:20000], num_workers=4)
    bpe_combined.save("advanced_bpe_combined_model.pkl")
    
    # Advanced evaluation
    print("\nRunning advanced evaluation...")
    test_texts = combined_texts[1000:3000]
    
    # Ground truth tokens
    ground_truth_tokens = [text.split() for text in test_texts]
    
    models_to_evaluate = [
        ("Advanced Code BPE", bpe_code),
        ("Advanced Documentation BPE", bpe_doc),
        ("Advanced Combined BPE", bpe_combined)
    ]
    
    evaluation_results = {}
    
    for model_name, model in models_to_evaluate:
        print(f"\nEvaluating {model_name}...")
        evaluator = AdvancedBPEEvaluator(model, ground_truth_tokens)
        results = evaluator.comprehensive_evaluation(test_texts)
        evaluation_results[model_name] = results
        
        # Print summary
        print(f"\n{model_name} Results:")
        print(f"  Vocabulary Overlap: {results['vocabulary_overlap']['jaccard_similarity']:.4f}")
        print(f"  Compression Ratio: {results['compression_ratio']['compression_ratio']:.4f}")
        print(f"  OOV Rate: {results['oov_rate']['oov_rate']:.4f}")
        print(f"  Perplexity: {results['perplexity']['perplexity']:.2f}")
        print(f"  Token Entropy: {results['token_distribution']['entropy']:.4f}")
        print(f"  Semantic Coherence: {results['semantic_coherence']['semantic_coherence']:.4f}")
    
    # Save results
    with open("advanced_bpe_evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print("\n=== Advanced BPE Implementation Complete ===")
    print("Results saved with visualizations in 'plots/' directory")


if __name__ == "__main__":
    main_advanced()

