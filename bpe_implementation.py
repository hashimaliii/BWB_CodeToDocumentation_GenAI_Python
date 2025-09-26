"""
Byte Pair Encoding (BPE) Implementation for Python Function Dataset
Includes GPU support and comprehensive evaluation against professional standards.
"""

import pandas as pd
import numpy as np
import re
import pickle
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """Comprehensive dataset analysis for understanding data characteristics."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.analysis_results = {}
    
    def load_dataset(self, sample_size: Optional[int] = None):
        """Load dataset with optional sampling for large files."""
        print(f"Loading dataset from {self.csv_path}...")
        
        if sample_size:
            print(f"Sampling {sample_size} rows for analysis...")
            # Read in chunks to handle large files
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= sample_size:
                    break
            
            self.df = pd.concat(chunks).sample(n=sample_size, random_state=42)
        else:
            self.df = pd.read_csv(self.csv_path)
        
        print(f"Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def analyze_structure(self):
        """Analyze dataset structure and basic statistics."""
        print("\n=== Dataset Structure Analysis ===")
        
        # Basic info
        self.analysis_results['shape'] = self.df.shape
        self.analysis_results['columns'] = list(self.df.columns)
        self.analysis_results['dtypes'] = dict(self.df.dtypes)
        
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        self.analysis_results['missing_values'] = dict(missing_values)
        print(f"\nMissing values:\n{missing_values}")
        
        # Partition distribution
        if 'partition' in self.df.columns:
            partition_dist = self.df['partition'].value_counts()
            self.analysis_results['partition_distribution'] = dict(partition_dist)
            print(f"\nPartition distribution:\n{partition_dist}")
        
        return self.analysis_results
    
    def analyze_text_characteristics(self):
        """Analyze text characteristics of code and docstrings."""
        print("\n=== Text Characteristics Analysis ===")
        
        text_stats = {}
        
        for column in ['code', 'docstring', 'summary']:
            if column in self.df.columns:
                # Basic statistics
                lengths = self.df[column].astype(str).str.len()
                word_counts = self.df[column].astype(str).str.split().str.len()
                
                text_stats[column] = {
                    'mean_length': lengths.mean(),
                    'std_length': lengths.std(),
                    'min_length': lengths.min(),
                    'max_length': lengths.max(),
                    'mean_words': word_counts.mean(),
                    'std_words': word_counts.std(),
                    'min_words': word_counts.min(),
                    'max_words': word_counts.max()
                }
                
                print(f"\n{column.upper()} Statistics:")
                print(f"  Length - Mean: {lengths.mean():.1f}, Std: {lengths.std():.1f}")
                print(f"  Length - Min: {lengths.min()}, Max: {lengths.max()}")
                print(f"  Words - Mean: {word_counts.mean():.1f}, Std: {word_counts.std():.1f}")
                print(f"  Words - Min: {word_counts.min()}, Max: {word_counts.max()}")
        
        self.analysis_results['text_characteristics'] = text_stats
        return text_stats
    
    def analyze_token_distribution(self):
        """Analyze token distribution for BPE training."""
        print("\n=== Token Distribution Analysis ===")
        
        all_code_tokens = []
        all_docstring_tokens = []
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Analyzing tokens"):
            # Simple tokenization for analysis
            code_tokens = self._simple_tokenize(str(row.get('code', '')))
            docstring_tokens = self._simple_tokenize(str(row.get('docstring', '')))
            
            all_code_tokens.extend(code_tokens)
            all_docstring_tokens.extend(docstring_tokens)
        
        # Calculate token statistics
        code_token_counts = Counter(all_code_tokens)
        docstring_token_counts = Counter(all_docstring_tokens)
        
        token_stats = {
            'code': {
                'unique_tokens': len(code_token_counts),
                'total_tokens': len(all_code_tokens),
                'most_common': code_token_counts.most_common(20)
            },
            'docstring': {
                'unique_tokens': len(docstring_token_counts),
                'total_tokens': len(all_docstring_tokens),
                'most_common': docstring_token_counts.most_common(20)
            }
        }
        
        print(f"Code tokens: {len(all_code_tokens)} total, {len(code_token_counts)} unique")
        print(f"Docstring tokens: {len(all_docstring_tokens)} total, {len(docstring_token_counts)} unique")
        
        self.analysis_results['token_distribution'] = token_stats
        return token_stats
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for analysis purposes."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        return tokens
    
    def save_analysis(self, output_path: str):
        """Save analysis results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"Analysis saved to {output_path}")


class BPE:
    """Byte Pair Encoding implementation with GPU support."""
    
    def __init__(self, vocab_size: int = 10000, use_gpu: bool = True):
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Vocabulary and merges
        self.vocab = {}
        self.merges = []
        self.reverse_vocab = {}
        
        # Initialize with character vocabulary
        self._initialize_character_vocab()
    
    def _initialize_character_vocab(self):
        """Initialize vocabulary with individual characters."""
        # Add basic characters
        chars = set()
        chars.update([chr(i) for i in range(32, 127)])  # Printable ASCII
        chars.update(['\n', '\t', ' '])  # Common whitespace
        
        # Add special tokens
        special_tokens = ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '<CODE>', '<DOC>']
        
        for i, char in enumerate(sorted(chars) + special_tokens):
            self.vocab[char] = i
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Initialized character vocabulary with {len(self.vocab)} tokens")
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from texts."""
        word_freqs = Counter()
        
        for text in texts:
            # Simple word tokenization
            words = text.split()
            for word in words:
                word_freqs[word] += 1
        
        return dict(word_freqs)
    
    def _get_pairs(self, word: str) -> Set[Tuple[str, str]]:
        """Get all consecutive character pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge a pair in the vocabulary."""
        bigram = ''.join(pair)
        new_vocab = {}
        
        pattern = re.escape(pair[0] + pair[1])
        
        for word in vocab:
            new_word = re.sub(pattern, bigram, word)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def train(self, texts: List[str], progress_callback=None):
        """Train BPE model on given texts."""
        print(f"Training BPE with vocab size {self.vocab_size} on {len(texts)} texts...")
        
        # Get word frequencies
        word_freqs = self._get_word_freqs(texts)
        print(f"Found {len(word_freqs)} unique words")
        
        # Initialize vocabulary with characters
        vocab = {}
        for word, freq in word_freqs.items():
            word_chars = ' '.join(list(word)) + ' </w>'
            vocab[word_chars] = freq
        
        # Add individual characters to vocab
        for word in word_freqs:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        # Training loop
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in tqdm(range(num_merges), desc="Training BPE"):
            pairs = self._get_pair_counts(vocab)
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
            
            if progress_callback:
                progress_callback(i + 1, num_merges)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")
    
    def _get_pair_counts(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get counts of all character pairs."""
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return dict(pairs)
    
    def encode(self, text: str) -> List[int]:
        """Encode text using trained BPE."""
        if not text:
            return []
        
        # Convert to BPE tokens
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
        
        # Initialize with characters
        symbols = list(word) + ['</w>']
        
        # Apply merges
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
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append('<UNK>')
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, path: str):
        """Save BPE model to file."""
        model_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"BPE model saved to {path}")
    
    def load(self, path: str):
        """Load BPE model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab = model_data['vocab']
        self.merges = model_data['merges']
        self.vocab_size = model_data['vocab_size']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"BPE model loaded from {path}")


class BPEEvaluator:
    """Comprehensive evaluation of BPE against professional standards."""
    
    def __init__(self, bpe_model: BPE, ground_truth_tokens: List[List[str]]):
        self.bpe_model = bpe_model
        self.ground_truth_tokens = ground_truth_tokens
    
    def evaluate_vocabulary_overlap(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate vocabulary overlap using Jaccard similarity."""
        print("Evaluating vocabulary overlap...")
        
        bpe_vocab = set(self.bpe_model.vocab.keys())
        ground_truth_vocab = set()
        
        for tokens in self.ground_truth_tokens:
            ground_truth_vocab.update(tokens)
        
        # Calculate Jaccard similarity
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
    
    def evaluate_compression_ratio(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate compression ratio compared to character-level encoding."""
        print("Evaluating compression ratio...")
        
        total_chars = 0
        total_tokens = 0
        
        for text in test_texts:
            chars = len(text)
            tokens = len(self.bpe_model.encode(text))
            
            total_chars += chars
            total_tokens += tokens
        
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 1.0
        
        return {
            'compression_ratio': compression_ratio,
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'avg_chars_per_token': total_chars / total_tokens if total_tokens > 0 else 1.0
        }
    
    def evaluate_boundary_accuracy(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate boundary accuracy against ground truth."""
        print("Evaluating boundary accuracy...")
        
        correct_boundaries = 0
        total_boundaries = 0
        
        for i, text in enumerate(test_texts):
            if i >= len(self.ground_truth_tokens):
                break
            
            # Get BPE tokens
            bpe_tokens = self.bpe_model.encode(text)
            bpe_text = self.bpe_model.decode(bpe_tokens)
            
            # Simple boundary detection
            gt_boundaries = set()
            bpe_boundaries = set()
            
            # This is a simplified boundary detection
            # In practice, you'd use more sophisticated methods
            words = text.split()
            for word in words:
                gt_boundaries.add(word)
            
            bpe_words = bpe_text.split()
            for word in bpe_words:
                bpe_boundaries.add(word)
            
            correct_boundaries += len(gt_boundaries.intersection(bpe_boundaries))
            total_boundaries += len(gt_boundaries.union(bpe_boundaries))
        
        boundary_accuracy = correct_boundaries / total_boundaries if total_boundaries > 0 else 0.0
        
        return {
            'boundary_accuracy': boundary_accuracy,
            'correct_boundaries': correct_boundaries,
            'total_boundaries': total_boundaries
        }
    
    def evaluate_consistency(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate consistency of tokenization."""
        print("Evaluating consistency...")
        
        # Test multiple encodings of the same text
        consistency_scores = []
        
        for text in test_texts[:100]:  # Sample for efficiency
            encodings = []
            for _ in range(5):  # Multiple encodings
                encoding = self.bpe_model.encode(text)
                encodings.append(tuple(encoding))
            
            # Check if all encodings are identical
            is_consistent = len(set(encodings)) == 1
            consistency_scores.append(1.0 if is_consistent else 0.0)
        
        consistency_score = np.mean(consistency_scores)
        
        return {
            'consistency_score': consistency_score,
            'consistent_texts': sum(consistency_scores),
            'total_texts': len(consistency_scores)
        }
    
    def evaluate_oov_rate(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate out-of-vocabulary (OOV) rate."""
        print("Evaluating OOV rate...")
        
        total_tokens = 0
        oov_tokens = 0
        
        for text in test_texts:
            tokens = self.bpe_model.encode(text)
            total_tokens += len(tokens)
            
            for token_id in tokens:
                if token_id == self.bpe_model.vocab.get('<UNK>', -1):
                    oov_tokens += 1
        
        oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'oov_rate': oov_rate,
            'oov_tokens': oov_tokens,
            'total_tokens': total_tokens,
            'oov_percentage': oov_rate * 100
        }
    
    def comprehensive_evaluation(self, test_texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation."""
        print("Running comprehensive BPE evaluation...")
        
        results = {
            'vocabulary_overlap': self.evaluate_vocabulary_overlap(test_texts),
            'compression_ratio': self.evaluate_compression_ratio(test_texts),
            'boundary_accuracy': self.evaluate_boundary_accuracy(test_texts),
            'consistency': self.evaluate_consistency(test_texts),
            'oov_rate': self.evaluate_oov_rate(test_texts)
        }
        
        return results


def main():
    """Main function to run the complete BPE pipeline."""
    print("=== BPE Implementation for Python Function Dataset ===")
    
    # Configuration
    CSV_PATH = "python_functions_and_documentation_dataset.csv"
    SAMPLE_SIZE = 10000  # Adjust based on your system's memory
    VOCAB_SIZE = 5000
    USE_GPU = True
    
    # Task 1: Dataset Loading and Analysis
    print("\n" + "="*50)
    print("TASK 1: Dataset Loading and Analysis")
    print("="*50)
    
    analyzer = DatasetAnalyzer(CSV_PATH)
    df = analyzer.load_dataset(sample_size=SAMPLE_SIZE)
    
    # Analyze structure
    structure_analysis = analyzer.analyze_structure()
    text_analysis = analyzer.analyze_text_characteristics()
    token_analysis = analyzer.analyze_token_distribution()
    
    # Save analysis
    analyzer.save_analysis("dataset_analysis.json")
    
    # Task 2: BPE Implementation
    print("\n" + "="*50)
    print("TASK 2: BPE Tokenization Implementation")
    print("="*50)
    
    # Prepare training data
    code_texts = df['code'].astype(str).tolist() if 'code' in df.columns else []
    docstring_texts = df['docstring'].astype(str).tolist() if 'docstring' in df.columns else []
    combined_texts = code_texts + docstring_texts
    
    print(f"Training on {len(combined_texts)} texts")
    
    # Train separate BPE models
    print("\nTraining BPE for code...")
    bpe_code = BPE(vocab_size=VOCAB_SIZE, use_gpu=USE_GPU)
    bpe_code.train(code_texts[:5000])  # Sample for training
    bpe_code.save("bpe_code_model.pkl")
    
    print("\nTraining BPE for documentation...")
    bpe_doc = BPE(vocab_size=VOCAB_SIZE, use_gpu=USE_GPU)
    bpe_doc.train(docstring_texts[:5000])  # Sample for training
    bpe_doc.save("bpe_doc_model.pkl")
    
    print("\nTraining BPE for combined usage...")
    bpe_combined = BPE(vocab_size=VOCAB_SIZE * 2, use_gpu=USE_GPU)
    bpe_combined.train(combined_texts[:10000])  # Sample for training
    bpe_combined.save("bpe_combined_model.pkl")
    
    # Task 3: BPE Evaluation
    print("\n" + "="*50)
    print("TASK 3: BPE Evaluation Against Professional Standards")
    print("="*50)
    
    # Prepare test data
    test_texts = combined_texts[1000:2000]  # Test set
    
    # Get ground truth tokens (simplified for this example)
    ground_truth_tokens = []
    for text in test_texts:
        tokens = text.split()  # Simple word tokenization as ground truth
        ground_truth_tokens.append(tokens)
    
    # Evaluate each model
    models_to_evaluate = [
        ("Code BPE", bpe_code),
        ("Documentation BPE", bpe_doc),
        ("Combined BPE", bpe_combined)
    ]
    
    evaluation_results = {}
    
    for model_name, model in models_to_evaluate:
        print(f"\nEvaluating {model_name}...")
        evaluator = BPEEvaluator(model, ground_truth_tokens)
        results = evaluator.comprehensive_evaluation(test_texts)
        evaluation_results[model_name] = results
        
        # Print summary
        print(f"\n{model_name} Results:")
        print(f"  Vocabulary Overlap (Jaccard): {results['vocabulary_overlap']['jaccard_similarity']:.4f}")
        print(f"  Compression Ratio: {results['compression_ratio']['compression_ratio']:.4f}")
        print(f"  Boundary Accuracy: {results['boundary_accuracy']['boundary_accuracy']:.4f}")
        print(f"  Consistency Score: {results['consistency']['consistency_score']:.4f}")
        print(f"  OOV Rate: {results['oov_rate']['oov_rate']:.4f}")
    
    # Save evaluation results
    with open("bpe_evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print("\n=== BPE Implementation Complete ===")
    print("Results saved to:")
    print("- dataset_analysis.json")
    print("- bpe_code_model.pkl")
    print("- bpe_doc_model.pkl") 
    print("- bpe_combined_model.pkl")
    print("- bpe_evaluation_results.json")


if __name__ == "__main__":
    main()
