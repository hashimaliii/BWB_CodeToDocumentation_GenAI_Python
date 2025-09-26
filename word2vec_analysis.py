"""
Word2Vec Analysis Script
Comprehensive analysis of code and documentation embeddings
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try to import gensim for Word2Vec
try:
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    GENSIM_AVAILABLE = True
    print("Gensim available - using optimized Word2Vec")
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not available - using custom implementation")

class Word2VecCallback(CallbackAny2Vec):
    """Callback to track training progress"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch} completed")

class CustomWord2Vec:
    """
    Custom Word2Vec implementation using Skip-gram architecture
    """
    
    def __init__(self, vector_size=100, window=5, min_count=5, workers=4, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.vocab = {}
        self.word_vectors = {}
        self.context_vectors = {}
        self.vocab_size = 0
        
    def build_vocab(self, sentences):
        """Build vocabulary from sentences"""
        word_counts = Counter()
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
        
        # Filter by min_count
        self.vocab = {word: idx for idx, (word, count) in enumerate(word_counts.most_common()) 
                     if count >= self.min_count}
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        
    def train(self, sentences):
        """Train Word2Vec model using Skip-gram"""
        print("Building vocabulary...")
        self.build_vocab(sentences)
        
        # Initialize word and context vectors
        self.word_vectors = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.vector_size))
        self.context_vectors = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.vector_size))
        
        print("Training Word2Vec model...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for sentence in sentences:
                for i, word in enumerate(sentence):
                    if word not in self.vocab:
                        continue
                    word_idx = self.vocab[word]
                    
                    # Get context words
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)
                    
                    for j in range(start, end):
                        if j != i and j < len(sentence):
                            context_word = sentence[j]
                            if context_word in self.vocab:
                                context_idx = self.vocab[context_word]
                                # Simple skip-gram update (simplified)
                                self._update_vectors(word_idx, context_idx)
        
        print("Training completed!")
    
    def _update_vectors(self, word_idx, context_idx):
        """Update word and context vectors (simplified version)"""
        # This is a simplified update - in practice, you'd use proper backpropagation
        learning_rate = 0.01
        word_vec = self.word_vectors[word_idx]
        context_vec = self.context_vectors[context_idx]
        
        # Simple gradient update
        self.word_vectors[word_idx] += learning_rate * context_vec
        self.context_vectors[context_idx] += learning_rate * word_vec
    
    def get_vector(self, word):
        """Get vector for a word"""
        if word in self.vocab:
            return self.word_vectors[self.vocab[word]]
        return None
    
    def similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def most_similar(self, word, topn=10):
        """Find most similar words"""
        if word not in self.vocab:
            return []
        
        word_vec = self.get_vector(word)
        similarities = []
        
        for other_word, idx in self.vocab.items():
            if other_word != word:
                other_vec = self.word_vectors[idx]
                sim = cosine_similarity([word_vec], [other_vec])[0][0]
                similarities.append((other_word, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

def load_bpe_models():
    """Load BPE models from pickle files"""
    print("Loading BPE models...")
    
    models = {}
    model_files = {
        'code': 'bpe_code_model.pkl',
        'doc': 'bpe_doc_model.pkl',
        'combined': 'bpe_combined_model.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"Loaded {name} BPE model")
        else:
            print(f"Warning: {path} not found")
    
    return models

def load_dataset(max_samples=5000):
    """Load the dataset"""
    print("Loading dataset...")
    csv_path = 'python_functions_and_documentation_dataset.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Limit samples if specified
        if max_samples is not None and len(df) > max_samples:
            df = df.head(max_samples)
            print(f"Dataset limited to first {max_samples} samples")
        elif max_samples is None:
            print("Using full dataset")
        print(f"Dataset loaded: {len(df)} samples")
        return df
    else:
        print(f"Error: {csv_path} not found")
        return None

def prepare_training_data(df, bpe_models, data_type="combined"):
    """Prepare training data for Word2Vec"""
    print(f"Preparing training data for {data_type}...")
    
    sentences = []
    
    if data_type == "code" and "code" in bpe_models:
        # Use code BPE model
        for _, row in df.iterrows():
            code_text = str(row['code'])
            tokens = bpe_models['code'].fast_tokenize(code_text)
            if len(tokens) > 0:
                sentences.append(tokens)
    
    elif data_type == "doc" and "doc" in bpe_models:
        # Use documentation BPE model
        for _, row in df.iterrows():
            doc_text = str(row['docstring'])
            tokens = bpe_models['doc'].fast_tokenize(doc_text)
            if len(tokens) > 0:
                sentences.append(tokens)
    
    elif data_type == "combined" and "combined" in bpe_models:
        # Use combined BPE model for both code and documentation
        for _, row in df.iterrows():
            # Combine code and documentation
            code_text = str(row['code'])
            doc_text = str(row['docstring'])
            combined_text = f"{code_text} {doc_text}"
            tokens = bpe_models['combined'].fast_tokenize(combined_text)
            if len(tokens) > 0:
                sentences.append(tokens)
    
    print(f"Prepared {len(sentences)} sentences for training")
    return sentences

def train_word2vec_models(df, bpe_models):
    """Train Word2Vec models for different data types"""
    print("Training Word2Vec models...")
    
    # Train models for different data types
    data_types = ["code", "doc", "combined"]
    models = {}
    
    for data_type in data_types:
        print(f"\nTraining {data_type} Word2Vec model...")
        sentences = prepare_training_data(df, bpe_models, data_type)
        
        if len(sentences) == 0:
            print(f"No sentences available for {data_type}, skipping...")
            continue
        
        if GENSIM_AVAILABLE:
            # Use Gensim Word2Vec
            model = Word2Vec(
                sentences=sentences,
                vector_size=100,
                window=5,
                min_count=5,
                workers=4,
                epochs=10,
                sg=1,  # Skip-gram
                callbacks=[Word2VecCallback()]
            )
        else:
            # Use custom implementation
            model = CustomWord2Vec(
                vector_size=100,
                window=5,
                min_count=5,
                workers=4,
                epochs=10
            )
            model.train(sentences)
        
        models[data_type] = model
        print(f"{data_type} model trained successfully!")
    
    return models

def evaluate_semantic_similarity(models):
    """Evaluate semantic similarity tasks"""
    print("Evaluating semantic similarity...")
    
    similarity_results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} model...")
        
        # Get vocabulary
        if hasattr(model, 'wv'):
            vocab = list(model.wv.key_to_index.keys())
        else:
            vocab = list(model.vocab.keys())
        
        # Sample some words for similarity testing
        test_words = vocab[:min(50, len(vocab))]
        
        similarities = []
        for word in test_words:
            if hasattr(model, 'wv'):
                # Gensim model
                try:
                    similar_words = model.wv.most_similar(word, topn=5)
                    similarities.extend([(word, sim_word, sim_score) for sim_word, sim_score in similar_words])
                except:
                    pass
            else:
                # Custom model
                similar_words = model.most_similar(word, topn=5)
                similarities.extend([(word, sim_word, sim_score) for sim_word, sim_score in similar_words])
        
        similarity_results[model_name] = {
            "total_similarities": len(similarities),
            "avg_similarity": np.mean([sim[2] for sim in similarities]) if similarities else 0.0,
            "top_similarities": similarities[:10]
        }
    
    return similarity_results

def evaluate_code_completion(models, df, bpe_models):
    """Evaluate code completion accuracy"""
    print("Evaluating code completion...")
    
    completion_results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} for code completion...")
        
        # Test code completion on a subset of data
        test_samples = df.sample(min(50, len(df)))
        completion_scores = []
        
        for _, row in test_samples.iterrows():
            code_text = str(row['code'])
            
            # Split code into tokens and test completion
            if model_name == "code" and "code" in bpe_models:
                tokens = bpe_models['code'].fast_tokenize(code_text)
            elif model_name == "doc" and "doc" in bpe_models:
                tokens = bpe_models['doc'].fast_tokenize(str(row['docstring']))
            elif model_name == "combined" and "combined" in bpe_models:
                combined_text = f"{code_text} {str(row['docstring'])}"
                tokens = bpe_models['combined'].fast_tokenize(combined_text)
            else:
                continue
            
            if len(tokens) > 1:
                # Test predicting next token
                context = tokens[:-1]
                target = tokens[-1]
                
                if hasattr(model, 'wv'):
                    # Gensim model
                    try:
                        predicted = model.wv.most_similar(positive=context, topn=1)[0][0]
                        score = 1.0 if predicted == target else 0.0
                    except:
                        score = 0.0
                else:
                    # Custom model
                    if target in model.vocab:
                        similar_words = model.most_similar(target, topn=1)
                        score = 1.0 if similar_words and similar_words[0][0] == target else 0.0
                    else:
                        score = 0.0
                
                completion_scores.append(score)
        
        completion_results[model_name] = {
            "accuracy": np.mean(completion_scores) if completion_scores else 0.0,
            "total_tests": len(completion_scores),
            "scores": completion_scores
        }
    
    return completion_results

def evaluate_documentation_relevance(models, df, bpe_models):
    """Evaluate documentation relevance scoring"""
    print("Evaluating documentation relevance...")
    
    relevance_results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} for documentation relevance...")
        
        # Test documentation relevance
        test_samples = df.sample(min(25, len(df)))
        relevance_scores = []
        
        for _, row in test_samples.iterrows():
            code_text = str(row['code'])
            doc_text = str(row['docstring'])
            
            # Calculate relevance between code and documentation
            if model_name == "combined" and "combined" in bpe_models:
                # For combined model, test if code and doc are similar
                code_tokens = bpe_models['combined'].fast_tokenize(code_text)
                doc_tokens = bpe_models['combined'].fast_tokenize(doc_text)
                
                if len(code_tokens) > 0 and len(doc_tokens) > 0:
                    # Calculate average similarity between code and doc tokens
                    similarities = []
                    for code_token in code_tokens[:3]:  # Limit for efficiency
                        for doc_token in doc_tokens[:3]:
                            if hasattr(model, 'wv'):
                                try:
                                    sim = model.wv.similarity(code_token, doc_token)
                                    similarities.append(sim)
                                except:
                                    similarities.append(0.0)
                            else:
                                sim = model.similarity(code_token, doc_token)
                                similarities.append(sim)
                    
                    relevance_score = np.mean(similarities) if similarities else 0.0
                    relevance_scores.append(relevance_score)
        
        relevance_results[model_name] = {
            "avg_relevance": np.mean(relevance_scores) if relevance_scores else 0.0,
            "total_tests": len(relevance_scores),
            "scores": relevance_scores
        }
    
    return relevance_results

def create_visualizations(models, output_dir="word2vec"):
    """Create visualization analysis"""
    print("Creating visualizations...")
    
    for model_name, model in models.items():
        print(f"Creating visualizations for {model_name}...")
        
        # Get word vectors
        if hasattr(model, 'wv'):
            words = list(model.wv.key_to_index.keys())[:50]  # Limit for visualization
            vectors = np.array([model.wv[word] for word in words])
        else:
            words = list(model.vocab.keys())[:50]
            vectors = np.array([model.word_vectors[model.vocab[word]] for word in words])
        
        if len(vectors) == 0:
            print(f"No vectors available for {model_name}, skipping visualization")
            continue
        
        # PCA visualization
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
        
        # Annotate some points
        for i, word in enumerate(words[:15]):  # Annotate first 15 words
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'Word2Vec PCA Visualization - {model_name.title()}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/word2vec_pca_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # t-SNE visualization
        if len(vectors) > 10:  # t-SNE needs sufficient data
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)//4))
            vectors_tsne = tsne.fit_transform(vectors)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.6)
            
            # Annotate some points
            for i, word in enumerate(words[:15]):
                plt.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.title(f'Word2Vec t-SNE Visualization - {model_name.title()}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/word2vec_tsne_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Visualizations created successfully!")

def save_results(results, output_dir="word2vec"):
    """Save all results to JSON"""
    print("Saving results...")
    
    # Save to JSON
    with open(f'{output_dir}/word2vec_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/word2vec_results.json")

def main():
    """Main function to run Word2Vec analysis"""
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Word2Vec Analysis for Code and Documentation')
    parser.add_argument('--samples', type=int, default=5000, 
                       help='Number of samples to use (default: 5000, use 0 for full dataset)')
    parser.add_argument('--output-dir', type=str, default='word2vec',
                       help='Output directory for results (default: word2vec)')
    
    args = parser.parse_args()
    
    print("Starting Word2Vec Analysis")
    print("=" * 50)
    print(f"Sample size: {'Full dataset' if args.samples == 0 else f'{args.samples} samples'}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data with specified sample size
        max_samples = None if args.samples == 0 else args.samples
        df = load_dataset(max_samples=max_samples)
        if df is None:
            return
        
        bpe_models = load_bpe_models()
        if not bpe_models:
            print("No BPE models loaded, exiting...")
            return
        
        # Train models
        models = train_word2vec_models(df, bpe_models)
        if not models:
            print("No models trained, exiting...")
            return
        
        # Evaluate models
        print("\nEvaluating models...")
        similarity_results = evaluate_semantic_similarity(models)
        completion_results = evaluate_code_completion(models, df, bpe_models)
        relevance_results = evaluate_documentation_relevance(models, df, bpe_models)
        
        # Create visualizations
        create_visualizations(models, output_dir)
        
        # Compile results
        results = {
            'semantic_similarity': similarity_results,
            'code_completion': completion_results,
            'documentation_relevance': relevance_results,
            'model_info': {
                name: {
                    "vocab_size": len(model.wv.key_to_index) if hasattr(model, 'wv') else len(model.vocab),
                    "vector_size": model.wv.vector_size if hasattr(model, 'wv') else model.vector_size,
                    "model_type": "gensim" if hasattr(model, 'wv') else "custom"
                } for name, model in models.items()
            }
        }
        
        # Save results
        save_results(results, output_dir)
        
        # Print summary
        print("\nAnalysis Summary:")
        print("-" * 30)
        
        if 'model_info' in results:
            print("Model Information:")
            for model_name, info in results['model_info'].items():
                print(f"  {model_name}: {info['vocab_size']} vocab, {info['vector_size']}D vectors")
        
        if 'semantic_similarity' in results:
            print("\nSemantic Similarity Results:")
            for model_name, metrics in results['semantic_similarity'].items():
                print(f"  {model_name}: {metrics['total_similarities']} similarities, "
                      f"avg similarity: {metrics['avg_similarity']:.4f}")
        
        if 'code_completion' in results:
            print("\nCode Completion Results:")
            for model_name, metrics in results['code_completion'].items():
                print(f"  {model_name}: {metrics['accuracy']:.4f} accuracy "
                      f"({metrics['total_tests']} tests)")
        
        if 'documentation_relevance' in results:
            print("\nDocumentation Relevance Results:")
            for model_name, metrics in results['documentation_relevance'].items():
                print(f"  {model_name}: {metrics['avg_relevance']:.4f} relevance "
                      f"({metrics['total_tests']} tests)")
        
        print(f"\nVisualizations saved in: {output_dir}/")
        print("Files created:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
        
        print("\nWord2Vec analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
