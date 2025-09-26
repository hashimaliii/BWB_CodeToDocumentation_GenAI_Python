"""
Word2Vec Implementation for Code and Documentation Analysis
Implements Skip-gram Word2Vec model for code semantics, documentation language,
and joint code-documentation relationships.
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
        learning_rate = 0.001
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
class Word2VecAnalyzer:
    """
    Comprehensive Word2Vec analysis for code and documentation
    """
   
    def __init__(self, output_dir="word2vec"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
       
    def load_bpe_data(self, code_model_path, doc_model_path, combined_model_path):
        """Load BPE tokenized data from pickle files"""
        print("Loading BPE models...")
       
        # Load BPE models
        with open(code_model_path, 'rb') as f:
            self.code_bpe = pickle.load(f)
       
        with open(doc_model_path, 'rb') as f:
            self.doc_bpe = pickle.load(f)
           
        with open(combined_model_path, 'rb') as f:
            self.combined_bpe = pickle.load(f)
       
        print("BPE models loaded successfully!")
       
    def load_dataset(self, csv_path, max_samples=5000):
        """Load the original dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(csv_path)
        # Limit to first 5k samples for efficiency
        if len(self.df) > max_samples:
            self.df = self.df.head(max_samples)
            print(f"Dataset limited to first {max_samples} samples for efficiency")
        print(f"Dataset loaded: {len(self.df)} samples")
        return self.df
   
    def prepare_training_data(self, data_type="combined"):
        """Prepare training data for Word2Vec"""
        print(f"Preparing training data for {data_type}...")
       
        sentences = []
       
        if data_type == "code":
            # Use code BPE model
            for _, row in self.df.iterrows():
                code_text = str(row['code'])
                tokens = code_text.split() # Fallback to simple split since BPE is dict
                if len(tokens) > 0:
                    sentences.append(tokens)
       
        elif data_type == "doc":
            # Use documentation BPE model
            for _, row in self.df.iterrows():
                doc_text = str(row['docstring'])
                tokens = doc_text.split() # Fallback to simple split since BPE is dict
                if len(tokens) > 0:
                    sentences.append(tokens)
       
        elif data_type == "combined":
            # Use combined BPE model for both code and documentation
            for _, row in self.df.iterrows():
                # Combine code and documentation
                code_text = str(row['code'])
                doc_text = str(row['docstring'])
                combined_text = f"{code_text} {doc_text}"
                tokens = combined_text.split() # Fallback to simple split since BPE is dict
                if len(tokens) > 0:
                    sentences.append(tokens)
       
        print(f"Prepared {len(sentences)} sentences for training")
        return sentences
   
    def train_word2vec_models(self):
        """Train Word2Vec models for different data types"""
        print("Training Word2Vec models...")
       
        # Train models for different data types
        data_types = ["code", "doc", "combined"]
        self.models = {}
       
        for data_type in data_types:
            print(f"\nTraining {data_type} Word2Vec model...")
            sentences = self.prepare_training_data(data_type)
           
            if GENSIM_AVAILABLE:
                # Use Gensim Word2Vec
                model = Word2Vec(
                    sentences=sentences,
                    vector_size=100,
                    window=5,
                    min_count=5,
                    workers=4,
                    epochs=15,
                    sg=1, # Skip-gram
                    callbacks=[Word2VecCallback()]
                )
            else:
                # Use custom implementation
                model = CustomWord2Vec(
                    vector_size=100,
                    window=5,
                    min_count=5,
                    workers=4,
                    epochs=15
                )
                model.train(sentences)
           
            self.models[data_type] = model
            print(f"{data_type} model trained successfully!")
       
        return self.models
   
    def evaluate_semantic_similarity(self):
        """Evaluate semantic similarity tasks"""
        print("Evaluating semantic similarity...")
       
        similarity_results = {}
       
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} model...")
           
            # Get vocabulary
            if hasattr(model, 'wv'):
                vocab = list(model.wv.key_to_index.keys())
            else:
                vocab = list(model.vocab.keys())
           
            # Sample some words for similarity testing
            test_words = vocab[:min(100, len(vocab))]
           
            similarities = []
            for word in test_words:
                if hasattr(model, 'wv'):
                    # Gensim model
                    similar_words = model.wv.most_similar(word, topn=5)
                    similarities.extend([(word, sim_word, sim_score) for sim_word, sim_score in similar_words])
                else:
                    # Custom model
                    similar_words = model.most_similar(word, topn=5)
                    similarities.extend([(word, sim_word, sim_score) for sim_word, sim_score in similar_words])
           
            similarity_results[model_name] = {
                "total_similarities": len(similarities),
                "avg_similarity": np.mean([sim[2] for sim in similarities]),
                "top_similarities": similarities[:10]
            }
       
        self.results['semantic_similarity'] = similarity_results
        return similarity_results
   
    def evaluate_code_completion(self):
        """Evaluate code completion accuracy"""
        print("Evaluating code completion...")
       
        completion_results = {}
       
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} for code completion...")
           
            # Test code completion on a subset of data
            test_samples = self.df.sample(min(100, len(self.df)))
            completion_scores = []
           
            for _, row in test_samples.iterrows():
                code_text = str(row['code'])
               
                # Split code into tokens and test completion
                if model_name == "code":
                    tokens = code_text.split() # Fallback to simple split
                elif model_name == "doc":
                    tokens = str(row['docstring']).split() # Fallback to simple split
                else:
                    combined_text = f"{code_text} {str(row['docstring'])}"
                    tokens = combined_text.split() # Fallback to simple split
               
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
       
        self.results['code_completion'] = completion_results
        return completion_results
   
    def evaluate_documentation_relevance(self):
        """Evaluate documentation relevance scoring"""
        print("Evaluating documentation relevance...")
       
        relevance_results = {}
       
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} for documentation relevance...")
           
            # Test documentation relevance
            test_samples = self.df.sample(min(50, len(self.df)))
            relevance_scores = []
           
            for _, row in test_samples.iterrows():
                code_text = str(row['code'])
                doc_text = str(row['docstring'])
               
                # Calculate relevance between code and documentation
                if model_name == "combined":
                    # For combined model, test if code and doc are similar
                    code_tokens = code_text.split() # Fallback to simple split
                    doc_tokens = doc_text.split() # Fallback to simple split
                   
                    if len(code_tokens) > 0 and len(doc_tokens) > 0:
                        # Calculate average similarity between code and doc tokens
                        similarities = []
                        for code_token in code_tokens[:5]: # Limit for efficiency
                            for doc_token in doc_tokens[:5]:
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
       
        self.results['documentation_relevance'] = relevance_results
        return relevance_results
   
    def create_visualizations(self):
        """Create visualization analysis"""
        print("Creating visualizations...")
       
        for model_name, model in self.models.items():
            print(f"Creating visualizations for {model_name}...")
           
            # Get word vectors
            if hasattr(model, 'wv'):
                words = list(model.wv.key_to_index.keys())[:100] # Limit for visualization
                vectors = np.array([model.wv[word] for word in words])
            else:
                words = list(model.vocab.keys())[:100]
                vectors = np.array([model.word_vectors[model.vocab[word]] for word in words])
           
            # PCA visualization
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
           
            plt.figure(figsize=(12, 8))
            plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
           
            # Annotate some points
            for i, word in enumerate(words[:20]): # Annotate first 20 words
                plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
           
            plt.title(f'Word2Vec PCA Visualization - {model_name.title()}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/word2vec_pca_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
           
            # t-SNE visualization
            if len(vectors) > 10: # t-SNE needs sufficient data
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)//4))
                vectors_tsne = tsne.fit_transform(vectors)
               
                plt.figure(figsize=(12, 8))
                plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.6)
               
                # Annotate some points
                for i, word in enumerate(words[:20]):
                    plt.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
               
                plt.title(f'Word2Vec t-SNE Visualization - {model_name.title()}')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/word2vec_tsne_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
       
        print("Visualizations created successfully!")
   
    def save_results(self):
        """Save all results to JSON and pickle files"""
        print("Saving results...")
        
        # Add model information
        model_info = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'wv'):
                model_info[model_name] = {
                    "vocab_size": len(model.wv.key_to_index),
                    "vector_size": model.wv.vector_size,
                    "model_type": "gensim"
                }
            else:
                model_info[model_name] = {
                    "vocab_size": len(model.vocab),
                    "vector_size": model.vector_size,
                    "model_type": "custom"
                }
        
        self.results['model_info'] = model_info
        
        # Save to JSON
        json_path = f'{self.output_dir}/word2vec_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save to pickle
        pickle_path = f'{self.output_dir}/word2vec_results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Results saved to {json_path} and {pickle_path}")
   
    def run_complete_analysis(self, csv_path, code_model_path, doc_model_path, combined_model_path, max_samples=5000):
        """Run complete Word2Vec analysis"""
        print("Starting complete Word2Vec analysis...")
       
        # Load data with sample limit
        self.load_dataset(csv_path, max_samples=max_samples)
        self.load_bpe_data(code_model_path, doc_model_path, combined_model_path)
       
        # Train models
        self.train_word2vec_models()
       
        # Evaluate models
        self.evaluate_semantic_similarity()
        self.evaluate_code_completion()
        self.evaluate_documentation_relevance()
       
        # Create visualizations
        self.create_visualizations()
       
        # Save results
        self.save_results()
       
        print("Word2Vec analysis completed successfully!")
        return self.results
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
   
    # Initialize analyzer
    analyzer = Word2VecAnalyzer(output_dir=args.output_dir)
   
    # Run complete analysis
    max_samples = None if args.samples == 0 else args.samples
    results = analyzer.run_complete_analysis(
        csv_path="python_functions_and_documentation_dataset.csv",
        code_model_path="bpe_code_model.pkl",
        doc_model_path="bpe_doc_model.pkl",
        combined_model_path="bpe_combined_model.pkl",
        max_samples=max_samples
    )
   
    print("\nAnalysis Summary:")
    for key, value in results.items():
        print(f"{key}: {type(value)}")
if __name__ == "__main__":
    main()


