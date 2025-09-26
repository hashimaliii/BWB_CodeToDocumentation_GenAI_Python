"""
Example usage of the BPE implementation for Python function dataset.
Demonstrates how to use the basic and advanced BPE implementations.
"""

import pandas as pd
import json
from bpe_implementation import DatasetAnalyzer, BPE, BPEEvaluator
from gpu_bpe_advanced import GPUOptimizedBPE, AdvancedBPEEvaluator

def example_basic_bpe():
    """Example using the basic BPE implementation."""
    print("=== Basic BPE Example ===")
    
    # Load dataset
    analyzer = DatasetAnalyzer("python_functions_and_documentation_dataset.csv")
    df = analyzer.load_dataset(sample_size=5000)
    
    # Prepare data
    code_texts = df['code'].astype(str).tolist()[:1000]
    docstring_texts = df['docstring'].astype(str).tolist()[:1000]
    
    # Train BPE model
    print("Training basic BPE model...")
    bpe = BPE(vocab_size=2000, use_gpu=False)  # Use CPU for basic example
    bpe.train(code_texts + docstring_texts)
    
    # Test encoding/decoding
    test_text = code_texts[0]
    print(f"\nOriginal text: {test_text[:100]}...")
    
    encoded = bpe.encode(test_text)
    print(f"Encoded tokens: {encoded[:10]}...")
    
    decoded = bpe.decode(encoded)
    print(f"Decoded text: {decoded[:100]}...")
    
    # Save model
    bpe.save("example_bpe_model.pkl")
    
    # Load model
    new_bpe = BPE()
    new_bpe.load("example_bpe_model.pkl")
    
    print("Basic BPE example completed!")

def example_advanced_bpe():
    """Example using the advanced GPU-optimized BPE implementation."""
    print("\n=== Advanced BPE Example ===")
    
    # Load dataset
    analyzer = DatasetAnalyzer("python_functions_and_documentation_dataset.csv")
    df = analyzer.load_dataset(sample_size=10000)
    
    # Prepare data
    code_texts = df['code'].astype(str).tolist()
    docstring_texts = df['docstring'].astype(str).tolist()
    
    # Train advanced BPE model
    print("Training advanced BPE model...")
    bpe = GPUOptimizedBPE(vocab_size=5000, use_gpu=True, batch_size=512)
    bpe.train_parallel(code_texts + docstring_texts, num_workers=2)
    
    # Batch encoding example
    test_texts = code_texts[:10]
    print(f"\nBatch encoding {len(test_texts)} texts...")
    
    batch_encoded = bpe.encode_batch(test_texts)
    print(f"Batch encoded {len(batch_encoded)} texts")
    
    # Individual encoding example
    test_text = code_texts[0]
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"\nOriginal: {test_text[:100]}...")
    print(f"Decoded: {decoded[:100]}...")
    
    # Save model
    bpe.save("example_advanced_bpe_model.pkl")
    
    print("Advanced BPE example completed!")

def example_evaluation():
    """Example of comprehensive BPE evaluation."""
    print("\n=== BPE Evaluation Example ===")
    
    # Load dataset
    analyzer = DatasetAnalyzer("python_functions_and_documentation_dataset.csv")
    df = analyzer.load_dataset(sample_size=2000)
    
    # Prepare data
    code_texts = df['code'].astype(str).tolist()
    docstring_texts = df['docstring'].astype(str).tolist()
    combined_texts = code_texts + docstring_texts
    
    # Train model for evaluation
    print("Training model for evaluation...")
    bpe = BPE(vocab_size=3000, use_gpu=False)
    bpe.train(combined_texts[:1500])
    
    # Prepare test data and ground truth
    test_texts = combined_texts[1500:2000]
    ground_truth_tokens = [text.split() for text in test_texts]
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    evaluator = BPEEvaluator(bpe, ground_truth_tokens)
    results = evaluator.comprehensive_evaluation(test_texts)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, values in results.items():
        print(f"\n{metric.upper()}:")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Save results
    with open("example_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nEvaluation example completed!")

def example_advanced_evaluation():
    """Example of advanced BPE evaluation with visualizations."""
    print("\n=== Advanced BPE Evaluation Example ===")
    
    # Load dataset
    analyzer = DatasetAnalyzer("python_functions_and_documentation_dataset.csv")
    df = analyzer.load_dataset(sample_size=3000)
    
    # Prepare data
    code_texts = df['code'].astype(str).tolist()
    docstring_texts = df['docstring'].astype(str).tolist()
    combined_texts = code_texts + docstring_texts
    
    # Train advanced model
    print("Training advanced model for evaluation...")
    bpe = GPUOptimizedBPE(vocab_size=4000, use_gpu=True, batch_size=256)
    bpe.train_parallel(combined_texts[:2000], num_workers=2)
    
    # Prepare test data
    test_texts = combined_texts[2000:2500]
    ground_truth_tokens = [text.split() for text in test_texts]
    
    # Run advanced evaluation
    print("Running advanced evaluation with visualizations...")
    evaluator = AdvancedBPEEvaluator(bpe, ground_truth_tokens)
    results = evaluator.comprehensive_evaluation(test_texts)
    
    # Print results
    print("\nAdvanced Evaluation Results:")
    for metric, values in results.items():
        print(f"\n{metric.upper()}:")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")
    
    # Save results
    with open("example_advanced_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nAdvanced evaluation example completed!")
    print("Check the 'plots/' directory for visualizations.")

def example_comparison():
    """Example comparing different BPE configurations."""
    print("\n=== BPE Configuration Comparison ===")
    
    # Load dataset
    analyzer = DatasetAnalyzer("python_functions_and_documentation_dataset.csv")
    df = analyzer.load_dataset(sample_size=2000)
    
    code_texts = df['code'].astype(str).tolist()[:1000]
    docstring_texts = df['docstring'].astype(str).tolist()[:1000]
    combined_texts = code_texts + docstring_texts
    
    # Test different configurations
    configs = [
        ("Small Vocab", 1000),
        ("Medium Vocab", 3000),
        ("Large Vocab", 5000)
    ]
    
    comparison_results = {}
    
    for config_name, vocab_size in configs:
        print(f"\nTraining {config_name} (vocab_size={vocab_size})...")
        
        # Train model
        bpe = BPE(vocab_size=vocab_size, use_gpu=False)
        bpe.train(combined_texts)
        
        # Quick evaluation
        test_texts = combined_texts[100:200]
        ground_truth_tokens = [text.split() for text in test_texts]
        
        evaluator = BPEEvaluator(bpe, ground_truth_tokens)
        results = evaluator.comprehensive_evaluation(test_texts)
        
        # Store results
        comparison_results[config_name] = {
            'vocab_size': vocab_size,
            'jaccard_similarity': results['vocabulary_overlap']['jaccard_similarity'],
            'compression_ratio': results['compression_ratio']['compression_ratio'],
            'oov_rate': results['oov_rate']['oov_rate']
        }
    
    # Print comparison
    print("\nConfiguration Comparison:")
    print(f"{'Config':<15} {'Vocab Size':<10} {'Jaccard':<10} {'Compression':<12} {'OOV Rate':<10}")
    print("-" * 65)
    
    for config_name, results in comparison_results.items():
        print(f"{config_name:<15} {results['vocab_size']:<10} "
              f"{results['jaccard_similarity']:<10.4f} "
              f"{results['compression_ratio']:<12.4f} "
              f"{results['oov_rate']:<10.4f}")
    
    # Save comparison
    with open("bpe_configuration_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print("\nConfiguration comparison completed!")

def main():
    """Run all examples."""
    print("=== BPE Implementation Examples ===")
    
    try:
        # Run examples
        example_basic_bpe()
        example_advanced_bpe()
        example_evaluation()
        example_advanced_evaluation()
        example_comparison()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the dataset file exists and you have the required dependencies installed.")

if __name__ == "__main__":
    main()
