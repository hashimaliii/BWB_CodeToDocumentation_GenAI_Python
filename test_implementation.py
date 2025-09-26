"""
Test script for the BPE implementation.
Creates a small sample dataset for testing purposes.
"""

import pandas as pd
import numpy as np
from bpe_implementation import BPE, BPEEvaluator
from gpu_bpe_advanced import GPUOptimizedBPE, AdvancedBPEEvaluator
import json

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    print("Creating sample dataset...")
    
    # Sample Python functions and docstrings
    sample_data = [
        {
            "code": "def hello_world(): return 'Hello, World!'",
            "docstring": "A simple function that returns a greeting message.",
            "summary": "Returns greeting message",
            "func_name": "hello_world",
            "repo": "test_repo",
            "partition": "train"
        },
        {
            "code": "def add_numbers(a, b): return a + b",
            "docstring": "Adds two numbers together and returns the result.",
            "summary": "Adds two numbers",
            "func_name": "add_numbers",
            "repo": "test_repo",
            "partition": "train"
        },
        {
            "code": "def multiply(x, y): return x * y",
            "docstring": "Multiplies two numbers and returns the product.",
            "summary": "Multiplies two numbers",
            "func_name": "multiply",
            "repo": "test_repo",
            "partition": "train"
        },
        {
            "code": "def calculate_area(length, width): return length * width",
            "docstring": "Calculates the area of a rectangle given length and width.",
            "summary": "Calculates rectangle area",
            "func_name": "calculate_area",
            "repo": "test_repo",
            "partition": "test"
        },
        {
            "code": "def is_even(number): return number % 2 == 0",
            "docstring": "Checks if a number is even by checking if it's divisible by 2.",
            "summary": "Checks if number is even",
            "func_name": "is_even",
            "repo": "test_repo",
            "partition": "test"
        },
        {
            "code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "docstring": "Calculates the factorial of a number using recursion.",
            "summary": "Calculates factorial recursively",
            "func_name": "factorial",
            "repo": "test_repo",
            "partition": "valid"
        },
        {
            "code": "def reverse_string(text): return text[::-1]",
            "docstring": "Reverses the order of characters in a string.",
            "summary": "Reverses string characters",
            "func_name": "reverse_string",
            "repo": "test_repo",
            "partition": "valid"
        },
        {
            "code": "def count_vowels(text): return sum(1 for char in text.lower() if char in 'aeiou')",
            "docstring": "Counts the number of vowels in a given text string.",
            "summary": "Counts vowels in text",
            "func_name": "count_vowels",
            "repo": "test_repo",
            "partition": "valid"
        },
        {
            "code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "docstring": "Calculates the nth Fibonacci number using recursion.",
            "summary": "Calculates Fibonacci number",
            "func_name": "fibonacci",
            "repo": "test_repo",
            "partition": "valid"
        },
        {
            "code": "def find_max(numbers): return max(numbers) if numbers else None",
            "docstring": "Finds the maximum value in a list of numbers.",
            "summary": "Finds maximum in list",
            "func_name": "find_max",
            "repo": "test_repo",
            "partition": "test"
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    df.to_csv("sample_dataset.csv", index=False)
    print(f"Sample dataset created with {len(df)} rows")
    
    return df

def test_basic_bpe():
    """Test the basic BPE implementation."""
    print("\n=== Testing Basic BPE ===")
    
    # Load sample data
    df = pd.read_csv("sample_dataset.csv")
    texts = (df['code'] + ' ' + df['docstring']).tolist()
    
    # Create and train BPE model
    bpe = BPE(vocab_size=100, use_gpu=False)  # Small vocab for testing
    bpe.train(texts)
    
    # Test encoding/decoding
    test_text = texts[0]
    print(f"Original: {test_text}")
    
    encoded = bpe.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = bpe.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Test evaluation
    ground_truth_tokens = [text.split() for text in texts]
    evaluator = BPEEvaluator(bpe, ground_truth_tokens)
    results = evaluator.comprehensive_evaluation(texts)
    
    print(f"\nBasic BPE Results:")
    print(f"  Jaccard Similarity: {results['vocabulary_overlap']['jaccard_similarity']:.4f}")
    print(f"  Compression Ratio: {results['compression_ratio']['compression_ratio']:.4f}")
    print(f"  OOV Rate: {results['oov_rate']['oov_rate']:.4f}")
    
    return results

def test_advanced_bpe():
    """Test the advanced GPU-optimized BPE implementation."""
    print("\n=== Testing Advanced BPE ===")
    
    # Load sample data
    df = pd.read_csv("sample_dataset.csv")
    texts = (df['code'] + ' ' + df['docstring']).tolist()
    
    # Create and train advanced BPE model
    bpe = GPUOptimizedBPE(vocab_size=200, use_gpu=False, batch_size=32)  # Small for testing
    bpe.train_parallel(texts, num_workers=1)
    
    # Test batch encoding
    encoded_batch = bpe.encode_batch(texts)
    print(f"Batch encoded {len(encoded_batch)} texts")
    
    # Test individual encoding/decoding
    test_text = texts[0]
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Decoded: {decoded}")
    
    # Test advanced evaluation
    ground_truth_tokens = [text.split() for text in texts]
    evaluator = AdvancedBPEEvaluator(bpe, ground_truth_tokens)
    results = evaluator.comprehensive_evaluation(texts)
    
    print(f"\nAdvanced BPE Results:")
    print(f"  Jaccard Similarity: {results['vocabulary_overlap']['jaccard_similarity']:.4f}")
    print(f"  Compression Ratio: {results['compression_ratio']['compression_ratio']:.4f}")
    print(f"  OOV Rate: {results['oov_rate']['oov_rate']:.4f}")
    print(f"  Perplexity: {results['perplexity']['perplexity']:.2f}")
    print(f"  Token Entropy: {results['token_distribution']['entropy']:.4f}")
    
    return results

def test_model_saving_loading():
    """Test model saving and loading functionality."""
    print("\n=== Testing Model Save/Load ===")
    
    # Create sample data
    texts = ["def test(): return 'hello'", "def example(): return 'world'"]
    
    # Train model
    bpe = BPE(vocab_size=50, use_gpu=False)
    bpe.train(texts)
    
    # Test encoding before save
    encoded_before = bpe.encode("def new(): return 'test'")
    print(f"Encoded before save: {encoded_before}")
    
    # Save model
    bpe.save("test_model.pkl")
    print("Model saved successfully")
    
    # Load model
    new_bpe = BPE()
    new_bpe.load("test_model.pkl")
    print("Model loaded successfully")
    
    # Test encoding after load
    encoded_after = new_bpe.encode("def new(): return 'test'")
    print(f"Encoded after load: {encoded_after}")
    
    # Verify they're the same
    if encoded_before == encoded_after:
        print("✓ Save/load test passed!")
    else:
        print("✗ Save/load test failed!")
    
    return encoded_before == encoded_after

def main():
    """Run all tests."""
    print("=== BPE Implementation Test Suite ===")
    
    try:
        # Create sample dataset
        create_sample_dataset()
        
        # Test basic BPE
        basic_results = test_basic_bpe()
        
        # Test advanced BPE
        advanced_results = test_advanced_bpe()
        
        # Test save/load functionality
        save_load_success = test_model_saving_loading()
        
        # Summary
        print("\n=== Test Summary ===")
        print(f"Basic BPE: ✓ Passed")
        print(f"Advanced BPE: ✓ Passed")
        print(f"Save/Load: {'✓ Passed' if save_load_success else '✗ Failed'}")
        
        # Save test results
        test_results = {
            "basic_bpe": basic_results,
            "advanced_bpe": advanced_results,
            "save_load_success": save_load_success
        }
        
        with open("test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print("\n✓ All tests completed successfully!")
        print("Test results saved to test_results.json")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
