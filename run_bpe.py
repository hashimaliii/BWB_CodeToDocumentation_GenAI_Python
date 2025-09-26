#!/usr/bin/env python3
"""
Main script to run the BPE implementation on your dataset.
This script provides a simple interface to run the complete BPE pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run BPE implementation on Python function dataset')
    parser.add_argument('--csv', type=str, default='python_functions_and_documentation_dataset.csv',
                       help='Path to the CSV dataset file')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of samples to use for training (default: 10000)')
    parser.add_argument('--vocab-size', type=int, default=5000,
                       help='Target vocabulary size (default: 5000)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--mode', type=str, choices=['basic', 'advanced', 'test', 'simple'], default='basic',
                       help='Run mode: basic, advanced, test, or simple (default: basic)')
    parser.add_argument('--full-dataset', action='store_true',
                       help='Use full dataset instead of sampling')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found!")
        print("Please make sure the dataset file exists in the current directory.")
        sys.exit(1)
    
    # Determine GPU usage
    use_gpu = args.gpu and not args.no_gpu
    
    print("=== BPE Implementation Runner ===")
    print(f"Dataset: {args.csv}")
    print(f"Sample size: {args.sample_size}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"GPU acceleration: {use_gpu}")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers}")
    print()
    
    try:
        if args.mode == 'test':
            print("Running test mode with sample dataset...")
            from test_implementation import main as test_main
            test_main()
            
        elif args.mode == 'basic':
            print("Running basic BPE implementation...")
            from bpe_implementation import main as basic_main
            
            # Modify the main function parameters
            import bpe_implementation
            original_main = bpe_implementation.main
            
            def custom_main():
                # Update global variables in the module
                bpe_implementation.CSV_PATH = args.csv
                bpe_implementation.SAMPLE_SIZE = args.sample_size
                bpe_implementation.VOCAB_SIZE = args.vocab_size
                bpe_implementation.USE_GPU = use_gpu
                original_main()
            
            custom_main()
            
        elif args.mode == 'advanced':
            print("Running advanced GPU-optimized BPE implementation...")
            from gpu_bpe_advanced import main_advanced as advanced_main
            
            # Modify the main function parameters
            import gpu_bpe_advanced
            original_main = gpu_bpe_advanced.main_advanced
            
            def custom_main():
                # Update global variables in the module
                gpu_bpe_advanced.CSV_PATH = args.csv
                # Handle full dataset flag
                if args.full_dataset:
                    gpu_bpe_advanced.SAMPLE_SIZE = None  # Use full dataset
                    print("Using full dataset (no sampling)")
                else:
                    gpu_bpe_advanced.SAMPLE_SIZE = args.sample_size
                    print(f"Using sample size: {args.sample_size}")
                gpu_bpe_advanced.VOCAB_SIZE = args.vocab_size
                gpu_bpe_advanced.USE_GPU = use_gpu
                gpu_bpe_advanced.BATCH_SIZE = 1024
                original_main()
            
            custom_main()
            
        elif args.mode == 'simple':
            print("Running simple BPE implementation...")
            from bpe_single_file import main as simple_main
            
            # Determine if using full dataset or sample
            use_sample = not args.full_dataset
            
            print(f"Using {'full dataset' if not use_sample else f'sample of {args.sample_size} rows'}")
            
            # Run simple BPE with parameters
            simple_main(
                use_sample=use_sample,
                sample_size=args.sample_size,
                vocab_size=args.vocab_size,
                min_frequency=2,
                csv_path=args.csv
            )
        
        print("\n✓ BPE implementation completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error running BPE implementation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

