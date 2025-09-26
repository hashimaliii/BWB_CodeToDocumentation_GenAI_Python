#!/usr/bin/env python3
"""
Setup script for the BPE implementation.
Installs dependencies and checks system requirements.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed - cannot check CUDA")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def check_dataset():
    """Check if dataset file exists."""
    dataset_files = [
        "python_functions_and_documentation_dataset.csv",
        "sample_dataset.csv"
    ]
    
    for dataset_file in dataset_files:
        if os.path.exists(dataset_file):
            print(f"✓ Dataset found: {dataset_file}")
            return True
    
    print("⚠ No dataset file found")
    print("Please ensure your dataset CSV file is in the current directory")
    return False

def run_tests():
    """Run basic tests to verify installation."""
    print("Running basic tests...")
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import torch
        from bpe_implementation import BPE
        from gpu_bpe_advanced import GPUOptimizedBPE
        print("✓ All imports successful")
        
        # Test basic functionality
        bpe = BPE(vocab_size=100, use_gpu=False)
        test_texts = ["def hello(): return 'world'", "def add(a, b): return a + b"]
        bpe.train(test_texts)
        
        encoded = bpe.encode("def test(): return 'success'")
        decoded = bpe.decode(encoded)
        
        if "test" in decoded and "success" in decoded:
            print("✓ Basic BPE functionality working")
        else:
            print("✗ Basic BPE functionality test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=== BPE Implementation Setup ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("Please install dependencies manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Check dataset
    check_dataset()
    
    # Run tests
    if not run_tests():
        print("Setup completed with warnings. Some tests failed.")
        sys.exit(1)
    
    print("\n=== Setup Complete ===")
    print("You can now run the BPE implementation:")
    print("  python run_bpe.py --help                    # See all options")
    print("  python run_bpe.py --mode test              # Run tests")
    print("  python run_bpe.py --mode basic             # Run basic implementation")
    print("  python run_bpe.py --mode advanced          # Run advanced implementation")
    print()
    print("Or run examples directly:")
    print("  python example_usage.py                    # Run usage examples")
    print("  python test_implementation.py              # Run test suite")

if __name__ == "__main__":
    main()
