"""
Configuration file for BPE implementation.
Contains various configuration options for different use cases.
"""

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # Dataset settings
    "csv_path": "python_functions_and_documentation_dataset.csv",
    "sample_size": 10000,
    "test_size": 1000,
    
    # BPE settings
    "vocab_size": 5000,
    "use_gpu": True,
    "batch_size": 1024,
    "num_workers": 4,
    
    # Training settings
    "max_epochs": 100,
    "early_stopping_patience": 10,
    
    # Evaluation settings
    "evaluation_metrics": [
        "vocabulary_overlap",
        "compression_ratio", 
        "boundary_accuracy",
        "consistency",
        "oov_rate"
    ],
    
    # Output settings
    "output_dir": "outputs",
    "save_models": True,
    "save_visualizations": True,
    "save_analysis": True
}

# Configuration for different scenarios
CONFIGURATIONS = {
    "small": {
        "sample_size": 1000,
        "vocab_size": 1000,
        "batch_size": 256,
        "num_workers": 2,
        "use_gpu": False
    },
    
    "medium": {
        "sample_size": 5000,
        "vocab_size": 3000,
        "batch_size": 512,
        "num_workers": 4,
        "use_gpu": True
    },
    
    "large": {
        "sample_size": 20000,
        "vocab_size": 8000,
        "batch_size": 1024,
        "num_workers": 8,
        "use_gpu": True
    },
    
    "research": {
        "sample_size": 50000,
        "vocab_size": 15000,
        "batch_size": 2048,
        "num_workers": 12,
        "use_gpu": True,
        "evaluation_metrics": [
            "vocabulary_overlap",
            "compression_ratio",
            "boundary_accuracy", 
            "consistency",
            "oov_rate",
            "perplexity",
            "token_distribution",
            "semantic_coherence"
        ]
    },
    
    "production": {
        "sample_size": None,  # Use full dataset
        "vocab_size": 10000,
        "batch_size": 4096,
        "num_workers": 16,
        "use_gpu": True,
        "max_epochs": 200,
        "early_stopping_patience": 20
    }
}

# Specialized configurations for different data types
DATA_TYPE_CONFIGS = {
    "code_only": {
        "focus_columns": ["code"],
        "vocab_size": 4000,
        "special_tokens": ["<FUNC>", "<CLASS>", "<VAR>", "<NUM>", "<STR>"]
    },
    
    "docstring_only": {
        "focus_columns": ["docstring"],
        "vocab_size": 3000,
        "special_tokens": ["<SENT>", "<PARA>", "<LIST>", "<CODE>"]
    },
    
    "combined": {
        "focus_columns": ["code", "docstring"],
        "vocab_size": 6000,
        "special_tokens": ["<CODE>", "<DOC>", "<FUNC>", "<CLASS>", "<VAR>", "<NUM>", "<STR>"]
    }
}

# GPU-specific configurations
GPU_CONFIGS = {
    "no_gpu": {
        "use_gpu": False,
        "batch_size": 256,
        "num_workers": 4
    },
    
    "single_gpu": {
        "use_gpu": True,
        "batch_size": 1024,
        "num_workers": 4
    },
    
    "multi_gpu": {
        "use_gpu": True,
        "batch_size": 2048,
        "num_workers": 8,
        "distributed": True
    }
}

def get_config(config_name: str = "default", **overrides) -> Dict[str, Any]:
    """
    Get configuration by name with optional overrides.
    
    Args:
        config_name: Name of the configuration preset
        **overrides: Additional configuration overrides
    
    Returns:
        Configuration dictionary
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Apply preset configuration
    if config_name in CONFIGURATIONS:
        config.update(CONFIGURATIONS[config_name])
    
    # Apply overrides
    config.update(overrides)
    
    # Apply environment variables if they exist
    env_overrides = {}
    for key in config:
        env_key = f"BPE_{key.upper()}"
        if env_key in os.environ:
            value = os.environ[env_key]
            # Try to convert to appropriate type
            if config[key] is None:
                env_overrides[key] = value
            elif isinstance(config[key], bool):
                env_overrides[key] = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(config[key], int):
                env_overrides[key] = int(value)
            elif isinstance(config[key], float):
                env_overrides[key] = float(value)
            else:
                env_overrides[key] = value
    
    config.update(env_overrides)
    
    return config

def get_data_type_config(data_type: str, **overrides) -> Dict[str, Any]:
    """Get configuration for specific data type."""
    config = DEFAULT_CONFIG.copy()
    if data_type in DATA_TYPE_CONFIGS:
        config.update(DATA_TYPE_CONFIGS[data_type])
    config.update(overrides)
    return config

def get_gpu_config(gpu_setup: str, **overrides) -> Dict[str, Any]:
    """Get GPU-specific configuration."""
    config = DEFAULT_CONFIG.copy()
    if gpu_setup in GPU_CONFIGS:
        config.update(GPU_CONFIGS[gpu_setup])
    config.update(overrides)
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["csv_path", "vocab_size", "use_gpu"]
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    if config["vocab_size"] <= 0:
        print("Error: vocab_size must be positive")
        return False
    
    if config["batch_size"] <= 0:
        print("Error: batch_size must be positive")
        return False
    
    if config["num_workers"] < 1:
        print("Error: num_workers must be at least 1")
        return False
    
    return True

def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format."""
    print("=== BPE Configuration ===")
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")
    print("=" * 25)

# Example usage configurations
EXAMPLE_CONFIGS = {
    "quick_test": get_config("small", sample_size=100, vocab_size=500),
    "development": get_config("medium", sample_size=2000, vocab_size=2000),
    "benchmark": get_config("research", sample_size=10000, vocab_size=5000),
    "production_ready": get_config("production", sample_size=None, vocab_size=10000)
}

if __name__ == "__main__":
    # Example usage
    print("Available configurations:")
    for name, config in EXAMPLE_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print_config(config)
