"""
Runner script for Word2Vec analysis
Executes Word2Vec training and evaluation on BPE tokenized data
"""

import os
import sys
import time
from word2vec_implementation import Word2VecAnalyzer

def main():
    """Run Word2Vec analysis"""
    print("Starting Word2Vec Analysis")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "python_functions_and_documentation_dataset.csv",
        "bpe_code_model.pkl",
        "bpe_doc_model.pkl", 
        "bpe_combined_model.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all BPE models and dataset are available.")
        return
    
    # Initialize analyzer
    analyzer = Word2VecAnalyzer(output_dir="word2vec")
    
    try:
        # Run complete analysis
        start_time = time.time()
        
        results = analyzer.run_complete_analysis(
            csv_path="python_functions_and_documentation_dataset.csv",
            code_model_path="bpe_code_model.pkl",
            doc_model_path="bpe_doc_model.pkl",
            combined_model_path="bpe_combined_model.pkl"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nAnalysis completed in {duration:.2f} seconds")
        print(f"Results saved in: {analyzer.output_dir}/")
        
        # Print summary
        print("\nResults Summary:")
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
        
        print(f"\nVisualizations saved in: {analyzer.output_dir}/")
        print("Files created:")
        for file in os.listdir(analyzer.output_dir):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
