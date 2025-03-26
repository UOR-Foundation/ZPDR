#!/usr/bin/env python3
"""
Example demonstrating the optimized Phase 3 features of ZPDR.

This example shows:
1. Processing large files with streaming capabilities
2. Using parallel processing for improved performance
3. Benchmarking different configuration options
"""

import os
import time
import tempfile
import random
from pathlib import Path
import matplotlib.pyplot as plt

from zpdr.core.zpdr_processor import ZPDRProcessor
from zpdr.core.streaming_zpdr_processor import StreamingZPDRProcessor
from zpdr.utils.benchmarking import ZPDRBenchmark

def generate_test_file(size_mb: int, file_path: str = None) -> str:
    """
    Generate a test file of the specified size.
    
    Args:
        size_mb: Size of the file in MB
        file_path: Optional file path to save to
        
    Returns:
        Path to the generated file
    """
    size_bytes = size_mb * 1024 * 1024
    
    if file_path is None:
        fd, path = tempfile.mkstemp(prefix="zpdr_test_", suffix=".dat")
        os.close(fd)
    else:
        path = file_path
    
    print(f"Generating {size_mb}MB test file at {path}")
    
    # Generate data in chunks to avoid memory issues with large files
    chunk_size = 1024 * 1024  # 1MB chunks
    with open(path, 'wb') as f:
        bytes_written = 0
        while bytes_written < size_bytes:
            # Determine chunk size for this iteration
            current_chunk_size = min(chunk_size, size_bytes - bytes_written)
            
            # Generate random data
            data = os.urandom(current_chunk_size)
            f.write(data)
            
            bytes_written += current_chunk_size
    
    print(f"Test file generated: {os.path.getsize(path) / (1024*1024):.2f}MB")
    return path

def compare_standard_vs_streaming(file_path: str) -> None:
    """
    Compare standard and streaming processors on the same file.
    
    Args:
        file_path: Path to the test file
    """
    print("\n=== Standard vs Streaming Processor Comparison ===")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f}MB")
    
    # Test with standard processor
    standard_processor = ZPDRProcessor()
    
    print("\nStandard Processor:")
    start_time = time.time()
    standard_manifest = standard_processor.process_file(file_path)
    standard_time = time.time() - start_time
    print(f"  Encoding time: {standard_time:.2f} seconds")
    
    standard_manifest_path = f"{file_path}.standard.zpdr"
    standard_processor.save_manifest(standard_manifest, standard_manifest_path)
    
    output_path = f"{file_path}.standard.out"
    start_time = time.time()
    standard_processor.reconstruct_file(standard_manifest_path, output_path)
    standard_decode_time = time.time() - start_time
    print(f"  Decoding time: {standard_decode_time:.2f} seconds")
    
    # Test with streaming processor (with different worker counts)
    for workers in [1, 2, 4]:
        print(f"\nStreaming Processor ({workers} workers):")
        streaming_processor = StreamingZPDRProcessor(
            chunk_size=1*1024*1024,  # 1MB chunks
            max_workers=workers
        )
        
        start_time = time.time()
        streaming_manifest = streaming_processor.process_file(file_path)
        streaming_time = time.time() - start_time
        print(f"  Encoding time: {streaming_time:.2f} seconds")
        print(f"  Speedup: {standard_time / streaming_time:.2f}x")
        
        streaming_manifest_path = f"{file_path}.streaming{workers}.zpdr"
        streaming_processor.save_manifest(streaming_manifest, streaming_manifest_path)
        
        output_path = f"{file_path}.streaming{workers}.out"
        start_time = time.time()
        streaming_processor.reconstruct_file(streaming_manifest_path, output_path)
        streaming_decode_time = time.time() - start_time
        print(f"  Decoding time: {streaming_decode_time:.2f} seconds")
        print(f"  Speedup: {standard_decode_time / streaming_decode_time:.2f}x")

def run_mini_benchmark(output_dir: str = None) -> None:
    """
    Run a mini benchmark to demonstrate the benchmarking functionality.
    
    Args:
        output_dir: Directory to save benchmark results
    """
    print("\n=== Running Mini Benchmark ===")
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="zpdr_benchmark_")
    
    print(f"Benchmark results will be saved to: {output_dir}")
    
    # Initialize benchmark tool
    benchmark = ZPDRBenchmark(output_dir=output_dir)
    
    # Run a small file size scaling benchmark
    print("\nRunning file size scaling benchmark...")
    size_results = benchmark.benchmark_file_size_scaling(
        size_range=[10*1024, 100*1024, 1024*1024],  # 10KB, 100KB, 1MB
        repetitions=2
    )
    
    # Print summary
    print("\nFile Size Scaling Results:")
    for size in size_results['size_range']:
        size_text = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
        print(f"Size: {size_text}")
        
        # Standard processor
        std_encode = size_results['averages']['standard'][size]['encode']
        std_decode = size_results['averages']['standard'][size]['decode']
        print(f"  Standard:  Encode = {std_encode:.3f}s, Decode = {std_decode:.3f}s")
        
        # Streaming processor
        stream_encode = size_results['averages']['streaming'][size]['encode']
        stream_decode = size_results['averages']['streaming'][size]['decode']
        print(f"  Streaming: Encode = {stream_encode:.3f}s, Decode = {stream_decode:.3f}s")
        
        # Speedup
        encode_speedup = size_results['speedups']['encode'][size]
        decode_speedup = size_results['speedups']['decode'][size]
        print(f"  Speedup:   Encode = {encode_speedup:.2f}x, Decode = {decode_speedup:.2f}x")
    
    print(f"\nDetailed results and plots saved to: {output_dir}")

def memory_efficiency_demo(file_size_mb: int = 10) -> None:
    """
    Demonstrate memory efficiency options.
    
    Args:
        file_size_mb: Size of the test file in MB
    """
    print("\n=== Memory Efficiency Demonstration ===")
    
    # Generate a test file
    test_file = generate_test_file(file_size_mb)
    
    print("\nTesting different memory limits:")
    
    for memory_limit_mb in [5, 10, 50]:
        print(f"\nMemory limit: {memory_limit_mb}MB")
        
        # Create memory-limited processor
        processor = StreamingZPDRProcessor(
            max_memory_mb=memory_limit_mb,
            chunk_size=1*1024*1024  # 1MB chunks
        )
        
        try:
            start_time = time.time()
            manifest = processor.process_file(test_file)
            processing_time = time.time() - start_time
            
            manifest_path = f"{test_file}.mem{memory_limit_mb}.zpdr"
            processor.save_manifest(manifest, manifest_path)
            
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Manifest saved to: {manifest_path}")
            
            # Reconstruct to verify
            output_path = f"{test_file}.mem{memory_limit_mb}.out"
            start_time = time.time()
            processor.reconstruct_file(manifest_path, output_path)
            decode_time = time.time() - start_time
            
            print(f"  Reconstruction time: {decode_time:.2f} seconds")
            print(f"  Output file: {output_path}")
            
            # Verify files match
            import hashlib
            
            with open(test_file, 'rb') as f:
                original_hash = hashlib.sha256(f.read()).hexdigest()
                
            with open(output_path, 'rb') as f:
                output_hash = hashlib.sha256(f.read()).hexdigest()
                
            if original_hash == output_hash:
                print(f"  Verification: PASS")
            else:
                print(f"  Verification: FAIL")
                
        except Exception as e:
            print(f"  Error with {memory_limit_mb}MB limit: {e}")
    
    # Clean up
    try:
        os.unlink(test_file)
    except:
        pass

def main():
    """Main entry point."""
    print("ZPDR Phase 3 Optimization Example")
    
    # Create a moderate-sized test file (adjust based on your system memory)
    test_file = generate_test_file(10)  # 10MB
    
    try:
        # Compare standard and streaming processors
        compare_standard_vs_streaming(test_file)
        
        # Run a mini benchmark
        run_mini_benchmark()
        
        # Demonstrate memory efficiency
        memory_efficiency_demo(5)  # 5MB
        
    finally:
        # Clean up test files
        try:
            if os.path.exists(test_file):
                os.unlink(test_file)
                
            # Clean up output files
            for ext in ['.standard.zpdr', '.standard.out', 
                       '.streaming1.zpdr', '.streaming1.out',
                       '.streaming2.zpdr', '.streaming2.out',
                       '.streaming4.zpdr', '.streaming4.out']:
                path = test_file + ext
                if os.path.exists(path):
                    os.unlink(path)
        except:
            pass

if __name__ == '__main__':
    main()