#!/usr/bin/env python3
"""
Benchmarking utilities for ZPDR.

This module provides benchmarking tools for measuring the performance
of the ZPDR system under various configurations and workloads.
"""

import os
import time
import tempfile
import random
import gc
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import json
import matplotlib.pyplot as plt
import numpy as np

from ..core.zpdr_processor import ZPDRProcessor
from ..core.streaming_zpdr_processor import StreamingZPDRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZPDRBenchmark:
    """
    Benchmarking tool for ZPDR performance analysis.
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 generate_plots: bool = True):
        """
        Initialize the benchmarking tool.
        
        Args:
            output_dir: Directory to save benchmark results and plots
            generate_plots: Whether to generate plots for benchmark results
        """
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="zpdr_benchmark_")
        self.generate_plots = generate_plots
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized ZPDR benchmark with output directory: {self.output_dir}")
    
    def benchmark_file_size_scaling(self, 
                                 size_range: List[int] = None,
                                 repetitions: int = 3) -> Dict[str, Any]:
        """
        Benchmark how ZPDR performance scales with file size.
        
        Args:
            size_range: List of file sizes to test (in bytes)
            repetitions: Number of times to repeat each test
            
        Returns:
            Dictionary of benchmark results
        """
        size_range = size_range or [1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024]
        
        logger.info(f"Running file size scaling benchmark with sizes: {size_range}")
        
        results = {
            'standard': {size: {'encode': [], 'decode': []} for size in size_range},
            'streaming': {size: {'encode': [], 'decode': []} for size in size_range},
        }
        
        # Create processor instances
        standard_processor = ZPDRProcessor()
        streaming_processor = StreamingZPDRProcessor()
        
        for size in size_range:
            logger.info(f"Benchmarking file size: {size} bytes")
            
            for _ in range(repetitions):
                # Generate a random test file
                test_file, test_data = self._generate_test_file(size)
                
                try:
                    # Benchmark standard processor
                    gc.collect()  # Force garbage collection
                    
                    # Encoding
                    encode_start = time.time()
                    manifest = standard_processor.process_file(test_file)
                    manifest_file = Path(self.output_dir) / f"manifest_{size}.zpdr"
                    standard_processor.save_manifest(manifest, manifest_file)
                    encode_time = time.time() - encode_start
                    
                    results['standard'][size]['encode'].append(encode_time)
                    
                    # Decoding
                    output_file = Path(self.output_dir) / f"output_{size}.bin"
                    
                    gc.collect()  # Force garbage collection
                    
                    decode_start = time.time()
                    standard_processor.reconstruct_file(manifest_file, output_file)
                    decode_time = time.time() - decode_start
                    
                    results['standard'][size]['decode'].append(decode_time)
                    
                    # Verify reconstruction
                    with open(output_file, 'rb') as f:
                        output_data = f.read()
                    
                    if hashlib.sha256(output_data).hexdigest() != hashlib.sha256(test_data).hexdigest():
                        logger.warning(f"Standard processor verification failed for size {size}")
                    
                    # Benchmark streaming processor
                    gc.collect()  # Force garbage collection
                    
                    # Encoding
                    encode_start = time.time()
                    manifest = streaming_processor.process_file(test_file)
                    manifest_file = Path(self.output_dir) / f"manifest_stream_{size}.zpdr"
                    standard_processor.save_manifest(manifest, manifest_file)
                    encode_time = time.time() - encode_start
                    
                    results['streaming'][size]['encode'].append(encode_time)
                    
                    # Decoding
                    output_file = Path(self.output_dir) / f"output_stream_{size}.bin"
                    
                    gc.collect()  # Force garbage collection
                    
                    decode_start = time.time()
                    streaming_processor.reconstruct_file(manifest_file, output_file)
                    decode_time = time.time() - decode_start
                    
                    results['streaming'][size]['decode'].append(decode_time)
                    
                    # Verify reconstruction
                    with open(output_file, 'rb') as f:
                        output_data = f.read()
                    
                    if hashlib.sha256(output_data).hexdigest() != hashlib.sha256(test_data).hexdigest():
                        logger.warning(f"Streaming processor verification failed for size {size}")
                    
                finally:
                    # Clean up test file
                    try:
                        os.unlink(test_file)
                    except:
                        pass
        
        # Calculate averages
        averages = {
            'standard': {size: {
                'encode': sum(results['standard'][size]['encode']) / len(results['standard'][size]['encode']),
                'decode': sum(results['standard'][size]['decode']) / len(results['standard'][size]['decode'])
            } for size in size_range},
            'streaming': {size: {
                'encode': sum(results['streaming'][size]['encode']) / len(results['streaming'][size]['encode']),
                'decode': sum(results['streaming'][size]['decode']) / len(results['streaming'][size]['decode'])
            } for size in size_range},
        }
        
        # Calculate speedups
        speedups = {
            'encode': {size: averages['standard'][size]['encode'] / averages['streaming'][size]['encode'] 
                      for size in size_range},
            'decode': {size: averages['standard'][size]['decode'] / averages['streaming'][size]['decode']
                      for size in size_range}
        }
        
        # Compile results
        benchmark_results = {
            'raw_results': results,
            'averages': averages,
            'speedups': speedups,
            'size_range': size_range,
            'repetitions': repetitions,
            'timestamp': time.time()
        }
        
        # Save results
        self._save_results(benchmark_results, 'file_size_scaling')
        
        # Generate plots
        if self.generate_plots:
            self._plot_file_size_scaling(benchmark_results)
        
        return benchmark_results
    
    def benchmark_parallel_scaling(self, 
                                  file_size: int = 10*1024*1024,
                                  worker_range: List[int] = None,
                                  repetitions: int = 3) -> Dict[str, Any]:
        """
        Benchmark how performance scales with number of parallel workers.
        
        Args:
            file_size: Size of test file in bytes
            worker_range: List of worker counts to test
            repetitions: Number of times to repeat each test
            
        Returns:
            Dictionary of benchmark results
        """
        worker_range = worker_range or [1, 2, 4, 8, 16]
        
        logger.info(f"Running parallel scaling benchmark with {file_size} byte file and workers: {worker_range}")
        
        results = {workers: {'encode': [], 'decode': []} for workers in worker_range}
        
        # Generate a single test file for all tests
        test_file, test_data = self._generate_test_file(file_size)
        
        try:
            for workers in worker_range:
                logger.info(f"Benchmarking with {workers} workers")
                
                # Create streaming processor with specified workers
                processor = StreamingZPDRProcessor(max_workers=workers)
                
                for _ in range(repetitions):
                    gc.collect()  # Force garbage collection
                    
                    # Encoding
                    encode_start = time.time()
                    manifest = processor.process_file(test_file)
                    manifest_file = Path(self.output_dir) / f"manifest_workers_{workers}.zpdr"
                    processor.save_manifest(manifest, manifest_file)
                    encode_time = time.time() - encode_start
                    
                    results[workers]['encode'].append(encode_time)
                    
                    # Decoding
                    output_file = Path(self.output_dir) / f"output_workers_{workers}.bin"
                    
                    gc.collect()  # Force garbage collection
                    
                    decode_start = time.time()
                    processor.reconstruct_file(manifest_file, output_file)
                    decode_time = time.time() - decode_start
                    
                    results[workers]['decode'].append(decode_time)
                    
                    # Verify reconstruction
                    with open(output_file, 'rb') as f:
                        output_data = f.read()
                    
                    if hashlib.sha256(output_data).hexdigest() != hashlib.sha256(test_data).hexdigest():
                        logger.warning(f"Verification failed for {workers} workers")
        finally:
            # Clean up test file
            try:
                os.unlink(test_file)
            except:
                pass
        
        # Calculate averages
        averages = {workers: {
            'encode': sum(results[workers]['encode']) / len(results[workers]['encode']),
            'decode': sum(results[workers]['decode']) / len(results[workers]['decode'])
        } for workers in worker_range}
        
        # Calculate speedups relative to single worker
        speedups = {
            'encode': {workers: averages[1]['encode'] / averages[workers]['encode'] for workers in worker_range},
            'decode': {workers: averages[1]['decode'] / averages[workers]['decode'] for workers in worker_range}
        }
        
        # Compile results
        benchmark_results = {
            'raw_results': results,
            'averages': averages,
            'speedups': speedups,
            'worker_range': worker_range,
            'file_size': file_size,
            'repetitions': repetitions,
            'timestamp': time.time()
        }
        
        # Save results
        self._save_results(benchmark_results, 'parallel_scaling')
        
        # Generate plots
        if self.generate_plots:
            self._plot_parallel_scaling(benchmark_results)
        
        return benchmark_results
    
    def benchmark_memory_usage(self, 
                             size_range: List[int] = None,
                             memory_limits: List[int] = None,
                             repetitions: int = 3) -> Dict[str, Any]:
        """
        Benchmark memory usage under different memory limits.
        
        Args:
            size_range: List of file sizes to test (in bytes)
            memory_limits: List of memory limits to test (in MB)
            repetitions: Number of times to repeat each test
            
        Returns:
            Dictionary of benchmark results
        """
        size_range = size_range or [1024*1024, 10*1024*1024, 100*1024*1024]
        memory_limits = memory_limits or [10, 50, 100, 500]
        
        logger.info(f"Running memory usage benchmark with sizes: {size_range} and limits: {memory_limits}")
        
        results = {
            size: {limit: {'encode': [], 'decode': []} 
                  for limit in memory_limits}
            for size in size_range
        }
        
        for size in size_range:
            # Generate test file for this size
            test_file, test_data = self._generate_test_file(size)
            
            try:
                for limit in memory_limits:
                    logger.info(f"Benchmarking file size {size} with {limit}MB limit")
                    
                    # Create processor with specified memory limit
                    processor = StreamingZPDRProcessor(max_memory_mb=limit)
                    
                    for _ in range(repetitions):
                        gc.collect()  # Force garbage collection
                        
                        # Encoding
                        encode_start = time.time()
                        try:
                            manifest = processor.process_file(test_file)
                            manifest_file = Path(self.output_dir) / f"manifest_mem_{size}_{limit}.zpdr"
                            processor.save_manifest(manifest, manifest_file)
                            encode_time = time.time() - encode_start
                            
                            results[size][limit]['encode'].append(encode_time)
                            
                            # Decoding
                            output_file = Path(self.output_dir) / f"output_mem_{size}_{limit}.bin"
                            
                            gc.collect()  # Force garbage collection
                            
                            decode_start = time.time()
                            processor.reconstruct_file(manifest_file, output_file)
                            decode_time = time.time() - decode_start
                            
                            results[size][limit]['decode'].append(decode_time)
                            
                            # Verify reconstruction
                            with open(output_file, 'rb') as f:
                                output_data = f.read()
                            
                            if hashlib.sha256(output_data).hexdigest() != hashlib.sha256(test_data).hexdigest():
                                logger.warning(f"Verification failed for size {size} and limit {limit}MB")
                                
                        except Exception as e:
                            logger.error(f"Error with size {size} and limit {limit}MB: {e}")
                            results[size][limit]['encode'].append(None)
                            results[size][limit]['decode'].append(None)
            finally:
                # Clean up test file
                try:
                    os.unlink(test_file)
                except:
                    pass
        
        # Calculate averages (ignoring None values)
        averages = {
            size: {limit: {
                'encode': sum(filter(None, results[size][limit]['encode'])) / len(list(filter(None, results[size][limit]['encode']))) if list(filter(None, results[size][limit]['encode'])) else None,
                'decode': sum(filter(None, results[size][limit]['decode'])) / len(list(filter(None, results[size][limit]['decode']))) if list(filter(None, results[size][limit]['decode'])) else None
            } for limit in memory_limits}
            for size in size_range
        }
        
        # Compile results
        benchmark_results = {
            'raw_results': results,
            'averages': averages,
            'size_range': size_range,
            'memory_limits': memory_limits,
            'repetitions': repetitions,
            'timestamp': time.time()
        }
        
        # Save results
        self._save_results(benchmark_results, 'memory_usage')
        
        # Generate plots
        if self.generate_plots:
            self._plot_memory_usage(benchmark_results)
        
        return benchmark_results
    
    def _generate_test_file(self, size: int) -> Tuple[str, bytes]:
        """
        Generate a test file of the specified size.
        
        Args:
            size: Size of the file in bytes
            
        Returns:
            Tuple of (file path, file data)
        """
        # Create a temporary file
        fd, path = tempfile.mkstemp(prefix="zpdr_test_", suffix=".dat")
        os.close(fd)
        
        # Generate random data
        if size <= 1024 * 1024:  # For files up to 1MB, use pure random data
            data = os.urandom(size)
        else:
            # For larger files, generate patterns for better compression
            chunk_size = 1024
            chunks = []
            
            while sum(len(c) for c in chunks) < size:
                # Add some random data
                if random.random() < 0.6:
                    chunks.append(os.urandom(min(chunk_size, size - sum(len(c) for c in chunks))))
                # Add some repeating patterns
                else:
                    pattern = os.urandom(16)
                    pattern_size = min(chunk_size, size - sum(len(c) for c in chunks))
                    chunks.append((pattern * (pattern_size // len(pattern) + 1))[:pattern_size])
                
                if sum(len(c) for c in chunks) >= size:
                    break
            
            data = b''.join(chunks)[:size]
        
        # Write data to file
        with open(path, 'wb') as f:
            f.write(data)
        
        return path, data
    
    def _save_results(self, results: Dict[str, Any], benchmark_name: str) -> str:
        """
        Save benchmark results to a JSON file.
        
        Args:
            results: Benchmark results to save
            benchmark_name: Name of the benchmark
            
        Returns:
            Path to the saved file
        """
        # Convert non-serializable types (like Decimal)
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            elif isinstance(obj, Decimal):
                return float(obj)
            else:
                return obj
        
        # Convert results to JSON-serializable format
        serializable_results = convert_for_json(results)
        
        # Create filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_name}_{timestamp}.json"
        file_path = Path(self.output_dir) / filename
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {file_path}")
        
        return str(file_path)
    
    def _plot_file_size_scaling(self, results: Dict[str, Any]) -> str:
        """
        Generate plots for file size scaling benchmark.
        
        Args:
            results: Benchmark results
            
        Returns:
            Path to the saved plot file
        """
        plt.figure(figsize=(12, 8))
        
        # Convert sizes to KB/MB for readability
        sizes = results['size_range']
        size_labels = []
        for size in sizes:
            if size < 1024:
                size_labels.append(f"{size}B")
            elif size < 1024 * 1024:
                size_labels.append(f"{size/1024:.1f}KB")
            else:
                size_labels.append(f"{size/(1024*1024):.1f}MB")
        
        # Plot encoding times
        plt.subplot(2, 2, 1)
        plt.plot(size_labels, [results['averages']['standard'][size]['encode'] for size in sizes], 'b-o', label='Standard')
        plt.plot(size_labels, [results['averages']['streaming'][size]['encode'] for size in sizes], 'r-o', label='Streaming')
        plt.xlabel('File Size')
        plt.ylabel('Time (seconds)')
        plt.title('Encoding Time vs File Size')
        plt.legend()
        plt.grid(True)
        
        # Plot decoding times
        plt.subplot(2, 2, 2)
        plt.plot(size_labels, [results['averages']['standard'][size]['decode'] for size in sizes], 'b-o', label='Standard')
        plt.plot(size_labels, [results['averages']['streaming'][size]['decode'] for size in sizes], 'r-o', label='Streaming')
        plt.xlabel('File Size')
        plt.ylabel('Time (seconds)')
        plt.title('Decoding Time vs File Size')
        plt.legend()
        plt.grid(True)
        
        # Plot encoding speedup
        plt.subplot(2, 2, 3)
        plt.plot(size_labels, [results['speedups']['encode'][size] for size in sizes], 'g-o')
        plt.xlabel('File Size')
        plt.ylabel('Speedup (ratio)')
        plt.title('Encoding Speedup: Standard vs Streaming')
        plt.axhline(y=1, color='k', linestyle='--')
        plt.grid(True)
        
        # Plot decoding speedup
        plt.subplot(2, 2, 4)
        plt.plot(size_labels, [results['speedups']['decode'][size] for size in sizes], 'g-o')
        plt.xlabel('File Size')
        plt.ylabel('Speedup (ratio)')
        plt.title('Decoding Speedup: Standard vs Streaming')
        plt.axhline(y=1, color='k', linestyle='--')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = Path(self.output_dir) / f"file_size_scaling_{timestamp}.png"
        plt.savefig(file_path)
        
        logger.info(f"Saved plot to {file_path}")
        
        return str(file_path)
    
    def _plot_parallel_scaling(self, results: Dict[str, Any]) -> str:
        """
        Generate plots for parallel scaling benchmark.
        
        Args:
            results: Benchmark results
            
        Returns:
            Path to the saved plot file
        """
        plt.figure(figsize=(12, 8))
        
        workers = results['worker_range']
        
        # Plot encoding times
        plt.subplot(2, 2, 1)
        plt.plot(workers, [results['averages'][w]['encode'] for w in workers], 'b-o')
        plt.xlabel('Number of Workers')
        plt.ylabel('Time (seconds)')
        plt.title('Encoding Time vs Number of Workers')
        plt.grid(True)
        
        # Plot decoding times
        plt.subplot(2, 2, 2)
        plt.plot(workers, [results['averages'][w]['decode'] for w in workers], 'r-o')
        plt.xlabel('Number of Workers')
        plt.ylabel('Time (seconds)')
        plt.title('Decoding Time vs Number of Workers')
        plt.grid(True)
        
        # Plot encoding speedup
        plt.subplot(2, 2, 3)
        plt.plot(workers, [results['speedups']['encode'][w] for w in workers], 'g-o')
        plt.plot(workers, workers, 'k--', label='Ideal')
        plt.xlabel('Number of Workers')
        plt.ylabel('Speedup (ratio)')
        plt.title('Encoding Speedup vs Number of Workers')
        plt.legend()
        plt.grid(True)
        
        # Plot decoding speedup
        plt.subplot(2, 2, 4)
        plt.plot(workers, [results['speedups']['decode'][w] for w in workers], 'g-o')
        plt.plot(workers, workers, 'k--', label='Ideal')
        plt.xlabel('Number of Workers')
        plt.ylabel('Speedup (ratio)')
        plt.title('Decoding Speedup vs Number of Workers')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = Path(self.output_dir) / f"parallel_scaling_{timestamp}.png"
        plt.savefig(file_path)
        
        logger.info(f"Saved plot to {file_path}")
        
        return str(file_path)
    
    def _plot_memory_usage(self, results: Dict[str, Any]) -> str:
        """
        Generate plots for memory usage benchmark.
        
        Args:
            results: Benchmark results
            
        Returns:
            Path to the saved plot file
        """
        plt.figure(figsize=(12, 8))
        
        sizes = results['size_range']
        limits = results['memory_limits']
        
        # Convert sizes to KB/MB for readability
        size_labels = []
        for size in sizes:
            if size < 1024:
                size_labels.append(f"{size}B")
            elif size < 1024 * 1024:
                size_labels.append(f"{size/1024:.1f}KB")
            else:
                size_labels.append(f"{size/(1024*1024):.1f}MB")
        
        # Plot encoding times for each size
        plt.subplot(2, 1, 1)
        for i, size in enumerate(sizes):
            encode_times = []
            for limit in limits:
                time_value = results['averages'][size][limit]['encode']
                encode_times.append(time_value if time_value is not None else float('nan'))
            plt.plot(limits, encode_times, marker='o', label=size_labels[i])
        
        plt.xlabel('Memory Limit (MB)')
        plt.ylabel('Time (seconds)')
        plt.title('Encoding Time vs Memory Limit')
        plt.legend()
        plt.grid(True)
        
        # Plot decoding times for each size
        plt.subplot(2, 1, 2)
        for i, size in enumerate(sizes):
            decode_times = []
            for limit in limits:
                time_value = results['averages'][size][limit]['decode']
                decode_times.append(time_value if time_value is not None else float('nan'))
            plt.plot(limits, decode_times, marker='o', label=size_labels[i])
        
        plt.xlabel('Memory Limit (MB)')
        plt.ylabel('Time (seconds)')
        plt.title('Decoding Time vs Memory Limit')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = Path(self.output_dir) / f"memory_usage_{timestamp}.png"
        plt.savefig(file_path)
        
        logger.info(f"Saved plot to {file_path}")
        
        return str(file_path)

# Example usage
if __name__ == "__main__":
    benchmark = ZPDRBenchmark()
    
    # Run a simple file size scaling benchmark with small files
    results = benchmark.benchmark_file_size_scaling(
        size_range=[1024, 10*1024, 100*1024],  # 1KB, 10KB, 100KB
        repetitions=2
    )
    
    print(f"Benchmark results saved to: {benchmark.output_dir}")
    print(f"Standard vs Streaming speedup for 100KB file:")
    print(f"  Encoding: {results['speedups']['encode'][100*1024]:.2f}x")
    print(f"  Decoding: {results['speedups']['decode'][100*1024]:.2f}x")