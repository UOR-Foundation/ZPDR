#!/usr/bin/env python3
"""
Command-line interface for the ZPDR system.

This module provides a command-line interface for encoding and decoding
files using the Zero-Point Data Reconstruction (ZPDR) system.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from zpdr.core.zpdr_processor import ZPDRProcessor
from zpdr.core.streaming_zpdr_processor import StreamingZPDRProcessor
from zpdr.utils.benchmarking import ZPDRBenchmark

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: List of command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Zero-Point Data Reconstruction (ZPDR) system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Common options for all subcommands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    common_parser.add_argument('--threshold', '-t', type=float, default=0.99,
                       help='Coherence threshold for verification')
    
    # Processor options
    processor_parser = argparse.ArgumentParser(add_help=False, parents=[common_parser])
    processor_parser.add_argument('--streaming', '-s', action='store_true',
                         help='Use streaming processor for better memory efficiency')
    processor_parser.add_argument('--chunk-size', type=int, default=1024*1024,
                         help='Size of chunks for processing large files (bytes)')
    processor_parser.add_argument('--max-workers', type=int, default=None,
                         help='Maximum number of parallel workers')
    processor_parser.add_argument('--max-memory', type=int, default=100,
                         help='Maximum memory usage in MB')
    processor_parser.add_argument('--auto-correct', '-a', action='store_true',
                         help='Automatically correct errors during processing')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', parents=[processor_parser], 
                                        help='Encode a file to zero-point coordinates')
    encode_parser.add_argument('input_file', help='Path to the input file')
    encode_parser.add_argument('--output', '-o', help='Path to save the manifest (defaults to input_file.zpdr)')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', parents=[processor_parser],
                                        help='Decode a file from zero-point coordinates')
    decode_parser.add_argument('manifest', help='Path to the manifest file')
    decode_parser.add_argument('--output', '-o', help='Path to save the decoded file (defaults to original filename)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', parents=[common_parser],
                                        help='Verify a manifest')
    verify_parser.add_argument('manifest', help='Path to the manifest file')
    verify_parser.add_argument('--comprehensive', '-c', action='store_true',
                             help='Perform comprehensive verification including invariants')
    
    # Correct command
    correct_parser = subparsers.add_parser('correct', parents=[common_parser],
                                         help='Correct a manifest to improve coherence')
    correct_parser.add_argument('manifest', help='Path to the manifest file')
    correct_parser.add_argument('--output', '-o', help='Path to save the corrected manifest (defaults to manifest_corrected.zpdr)')
    
    # Info command
    info_parser = subparsers.add_parser('info', parents=[common_parser],
                                      help='Display information about a manifest')
    info_parser.add_argument('manifest', help='Path to the manifest file')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', parents=[common_parser],
                                           help='Run benchmarks for ZPDR performance')
    benchmark_parser.add_argument('--output-dir', '-o', 
                                help='Directory to save benchmark results (defaults to temporary directory)')
    benchmark_parser.add_argument('--no-plots', action='store_true',
                                help='Disable plot generation')
    
    # Benchmark subcommands
    benchmark_subparsers = benchmark_parser.add_subparsers(dest='benchmark_type', help='Type of benchmark to run')
    
    # File size benchmark
    size_parser = benchmark_subparsers.add_parser('filesize', 
                                                help='Benchmark performance with different file sizes')
    size_parser.add_argument('--sizes', nargs='+', type=int, 
                           help='List of file sizes to test in bytes (default: 1KB, 10KB, 100KB, 1MB, 10MB)')
    size_parser.add_argument('--repetitions', '-r', type=int, default=3,
                           help='Number of times to repeat each test')
    
    # Parallel workers benchmark
    parallel_parser = benchmark_subparsers.add_parser('parallel',
                                                    help='Benchmark performance with different numbers of workers')
    parallel_parser.add_argument('--file-size', type=int, default=10*1024*1024,
                               help='Size of test file in bytes (default: 10MB)')
    parallel_parser.add_argument('--workers', nargs='+', type=int,
                               help='List of worker counts to test (default: 1, 2, 4, 8, 16)')
    parallel_parser.add_argument('--repetitions', '-r', type=int, default=3,
                               help='Number of times to repeat each test')
    
    # Memory usage benchmark
    memory_parser = benchmark_subparsers.add_parser('memory',
                                                  help='Benchmark performance with different memory limits')
    memory_parser.add_argument('--sizes', nargs='+', type=int,
                             help='List of file sizes to test in bytes (default: 1MB, 10MB, 100MB)')
    memory_parser.add_argument('--limits', nargs='+', type=int,
                             help='List of memory limits to test in MB (default: 10, 50, 100, 500)')
    memory_parser.add_argument('--repetitions', '-r', type=int, default=3,
                             help='Number of times to repeat each test')
    
    return parser.parse_args(args)

def encode_file(args: argparse.Namespace) -> int:
    """
    Encode a file to zero-point coordinates.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Determine output path
        input_path = Path(args.input_file)
        output_path = args.output
        if not output_path:
            output_path = str(input_path) + '.zpdr'
        
        if args.verbose:
            print(f"Encoding file: {input_path}")
            print(f"Output manifest: {output_path}")
            if args.streaming:
                print(f"Using streaming processor with {args.chunk_size} byte chunks")
                print(f"Max workers: {args.max_workers or 'auto'}")
                print(f"Max memory: {args.max_memory}MB")
        
        # Create appropriate processor based on arguments
        if args.streaming:
            processor = StreamingZPDRProcessor(
                coherence_threshold=args.threshold,
                chunk_size=args.chunk_size,
                max_workers=args.max_workers,
                max_memory_mb=args.max_memory,
                auto_correct=args.auto_correct
            )
        else:
            processor = ZPDRProcessor(
                coherence_threshold=args.threshold,
                auto_correct=args.auto_correct
            )
        
        # Process file
        start_time = time.time()
        manifest = processor.process_file(input_path)
        processing_time = time.time() - start_time
        
        # Save manifest
        processor.save_manifest(manifest, output_path)
        
        if args.verbose:
            print(f"File encoded successfully in {processing_time:.2f} seconds.")
            print(f"Original file size: {manifest.file_size} bytes")
            print(f"Checksum: {manifest.checksum}")
            print(f"Global coherence: {manifest.trivector_digest.get_global_coherence()}")
        
        return 0
    
    except Exception as e:
        print(f"Error encoding file: {e}", file=sys.stderr)
        return 1

def decode_file(args: argparse.Namespace) -> int:
    """
    Decode a file from zero-point coordinates.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load manifest first (we need this regardless of processor type)
        manifest_path = Path(args.manifest)
        
        # Create temporary processor to load manifest
        temp_processor = ZPDRProcessor(coherence_threshold=args.threshold)
        manifest = temp_processor.load_manifest(manifest_path)
        
        # Determine output path
        output_path = args.output
        if not output_path:
            output_path = manifest.original_filename
        
        if args.verbose:
            print(f"Decoding manifest: {manifest_path}")
            print(f"Output file: {output_path}")
            if args.streaming:
                print(f"Using streaming processor with {args.chunk_size} byte chunks")
                print(f"Max workers: {args.max_workers or 'auto'}")
                print(f"Max memory: {args.max_memory}MB")
        
        # Create appropriate processor based on arguments
        if args.streaming:
            processor = StreamingZPDRProcessor(
                coherence_threshold=args.threshold,
                chunk_size=args.chunk_size,
                max_workers=args.max_workers,
                max_memory_mb=args.max_memory,
                auto_correct=args.auto_correct
            )
        else:
            processor = ZPDRProcessor(
                coherence_threshold=args.threshold,
                auto_correct=args.auto_correct
            )
        
        # Reconstruct file
        start_time = time.time()
        processor.reconstruct_file(manifest_path, output_path)
        processing_time = time.time() - start_time
        
        if args.verbose:
            print(f"File decoded successfully in {processing_time:.2f} seconds.")
            print(f"Output file size: {os.path.getsize(output_path)} bytes")
            print(f"Original checksum: {manifest.checksum}")
            
            # Verify reconstruction
            original_hash = manifest.checksum.split(':')[1] if manifest.checksum.startswith('sha256:') else None
            if original_hash:
                import hashlib
                with open(output_path, 'rb') as f:
                    data = f.read()
                actual_hash = hashlib.sha256(data).hexdigest()
                
                if original_hash == actual_hash:
                    print(f"Checksum verification: PASS")
                else:
                    print(f"Checksum verification: FAIL")
                    print(f"  Expected: {original_hash}")
                    print(f"  Actual:   {actual_hash}")
                    return 1
        
        return 0
    
    except Exception as e:
        print(f"Error decoding file: {e}", file=sys.stderr)
        return 1

def verify_manifest(args: argparse.Namespace) -> int:
    """
    Verify a manifest.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Create processor
        processor = ZPDRProcessor(coherence_threshold=args.threshold)
        
        # Load manifest
        manifest_path = Path(args.manifest)
        manifest = processor.load_manifest(manifest_path)
        
        if args.verbose:
            print(f"Verifying manifest: {manifest_path}")
        
        if args.comprehensive:
            # Perform comprehensive verification
            results = processor.verify_manifest(manifest)
            
            if results['overall_passed']:
                if args.verbose:
                    print("Comprehensive verification: PASS")
                    print("\nCoherence verification:")
                    print(f"  Global coherence: {results['coherence_verification']['calculated_coherence']}")
                    print(f"  Threshold: {results['coherence_verification']['threshold']}")
                    print("\nInvariants verification:")
                    print(f"  Maximum difference: {results['invariants_verification']['max_invariant_difference']}")
                    print(f"  Epsilon: {results['invariants_verification']['epsilon']}")
            else:
                print("Comprehensive verification: FAIL", file=sys.stderr)
                
                if not results['coherence_verification']['passed']:
                    print("\nCoherence verification: FAIL", file=sys.stderr)
                    print(f"  Global coherence: {results['coherence_verification']['calculated_coherence']}", file=sys.stderr)
                    print(f"  Threshold: {results['coherence_verification']['threshold']}", file=sys.stderr)
                    print(f"  Weakest link: {results['coherence_verification']['weakest_coherence_link']}", file=sys.stderr)
                    print(f"  Recommendation: {results['coherence_verification']['recommendation']}", file=sys.stderr)
                
                if not results['invariants_verification']['passed']:
                    print("\nInvariants verification: FAIL", file=sys.stderr)
                    print(f"  Maximum difference: {results['invariants_verification']['max_invariant_difference']}", file=sys.stderr)
                    print(f"  Epsilon: {results['invariants_verification']['epsilon']}", file=sys.stderr)
                
                return 1
        else:
            # Just verify coherence
            coherence_result = processor.verifier.verify_coherence(manifest)
            
            if coherence_result['passed']:
                if args.verbose:
                    print(f"Coherence verification: PASS")
                    print(f"  Global coherence: {coherence_result['calculated_coherence']}")
                    print(f"  Threshold: {coherence_result['threshold']}")
            else:
                print(f"Coherence verification: FAIL", file=sys.stderr)
                print(f"  Global coherence: {coherence_result['calculated_coherence']}", file=sys.stderr)
                print(f"  Threshold: {coherence_result['threshold']}", file=sys.stderr)
                print(f"  Weakest link: {coherence_result['weakest_coherence_link']}", file=sys.stderr)
                print(f"  Recommendation: {coherence_result['recommendation']}", file=sys.stderr)
                return 1
        
        return 0
    
    except Exception as e:
        print(f"Error verifying manifest: {e}", file=sys.stderr)
        return 1

def correct_manifest(args: argparse.Namespace) -> int:
    """
    Correct a manifest to improve coherence.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Create processor
        processor = ZPDRProcessor(coherence_threshold=args.threshold)
        
        # Load manifest
        manifest_path = Path(args.manifest)
        manifest = processor.load_manifest(manifest_path)
        
        # Determine output path
        output_path = args.output
        if not output_path:
            output_path = str(manifest_path) + '_corrected'
        
        if args.verbose:
            print(f"Correcting manifest: {manifest_path}")
            print(f"Output manifest: {output_path}")
            print(f"Initial coherence: {manifest.trivector_digest.get_global_coherence()}")
        
        # Apply correction
        corrected_manifest, correction_details = processor.correct_coordinates(manifest)
        
        # Save corrected manifest
        processor.save_manifest(corrected_manifest, output_path)
        
        if args.verbose:
            print("\nCorrection details:")
            print(f"  Iterations: {correction_details['iterations']}")
            print(f"  Initial coherence: {correction_details['initial_coherence']}")
            print(f"  Final coherence: {correction_details['final_coherence']}")
            print(f"  Improvement: {correction_details['improvement']}")
            
            if correction_details['improvement'] > 0:
                print("\nManifest successfully corrected.")
            else:
                print("\nNo improvement achieved. Manifest may already be optimal.")
        
        return 0
    
    except Exception as e:
        print(f"Error correcting manifest: {e}", file=sys.stderr)
        return 1

def display_info(args: argparse.Namespace) -> int:
    """
    Display information about a manifest.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Create processor
        processor = ZPDRProcessor(coherence_threshold=args.threshold)
        
        # Load manifest
        manifest_path = Path(args.manifest)
        manifest = processor.load_manifest(manifest_path)
        
        # Display information
        print(f"ZPDR Manifest Information:")
        print(f"  Manifest file: {manifest_path}")
        print(f"  Original filename: {manifest.original_filename}")
        print(f"  Original file size: {manifest.file_size} bytes")
        print(f"  Checksum: {manifest.checksum}")
        print(f"  Base fiber ID: {manifest.base_fiber_id}")
        print(f"  Structure ID: {manifest.structure_id}")
        print(f"  Coherence:")
        print(f"    Global coherence: {manifest.trivector_digest.get_global_coherence()}")
        print(f"    Coherence threshold: {manifest.coherence_threshold}")
        print(f"  Coordinates:")
        print(f"    Hyperbolic vector: {[float(v) for v in manifest.trivector_digest.hyperbolic]}")
        print(f"    Elliptical vector: {[float(v) for v in manifest.trivector_digest.elliptical]}")
        print(f"    Euclidean vector: {[float(v) for v in manifest.trivector_digest.euclidean]}")
        
        # Display additional metadata if present
        if manifest.additional_metadata:
            print(f"  Additional metadata:")
            for key, value in manifest.additional_metadata.items():
                print(f"    {key}: {value}")
        
        return 0
    
    except Exception as e:
        print(f"Error displaying manifest information: {e}", file=sys.stderr)
        return 1

def run_benchmark(args: argparse.Namespace) -> int:
    """
    Run benchmarks for ZPDR performance.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Initialize benchmarking tool
        benchmark = ZPDRBenchmark(
            output_dir=args.output_dir,
            generate_plots=not args.no_plots
        )
        
        if args.verbose:
            print(f"Running benchmark: {args.benchmark_type}")
            print(f"Output directory: {benchmark.output_dir}")
        
        # Run appropriate benchmark based on type
        if args.benchmark_type == 'filesize':
            # Convert size input if provided
            sizes = None
            if args.sizes:
                sizes = [int(size) for size in args.sizes]
            
            if args.verbose:
                size_text = "default sizes" if not sizes else f"sizes: {sizes}"
                print(f"File size scaling benchmark with {size_text}")
                print(f"Repetitions: {args.repetitions}")
            
            results = benchmark.benchmark_file_size_scaling(
                size_range=sizes,
                repetitions=args.repetitions
            )
            
            # Print summary
            print("\nBenchmark Results:")
            for size, avg in results['averages']['streaming'].items():
                size_text = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                print(f"Size: {size_text}")
                print(f"  Encode: {avg['encode']:.3f}s")
                print(f"  Decode: {avg['decode']:.3f}s")
            
            print("\nSpeedups (Streaming vs Standard):")
            for size, speedup in results['speedups']['encode'].items():
                size_text = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                print(f"Size: {size_text}")
                print(f"  Encode: {speedup:.2f}x")
                print(f"  Decode: {results['speedups']['decode'][size]:.2f}x")
            
        elif args.benchmark_type == 'parallel':
            # Convert worker input if provided
            workers = None
            if args.workers:
                workers = [int(w) for w in args.workers]
            
            file_size = args.file_size
            
            if args.verbose:
                workers_text = "default worker counts" if not workers else f"workers: {workers}"
                size_text = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f}MB"
                print(f"Parallel scaling benchmark with {workers_text} on {size_text} file")
                print(f"Repetitions: {args.repetitions}")
            
            results = benchmark.benchmark_parallel_scaling(
                file_size=file_size,
                worker_range=workers,
                repetitions=args.repetitions
            )
            
            # Print summary
            print("\nBenchmark Results:")
            for workers, avg in results['averages'].items():
                print(f"Workers: {workers}")
                print(f"  Encode: {avg['encode']:.3f}s")
                print(f"  Decode: {avg['decode']:.3f}s")
            
            print("\nSpeedups (vs Single Worker):")
            for workers, speedup in results['speedups']['encode'].items():
                print(f"Workers: {workers}")
                print(f"  Encode: {speedup:.2f}x")
                print(f"  Decode: {results['speedups']['decode'][workers]:.2f}x")
            
        elif args.benchmark_type == 'memory':
            # Convert inputs if provided
            sizes = None
            if args.sizes:
                sizes = [int(size) for size in args.sizes]
                
            limits = None
            if args.limits:
                limits = [int(limit) for limit in args.limits]
            
            if args.verbose:
                sizes_text = "default sizes" if not sizes else f"sizes: {sizes}"
                limits_text = "default limits" if not limits else f"limits: {limits}MB"
                print(f"Memory usage benchmark with {sizes_text} and {limits_text}")
                print(f"Repetitions: {args.repetitions}")
            
            results = benchmark.benchmark_memory_usage(
                size_range=sizes,
                memory_limits=limits,
                repetitions=args.repetitions
            )
            
            # Print summary
            print("\nBenchmark Results:")
            for size in results['size_range']:
                size_text = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                print(f"\nFile Size: {size_text}")
                
                for limit in results['memory_limits']:
                    avgs = results['averages'][size][limit]
                    if avgs['encode'] is not None and avgs['decode'] is not None:
                        print(f"  Memory Limit: {limit}MB")
                        print(f"    Encode: {avgs['encode']:.3f}s")
                        print(f"    Decode: {avgs['decode']:.3f}s")
                    else:
                        print(f"  Memory Limit: {limit}MB - Failed (insufficient memory)")
        
        else:
            print(f"Unknown benchmark type: {args.benchmark_type}. Use --help for more information.", file=sys.stderr)
            return 1
        
        print(f"\nBenchmark results saved to: {benchmark.output_dir}")
        
        return 0
    
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        return 1

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the ZPDR command-line interface.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == 'encode':
        return encode_file(parsed_args)
    elif parsed_args.command == 'decode':
        return decode_file(parsed_args)
    elif parsed_args.command == 'verify':
        return verify_manifest(parsed_args)
    elif parsed_args.command == 'correct':
        return correct_manifest(parsed_args)
    elif parsed_args.command == 'info':
        return display_info(parsed_args)
    elif parsed_args.command == 'benchmark':
        return run_benchmark(parsed_args)
    else:
        print("Please specify a command. Use --help for more information.", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())