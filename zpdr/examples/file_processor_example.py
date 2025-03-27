"""
File Processing Example using ZPDR

This example demonstrates a complete file encoding and decoding workflow using
Zero-Point Data Resolution (ZPDR). It showcases a practical, real-world application
of the ZPDR framework for processing arbitrary files using its robust mathematical
representation system.

Key Features:

1. Complete File Processing Pipeline:
   - Reading files into memory or processing in streaming mode for large files
   - Encoding file content using the ZPDR trilateral representation system
   - Storing the ZPA manifests to disk in full or compact formats
   - Loading and parsing ZPA manifests from disk
   - Reconstructing the original files with byte-perfect accuracy
   - Verifying reconstruction integrity through hash validation

2. Performance Optimization:
   - Automatic streaming/chunking for large files (>10MB)
   - Support for multi-threaded encoding/decoding
   - Performance metrics tracking and reporting
   - Configurable chunk sizes for streaming processing

3. Error Handling and Verification:
   - Built-in error correction capabilities
   - Verification of reconstruction integrity
   - Configurable verification levels
   - Detailed logging and metrics reporting

4. Mathematical Foundation:
   - Implementation of the Prime Framework's mathematical principles for data representation
   - Representation of binary data in trilateral form across three geometric spaces
   - Coherence-based validation ensuring data integrity
   - Base-agnostic encoding decoupled from specific numerical bases

Usage:
    python -m zpdr.examples.file_processor_example encode <input_file> [-o <output_file>]
    python -m zpdr.examples.file_processor_example decode <zpdr_manifest> [-o <output_file>]

Additional options:
    -c, --compact: Use compact manifest format (smaller but less human-readable)
    -s, --streaming: Force streaming mode even for small files
    -q, --quiet: Suppress verbose output

This example demonstrates how the ZPDR framework can be applied to real-world data
processing tasks, providing a robust, mathematically sound approach to data representation
and transformation that maintains integrity across encoding and decoding operations.
"""

import os
import time
import argparse
import hashlib
from pathlib import Path
import numpy as np

# Import ZPDR core components
from zpdr.core.zpdr_processor import ZPDRProcessor, StreamingZPDRProcessor
from zpdr.core.zpa_manifest import ZPAManifest


class ZPDRFileProcessor:
    """
    File processor that uses ZPDR for encoding and decoding files.
    
    This class encapsulates the functionality needed to process files through the ZPDR
    framework, providing a complete pipeline for encoding, storing, retrieving, and
    reconstructing data. It implements the practical application of ZPDR's mathematical
    principles for real-world file processing.
    
    Key capabilities:
    
    1. File Encoding:
       - Reading files into memory or processing in chunks (streaming mode)
       - Converting binary data to trilateral ZPDR representation (H, E, U vectors)
       - Generating ZPA manifests with file metadata and coherence information
       - Saving manifests to disk in either full or compact format
    
    2. File Decoding:
       - Loading and parsing ZPA manifests from disk
       - Validating manifest coherence and integrity
       - Reconstructing the original binary data through reverse transformation
       - Verifying reconstruction correctness through hash validation
    
    3. Performance Management:
       - Automatic selection between standard and streaming processing based on file size
       - Configurable chunk size for efficient memory usage with large files
       - Multi-threaded processing for improved performance
       - Comprehensive metrics tracking for performance analysis
    
    4. Error Handling:
       - Built-in error correction for corrupted manifests
       - Configurable verification levels for different security requirements
       - Detailed logging and error reporting
    
    The processor implements the mathematical principles of the Prime Framework,
    using the trilateral representation system (hyperbolic, elliptical, and Euclidean
    vectors) to encode and decode data in a robust, base-agnostic manner. This provides
    inherent error detection and correction capabilities while maintaining mathematical
    coherence throughout the processing pipeline.
    """
    
    def __init__(self, config=None):
        """
        Initialize the file processor with configuration options.
        
        Args:
            config: Optional configuration dictionary overriding defaults
        """
        # Set up configuration with defaults
        self.config = {
            'coherence_threshold': 0.95,
            'verification_level': 'standard',  # 'minimal', 'standard', 'strict'
            'error_correction_enabled': True,
            'multithreaded': True,
            'chunk_size': 1024 * 1024,  # 1MB chunks for streaming
            'manifest_format': 'full',  # 'full', 'compact'
            'verbose': True
        }
        
        # Update with provided config if any
        if config:
            self.config.update(config)
        
        # Initialize processors
        self.processor = ZPDRProcessor({
            'coherence_threshold': self.config['coherence_threshold'],
            'verification_level': self.config['verification_level'],
            'error_correction_enabled': self.config['error_correction_enabled'],
            'multithreaded': self.config['multithreaded']
        })
        
        self.streaming_processor = StreamingZPDRProcessor({
            'coherence_threshold': self.config['coherence_threshold'],
            'verification_level': self.config['verification_level'],
            'error_correction_enabled': self.config['error_correction_enabled'],
            'chunk_size': self.config['chunk_size']
        })
        
        # Initialize metrics tracking
        self.metrics = {
            'processed_files': 0,
            'total_bytes': 0,
            'encoding_time_ms': 0,
            'decoding_time_ms': 0,
            'manifest_size_bytes': 0,
            'compression_ratio': 0,
            'verification_failures': 0
        }
    
    def encode_file(self, input_path, output_path=None):
        """
        Encode a file to ZPDR representation and save the manifest.
        
        Args:
            input_path: Path to the file to encode
            output_path: Path to save the manifest (defaults to input_path + '.zpdr')
            
        Returns:
            Path to the saved manifest file
        """
        # Normalize paths
        input_path = Path(input_path)
        if not output_path:
            output_path = str(input_path) + '.zpdr'
        output_path = Path(output_path)
        
        # Check if input file exists
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Determine file size
        file_size = input_path.stat().st_size
        self.log(f"Processing file: {input_path} ({file_size} bytes)")
        
        # Choose processing method based on file size
        if file_size > 10 * 1024 * 1024:  # 10MB threshold for streaming
            return self._encode_file_streaming(input_path, output_path)
        else:
            return self._encode_file_standard(input_path, output_path)
    
    def _encode_file_standard(self, input_path, output_path):
        """
        Encode a file using the standard processor (full in-memory processing).
        
        Args:
            input_path: Path to the file to encode
            output_path: Path to save the manifest
            
        Returns:
            Path to the saved manifest file
        """
        # Read the entire file into memory
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        self.log(f"Read {len(file_data)} bytes into memory")
        
        # Calculate original file hash for verification
        original_hash = hashlib.sha256(file_data).hexdigest()
        
        # Track metrics
        self.metrics['total_bytes'] += len(file_data)
        self.metrics['processed_files'] += 1
        
        # Start encoding timer
        start_time = time.time()
        
        # Encode the file using the ZPDR processor
        manifest = self.processor.encode(file_data, description=f"ZPDR encoding of {input_path.name}")
        
        # Calculate encoding time
        encoding_time = time.time() - start_time
        self.metrics['encoding_time_ms'] += encoding_time * 1000
        
        # Add file metadata to the manifest
        manifest.metadata.update({
            'original_filename': input_path.name,
            'file_size': len(file_data),
            'file_hash': f"sha256:{original_hash}",
            'encoding_time_ms': encoding_time * 1000
        })
        
        # Serialize the manifest
        compact = self.config['manifest_format'] == 'compact'
        manifest_data = manifest.serialize(compact=compact)
        
        # Save the manifest to the output file
        with open(output_path, 'w') as f:
            f.write(manifest_data)
        
        # Update metrics
        manifest_size = len(manifest_data)
        self.metrics['manifest_size_bytes'] += manifest_size
        compression_ratio = len(file_data) / manifest_size if manifest_size > 0 else 0
        self.metrics['compression_ratio'] = (self.metrics['compression_ratio'] * 
                                            (self.metrics['processed_files'] - 1) + 
                                            compression_ratio) / self.metrics['processed_files']
        
        self.log(f"Encoded file saved to {output_path} ({manifest_size} bytes)")
        self.log(f"Encoding time: {encoding_time*1000:.2f}ms")
        self.log(f"Compression ratio: {compression_ratio:.2f}x")
        
        return output_path
    
    def _encode_file_streaming(self, input_path, output_path):
        """
        Encode a file using the streaming processor (chunk-by-chunk processing).
        
        Args:
            input_path: Path to the file to encode
            output_path: Path to save the manifest
            
        Returns:
            Path to the saved manifest file
        """
        # Open the file for streaming
        with open(input_path, 'rb') as f:
            file_size = input_path.stat().st_size
            
            # Track metrics
            self.metrics['total_bytes'] += file_size
            self.metrics['processed_files'] += 1
            
            # Start encoding timer
            start_time = time.time()
            
            # Encode the file using the streaming ZPDR processor
            manifests = self.streaming_processor.encode_stream(f, self.config['chunk_size'])
            
            # Calculate encoding time
            encoding_time = time.time() - start_time
            self.metrics['encoding_time_ms'] += encoding_time * 1000
        
        # Add file metadata to the first manifest
        if manifests:
            manifests[0].metadata.update({
                'original_filename': input_path.name,
                'file_size': file_size,
                'chunks': len(manifests),
                'encoding_time_ms': encoding_time * 1000
            })
        
        # Save manifest metadata
        manifest_data = []
        for i, manifest in enumerate(manifests):
            # Add chunk index to each manifest
            manifest.metadata['chunk_index'] = i
            manifest.metadata['total_chunks'] = len(manifests)
            
            # Serialize the manifest
            compact = self.config['manifest_format'] == 'compact'
            serialized = manifest.serialize(compact=compact)
            manifest_data.append(serialized)
        
        # For streaming, we save a list of manifests
        combined_manifest = "[\n" + ",\n".join(manifest_data) + "\n]"
        
        # Save the combined manifest to the output file
        with open(output_path, 'w') as f:
            f.write(combined_manifest)
        
        # Update metrics
        manifest_size = len(combined_manifest)
        self.metrics['manifest_size_bytes'] += manifest_size
        compression_ratio = file_size / manifest_size if manifest_size > 0 else 0
        self.metrics['compression_ratio'] = (self.metrics['compression_ratio'] * 
                                            (self.metrics['processed_files'] - 1) + 
                                            compression_ratio) / self.metrics['processed_files']
        
        self.log(f"Encoded file saved to {output_path} ({manifest_size} bytes)")
        self.log(f"Encoding time: {encoding_time*1000:.2f}ms for {len(manifests)} chunks")
        self.log(f"Compression ratio: {compression_ratio:.2f}x")
        
        return output_path
    
    def decode_file(self, manifest_path, output_path=None):
        """
        Decode a ZPDR manifest back to the original file.
        
        Args:
            manifest_path: Path to the ZPDR manifest
            output_path: Path to save the reconstructed file (defaults to original filename)
            
        Returns:
            Path to the reconstructed file
        """
        # Normalize paths
        manifest_path = Path(manifest_path)
        
        # Check if manifest file exists
        if not manifest_path.exists() or not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        # Read the manifest file
        with open(manifest_path, 'r') as f:
            manifest_data = f.read()
        
        self.log(f"Read manifest from {manifest_path} ({len(manifest_data)} bytes)")
        
        # Check if this is a streaming manifest (list of manifests)
        if manifest_data.strip().startswith('[') and manifest_data.strip().endswith(']'):
            return self._decode_file_streaming(manifest_data, output_path)
        else:
            return self._decode_file_standard(manifest_data, output_path)
    
    def _decode_file_standard(self, manifest_data, output_path=None):
        """
        Decode a standard (single) manifest back to the original file.
        
        Args:
            manifest_data: Serialized manifest data
            output_path: Path to save the reconstructed file
            
        Returns:
            Path to the reconstructed file
        """
        # Start decoding timer
        start_time = time.time()
        
        # Parse the manifest
        manifest = ZPAManifest.deserialize(manifest_data)
        
        # Determine output path if not specified
        if not output_path:
            if 'original_filename' in manifest.metadata:
                output_path = manifest.metadata['original_filename']
            else:
                # Generate a default name based on the manifest file
                output_path = "reconstructed_file"
        
        # Decode the file using the ZPDR processor
        file_data = self.processor.decode(manifest)
        
        # Calculate decoding time
        decoding_time = time.time() - start_time
        self.metrics['decoding_time_ms'] += decoding_time * 1000
        
        # Save the reconstructed file
        with open(output_path, 'wb') as f:
            if isinstance(file_data, str):
                f.write(file_data.encode('utf-8'))
            elif isinstance(file_data, int):
                # Convert integer to bytes
                byte_length = (file_data.bit_length() + 7) // 8
                f.write(file_data.to_bytes(max(1, byte_length), byteorder='big'))
            else:
                f.write(file_data)
        
        # Verify the reconstruction
        if 'file_hash' in manifest.metadata and manifest.metadata['file_hash'].startswith('sha256:'):
            original_hash = manifest.metadata['file_hash'][7:]  # Remove 'sha256:' prefix
            reconstructed_hash = hashlib.sha256(file_data).hexdigest()
            
            if original_hash != reconstructed_hash:
                self.log(f"Warning: Reconstructed file hash does not match original")
                self.metrics['verification_failures'] += 1
        
        self.log(f"Decoded file saved to {output_path} ({len(file_data)} bytes)")
        self.log(f"Decoding time: {decoding_time*1000:.2f}ms")
        
        return output_path
    
    def _decode_file_streaming(self, manifest_data, output_path=None):
        """
        Decode a streaming manifest (list of manifests) back to the original file.
        
        Args:
            manifest_data: Serialized manifest data (as a JSON array)
            output_path: Path to save the reconstructed file
            
        Returns:
            Path to the reconstructed file
        """
        import json
        
        # Parse the list of manifests
        manifest_list = json.loads(manifest_data)
        
        # Convert list to individual manifest objects
        manifests = []
        for manifest_dict in manifest_list:
            # If it's already serialized, deserialize it
            if isinstance(manifest_dict, str):
                manifest = ZPAManifest.deserialize(manifest_dict)
            else:
                # Otherwise, use from_dict
                manifest = ZPAManifest.from_dict(manifest_dict)
            manifests.append(manifest)
        
        # Determine output path if not specified
        if not output_path and manifests:
            if 'original_filename' in manifests[0].metadata:
                output_path = manifests[0].metadata['original_filename']
            else:
                # Generate a default name
                output_path = "reconstructed_file"
        
        # Start decoding timer
        start_time = time.time()
        
        # Decode the file using the streaming ZPDR processor
        file_data = self.streaming_processor.decode_stream(manifests)
        
        # Calculate decoding time
        decoding_time = time.time() - start_time
        self.metrics['decoding_time_ms'] += decoding_time * 1000
        
        # Save the reconstructed file
        with open(output_path, 'wb') as f:
            f.write(file_data)
        
        self.log(f"Decoded file saved to {output_path} ({len(file_data)} bytes)")
        self.log(f"Decoding time: {decoding_time*1000:.2f}ms for {len(manifests)} chunks")
        
        return output_path
    
    def get_metrics(self):
        """
        Get the performance metrics from the processor.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Get processor-specific metrics
        processor_metrics = self.processor.get_performance_metrics()
        
        # Combine with our metrics
        combined_metrics = {**self.metrics, **processor_metrics}
        
        # Calculate averages
        if self.metrics['processed_files'] > 0:
            combined_metrics['avg_encoding_time_ms'] = (
                self.metrics['encoding_time_ms'] / self.metrics['processed_files']
            )
            
            combined_metrics['avg_processing_speed_mbps'] = (
                (self.metrics['total_bytes'] / 1024 / 1024) / 
                (self.metrics['encoding_time_ms'] / 1000)
            ) if self.metrics['encoding_time_ms'] > 0 else 0
        
        return combined_metrics
    
    def log(self, message):
        """Output a log message if verbose mode is enabled."""
        if self.config['verbose']:
            print(message)


def main():
    """
    Main function for the file processor example.
    
    This function implements the command-line interface for the ZPDR file processor,
    handling arguments, setting up the processing environment, and executing the
    requested encoding or decoding operations.
    
    The function provides a user-friendly interface to the ZPDR framework's file
    processing capabilities, with support for:
    
    1. Command-line Arguments:
       - Operation mode selection (encode/decode)
       - Input/output file path specification
       - Format configuration (compact/full)
       - Processing mode selection (standard/streaming)
       - Verbosity control
       
    2. Error Handling:
       - Input file validation
       - Exception catching and reporting
       - Proper exit code management
       
    3. Performance Reporting:
       - Encoding/decoding time measurements
       - Compression ratio calculations
       - Processing speed metrics
       - Verification status reporting
    
    This demonstrates how the ZPDR framework can be integrated into practical
    applications with a user-friendly interface, making its mathematical capabilities
    accessible for real-world data processing tasks.
    
    Returns:
        0 on success, 1 on error (suitable for exit code usage)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ZPDR File Processor Example')
    parser.add_argument('mode', choices=['encode', 'decode'], help='Operation mode')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-c', '--compact', action='store_true', help='Use compact manifest format')
    parser.add_argument('-s', '--streaming', action='store_true', help='Force streaming mode')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Set up configuration based on arguments
    config = {
        'verbose': not args.quiet,
        'manifest_format': 'compact' if args.compact else 'full',
        'chunk_size': 1024 * 1024  # 1MB chunks
    }
    
    # Initialize the processor
    processor = ZPDRFileProcessor(config)
    
    try:
        # Run the requested operation
        if args.mode == 'encode':
            output_path = processor.encode_file(args.input, args.output)
            
            # Print final metrics
            if not args.quiet:
                metrics = processor.get_metrics()
                print("\nPerformance Metrics:")
                print(f"- Encoding time: {metrics['encoding_time_ms']:.2f}ms")
                print(f"- Compression ratio: {metrics['compression_ratio']:.2f}x")
                print(f"- Processing speed: {metrics['avg_processing_speed_mbps']:.2f} MB/s")
        
        elif args.mode == 'decode':
            output_path = processor.decode_file(args.input, args.output)
            
            # Print final metrics
            if not args.quiet:
                metrics = processor.get_metrics()
                print("\nPerformance Metrics:")
                print(f"- Decoding time: {metrics['decoding_time_ms']:.2f}ms")
                if metrics['verification_failures'] > 0:
                    print(f"- Warning: {metrics['verification_failures']} verification failures")
        
        # Print success message
        if not args.quiet:
            print(f"\nSuccess! Output saved to: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())