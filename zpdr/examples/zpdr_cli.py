#!/usr/bin/env python3
"""
ZPDR Command Line Interface

This script provides a command-line interface to the ZPDR framework, enabling:
1. File encoding to ZPDR representation
2. ZPDR manifest decoding to original file
3. Verification of ZPDR manifests
4. Error correction for corrupted manifests
5. Information display about ZPDR manifests
6. Batch processing of multiple files

Usage examples:
  $ python zpdr_cli.py encode file.txt
  $ python zpdr_cli.py decode file.txt.zpdr
  $ python zpdr_cli.py verify manifest.zpdr
  $ python zpdr_cli.py correct corrupted.zpdr
  $ python zpdr_cli.py info manifest.zpdr
"""

import os
import sys
import time
import json
import argparse
import hashlib
from pathlib import Path

# Import ZPDR components
from zpdr.core.zpdr_processor import ZPDRProcessor, StreamingZPDRProcessor
from zpdr.core.zpa_manifest import ZPAManifest


class ZPDRCLI:
    """Command-line interface for ZPDR operations."""
    
    def __init__(self):
        """Initialize the CLI with default configuration."""
        # Default configuration
        self.config = {
            'coherence_threshold': 0.95,
            'verification_level': 'standard',
            'error_correction_enabled': True,
            'multithreaded': True,
            'chunk_size': 1024 * 1024,  # 1MB
            'manifest_format': 'full',
            'verbose': True,
            'output_dir': None
        }
        
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
        
        # Statistics tracking
        self.stats = {
            'processed_files': 0,
            'successful_files': 0,
            'total_bytes': 0,
            'total_time_ms': 0
        }
    
    def encode_file(self, input_path, output_path=None):
        """
        Encode a file to ZPDR representation.
        
        Args:
            input_path: Path to the file to encode
            output_path: Path to save the manifest (defaults to input_path + '.zpdr')
            
        Returns:
            Path to the created manifest file
        """
        # Normalize paths
        input_path = Path(input_path)
        if not output_path:
            output_path = str(input_path) + '.zpdr'
        else:
            output_path = Path(output_path)
            
            # Check if output is a directory
            if output_path.is_dir():
                output_path = output_path / (input_path.name + '.zpdr')
        
        # Check if input file exists
        if not input_path.exists():
            self.log(f"Error: Input file not found: {input_path}", error=True)
            return None
        
        # Start timing
        start_time = time.time()
        
        try:
            # Choose standard or streaming based on file size
            file_size = input_path.stat().st_size
            self.log(f"Encoding file: {input_path} ({file_size} bytes)")
            
            if file_size > 10 * 1024 * 1024:  # 10MB threshold
                # Use streaming processor for large files
                with open(input_path, 'rb') as f:
                    manifests = self.streaming_processor.encode_stream(f)
                
                # Add file metadata to the first manifest
                if manifests:
                    manifests[0].metadata.update({
                        'original_filename': input_path.name,
                        'file_size': file_size,
                        'encoding_time': time.time() - start_time
                    })
                
                # Save manifest data
                manifest_data = []
                for i, manifest in enumerate(manifests):
                    # Add chunk index to each manifest
                    manifest.metadata['chunk_index'] = i
                    manifest.metadata['total_chunks'] = len(manifests)
                    
                    # Serialize the manifest
                    serialized = manifest.serialize(
                        compact=(self.config['manifest_format'] == 'compact')
                    )
                    manifest_data.append(serialized)
                
                # For streaming, we save a list of manifests
                combined_manifest = "[\n" + ",\n".join(manifest_data) + "\n]"
                
                with open(output_path, 'w') as f:
                    f.write(combined_manifest)
                
                self.log(f"Saved streamed manifest ({len(manifests)} chunks) to {output_path}")
            else:
                # Use standard processor for small files
                with open(input_path, 'rb') as f:
                    data = f.read()
                
                # Encode the file
                manifest = self.processor.encode(data, description=f"ZPDR encoding of {input_path.name}")
                
                # Add file metadata
                manifest.metadata.update({
                    'original_filename': input_path.name,
                    'file_size': len(data),
                    'encoding_time': time.time() - start_time
                })
                
                # Serialize and save
                manifest_data = manifest.serialize(
                    compact=(self.config['manifest_format'] == 'compact')
                )
                
                with open(output_path, 'w') as f:
                    f.write(manifest_data)
                
                self.log(f"Saved manifest to {output_path}")
            
            # Update statistics
            self.stats['processed_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_bytes'] += file_size
            
            # Record total time
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            self.stats['total_time_ms'] += elapsed_ms
            
            self.log(f"Encoding completed in {elapsed_ms:.2f}ms")
            
            return output_path
            
        except Exception as e:
            self.log(f"Error encoding file: {e}", error=True)
            self.stats['processed_files'] += 1
            return None
    
    def decode_file(self, manifest_path, output_path=None):
        """
        Decode a ZPDR manifest back to the original file.
        
        Args:
            manifest_path: Path to the ZPDR manifest
            output_path: Path to save the decoded file (defaults to original filename)
            
        Returns:
            Path to the reconstructed file
        """
        # Normalize paths
        manifest_path = Path(manifest_path)
        
        # Check if manifest file exists
        if not manifest_path.exists():
            self.log(f"Error: Manifest file not found: {manifest_path}", error=True)
            return None
        
        # Start timing
        start_time = time.time()
        
        try:
            # Read the manifest file
            with open(manifest_path, 'r') as f:
                manifest_data = f.read()
            
            # Check if this is a streaming manifest
            is_streaming = manifest_data.strip().startswith('[') and manifest_data.strip().endswith(']')
            
            # Parse the manifest
            if is_streaming:
                self.log(f"Decoding streaming manifest: {manifest_path}")
                
                # Parse the list of manifests
                manifest_list = json.loads(manifest_data)
                
                # Convert list to manifest objects
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
                if not output_path and manifests and 'original_filename' in manifests[0].metadata:
                    output_path = manifests[0].metadata['original_filename']
            else:
                self.log(f"Decoding single manifest: {manifest_path}")
                
                # Parse single manifest
                manifest = ZPAManifest.deserialize(manifest_data)
                
                # Determine output path if not specified
                if not output_path and 'original_filename' in manifest.metadata:
                    output_path = manifest.metadata['original_filename']
            
            # If we still don't have an output path, use a default name
            if not output_path:
                output_path = "reconstructed_file"
            
            # If output_path is a directory, use the original filename
            output_path = Path(output_path)
            if output_path.is_dir():
                if is_streaming and manifests and 'original_filename' in manifests[0].metadata:
                    output_path = output_path / manifests[0].metadata['original_filename']
                elif not is_streaming and 'original_filename' in manifest.metadata:
                    output_path = output_path / manifest.metadata['original_filename']
                else:
                    output_path = output_path / "reconstructed_file"
            
            # Decode the file
            if is_streaming:
                # Use streaming processor
                file_data = self.streaming_processor.decode_stream(manifests)
                
                # Write to output file
                with open(output_path, 'wb') as f:
                    f.write(file_data)
                
                file_size = len(file_data)
            else:
                # Use standard processor
                decoded = self.processor.decode(manifest)
                
                # Handle different data types
                with open(output_path, 'wb') as f:
                    if isinstance(decoded, str):
                        f.write(decoded.encode('utf-8'))
                        file_size = len(decoded.encode('utf-8'))
                    elif isinstance(decoded, int):
                        # Convert integer to bytes
                        byte_length = (decoded.bit_length() + 7) // 8
                        bytes_data = decoded.to_bytes(max(1, byte_length), byteorder='big')
                        f.write(bytes_data)
                        file_size = len(bytes_data)
                    else:
                        f.write(decoded)
                        file_size = len(decoded)
            
            # Update statistics
            self.stats['processed_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_bytes'] += file_size
            
            # Record total time
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            self.stats['total_time_ms'] += elapsed_ms
            
            self.log(f"Decoded file saved to {output_path} ({file_size} bytes)")
            self.log(f"Decoding completed in {elapsed_ms:.2f}ms")
            
            return output_path
            
        except Exception as e:
            self.log(f"Error decoding manifest: {e}", error=True)
            self.stats['processed_files'] += 1
            return None
    
    def verify_manifest(self, manifest_path):
        """
        Verify the integrity of a ZPDR manifest.
        
        Args:
            manifest_path: Path to the ZPDR manifest
            
        Returns:
            Boolean indicating whether verification passed
        """
        # Normalize path
        manifest_path = Path(manifest_path)
        
        # Check if manifest file exists
        if not manifest_path.exists():
            self.log(f"Error: Manifest file not found: {manifest_path}", error=True)
            return False
        
        try:
            # Read the manifest file
            with open(manifest_path, 'r') as f:
                manifest_data = f.read()
            
            self.log(f"Verifying manifest: {manifest_path}")
            
            # Check if this is a streaming manifest
            if manifest_data.strip().startswith('[') and manifest_data.strip().endswith(']'):
                # Parse the list of manifests
                manifest_list = json.loads(manifest_data)
                
                # Verify each manifest
                all_valid = True
                for i, manifest_dict in enumerate(manifest_list):
                    # If it's already serialized, deserialize it
                    if isinstance(manifest_dict, str):
                        manifest = ZPAManifest.deserialize(manifest_dict)
                    else:
                        # Otherwise, use from_dict
                        manifest = ZPAManifest.from_dict(manifest_dict)
                    
                    # Verify the manifest
                    is_valid, results = self.processor.verify_manifest(manifest)
                    
                    # Report results
                    self.log(f"Chunk {i+1}/{len(manifest_list)}: " +
                             ("VALID" if is_valid else "INVALID"))
                    
                    # Print detailed results if verbosity is high
                    if self.config['verbose']:
                        self.log(f"  Global coherence: {results['coherence']['global']:.3f}")
                        if not is_valid:
                            self.log("  Verification failures:")
                            for key, value in results.items():
                                if isinstance(value, bool) and not value:
                                    self.log(f"    - {key}")
                    
                    all_valid = all_valid and is_valid
                
                # Final verdict
                self.log(f"Overall verification: " + 
                         ("PASSED" if all_valid else "FAILED"))
                
                return all_valid
                
            else:
                # Parse single manifest
                manifest = ZPAManifest.deserialize(manifest_data)
                
                # Verify the manifest
                is_valid, results = self.processor.verify_manifest(manifest)
                
                # Report results
                self.log(f"Verification: " + ("PASSED" if is_valid else "FAILED"))
                
                # Print detailed results
                if self.config['verbose']:
                    self.log(f"Global coherence: {results['coherence']['global']:.3f}")
                    self.log(f"Internal coherence:")
                    self.log(f"  H: {results['coherence']['H']:.3f}")
                    self.log(f"  E: {results['coherence']['E']:.3f}")
                    self.log(f"  U: {results['coherence']['U']:.3f}")
                    
                    self.log(f"Cross coherence:")
                    self.log(f"  HE: {results['coherence']['HE']:.3f}")
                    self.log(f"  EU: {results['coherence']['EU']:.3f}")
                    self.log(f"  HU: {results['coherence']['HU']:.3f}")
                    
                    # Print any verification failures
                    if not is_valid:
                        self.log("\nVerification failures:")
                        for key, value in results.items():
                            if isinstance(value, bool) and not value:
                                self.log(f"  - {key}")
                
                return is_valid
                
        except Exception as e:
            self.log(f"Error verifying manifest: {e}", error=True)
            return False
    
    def correct_manifest(self, manifest_path, output_path=None):
        """
        Apply error correction to a ZPDR manifest.
        
        Args:
            manifest_path: Path to the ZPDR manifest to correct
            output_path: Path to save the corrected manifest
            
        Returns:
            Path to the corrected manifest
        """
        # Normalize paths
        manifest_path = Path(manifest_path)
        if not output_path:
            output_path = str(manifest_path) + '.corrected'
        else:
            output_path = Path(output_path)
            
            # Check if output is a directory
            if output_path.is_dir():
                output_path = output_path / (manifest_path.name + '.corrected')
        
        # Check if manifest file exists
        if not manifest_path.exists():
            self.log(f"Error: Manifest file not found: {manifest_path}", error=True)
            return None
        
        try:
            # Read the manifest file
            with open(manifest_path, 'r') as f:
                manifest_data = f.read()
            
            self.log(f"Applying error correction to manifest: {manifest_path}")
            
            # Check if this is a streaming manifest
            if manifest_data.strip().startswith('[') and manifest_data.strip().endswith(']'):
                # Parse the list of manifests
                manifest_list = json.loads(manifest_data)
                
                # Correct each manifest
                corrected_manifests = []
                for i, manifest_dict in enumerate(manifest_list):
                    # If it's already serialized, deserialize it
                    if isinstance(manifest_dict, str):
                        manifest = ZPAManifest.deserialize(manifest_dict)
                    else:
                        # Otherwise, use from_dict
                        manifest = ZPAManifest.from_dict(manifest_dict)
                    
                    # Verify the manifest before correction
                    is_valid_before, results_before = self.processor.verify_manifest(manifest)
                    coherence_before = results_before['coherence']['global']
                    
                    # Apply error correction
                    corrected_manifest = self.processor.correct_errors(manifest)
                    
                    # Verify the manifest after correction
                    is_valid_after, results_after = self.processor.verify_manifest(corrected_manifest)
                    coherence_after = results_after['coherence']['global']
                    
                    # Report improvement
                    self.log(f"Chunk {i+1}/{len(manifest_list)}: " +
                             f"Coherence {coherence_before:.3f} → {coherence_after:.3f}")
                    
                    # Add to corrected manifests
                    corrected_manifests.append(corrected_manifest)
                
                # Serialize corrected manifests
                manifest_data = []
                for manifest in corrected_manifests:
                    serialized = manifest.serialize(
                        compact=(self.config['manifest_format'] == 'compact')
                    )
                    manifest_data.append(serialized)
                
                # Save as a list
                combined_manifest = "[\n" + ",\n".join(manifest_data) + "\n]"
                
                with open(output_path, 'w') as f:
                    f.write(combined_manifest)
                
                self.log(f"Corrected streaming manifest saved to {output_path}")
                
            else:
                # Parse single manifest
                manifest = ZPAManifest.deserialize(manifest_data)
                
                # Verify the manifest before correction
                is_valid_before, results_before = self.processor.verify_manifest(manifest)
                coherence_before = results_before['coherence']['global']
                
                # Apply error correction
                corrected_manifest = self.processor.correct_errors(manifest)
                
                # Verify the manifest after correction
                is_valid_after, results_after = self.processor.verify_manifest(corrected_manifest)
                coherence_after = results_after['coherence']['global']
                
                # Report improvement
                self.log(f"Correction applied: Coherence {coherence_before:.3f} → {coherence_after:.3f}")
                self.log(f"Validation: {is_valid_before} → {is_valid_after}")
                
                # Serialize and save
                corrected_data = corrected_manifest.serialize(
                    compact=(self.config['manifest_format'] == 'compact')
                )
                
                with open(output_path, 'w') as f:
                    f.write(corrected_data)
                
                self.log(f"Corrected manifest saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.log(f"Error correcting manifest: {e}", error=True)
            return None
    
    def show_manifest_info(self, manifest_path):
        """
        Display information about a ZPDR manifest.
        
        Args:
            manifest_path: Path to the ZPDR manifest
            
        Returns:
            Dictionary containing manifest information
        """
        # Normalize path
        manifest_path = Path(manifest_path)
        
        # Check if manifest file exists
        if not manifest_path.exists():
            self.log(f"Error: Manifest file not found: {manifest_path}", error=True)
            return None
        
        try:
            # Read the manifest file
            with open(manifest_path, 'r') as f:
                manifest_data = f.read()
            
            self.log(f"\n=== ZPDR Manifest Information ===")
            self.log(f"Manifest file: {manifest_path}")
            
            # Check file size
            file_size = manifest_path.stat().st_size
            self.log(f"File size: {file_size} bytes")
            
            # Check if this is a streaming manifest
            if manifest_data.strip().startswith('[') and manifest_data.strip().endswith(']'):
                # Parse the list of manifests
                manifest_list = json.loads(manifest_data)
                num_chunks = len(manifest_list)
                
                self.log(f"Type: Streaming manifest ({num_chunks} chunks)")
                
                # Parse first manifest to get metadata
                if num_chunks > 0:
                    # If it's already serialized, deserialize it
                    if isinstance(manifest_list[0], str):
                        first_manifest = ZPAManifest.deserialize(manifest_list[0])
                    else:
                        # Otherwise, use from_dict
                        first_manifest = ZPAManifest.from_dict(manifest_list[0])
                    
                    # Display manifest metadata
                    self.log("\nMetadata:")
                    for key, value in first_manifest.metadata.items():
                        self.log(f"  {key}: {value}")
                    
                    # Show coherence metrics
                    self.log("\nCoherence metrics (first chunk):")
                    self.log(f"  Global coherence: {first_manifest.global_coherence:.3f}")
                    self.log(f"  H coherence: {first_manifest.H_coherence:.3f}")
                    self.log(f"  E coherence: {first_manifest.E_coherence:.3f}")
                    self.log(f"  U coherence: {first_manifest.U_coherence:.3f}")
                
                # Verify each manifest and show summary
                if self.config['verbose']:
                    valid_chunks = 0
                    for i, manifest_dict in enumerate(manifest_list):
                        # If it's already serialized, deserialize it
                        if isinstance(manifest_dict, str):
                            manifest = ZPAManifest.deserialize(manifest_dict)
                        else:
                            # Otherwise, use from_dict
                            manifest = ZPAManifest.from_dict(manifest_dict)
                        
                        # Verify the manifest
                        is_valid, _ = self.processor.verify_manifest(manifest)
                        if is_valid:
                            valid_chunks += 1
                    
                    self.log(f"\nVerification summary: {valid_chunks}/{num_chunks} chunks valid")
                
            else:
                # Parse single manifest
                manifest = ZPAManifest.deserialize(manifest_data)
                
                self.log(f"Type: Single manifest")
                self.log(f"Version: {manifest.version}")
                
                # Display manifest metadata
                self.log("\nMetadata:")
                for key, value in manifest.metadata.items():
                    self.log(f"  {key}: {value}")
                
                # Show vector dimensions
                self.log("\nVector dimensions:")
                self.log(f"  Hyperbolic (H): {len(manifest.H)}")
                self.log(f"  Elliptical (E): {len(manifest.E)}")
                self.log(f"  Euclidean (U): {len(manifest.U)}")
                
                # Show coherence metrics
                self.log("\nCoherence metrics:")
                self.log(f"  Global coherence: {manifest.global_coherence:.3f}")
                self.log(f"  H coherence: {manifest.H_coherence:.3f}")
                self.log(f"  E coherence: {manifest.E_coherence:.3f}")
                self.log(f"  U coherence: {manifest.U_coherence:.3f}")
                self.log(f"  HE coherence: {manifest.HE_coherence:.3f}")
                self.log(f"  EU coherence: {manifest.EU_coherence:.3f}")
                self.log(f"  HU coherence: {manifest.HU_coherence:.3f}")
                
                # Verify the manifest
                is_valid, _ = self.processor.verify_manifest(manifest)
                self.log(f"\nVerification: {'PASSED' if is_valid else 'FAILED'}")
            
            return {
                'file_path': str(manifest_path),
                'file_size': file_size
            }
            
        except Exception as e:
            self.log(f"Error reading manifest: {e}", error=True)
            return None
    
    def batch_process(self, mode, files, output_dir=None):
        """
        Process multiple files in batch mode.
        
        Args:
            mode: Operation mode ('encode', 'decode', 'verify', 'correct', 'info')
            files: List of file paths to process
            output_dir: Directory to save output files
            
        Returns:
            List of processed file paths
        """
        # Set output directory
        if output_dir:
            self.config['output_dir'] = Path(output_dir)
            
            # Create directory if needed
            if not self.config['output_dir'].exists():
                self.config['output_dir'].mkdir(parents=True)
        
        # Track results
        results = []
        total_files = len(files)
        
        self.log(f"Batch processing {total_files} files in {mode} mode")
        
        # Process each file
        for i, file_path in enumerate(files):
            self.log(f"\nProcessing file {i+1}/{total_files}: {file_path}")
            
            # Determine output path if needed
            output_path = None
            if self.config['output_dir'] and mode in ('encode', 'decode', 'correct'):
                file_name = Path(file_path).name
                
                if mode == 'encode':
                    output_path = self.config['output_dir'] / (file_name + '.zpdr')
                elif mode == 'correct':
                    output_path = self.config['output_dir'] / (file_name + '.corrected')
                else:  # decode
                    # For decode, we'll use the original filename from the manifest
                    output_path = self.config['output_dir']
            
            # Process based on mode
            if mode == 'encode':
                result = self.encode_file(file_path, output_path)
            elif mode == 'decode':
                result = self.decode_file(file_path, output_path)
            elif mode == 'verify':
                result = self.verify_manifest(file_path)
            elif mode == 'correct':
                result = self.correct_manifest(file_path, output_path)
            elif mode == 'info':
                result = self.show_manifest_info(file_path)
            else:
                self.log(f"Unsupported mode: {mode}", error=True)
                result = None
            
            results.append(result)
        
        # Show batch summary
        if mode in ('encode', 'decode', 'correct'):
            success_count = sum(1 for r in results if r is not None)
            success_rate = success_count / total_files if total_files > 0 else 0
            
            self.log(f"\nBatch Summary:")
            self.log(f"  Files processed: {total_files}")
            self.log(f"  Successful: {success_count} ({success_rate*100:.1f}%)")
            
            if self.stats['total_time_ms'] > 0:
                avg_time = self.stats['total_time_ms'] / self.stats['processed_files']
                self.log(f"  Average processing time: {avg_time:.2f}ms per file")
            
            if self.stats['total_bytes'] > 0:
                total_mb = self.stats['total_bytes'] / (1024 * 1024)
                self.log(f"  Total data processed: {total_mb:.2f}MB")
                
                if self.stats['total_time_ms'] > 0:
                    throughput = total_mb / (self.stats['total_time_ms'] / 1000)
                    self.log(f"  Throughput: {throughput:.2f}MB/s")
        
        return results
    
    def log(self, message, error=False):
        """
        Log a message to the console.
        
        Args:
            message: Message to log
            error: Whether this is an error message
        """
        if error:
            print(message, file=sys.stderr)
        elif self.config['verbose']:
            print(message)


def main():
    """Main entry point for the CLI."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='ZPDR Command Line Interface',
        epilog='For more information, visit https://github.com/yourusername/zpdr'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode a file to ZPDR representation')
    encode_parser.add_argument('file', nargs='+', help='File path(s) to encode')
    encode_parser.add_argument('-o', '--output', help='Output file or directory')
    encode_parser.add_argument('-c', '--compact', action='store_true', help='Use compact manifest format')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode a ZPDR manifest to the original file')
    decode_parser.add_argument('file', nargs='+', help='Manifest file path(s) to decode')
    decode_parser.add_argument('-o', '--output', help='Output file or directory')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify the integrity of a ZPDR manifest')
    verify_parser.add_argument('file', nargs='+', help='Manifest file path(s) to verify')
    
    # Correct command
    correct_parser = subparsers.add_parser('correct', help='Apply error correction to a ZPDR manifest')
    correct_parser.add_argument('file', nargs='+', help='Manifest file path(s) to correct')
    correct_parser.add_argument('-o', '--output', help='Output file or directory')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display information about a ZPDR manifest')
    info_parser.add_argument('file', nargs='+', help='Manifest file path(s) to examine')
    
    # Global options
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress all output except errors')
    parser.add_argument('-t', '--threads', type=int, help='Number of threads to use for processing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize the CLI
    cli = ZPDRCLI()
    
    # Update configuration based on arguments
    cli.config['verbose'] = args.verbose and not args.quiet
    
    if args.threads:
        cli.config['multithreaded'] = True
        cli.processor.config['max_workers'] = args.threads
    
    if getattr(args, 'compact', False):
        cli.config['manifest_format'] = 'compact'
    
    # Determine if we're in batch mode (multiple files)
    is_batch = len(args.file) > 1
    output_dir = None
    
    if is_batch and args.output:
        # In batch mode with output specified, treat as directory
        output_dir = args.output
    
    # Execute command
    try:
        if is_batch:
            # Batch process multiple files
            results = cli.batch_process(args.command, args.file, output_dir)
            
            # Determine exit code based on results
            if args.command in ('encode', 'decode', 'correct'):
                success_count = sum(1 for r in results if r is not None)
                return 0 if success_count > 0 else 1
            elif args.command == 'verify':
                return 0 if all(results) else 1
            else:
                return 0
        else:
            # Process single file
            if args.command == 'encode':
                result = cli.encode_file(args.file[0], args.output)
                return 0 if result else 1
            
            elif args.command == 'decode':
                result = cli.decode_file(args.file[0], args.output)
                return 0 if result else 1
            
            elif args.command == 'verify':
                result = cli.verify_manifest(args.file[0])
                return 0 if result else 1
            
            elif args.command == 'correct':
                result = cli.correct_manifest(args.file[0], args.output)
                return 0 if result else 1
            
            elif args.command == 'info':
                result = cli.show_manifest_info(args.file[0])
                return 0 if result else 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130  # Standard Unix exit code for SIGINT
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())