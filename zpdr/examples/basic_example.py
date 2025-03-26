#!/usr/bin/env python3
"""
Basic example of using the ZPDR system.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the zpdr package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from zpdr.core.zpdr_processor import ZPDRProcessor
from zpdr.core.manifest import Manifest

def run_example():
    """Run a basic example of ZPDR encoding and decoding."""
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test file
        input_file = temp_path / "input.txt"
        manifest_file = temp_path / "input.txt.zpdr"
        output_file = temp_path / "output.txt"
        
        # Write some data to the input file
        test_data = b"This is a test of the Zero-Point Data Reconstruction system."
        with open(input_file, 'wb') as f:
            f.write(test_data)
        
        print(f"Created test file: {input_file}")
        print(f"Original data: {test_data.decode()}")
        print(f"Original size: {len(test_data)} bytes")
        print()
        
        # Create a ZPDR processor
        processor = ZPDRProcessor()
        
        # Process the input file
        print("Processing file to extract zero-point coordinates...")
        manifest = processor.process_file(input_file)
        
        # Save the manifest
        processor.save_manifest(manifest, manifest_file)
        print(f"Saved manifest to: {manifest_file}")
        
        # Display manifest information
        print("\nManifest information:")
        print(f"  File: {manifest.original_filename}")
        print(f"  Size: {manifest.file_size} bytes")
        print(f"  Checksum: {manifest.checksum}")
        print(f"  Coherence: {manifest.trivector_digest.get_global_coherence()}")
        
        # Display coordinates
        print("\nZero-point coordinates:")
        print("  Hyperbolic vector:")
        for i, v in enumerate(manifest.trivector_digest.hyperbolic):
            print(f"    H{i+1}: {float(v):.6f}")
        
        print("  Elliptical vector:")
        for i, v in enumerate(manifest.trivector_digest.elliptical):
            print(f"    E{i+1}: {float(v):.6f}")
        
        print("  Euclidean vector:")
        for i, v in enumerate(manifest.trivector_digest.euclidean):
            print(f"    U{i+1}: {float(v):.6f}")
        
        # Reconstruct the file
        print("\nReconstructing file from zero-point coordinates...")
        processor.reconstruct_file(manifest_file, output_file)
        
        # Verify the reconstruction
        is_correct = processor.verify_reconstruction(input_file, output_file)
        
        if is_correct:
            print("Reconstruction successful! Files are identical.")
        else:
            print("Reconstruction failed! Files are different.")
        
        # Print the reconstructed data
        with open(output_file, 'rb') as f:
            reconstructed_data = f.read()
        
        print(f"\nReconstructed data: {reconstructed_data.decode()}")
        print(f"Reconstructed size: {len(reconstructed_data)} bytes")

if __name__ == "__main__":
    run_example()