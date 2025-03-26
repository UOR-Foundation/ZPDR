#!/usr/bin/env python3
"""
Example of error correction in ZPDR.

This script demonstrates how to use ZPDR's error correction and verification capabilities
to detect and recover from errors in zero-point coordinates.
"""

import os
import sys
import tempfile
from pathlib import Path
from decimal import Decimal

# Add the parent directory to the path so we can import the zpdr package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from zpdr.core.zpdr_processor import ZPDRProcessor
from zpdr.core.trivector_digest import TrivectorDigest
from zpdr.core.manifest import Manifest

def run_example():
    """Run an example of error correction in ZPDR."""
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test file
        input_file = temp_path / "input.txt"
        manifest_file = temp_path / "input.txt.zpdr"
        corrupted_manifest_file = temp_path / "corrupted.zpdr"
        corrected_manifest_file = temp_path / "corrected.zpdr"
        output_file = temp_path / "output.txt"
        
        # Write some data to the input file
        test_data = b"This is a test of ZPDR's error correction and verification capabilities."
        with open(input_file, 'wb') as f:
            f.write(test_data)
        
        print(f"Created test file: {input_file}")
        print(f"Original data: {test_data.decode()}")
        print(f"Original size: {len(test_data)} bytes")
        print()
        
        # Create a ZPDR processor
        processor = ZPDRProcessor(coherence_threshold=0.99)
        
        # Process the input file
        print("1. Processing file to extract zero-point coordinates...")
        manifest = processor.process_file(input_file)
        
        # Save the manifest
        processor.save_manifest(manifest, manifest_file)
        print(f"   Saved manifest to: {manifest_file}")
        
        # Verify the manifest
        verification_results = processor.verify_manifest(manifest)
        print("\n2. Verifying the original manifest:")
        print(f"   Overall verification: {verification_results['overall_passed']}")
        print(f"   Coherence: {verification_results['coherence_verification']['calculated_coherence']}")
        
        # Create a corrupted manifest
        print("\n3. Creating a corrupted manifest...")
        
        # Load the manifest 
        loaded_manifest = processor.load_manifest(manifest_file)
        
        # Modify coordinates to introduce errors
        original_hyperbolic = loaded_manifest.trivector_digest.hyperbolic
        corrupt_hyperbolic = original_hyperbolic.copy()
        
        # Add noise to the first component
        corrupt_hyperbolic[0] += Decimal('0.15')  # Significant corruption
        
        # Normalize the corrupted vector
        h_norm = sum(v * v for v in corrupt_hyperbolic).sqrt()
        if h_norm > Decimal('0'):
            corrupt_hyperbolic = [h / h_norm for h in corrupt_hyperbolic]
        
        # Create a corrupted digest
        corrupt_digest = TrivectorDigest(
            corrupt_hyperbolic,
            loaded_manifest.trivector_digest.elliptical,
            loaded_manifest.trivector_digest.euclidean
        )
        
        # Create a manifest with the corrupted digest
        corrupt_manifest = Manifest(
            trivector_digest=corrupt_digest,
            original_filename=loaded_manifest.original_filename,
            file_size=loaded_manifest.file_size,
            checksum=loaded_manifest.checksum,
            base_fiber_id=loaded_manifest.base_fiber_id,
            structure_id=loaded_manifest.structure_id,
            coherence_threshold=float(loaded_manifest.coherence_threshold),
            additional_metadata=loaded_manifest.additional_metadata
        )
        
        # Save the corrupted manifest
        processor.save_manifest(corrupt_manifest, corrupted_manifest_file)
        print(f"   Saved corrupted manifest to: {corrupted_manifest_file}")
        
        # Verify the corrupted manifest
        corrupt_verification = processor.verify_manifest(corrupt_manifest)
        print("\n4. Verifying the corrupted manifest:")
        print(f"   Overall verification: {corrupt_verification['overall_passed']}")
        print(f"   Coherence: {corrupt_verification['coherence_verification']['calculated_coherence']}")
        print(f"   Weakest link: {corrupt_verification['coherence_verification']['weakest_coherence_link']}")
        print(f"   Recommendation: {corrupt_verification['coherence_verification']['recommendation']}")
        
        # Try to reconstruct from the corrupted manifest with auto-correction disabled
        print("\n5. Attempting reconstruction from corrupted manifest WITHOUT auto-correction...")
        no_correction_processor = ZPDRProcessor(coherence_threshold=0.99, auto_correct=False)
        
        try:
            no_correction_processor.reconstruct_file(corrupted_manifest_file, output_file)
            print("   Reconstruction succeeded unexpectedly!")
        except ValueError as e:
            print(f"   Reconstruction failed as expected: {e}")
        
        # Correct the manifest
        print("\n6. Applying error correction to the corrupted manifest...")
        corrected_manifest, correction_details = processor.correct_coordinates(corrupt_manifest)
        
        # Save the corrected manifest
        processor.save_manifest(corrected_manifest, corrected_manifest_file)
        print(f"   Saved corrected manifest to: {corrected_manifest_file}")
        
        # Print correction details
        print("\n7. Correction details:")
        print(f"   Iterations: {correction_details['iterations']}")
        print(f"   Initial coherence: {correction_details['initial_coherence']}")
        print(f"   Final coherence: {correction_details['final_coherence']}")
        print(f"   Improvement: {correction_details['improvement']}")
        
        # Verify the corrected manifest
        corrected_verification = processor.verify_manifest(corrected_manifest)
        print("\n8. Verifying the corrected manifest:")
        print(f"   Overall verification: {corrected_verification['overall_passed']}")
        print(f"   Coherence: {corrected_verification['coherence_verification']['calculated_coherence']}")
        
        # Try to reconstruct from the corrected manifest
        print("\n9. Attempting reconstruction from CORRECTED manifest...")
        try:
            processor.reconstruct_file(corrected_manifest_file, output_file)
            print("   Reconstruction succeeded!")
            
            # Read the reconstructed file
            with open(output_file, 'rb') as f:
                reconstructed_data = f.read()
            
            # Check if reconstructed matches original
            if reconstructed_data == test_data:
                print("   Verification: Original and reconstructed files are identical!")
            else:
                print("   Verification: Files differ!")
                
        except ValueError as e:
            print(f"   Reconstruction failed: {e}")
        
        # Demonstrate auto-correction
        print("\n10. Demonstrating automatic correction during reconstruction...")
        auto_correct_processor = ZPDRProcessor(coherence_threshold=0.99, auto_correct=True)
        
        try:
            # Reconstruct directly from the corrupted manifest
            auto_correct_processor.reconstruct_file(corrupted_manifest_file, output_file)
            print("   Reconstruction with auto-correction succeeded!")
            
            # Read the reconstructed file
            with open(output_file, 'rb') as f:
                reconstructed_data = f.read()
            
            # Check if reconstructed matches original
            if reconstructed_data == test_data:
                print("   Verification: Original and reconstructed files are identical!")
            else:
                print("   Verification: Files differ!")
                
        except ValueError as e:
            print(f"   Reconstruction failed despite auto-correction: {e}")
        
        print("\nExample complete! Error correction successfully recovered from corrupted coordinates.")

if __name__ == "__main__":
    run_example()