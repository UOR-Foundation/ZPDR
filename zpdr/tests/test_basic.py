#!/usr/bin/env python3
"""
Basic tests for the ZPDR system.
"""

import unittest
import tempfile
import os
from pathlib import Path

from zpdr.core.zpdr_processor import ZPDRProcessor
from zpdr.core.multivector import Multivector
from zpdr.core.trivector_digest import TrivectorDigest
from zpdr.core.manifest import Manifest

class TestBasic(unittest.TestCase):
    """
    Basic tests for the ZPDR system.
    """
    
    def setUp(self):
        """Set up the test environment."""
        self.processor = ZPDRProcessor(coherence_threshold=0.9)  # Lower threshold for testing
        self.test_data = b"Hello, ZPDR!"
        self.test_filename = "test.txt"
    
    def test_multivector_conversion(self):
        """Test conversion between binary data and multivector."""
        # Convert data to multivector
        multivector = Multivector.from_binary_data(self.test_data)
        
        # Extract components
        scalar = multivector.get_scalar_component()
        vector = multivector.get_vector_components()
        bivector = multivector.get_bivector_components()
        trivector = multivector.get_trivector_component()
        
        # Check that all components are populated
        self.assertNotEqual(scalar, 0)
        self.assertTrue(any(v != 0 for v in vector))
        self.assertTrue(any(v != 0 for v in bivector))
        self.assertNotEqual(trivector, 0)
    
    def test_trivector_digest(self):
        """Test creation and normalization of trivector digest."""
        # Create a multivector from test data
        multivector = Multivector.from_binary_data(self.test_data)
        
        # Extract trivector
        hyperbolic, elliptical, euclidean = multivector.extract_trivector()
        
        # Create trivector digest
        digest = TrivectorDigest(hyperbolic, elliptical, euclidean)
        
        # Check that coherence is calculated
        self.assertGreater(digest.get_global_coherence(), 0)
        
        # Test normalization
        normalized = digest.normalize()
        
        # Normalized vectors should have unit length
        h_norm = sum(v * v for v in normalized.hyperbolic).sqrt()
        e_norm = sum(v * v for v in normalized.elliptical).sqrt()
        u_norm = sum(v * v for v in normalized.euclidean).sqrt()
        
        self.assertAlmostEqual(float(h_norm), 1.0, places=6)
        self.assertAlmostEqual(float(e_norm), 1.0, places=6)
        self.assertAlmostEqual(float(u_norm), 1.0, places=6)
    
    def test_manifest_creation(self):
        """Test creation of manifest from data."""
        # Process data
        manifest = self.processor.process_data(self.test_data, self.test_filename)
        
        # Check manifest properties
        self.assertEqual(manifest.original_filename, self.test_filename)
        self.assertEqual(manifest.file_size, len(self.test_data))
        self.assertTrue(manifest.checksum.startswith("sha256:"))
        
        # Check coherence
        self.assertTrue(manifest.verify_coherence())
    
    def test_manifest_serialization(self):
        """Test serialization and deserialization of manifest."""
        # Create manifest
        manifest = self.processor.process_data(self.test_data, self.test_filename)
        
        # Convert to JSON and back
        json_str = manifest.to_json()
        loaded_manifest = Manifest.from_json(json_str)
        
        # Check that loaded manifest matches original
        self.assertEqual(loaded_manifest.original_filename, manifest.original_filename)
        self.assertEqual(loaded_manifest.file_size, manifest.file_size)
        self.assertEqual(loaded_manifest.checksum, manifest.checksum)
        
        # Check coordinates
        self.assertEqual(loaded_manifest.trivector_digest.hyperbolic, manifest.trivector_digest.hyperbolic)
        self.assertEqual(loaded_manifest.trivector_digest.elliptical, manifest.trivector_digest.elliptical)
        self.assertEqual(loaded_manifest.trivector_digest.euclidean, manifest.trivector_digest.euclidean)
    
    def test_reconstruction(self):
        """Test reconstruction of data from manifest."""
        # Create manifest
        manifest = self.processor.process_data(self.test_data, self.test_filename)
        
        # Reconstruct data
        reconstructed = self.processor.reconstruct_from_manifest(manifest)
        
        # Check that reconstructed data matches original
        self.assertEqual(reconstructed, self.test_data)
    
    def test_file_roundtrip(self):
        """Test full file encoding and decoding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary files
            input_file = Path(temp_dir) / "input.txt"
            manifest_file = Path(temp_dir) / "input.txt.zpdr"
            output_file = Path(temp_dir) / "output.txt"
            
            # Write test data to input file
            with open(input_file, 'wb') as f:
                f.write(self.test_data)
            
            # Process input file
            manifest = self.processor.process_file(input_file)
            
            # Save manifest
            self.processor.save_manifest(manifest, manifest_file)
            
            # Reconstruct file
            self.processor.reconstruct_file(manifest_file, output_file)
            
            # Read reconstructed file
            with open(output_file, 'rb') as f:
                reconstructed = f.read()
            
            # Check that reconstructed data matches original
            self.assertEqual(reconstructed, self.test_data)
            
            # Verify reconstruction
            self.assertTrue(self.processor.verify_reconstruction(input_file, output_file))

if __name__ == '__main__':
    unittest.main()