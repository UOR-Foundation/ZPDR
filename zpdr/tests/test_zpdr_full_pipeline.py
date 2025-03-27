import unittest
import numpy as np
from decimal import Decimal, getcontext
import time
import json

# Set high precision for Decimal calculations
getcontext().prec = 100

# Import core components from ZPDR
from zpdr.core.multivector import Multivector
from zpdr.core.geometric_spaces import (
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector,
    SpaceTransformer
)
from zpdr.core.zpa_manifest import (
    ZPAManifest, 
    create_zpa_manifest, 
    serialize_zpa,
    deserialize_zpa
)
from zpdr.utils import (
    to_decimal_array,
    calculate_internal_coherence,
    calculate_cross_coherence,
    calculate_global_coherence,
    validate_trilateral_coherence,
    normalize_with_invariants,
    denormalize_with_invariants,
    encode_to_zpdr_triple,
    reconstruct_from_zpdr_triple,
    COHERENCE_THRESHOLD
)

class TestZPDRFullPipeline(unittest.TestCase):
    """
    Test suite for validating the complete ZPDR processing pipeline.
    
    This test suite focuses on the full encoding-to-decoding workflow,
    including verification mechanisms, error correction, and performance.
    """
    
    def setUp(self):
        """Set up test fixtures and constants needed for pipeline testing."""
        # Define precision tolerance
        self.precision_tolerance = Decimal('1e-15')
        
        # Define test data
        self.test_values = [42, 101, 255, 1000, 65535, 16777216]
        self.test_string = "ZPDR_TEST_STRING"
        
        # Define expected coherence threshold
        self.coherence_threshold = COHERENCE_THRESHOLD
        
        # Create test multivector
        self.test_multivector = self._create_test_multivector()
        
    def test_end_to_end_encoding_decoding_natural_numbers(self):
        """
        Test the complete pipeline from encoding to decoding for natural numbers.
        
        This verifies that natural numbers can be encoded to ZPA and then
        decoded back to their original value with no loss of information.
        """
        # Process each test value
        for value in self.test_values:
            # Encode to ZPDR
            H, E, U = encode_to_zpdr_triple(value)
            
            # Create manifest
            manifest = create_zpa_manifest(H, E, U)
            
            # Serialize the manifest
            serialized = serialize_zpa(manifest)
            
            # Deserialize the manifest
            deserialized = deserialize_zpa(serialized)
            
            # Reconstruct the value
            decoded_value = reconstruct_from_zpdr_triple(
                deserialized.hyperbolic_vector,
                deserialized.elliptical_vector,
                deserialized.euclidean_vector
            )
            
            # Verify the decoded value matches the original
            self.assertEqual(decoded_value, value, 
                           f"Failed to correctly encode and decode value {value}")
    
    def test_verification_mechanism(self):
        """
        Test the verification mechanism for ZPA integrity.
        
        This verifies that the system correctly identifies valid and invalid ZPAs.
        """
        # Encode a valid test value
        H, E, U = encode_to_zpdr_triple(self.test_values[0])
        
        # Create valid manifest
        valid_manifest = create_zpa_manifest(H, E, U)
        
        # Create corrupted vectors
        corrupt_H = np.random.rand(3)  # Random hyperbolic vector
        corrupt_E = np.random.rand(3)  # Random elliptical vector
        corrupt_E = corrupt_E / np.linalg.norm(corrupt_E)  # Normalize to unit length
        corrupt_U = np.random.rand(3)  # Random euclidean vector
        
        # Create corrupted manifests
        corrupt_manifest_H = create_zpa_manifest(corrupt_H, E, U)
        corrupt_manifest_E = create_zpa_manifest(H, corrupt_E, U)
        corrupt_manifest_U = create_zpa_manifest(H, E, corrupt_U)
        corrupt_manifest_all = create_zpa_manifest(corrupt_H, corrupt_E, corrupt_U)
        
        # Verify valid manifest
        is_valid, coherence = validate_trilateral_coherence((H, E, U))
        self.assertTrue(is_valid, "Valid ZPA should be verified as valid")
        self.assertGreaterEqual(float(coherence), float(self.coherence_threshold),
                               "Valid ZPA should have coherence above threshold")
        
        # Verify corrupted manifests
        corrupt_tuples = [
            (corrupt_H, E, U),
            (H, corrupt_E, U),
            (H, E, corrupt_U),
            (corrupt_H, corrupt_E, corrupt_U)
        ]
        
        for i, corrupt_tuple in enumerate(corrupt_tuples):
            is_valid, coherence = validate_trilateral_coherence(corrupt_tuple)
            self.assertFalse(is_valid, f"Corrupted ZPA #{i+1} should be detected as invalid")
            self.assertLess(float(coherence), float(self.coherence_threshold),
                           f"Corrupted ZPA #{i+1} should have coherence below threshold")
    
    def test_error_correction(self):
        """
        Test the error correction capabilities of the ZPDR system.
        
        This verifies that the system can detect and correct small errors in ZPA.
        """
        # Encode a test value
        H, E, U = encode_to_zpdr_triple(self.test_values[1])
        
        # Add small perturbations (noise) to vectors
        noise_level = 0.01
        noisy_H = H + np.random.normal(0, noise_level, size=H.shape)
        noisy_E = E + np.random.normal(0, noise_level, size=E.shape)
        noisy_U = U + np.random.normal(0, noise_level, size=U.shape)
        
        # Renormalize E to unit length (as required for elliptical space)
        noisy_E = noisy_E / np.linalg.norm(noisy_E)
        
        # Calculate coherence of noisy vectors
        noisy_is_valid, noisy_coherence = validate_trilateral_coherence(
            (noisy_H, noisy_E, noisy_U)
        )
        
        # If coherence is already above threshold, increase noise until it falls below
        max_iterations = 10
        iteration = 0
        
        while noisy_is_valid and iteration < max_iterations:
            noise_level *= 2
            noisy_H = H + np.random.normal(0, noise_level, size=H.shape)
            noisy_E = E + np.random.normal(0, noise_level, size=E.shape)
            noisy_U = U + np.random.normal(0, noise_level, size=U.shape)
            
            # Renormalize E
            noisy_E = noisy_E / np.linalg.norm(noisy_E)
            
            noisy_is_valid, noisy_coherence = validate_trilateral_coherence(
                (noisy_H, noisy_E, noisy_U)
            )
            iteration += 1
        
        # Ensure we have a noisy version with low coherence
        self.assertLess(float(noisy_coherence), float(self.coherence_threshold),
                      "Noisy vectors should have coherence below threshold for this test")
        
        # Apply error correction by re-normalizing and aligning to canonical form
        corrected_H, H_invariants = normalize_with_invariants(noisy_H, "hyperbolic")
        corrected_E, E_invariants = normalize_with_invariants(noisy_E, "elliptical")
        corrected_U, U_invariants = normalize_with_invariants(noisy_U, "euclidean")
        
        # Verify corrected vectors have higher coherence
        corrected_is_valid, corrected_coherence = validate_trilateral_coherence(
            (corrected_H, corrected_E, corrected_U)
        )
        
        self.assertGreater(float(corrected_coherence), float(noisy_coherence),
                         "Error correction should improve coherence")
        
        # Reconstruct the original value from corrected ZPA
        corrected_H_orig = denormalize_with_invariants(corrected_H, H_invariants, "hyperbolic")
        corrected_E_orig = denormalize_with_invariants(corrected_E, E_invariants, "elliptical")
        corrected_U_orig = denormalize_with_invariants(corrected_U, U_invariants, "euclidean")
        
        # Calculate distance between original and corrected vectors
        h_distance = np.linalg.norm(H - corrected_H_orig)
        e_distance = np.linalg.norm(E - corrected_E_orig)
        u_distance = np.linalg.norm(U - corrected_U_orig)
        
        # Assert that the corrected vectors are closer to the original than the noisy ones
        self.assertLess(h_distance, np.linalg.norm(H - noisy_H),
                      "Corrected H vector should be closer to original than noisy vector")
        self.assertLess(e_distance, np.linalg.norm(E - noisy_E),
                      "Corrected E vector should be closer to original than noisy vector")
        self.assertLess(u_distance, np.linalg.norm(U - noisy_U),
                      "Corrected U vector should be closer to original than noisy vector")
    
    def test_encoding_performance(self):
        """
        Test the performance of the encoding process.
        
        This ensures that encoding operations meet reasonable performance criteria.
        """
        # Number of iterations for consistent timing
        iterations = 10
        
        # Test value to encode (using a larger value for performance testing)
        test_value = 16777216  # 2^24
        
        # Time the encoding process
        start_time = time.time()
        
        for _ in range(iterations):
            H, E, U = encode_to_zpdr_triple(test_value)
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Assert that encoding time is within reasonable bounds
        # For a development/testing environment, we'll set a conservative threshold
        self.assertLess(avg_time, 0.1, 
                      f"Encoding performance should be under 100ms per operation (was {avg_time*1000:.2f}ms)")
    
    def test_decoding_performance(self):
        """
        Test the performance of the decoding process.
        
        This ensures that decoding operations meet reasonable performance criteria.
        """
        # Number of iterations for consistent timing
        iterations = 10
        
        # Test value to encode/decode
        test_value = 16777216  # 2^24
        
        # Encode once to get the ZPA
        H, E, U = encode_to_zpdr_triple(test_value)
        
        # Time the decoding process
        start_time = time.time()
        
        for _ in range(iterations):
            decoded_value = reconstruct_from_zpdr_triple(H, E, U)
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Assert that decoding time is within reasonable bounds
        self.assertLess(avg_time, 0.1, 
                      f"Decoding performance should be under 100ms per operation (was {avg_time*1000:.2f}ms)")
    
    def test_manifest_serialization_performance(self):
        """
        Test the performance of ZPA manifest serialization.
        
        This ensures that serialization operations meet reasonable performance criteria.
        """
        # Number of iterations for consistent timing
        iterations = 10
        
        # Create a test manifest
        H, E, U = encode_to_zpdr_triple(self.test_values[2])
        manifest = create_zpa_manifest(H, E, U)
        
        # Time the serialization process
        start_time = time.time()
        
        for _ in range(iterations):
            serialized = serialize_zpa(manifest)
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Assert that serialization time is within reasonable bounds
        self.assertLess(avg_time, 0.01, 
                      f"Serialization performance should be under 10ms per operation (was {avg_time*1000:.2f}ms)")
    
    def test_manifest_deserialization_performance(self):
        """
        Test the performance of ZPA manifest deserialization.
        
        This ensures that deserialization operations meet reasonable performance criteria.
        """
        # Number of iterations for consistent timing
        iterations = 10
        
        # Create and serialize a test manifest
        H, E, U = encode_to_zpdr_triple(self.test_values[2])
        manifest = create_zpa_manifest(H, E, U)
        serialized = serialize_zpa(manifest)
        
        # Time the deserialization process
        start_time = time.time()
        
        for _ in range(iterations):
            deserialized = deserialize_zpa(serialized)
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Assert that deserialization time is within reasonable bounds
        self.assertLess(avg_time, 0.01, 
                      f"Deserialization performance should be under 10ms per operation (was {avg_time*1000:.2f}ms)")
    
    def test_parallel_processing_pipeline(self):
        """
        Test parallel processing of multiple values through the pipeline.
        
        This ensures that the pipeline can handle multiple encodings efficiently.
        """
        # Skip this test if parallel processing is not implemented
        try:
            from zpdr.utils.parallel_processor import process_batch
        except ImportError:
            self.skipTest("Parallel processing not implemented yet")
            return
        
        # Use a batch of test values
        batch_values = self.test_values
        
        # Process in parallel
        results = process_batch(batch_values)
        
        # Verify all results match the inputs
        for i, value in enumerate(batch_values):
            self.assertEqual(results[i]['decoded_value'], value, 
                           f"Parallel processing failed for value {value}")
    
    def test_zpdr_processor_multithreading(self):
        """
        Test the multithreaded capabilities of the ZPDR processor.
        
        This ensures that performance scales with multiple threads.
        """
        # Skip this test if multithreading is not implemented
        try:
            from zpdr.core.zpdr_processor import ZPDRProcessor
        except ImportError:
            self.skipTest("ZPDRProcessor not implemented yet")
            return
        
        # Create a ZPDR processor with multithreading
        processor = ZPDRProcessor(multithreaded=True)
        
        # Test values
        batch_values = self.test_values
        
        # Process in parallel
        start_time = time.time()
        results = processor.process_batch(batch_values)
        end_time = time.time()
        
        # Time per value
        parallel_time_per_value = (end_time - start_time) / len(batch_values)
        
        # Create a single-threaded processor for comparison
        processor_single = ZPDRProcessor(multithreaded=False)
        
        # Process sequentially
        start_time = time.time()
        for value in batch_values:
            processor_single.encode(value)
        end_time = time.time()
        
        # Time per value
        sequential_time_per_value = (end_time - start_time) / len(batch_values)
        
        # Verify multithreading provides a speedup
        # (allowing for some overhead in very small tests)
        self.assertLessEqual(parallel_time_per_value * 1.5, sequential_time_per_value,
                           "Multithreaded processing should be at least as fast as sequential processing")
    
    # Helper methods
    def _create_test_multivector(self):
        """Create a consistent test multivector for tests."""
        components = {
            "1": 0.5,           # Scalar part
            "e1": 0.1,          # Vector part
            "e2": 0.2,
            "e3": 0.3,
            "e12": 0.4,         # Bivector part
            "e23": 0.5,
            "e31": 0.6,
            "e123": 0.7         # Trivector part
        }
        
        return Multivector(components)


if __name__ == '__main__':
    unittest.main()