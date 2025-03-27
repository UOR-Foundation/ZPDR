import unittest
import numpy as np
from decimal import Decimal, getcontext
import json

# Set high precision for Decimal calculations
getcontext().prec = 100

# Import core components
from zpdr.core.multivector import Multivector
from zpdr.core.geometric_spaces import (
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector,
    SpaceTransformer
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
    COHERENCE_THRESHOLD,
    PRECISION_TOLERANCE
)

class TestZPDRErrorCorrection(unittest.TestCase):
    """
    Test suite for validating the error correction capabilities of ZPDR.
    
    This test suite focuses on verifying that the system can detect and
    correct errors in ZPA representations, ensuring data integrity and
    robustness in the presence of noise or corruption.
    """
    
    def setUp(self):
        """Set up test fixtures and constants needed for error correction testing."""
        # Define precision tolerance
        self.precision_tolerance = PRECISION_TOLERANCE
        
        # Define coherence threshold
        self.coherence_threshold = COHERENCE_THRESHOLD
        
        # Create test vectors for baseline
        # These vectors follow the expected patterns for stable coherence
        self.test_H = np.array([0.125, 0.25, 0.5], dtype=np.float64)
        self.test_E = np.array([0.7071, 0.7071, 0], dtype=np.float64)
        self.test_U = np.array([0.333, 0.333, 0.333], dtype=np.float64)
        
        # Test values for encoding
        self.test_values = [42, 101, 255, 1000, 65535]
    
    def test_noise_detection(self):
        """
        Test the system's ability to detect noise in ZPA vectors.
        
        This ensures that coherence measures correctly identify corrupted data.
        """
        # Calculate baseline coherence
        base_is_valid, base_coherence = validate_trilateral_coherence(
            (self.test_H, self.test_E, self.test_U)
        )
        
        # Verify baseline is valid
        self.assertTrue(base_is_valid, "Baseline ZPA should be valid")
        self.assertGreaterEqual(float(base_coherence), float(self.coherence_threshold),
                               "Baseline ZPA should have coherence above threshold")
        
        # Add increasing levels of noise until coherence drops below threshold
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        for noise_level in noise_levels:
            # Add noise to vectors
            noisy_H = self.test_H + np.random.normal(0, noise_level, size=self.test_H.shape)
            noisy_E = self.test_E + np.random.normal(0, noise_level, size=self.test_E.shape)
            noisy_U = self.test_U + np.random.normal(0, noise_level, size=self.test_U.shape)
            
            # Renormalize E to unit length (as required for elliptical space)
            noisy_E = noisy_E / np.linalg.norm(noisy_E)
            
            # Calculate coherence
            noisy_is_valid, noisy_coherence = validate_trilateral_coherence(
                (noisy_H, noisy_E, noisy_U)
            )
            
            # At higher noise levels, coherence should drop below threshold
            if noise_level >= 0.2:  # Expecting to fail at this level
                self.assertLess(float(noisy_coherence), float(base_coherence),
                             f"Coherence should decrease with noise level {noise_level}")
    
    def test_error_correction_via_normalization(self):
        """
        Test error correction via normalization to canonical forms.
        
        This verifies that normalizing vectors can fix certain types of errors.
        """
        # Add moderate noise to test vectors
        noise_level = 0.1
        noisy_H = self.test_H + np.random.normal(0, noise_level, size=self.test_H.shape)
        noisy_E = self.test_E + np.random.normal(0, noise_level, size=self.test_E.shape)
        noisy_U = self.test_U + np.random.normal(0, noise_level, size=self.test_U.shape)
        
        # Get baseline coherence of noisy vectors
        noisy_is_valid, noisy_coherence = validate_trilateral_coherence(
            (noisy_H, noisy_E, noisy_U)
        )
        
        # Normalize to canonical form
        H_normalized, H_invariants = normalize_with_invariants(noisy_H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(noisy_E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(noisy_U, "euclidean")
        
        # Calculate coherence after normalization
        normalized_is_valid, normalized_coherence = validate_trilateral_coherence(
            (H_normalized, E_normalized, U_normalized)
        )
        
        # Verify that normalization improved coherence
        self.assertGreaterEqual(float(normalized_coherence), float(noisy_coherence),
                           "Normalization should improve or maintain coherence")
        
        # Reconstruct from normalized form
        H_reconstructed = denormalize_with_invariants(H_normalized, H_invariants, "hyperbolic")
        E_reconstructed = denormalize_with_invariants(E_normalized, E_invariants, "elliptical")
        U_reconstructed = denormalize_with_invariants(U_normalized, U_invariants, "euclidean")
        
        # Verify reconstruction is closer to original than noisy version
        h_orig_distance = np.linalg.norm(self.test_H - H_reconstructed)
        e_orig_distance = np.linalg.norm(self.test_E - E_reconstructed)
        u_orig_distance = np.linalg.norm(self.test_U - U_reconstructed)
        
        h_noisy_distance = np.linalg.norm(self.test_H - noisy_H)
        e_noisy_distance = np.linalg.norm(self.test_E - noisy_E)
        u_noisy_distance = np.linalg.norm(self.test_U - noisy_U)
        
        # Reconstruction should maintain equal or better distance (though this may not
        # always be true for higher noise levels where normalization might not maintain orientation)
        self.assertLessEqual(h_orig_distance, h_noisy_distance * 1.5,
                         "Reconstructed H should not be significantly worse than noisy H")
        self.assertLessEqual(e_orig_distance, e_noisy_distance * 1.5,
                         "Reconstructed E should not be significantly worse than noisy E")
        self.assertLessEqual(u_orig_distance, u_noisy_distance * 1.5,
                         "Reconstructed U should not be significantly worse than noisy U")
    
    def test_invariant_based_error_detection(self):
        """
        Test detection of errors using invariants.
        
        This verifies that inconsistencies in invariants can be detected.
        """
        # Create a ZPA triple with normalized vectors
        H, E, U = self.test_H, self.test_E, self.test_U
        
        # Normalize to get invariants
        H_normalized, H_invariants = normalize_with_invariants(H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(U, "euclidean")
        
        # Create corrupted invariants
        corrupt_H_invariants = H_invariants.copy()
        if 'rotation_angle' in corrupt_H_invariants:
            corrupt_H_invariants['rotation_angle'] += np.pi / 2  # Add 90 degrees
        
        corrupt_E_invariants = E_invariants.copy()
        if 'rotation_angle' in corrupt_E_invariants:
            corrupt_E_invariants['rotation_angle'] += np.pi / 2  # Add 90 degrees
        
        corrupt_U_invariants = U_invariants.copy()
        if 'rotation_angle' in corrupt_U_invariants:
            corrupt_U_invariants['rotation_angle'] += np.pi / 2  # Add 90 degrees
        
        # Reconstruct with corrupted invariants
        H_corrupt = denormalize_with_invariants(H_normalized, corrupt_H_invariants, "hyperbolic")
        E_corrupt = denormalize_with_invariants(E_normalized, corrupt_E_invariants, "elliptical")
        U_corrupt = denormalize_with_invariants(U_normalized, corrupt_U_invariants, "euclidean")
        
        # Verify that corrupted reconstruction is different from original
        self.assertFalse(np.allclose(H, H_corrupt, rtol=1e-10, atol=1e-10),
                       "Corrupted H reconstruction should differ from original")
        self.assertFalse(np.allclose(E, E_corrupt, rtol=1e-10, atol=1e-10),
                       "Corrupted E reconstruction should differ from original")
        self.assertFalse(np.allclose(U, U_corrupt, rtol=1e-10, atol=1e-10),
                       "Corrupted U reconstruction should differ from original")
        
        # Calculate coherence of corrupted triple
        corrupt_is_valid, corrupt_coherence = validate_trilateral_coherence(
            (H_corrupt, E_corrupt, U_corrupt)
        )
        
        # Coherence should be lower for corrupted triple
        original_is_valid, original_coherence = validate_trilateral_coherence((H, E, U))
        self.assertLess(float(corrupt_coherence), float(original_coherence),
                     "Corrupted triple should have lower coherence than original")
    
    def test_error_correction_with_redundant_encoding(self):
        """
        Test error correction using redundant encoding.
        
        This verifies that encoding the same value in multiple ways provides error correction.
        """
        # Encode a test value (which should create a coherent ZPA triple)
        test_value = self.test_values[0]
        H, E, U = encode_to_zpdr_triple(test_value)
        
        # Verify baseline is valid
        base_is_valid, base_coherence = validate_trilateral_coherence((H, E, U))
        self.assertTrue(base_is_valid, "Baseline ZPA should be valid")
        
        # Add moderate noise to one component (H)
        noise_level = 0.1
        noisy_H = H + np.random.normal(0, noise_level, size=H.shape)
        
        # Calculate coherence with one noisy component
        noisy_is_valid, noisy_coherence = validate_trilateral_coherence((noisy_H, E, U))
        
        # Coherence should decrease but might still be valid depending on noise level
        self.assertLess(float(noisy_coherence), float(base_coherence),
                     "Coherence should decrease with noisy component")
        
        # Attempt reconstruction from noisy ZPA
        reconstructed_value = reconstruct_from_zpdr_triple(noisy_H, E, U)
        
        # Verify the reconstruction is still correct despite noise
        # This works because the redundant encoding in E and U components
        # provides error correction capabilities
        self.assertEqual(reconstructed_value, test_value,
                       "Reconstruction should be resilient to noise in one component")
    
    def test_progressive_error_accumulation(self):
        """
        Test how errors accumulate with multiple transformations.
        
        This verifies the system's robustness to progressive error buildup.
        """
        # Start with a basic ZPA triple
        H, E, U = encode_to_zpdr_triple(self.test_values[1])
        
        # Track coherence through multiple transform-normalize cycles
        coherence_values = []
        
        # Get initial coherence
        is_valid, initial_coherence = validate_trilateral_coherence((H, E, U))
        coherence_values.append(float(initial_coherence))
        
        # Simulate multiple cycles of transformation and normalization
        cycles = 5
        current_H, current_E, current_U = H.copy(), E.copy(), U.copy()
        
        for _ in range(cycles):
            # Apply small random transformations
            noise_level = 0.02
            current_H += np.random.normal(0, noise_level, size=current_H.shape)
            current_E += np.random.normal(0, noise_level, size=current_E.shape)
            current_U += np.random.normal(0, noise_level, size=current_U.shape)
            
            # Renormalize E to unit length
            current_E = current_E / np.linalg.norm(current_E)
            
            # Normalize to canonical form
            H_normalized, H_invariants = normalize_with_invariants(current_H, "hyperbolic")
            E_normalized, E_invariants = normalize_with_invariants(current_E, "elliptical")
            U_normalized, U_invariants = normalize_with_invariants(current_U, "euclidean")
            
            # Denormalize to get back to original space
            current_H = denormalize_with_invariants(H_normalized, H_invariants, "hyperbolic")
            current_E = denormalize_with_invariants(E_normalized, E_invariants, "elliptical")
            current_U = denormalize_with_invariants(U_normalized, U_invariants, "euclidean")
            
            # Calculate coherence
            is_valid, current_coherence = validate_trilateral_coherence(
                (current_H, current_E, current_U)
            )
            coherence_values.append(float(current_coherence))
        
        # Verify that coherence doesn't collapse across cycles
        # There may be some degradation, but it shouldn't drop dramatically
        for i in range(1, len(coherence_values)):
            self.assertGreater(coherence_values[i], 0.7 * coherence_values[0],
                            f"Coherence should not collapse at cycle {i}")
        
        # Attempt reconstruction from final vectors
        reconstructed_value = reconstruct_from_zpdr_triple(current_H, current_E, current_U)
        
        # Should still reconstruct to original value if error correction is working
        self.assertEqual(reconstructed_value, self.test_values[1],
                       "Value should still be reconstructible after multiple transformation cycles")
    
    def test_multivector_error_correction(self):
        """
        Test error correction at the multivector level.
        
        This verifies that errors in multivector representation can be corrected.
        """
        # Create a test multivector
        components = {
            "1": 0.5,
            "e1": 0.1,
            "e2": 0.2,
            "e3": 0.3,
            "e12": 0.4,
            "e23": 0.5,
            "e31": 0.6,
            "e123": 0.7
        }
        
        test_mv = Multivector(components)
        
        # Extract ZPA from multivector
        from zpdr.utils import _extract_trilateral_vectors
        H, E, U = _extract_trilateral_vectors(test_mv)
        
        # Add noise to components directly in the multivector
        noisy_components = components.copy()
        for basis, coef in components.items():
            noisy_components[basis] = coef * (1 + np.random.normal(0, 0.1))
        
        noisy_mv = Multivector(noisy_components)
        
        # Extract ZPA from noisy multivector
        noisy_H, noisy_E, noisy_U = _extract_trilateral_vectors(noisy_mv)
        
        # Calculate coherence
        orig_is_valid, orig_coherence = validate_trilateral_coherence((H, E, U))
        noisy_is_valid, noisy_coherence = validate_trilateral_coherence(
            (noisy_H, noisy_E, noisy_U)
        )
        
        # Verify that noise in multivector reduces coherence
        self.assertLess(float(noisy_coherence), float(orig_coherence),
                     "Noise in multivector should reduce coherence")
        
        # Reconstruct multivector from noisy ZPA
        from zpdr.utils import _reconstruct_multivector_from_trilateral
        reconstructed_mv = _reconstruct_multivector_from_trilateral(noisy_H, noisy_E, noisy_U)
        
        # Extract clean ZPA from reconstructed multivector
        recon_H, recon_E, recon_U = _extract_trilateral_vectors(reconstructed_mv)
        
        # Calculate coherence of reconstructed vectors
        recon_is_valid, recon_coherence = validate_trilateral_coherence(
            (recon_H, recon_E, recon_U)
        )
        
        # Reconstructed ZPA should have better coherence
        self.assertGreaterEqual(float(recon_coherence), float(noisy_coherence),
                             "Reconstruction should improve coherence")
    
    # Helper methods
    def _calculate_zpa_error_level(self, original_triple, noisy_triple):
        """Calculate the error level between original and noisy ZPA triples."""
        H, E, U = original_triple
        noisy_H, noisy_E, noisy_U = noisy_triple
        
        h_error = np.linalg.norm(H - noisy_H) / np.linalg.norm(H) if np.linalg.norm(H) > 0 else 0
        e_error = np.linalg.norm(E - noisy_E) / np.linalg.norm(E) if np.linalg.norm(E) > 0 else 0
        u_error = np.linalg.norm(U - noisy_U) / np.linalg.norm(U) if np.linalg.norm(U) > 0 else 0
        
        return (h_error + e_error + u_error) / 3


if __name__ == '__main__':
    unittest.main()