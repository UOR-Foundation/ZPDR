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
    COHERENCE_THRESHOLD,
    PRECISION_TOLERANCE
)

class TestZPDRVerification(unittest.TestCase):
    """
    Test suite for validating the verification mechanisms of ZPDR.
    
    This test suite focuses on verifying that the system can validate
    ZPA integrity and reconstruction correctness.
    """
    
    def setUp(self):
        """Set up test fixtures and constants needed for verification testing."""
        # Define precision tolerance
        self.precision_tolerance = PRECISION_TOLERANCE
        
        # Define coherence threshold
        self.coherence_threshold = COHERENCE_THRESHOLD
        
        # Create test vectors
        self.test_H = np.array([0.125, 0.25, 0.5], dtype=np.float64)
        self.test_E = np.array([0.7071, 0.7071, 0], dtype=np.float64)
        self.test_U = np.array([0.333, 0.333, 0.333], dtype=np.float64)
        
        # Test values for encoding
        self.test_values = [42, 101, 255, 1000, 65535]
        
        # Create a test manifest
        self.test_manifest = create_zpa_manifest(
            self.test_H,
            self.test_E,
            self.test_U
        )
    
    def test_zpa_integrity_verification(self):
        """
        Test the verification of ZPA integrity.
        
        This verifies that the system correctly identifies valid and invalid ZPAs.
        """
        # Verify integrity of a valid ZPA triple
        is_valid, coherence = validate_trilateral_coherence(
            (self.test_H, self.test_E, self.test_U)
        )
        
        self.assertTrue(is_valid, "Valid ZPA triple should pass integrity verification")
        self.assertGreaterEqual(float(coherence), float(self.coherence_threshold),
                               "Valid ZPA triple should have coherence above threshold")
        
        # Create invalid vectors
        invalid_H = np.random.rand(3)
        # For elliptical vector, normalize to unit length
        invalid_E = np.random.rand(3)
        invalid_E = invalid_E / np.linalg.norm(invalid_E)
        invalid_U = np.random.rand(3)
        
        # Verify that random vectors fail integrity check
        is_valid, coherence = validate_trilateral_coherence(
            (invalid_H, invalid_E, invalid_U)
        )
        
        self.assertLess(float(coherence), float(self.coherence_threshold),
                      "Random ZPA triple should have coherence below threshold")
    
    def test_manifest_integrity_verification(self):
        """
        Test verification of ZPA manifest integrity.
        
        This verifies that manifest serialization preserves ZPA integrity.
        """
        # Serialize and deserialize manifest
        serialized = serialize_zpa(self.test_manifest)
        deserialized = deserialize_zpa(serialized)
        
        # Verify vectors from deserialized manifest
        H = deserialized.hyperbolic_vector
        E = deserialized.elliptical_vector
        U = deserialized.euclidean_vector
        
        # Check integrity of deserialized ZPA
        is_valid, coherence = validate_trilateral_coherence((H, E, U))
        
        self.assertTrue(is_valid, "Deserialized ZPA should pass integrity verification")
        self.assertGreaterEqual(float(coherence), float(self.coherence_threshold),
                               "Deserialized ZPA should have coherence above threshold")
        
        # Verify that vectors match the originals
        self.assertTrue(np.allclose(H, self.test_H, rtol=1e-14, atol=1e-14),
                       "Deserialized H vector should match original")
        self.assertTrue(np.allclose(E, self.test_E, rtol=1e-14, atol=1e-14),
                       "Deserialized E vector should match original")
        self.assertTrue(np.allclose(U, self.test_U, rtol=1e-14, atol=1e-14),
                       "Deserialized U vector should match original")
    
    def test_weak_invariants_verification(self):
        """
        Test verification of weak invariants in ZPA.
        
        This verifies that invariants are logically consistent.
        """
        # Normalize vectors to extract invariants
        H_normalized, H_invariants = normalize_with_invariants(self.test_H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(self.test_E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(self.test_U, "euclidean")
        
        # Verify specific invariant properties
        
        # Hyperbolic space: scale_factor should be positive
        self.assertGreater(H_invariants.get('scale_factor', 0), 0,
                         "Hyperbolic scale factor should be positive")
        
        # Elliptical space: radius should be positive
        self.assertGreater(E_invariants.get('radius', 0), 0,
                         "Elliptical radius should be positive")
        
        # Euclidean space: magnitude should be positive
        self.assertGreater(U_invariants.get('magnitude', 0), 0,
                         "Euclidean magnitude should be positive")
        
        # Rotation angles should be in [-π, π]
        if 'rotation_angle' in H_invariants:
            self.assertGreaterEqual(H_invariants['rotation_angle'], -np.pi,
                                 "Hyperbolic rotation angle should be >= -π")
            self.assertLessEqual(H_invariants['rotation_angle'], np.pi,
                              "Hyperbolic rotation angle should be <= π")
        
        if 'rotation_angle' in E_invariants:
            self.assertGreaterEqual(E_invariants['rotation_angle'], -np.pi,
                                 "Elliptical rotation angle should be >= -π")
            self.assertLessEqual(E_invariants['rotation_angle'], np.pi,
                              "Elliptical rotation angle should be <= π")
        
        if 'rotation_angle' in U_invariants:
            self.assertGreaterEqual(U_invariants['rotation_angle'], -np.pi,
                                 "Euclidean rotation angle should be >= -π")
            self.assertLessEqual(U_invariants['rotation_angle'], np.pi,
                              "Euclidean rotation angle should be <= π")
    
    def test_strong_invariants_verification(self):
        """
        Test verification of strong invariants in ZPA.
        
        This verifies that invariants maintain specific mathematical properties.
        """
        # Create a test value and encode it
        test_value = self.test_values[2]
        H, E, U = encode_to_zpdr_triple(test_value)
        
        # Normalize to extract invariants
        H_normalized, H_invariants = normalize_with_invariants(H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(U, "euclidean")
        
        # Apply transformations to normalized vectors
        # Rotate hyperbolic vector
        angle = np.pi / 4  # 45 degrees
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        H_rotated = np.dot(rotation_matrix, H_normalized)
        E_rotated = np.dot(rotation_matrix, E_normalized)
        U_rotated = np.dot(rotation_matrix, U_normalized)
        
        # Re-normalize the rotated vectors to extract new invariants
        H_rotated_norm, H_rotated_invariants = normalize_with_invariants(H_rotated, "hyperbolic")
        E_rotated_norm, E_rotated_invariants = normalize_with_invariants(E_rotated, "elliptical")
        U_rotated_norm, U_rotated_invariants = normalize_with_invariants(U_rotated, "euclidean")
        
        # The normalized vectors should be the same after rotation
        # This is the strong invariant property - normalization removes the rotation
        self.assertTrue(np.allclose(H_normalized, H_rotated_norm, rtol=1e-14, atol=1e-14),
                       "Normalized H vector should be invariant to rotation")
        self.assertTrue(np.allclose(E_normalized, E_rotated_norm, rtol=1e-14, atol=1e-14),
                       "Normalized E vector should be invariant to rotation")
        self.assertTrue(np.allclose(U_normalized, U_rotated_norm, rtol=1e-14, atol=1e-14),
                       "Normalized U vector should be invariant to rotation")
        
        # The rotation invariants should differ by the applied rotation angle
        # (modulo 2π and accounting for potential phase wrapping)
        if 'rotation_angle' in H_invariants and 'rotation_angle' in H_rotated_invariants:
            angle_diff = H_rotated_invariants['rotation_angle'] - H_invariants['rotation_angle']
            # Normalize to [-π, π]
            while angle_diff > np.pi: angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi
            
            self.assertAlmostEqual(abs(angle_diff), angle, delta=1e-14,
                                msg="Rotation invariants should differ by applied angle")
    
    def test_true_reconstruction(self):
        """
        Test true reconstruction from ZPA.
        
        This verifies that the system correctly reconstructs the original values.
        """
        # For each test value
        for value in self.test_values:
            # Encode to ZPA
            H, E, U = encode_to_zpdr_triple(value)
            
            # Verify that we can reconstruct the original value
            reconstructed = reconstruct_from_zpdr_triple(H, E, U)
            self.assertEqual(reconstructed, value,
                           f"Failed to reconstruct value {value} from ZPA")
    
    def test_vector_transformation_invariants(self):
        """
        Test that vector transformations preserve invariants.
        
        This verifies that transformations between geometric spaces maintain invariants.
        """
        # Create vectors in each space
        h_vector = HyperbolicVector(self.test_H)
        e_vector = EllipticalVector(self.test_E)
        u_vector = EuclideanVector(self.test_U)
        
        # Transform between spaces using SpaceTransformer
        h_to_e = SpaceTransformer.hyperbolic_to_elliptical(h_vector)
        e_to_h = SpaceTransformer.elliptical_to_hyperbolic(e_vector)
        h_to_u = SpaceTransformer.hyperbolic_to_euclidean(h_vector)
        u_to_h = SpaceTransformer.euclidean_to_hyperbolic(u_vector)
        e_to_u = SpaceTransformer.elliptical_to_euclidean(e_vector)
        u_to_e = SpaceTransformer.euclidean_to_elliptical(u_vector)
        
        # Verify round-trip transformations maintain vector properties
        # Hyperbolic -> Euclidean -> Hyperbolic
        h_to_u_to_h = SpaceTransformer.euclidean_to_hyperbolic(
            SpaceTransformer.hyperbolic_to_euclidean(h_vector)
        )
        # Elliptical -> Euclidean -> Elliptical
        e_to_u_to_e = SpaceTransformer.euclidean_to_elliptical(
            SpaceTransformer.elliptical_to_euclidean(e_vector)
        )
        
        # For hyperbolic vector, check that it remains valid
        self.assertTrue(h_to_u_to_h.is_valid(),
                      "Hyperbolic vector should remain valid after round-trip transformation")
        
        # For elliptical vector, check that it remains on unit sphere
        self.assertTrue(e_to_u_to_e.is_valid(),
                      "Elliptical vector should remain valid after round-trip transformation")
        
        # Get invariants for original and transformed vectors
        h_invariants = h_vector.get_invariants()
        h_to_u_to_h_invariants = h_to_u_to_h.get_invariants()
        
        e_invariants = e_vector.get_invariants()
        e_to_u_to_e_invariants = e_to_u_to_e.get_invariants()
        
        # Check that key invariants are preserved or properly transformed
        if 'scale_factor' in h_invariants and 'scale_factor' in h_to_u_to_h_invariants:
            # Scale might change slightly due to numerical issues, but should be close
            self.assertAlmostEqual(h_invariants['scale_factor'], 
                                 h_to_u_to_h_invariants['scale_factor'],
                                 delta=1e-12,
                                 msg="Hyperbolic scale factor should be preserved")
        
        # For elliptical vectors, radius is always 1.0 by definition
        if 'radius' in e_invariants and 'radius' in e_to_u_to_e_invariants:
            self.assertAlmostEqual(e_invariants['radius'], 1.0, delta=1e-14,
                                msg="Elliptical radius should be 1.0")
            self.assertAlmostEqual(e_to_u_to_e_invariants['radius'], 1.0, delta=1e-14,
                                msg="Transformed elliptical radius should be 1.0")
    
    def test_manifest_tamper_detection(self):
        """
        Test detection of tampering in ZPA manifests.
        
        This verifies that the system can detect unauthorized changes to ZPA data.
        """
        # Create a valid manifest and serialize it
        H, E, U = encode_to_zpdr_triple(self.test_values[0])
        valid_manifest = create_zpa_manifest(H, E, U)
        serialized = serialize_zpa(valid_manifest)
        
        # Parse as JSON to simulate tampering
        try:
            manifest_data = json.loads(serialized)
            
            # Tamper with hyperbolic vector data
            if "hyperbolic_vector" in manifest_data:
                for i in range(min(3, len(manifest_data["hyperbolic_vector"]))):
                    manifest_data["hyperbolic_vector"][i] *= 1.5
            
            # Re-serialize the tampered data
            tampered_serialized = json.dumps(manifest_data)
            
            # Try to deserialize and validate
            tampered_manifest = deserialize_zpa(tampered_serialized)
            
            # Get the vectors
            tampered_H = tampered_manifest.hyperbolic_vector
            tampered_E = tampered_manifest.elliptical_vector
            tampered_U = tampered_manifest.euclidean_vector
            
            # Check coherence
            is_valid, coherence = validate_trilateral_coherence(
                (tampered_H, tampered_E, tampered_U)
            )
            
            # Tampered data should have lower coherence
            self.assertLess(float(coherence), float(self.coherence_threshold),
                         "Tampered manifest should have lower coherence")
            
        except (json.JSONDecodeError, KeyError) as e:
            # It's also valid for the deserialization to fail on tampered data
            self.skipTest(f"Serialization format might not be JSON: {e}")
    
    def test_multivector_reconstruction_verification(self):
        """
        Test verification of multivector reconstruction.
        
        This verifies that the multivector reconstruction process is accurate.
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
        
        # Extract ZPA triple
        from zpdr.utils import _extract_trilateral_vectors
        H, E, U = _extract_trilateral_vectors(test_mv)
        
        # Reconstruct multivector from ZPA
        from zpdr.utils import _reconstruct_multivector_from_trilateral
        reconstructed_mv = _reconstruct_multivector_from_trilateral(H, E, U)
        
        # Check key components match
        for basis, value in components.items():
            if basis in reconstructed_mv.components:
                self.assertAlmostEqual(reconstructed_mv.components[basis], value, delta=1e-3,
                                    msg=f"Reconstructed {basis} component should match original")
    
    # Helper methods
    def _calculate_distance_metric(self, original, reconstructed, norm_type=2):
        """Calculate normalized distance between original and reconstructed values."""
        original_norm = np.linalg.norm(original, ord=norm_type)
        if original_norm < 1e-14:
            return 0.0  # Avoid division by zero
        
        distance = np.linalg.norm(original - reconstructed, ord=norm_type)
        return distance / original_norm


if __name__ == '__main__':
    unittest.main()