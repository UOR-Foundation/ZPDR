import unittest
import numpy as np
from decimal import Decimal, getcontext

# Set high precision for Decimal calculations
getcontext().prec = 100

# Import ZPDR components
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
    PRECISION_TOLERANCE
)

class TestZPAExtraction(unittest.TestCase):
    """
    Test suite for validating ZPA (Zero-Point Address) extraction algorithms in ZPDR Phase 2.
    
    This test suite focuses on the algorithms that extract the three geometric vectors
    (H, E, U) that form the ZPA, ensuring the extraction process is mathematically
    sound and preserves the key properties required by the ZPDR framework.
    """
    
    def setUp(self):
        """Set up test fixtures and constants needed for ZPA extraction testing."""
        # Define precision tolerance
        self.precision_tolerance = Decimal('1e-15')
        
        # Define test data
        self.test_values = [42, 101, 255, 1000, 65535]
        
        # Define test multivector (a reference point for consistent tests)
        self.test_multivector = self._create_test_multivector()
        
        # Define expectation thresholds
        self.coherence_threshold = Decimal('0.95')

    def test_hyperbolic_extraction_algorithm(self):
        """
        Test the algorithm that extracts the hyperbolic vector component (H) from a multivector.
        
        This tests the extraction of the base transformation system in hyperbolic space.
        """
        # Extract H component
        H_vector = self._extract_hyperbolic_vector(self.test_multivector)
        
        # Verify it's a proper numpy array with correct dimension
        self.assertIsInstance(H_vector, np.ndarray)
        self.assertEqual(len(H_vector), 3, "H vector should be 3-dimensional in Phase 2")
        
        # Verify hyperbolic properties
        # In hyperbolic space (Poincaré disk model), vectors should have norm < 1
        norm = np.linalg.norm(H_vector)
        self.assertLess(norm, 1.0, "H vector should have norm < 1 for Poincaré disk model")
        
        # Verify extraction is deterministic
        H_vector2 = self._extract_hyperbolic_vector(self.test_multivector)
        self.assertTrue(np.allclose(H_vector, H_vector2, rtol=1e-14, atol=1e-14),
                       "H extraction should be deterministic")
        
        # Verify geometric consistency
        hyperbolic_vector = HyperbolicVector(H_vector)
        self.assertTrue(hyperbolic_vector.is_valid(), 
                      "Extracted H vector should be valid in hyperbolic space")

    def test_elliptical_extraction_algorithm(self):
        """
        Test the algorithm that extracts the elliptical vector component (E) from a multivector.
        
        This tests the extraction of the transformation span in elliptical space.
        """
        # Extract E component
        E_vector = self._extract_elliptical_vector(self.test_multivector)
        
        # Verify it's a proper numpy array with correct dimension
        self.assertIsInstance(E_vector, np.ndarray)
        self.assertEqual(len(E_vector), 3, "E vector should be 3-dimensional in Phase 2")
        
        # Verify elliptical properties
        # In elliptical space (spherical model), vectors should have norm = 1
        norm = np.linalg.norm(E_vector)
        self.assertAlmostEqual(norm, 1.0, delta=1e-14,
                            msg="E vector should have norm = 1 for spherical model")
        
        # Verify extraction is deterministic
        E_vector2 = self._extract_elliptical_vector(self.test_multivector)
        self.assertTrue(np.allclose(E_vector, E_vector2, rtol=1e-14, atol=1e-14),
                       "E extraction should be deterministic")
        
        # Verify geometric consistency
        elliptical_vector = EllipticalVector(E_vector)
        self.assertTrue(elliptical_vector.is_valid(), 
                      "Extracted E vector should be valid in elliptical space")

    def test_euclidean_extraction_algorithm(self):
        """
        Test the algorithm that extracts the Euclidean vector component (U) from a multivector.
        
        This tests the extraction of the transformed object in Euclidean space.
        """
        # Extract U component
        U_vector = self._extract_euclidean_vector(self.test_multivector)
        
        # Verify it's a proper numpy array with correct dimension
        self.assertIsInstance(U_vector, np.ndarray)
        self.assertEqual(len(U_vector), 3, "U vector should be 3-dimensional in Phase 2")
        
        # Verify extraction is deterministic
        U_vector2 = self._extract_euclidean_vector(self.test_multivector)
        self.assertTrue(np.allclose(U_vector, U_vector2, rtol=1e-14, atol=1e-14),
                       "U extraction should be deterministic")
        
        # Verify geometric consistency
        euclidean_vector = EuclideanVector(U_vector)
        self.assertTrue(euclidean_vector.is_valid(), 
                      "Extracted U vector should be valid in Euclidean space")

    def test_complete_zpa_extraction(self):
        """
        Test the extraction of the complete ZPA (all three vectors) from a multivector.
        
        This tests the extraction of the entire Zero-Point Address.
        """
        # Extract the complete ZPA
        H, E, U = self._extract_zpa(self.test_multivector)
        
        # Verify all three components are properly extracted
        self.assertIsInstance(H, np.ndarray)
        self.assertIsInstance(E, np.ndarray)
        self.assertIsInstance(U, np.ndarray)
        
        # Verify dimensions
        self.assertEqual(len(H), 3, "H vector should be 3-dimensional")
        self.assertEqual(len(E), 3, "E vector should be 3-dimensional")
        self.assertEqual(len(U), 3, "U vector should be 3-dimensional")
        
        # Verify geometric properties
        self.assertLess(np.linalg.norm(H), 1.0, "H vector should have norm < 1 for Poincaré disk model")
        self.assertAlmostEqual(np.linalg.norm(E), 1.0, delta=1e-14, msg="E vector should have norm = 1 for spherical model")
        
        # Verify extraction is deterministic
        H2, E2, U2 = self._extract_zpa(self.test_multivector)
        self.assertTrue(np.allclose(H, H2, rtol=1e-14, atol=1e-14), "H extraction should be deterministic")
        self.assertTrue(np.allclose(E, E2, rtol=1e-14, atol=1e-14), "E extraction should be deterministic")
        self.assertTrue(np.allclose(U, U2, rtol=1e-14, atol=1e-14), "U extraction should be deterministic")

    def test_zpa_normalization(self):
        """
        Test that ZPA vectors are properly normalized to their canonical orientations.
        
        This tests the "zero-point" normalization process that is central to ZPDR.
        """
        # Extract the ZPA
        H, E, U = self._extract_zpa(self.test_multivector)
        
        # Normalize each vector to its canonical orientation with invariants
        H_normalized, H_invariants = normalize_with_invariants(H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(U, "euclidean")
        
        # Verify normalization results are valid
        self.assertTrue(np.all(np.isfinite(H_normalized)), "Normalized H should have valid values")
        self.assertTrue(np.all(np.isfinite(E_normalized)), "Normalized E should have valid values")
        self.assertTrue(np.all(np.isfinite(U_normalized)), "Normalized U should have valid values")
        
        # Verify invariants are extracted
        self.assertIn('rotation_angle', H_invariants, "H invariants should include rotation angle")
        self.assertIn('rotation_angle', E_invariants, "E invariants should include rotation angle")
        self.assertIn('rotation_angle', U_invariants, "U invariants should include rotation angle")
        
        # Verify normalization is consistent
        H_normalized2, _ = normalize_with_invariants(H, "hyperbolic")
        E_normalized2, _ = normalize_with_invariants(E, "elliptical")
        U_normalized2, _ = normalize_with_invariants(U, "euclidean")
        
        self.assertTrue(np.allclose(H_normalized, H_normalized2, rtol=1e-14, atol=1e-14),
                      "H normalization should be consistent")
        self.assertTrue(np.allclose(E_normalized, E_normalized2, rtol=1e-14, atol=1e-14),
                      "E normalization should be consistent")
        self.assertTrue(np.allclose(U_normalized, U_normalized2, rtol=1e-14, atol=1e-14),
                      "U normalization should be consistent")
        
        # Verify we can denormalize back to original
        H_denormalized = denormalize_with_invariants(H_normalized, H_invariants, "hyperbolic")
        E_denormalized = denormalize_with_invariants(E_normalized, E_invariants, "elliptical")
        U_denormalized = denormalize_with_invariants(U_normalized, U_invariants, "euclidean")
        
        self.assertTrue(np.allclose(H, H_denormalized, rtol=1e-14, atol=1e-14),
                      "H denormalization should recover original vector")
        self.assertTrue(np.allclose(E, E_denormalized, rtol=1e-14, atol=1e-14),
                      "E denormalization should recover original vector")
        self.assertTrue(np.allclose(U, U_denormalized, rtol=1e-14, atol=1e-14),
                      "U denormalization should recover original vector")

    def test_extracted_zpa_coherence(self):
        """
        Test that extracted ZPA components have proper coherence.
        
        This tests that the H, E, U components form a coherent trilateral vector system.
        """
        # Extract the ZPA
        H, E, U = self._extract_zpa(self.test_multivector)
        
        # Calculate internal coherence for each vector
        H_coherence = calculate_internal_coherence(H)
        E_coherence = calculate_internal_coherence(E)
        U_coherence = calculate_internal_coherence(U)
        
        # Verify internal coherence meets threshold
        self.assertGreaterEqual(float(H_coherence), float(self.coherence_threshold),
                               "H vector should have high internal coherence")
        self.assertGreaterEqual(float(E_coherence), float(self.coherence_threshold),
                               "E vector should have high internal coherence")
        self.assertGreaterEqual(float(U_coherence), float(self.coherence_threshold),
                               "U vector should have high internal coherence")
        
        # Calculate cross-coherence between vector pairs
        HE_coherence = calculate_cross_coherence(H, E)
        EU_coherence = calculate_cross_coherence(E, U)
        HU_coherence = calculate_cross_coherence(H, U)
        
        # Verify cross-coherence meets threshold
        self.assertGreaterEqual(float(HE_coherence), float(self.coherence_threshold),
                               "H and E vectors should have high cross-coherence")
        self.assertGreaterEqual(float(EU_coherence), float(self.coherence_threshold),
                               "E and U vectors should have high cross-coherence")
        self.assertGreaterEqual(float(HU_coherence), float(self.coherence_threshold),
                               "H and U vectors should have high cross-coherence")
        
        # Calculate global coherence
        global_coherence = calculate_global_coherence(
            [H_coherence, E_coherence, U_coherence],
            [HE_coherence, EU_coherence, HU_coherence]
        )
        
        # Verify global coherence meets threshold
        self.assertGreaterEqual(float(global_coherence), float(self.coherence_threshold),
                               "ZPA should have high global coherence")
        
        # Verify using the validation utility
        is_valid, coherence = validate_trilateral_coherence((H, E, U))
        self.assertTrue(is_valid, "ZPA should be valid according to coherence validation")
        self.assertGreaterEqual(float(coherence), float(self.coherence_threshold),
                               "ZPA validation should report high coherence")

    def test_zpa_extraction_for_different_values(self):
        """
        Test ZPA extraction for different values to ensure consistent behavior.
        
        This tests that the extraction algorithm works across various inputs.
        """
        # Test with a range of values
        zpas = []
        for value in self.test_values:
            # Create a multivector for the value
            multivector = self._create_multivector_for_value(value)
            
            # Extract ZPA
            H, E, U = self._extract_zpa(multivector)
            
            # Verify basic properties
            self.assertEqual(len(H), 3, f"H vector should be 3D for value {value}")
            self.assertEqual(len(E), 3, f"E vector should be 3D for value {value}")
            self.assertEqual(len(U), 3, f"U vector should be 3D for value {value}")
            
            # Verify geometric properties
            self.assertLess(np.linalg.norm(H), 1.0, 
                          f"H vector should have norm < 1 for value {value}")
            self.assertAlmostEqual(np.linalg.norm(E), 1.0, delta=1e-14, 
                                 msg=f"E vector should have norm = 1 for value {value}")
            
            # Calculate coherence
            is_valid, coherence = validate_trilateral_coherence((H, E, U))
            self.assertTrue(is_valid, f"ZPA should be valid for value {value}")
            self.assertGreaterEqual(float(coherence), float(self.coherence_threshold),
                                  f"ZPA should have high coherence for value {value}")
            
            # Store the ZPA
            zpas.append((H, E, U))
        
        # Verify ZPAs are distinct for different values
        for i in range(len(zpas)):
            for j in range(i+1, len(zpas)):
                # Calculate distance between ZPAs
                distance = self._calculate_zpa_distance(zpas[i], zpas[j])
                
                # ZPAs for different values should be distinct
                self.assertGreater(distance, 0.1, 
                                 f"ZPAs for values {self.test_values[i]} and {self.test_values[j]} should be distinct")

    def test_zpa_extraction_with_rotations(self):
        """
        Test that ZPA extraction correctly handles rotations in the input multivector.
        
        This tests the rotation invariance property of the zero-point extraction.
        """
        # Extract the original ZPA
        original_H, original_E, original_U = self._extract_zpa(self.test_multivector)
        
        # Apply a rotation to the multivector
        rotated_multivector = self._apply_rotation_to_multivector(self.test_multivector, np.pi/4)
        
        # Extract ZPA from rotated multivector
        rotated_H, rotated_E, rotated_U = self._extract_zpa(rotated_multivector)
        
        # Normalize both ZPAs to zero-point orientation
        original_H_norm, _ = normalize_with_invariants(original_H, "hyperbolic")
        original_E_norm, _ = normalize_with_invariants(original_E, "elliptical")
        original_U_norm, _ = normalize_with_invariants(original_U, "euclidean")
        
        rotated_H_norm, _ = normalize_with_invariants(rotated_H, "hyperbolic")
        rotated_E_norm, _ = normalize_with_invariants(rotated_E, "elliptical")
        rotated_U_norm, _ = normalize_with_invariants(rotated_U, "euclidean")
        
        # Verify that normalized vectors are the same (rotation invariance)
        self.assertTrue(np.allclose(original_H_norm, rotated_H_norm, rtol=1e-14, atol=1e-14),
                      "Normalized H vectors should be the same regardless of input rotation")
        self.assertTrue(np.allclose(original_E_norm, rotated_E_norm, rtol=1e-14, atol=1e-14),
                      "Normalized E vectors should be the same regardless of input rotation")
        self.assertTrue(np.allclose(original_U_norm, rotated_U_norm, rtol=1e-14, atol=1e-14),
                      "Normalized U vectors should be the same regardless of input rotation")

    def test_multivector_reconstruction_from_zpa(self):
        """
        Test reconstruction of a multivector from its ZPA.
        
        This tests the inverse of the extraction process, ensuring the ZPA contains
        all necessary information to reconstruct the original multivector.
        """
        # Extract the ZPA
        H, E, U = self._extract_zpa(self.test_multivector)
        
        # Normalize with invariants
        H_normalized, H_invariants = normalize_with_invariants(H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(U, "euclidean")
        
        # Reconstruct the multivector
        reconstructed_multivector = self._reconstruct_multivector_from_zpa(
            H_normalized, E_normalized, U_normalized, 
            H_invariants, E_invariants, U_invariants
        )
        
        # Verify the reconstructed multivector has the correct structure
        self.assertIsInstance(reconstructed_multivector, Multivector)
        
        # Extract ZPA from reconstructed multivector
        reconstructed_H, reconstructed_E, reconstructed_U = self._extract_zpa(reconstructed_multivector)
        
        # Verify ZPA components are preserved
        self.assertTrue(np.allclose(H, reconstructed_H, rtol=1e-10, atol=1e-10),
                      "H vector should be preserved in reconstruction")
        self.assertTrue(np.allclose(E, reconstructed_E, rtol=1e-10, atol=1e-10),
                      "E vector should be preserved in reconstruction")
        self.assertTrue(np.allclose(U, reconstructed_U, rtol=1e-10, atol=1e-10),
                      "U vector should be preserved in reconstruction")
        
        # Calculate distance between original and reconstructed multivectors
        distance = self._calculate_multivector_distance(self.test_multivector, reconstructed_multivector)
        
        # Verify reconstruction is close to original
        self.assertLess(distance, 0.1, "Reconstructed multivector should be close to original")

    # Helper methods for the tests
    def _create_test_multivector(self):
        """Create a consistent test multivector for the tests."""
        # Create a multivector with non-zero components in all grades
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

    def _create_multivector_for_value(self, value):
        """Create a multivector representation for a given value."""
        # Convert to Decimal for high precision
        dec_value = Decimal(str(value))
        
        # Create base representations for the number
        base2_digits = self._number_to_base(value, 2)
        base10_digits = self._number_to_base(value, 10)
        
        # Initialize components dictionary
        components = {}
        
        # Scalar part: normalized magnitude
        bit_length = value.bit_length()
        scale_factor = Decimal(str(1.0)) / Decimal(str(max(1, bit_length)))
        components["1"] = float(scale_factor * dec_value)
        
        # Vector part: derived from binary representation
        if base2_digits:
            vector_scale = min(1.0, 3.0 / len(base2_digits))
            for i, digit in enumerate(base2_digits[:3]):  # Use first 3 digits
                if i < 3:  # Limit to 3D vector
                    basis = f"e{i+1}"
                    components[basis] = digit * vector_scale
        
        # Bivector part: derived from digit pairs
        if len(base10_digits) >= 2:
            bivector_bases = ["e12", "e23", "e31"]
            pairs = list(zip(base10_digits[:-1], base10_digits[1:]))
            for i, (d1, d2) in enumerate(pairs[:3]):  # Use first 3 pairs
                if i < 3:  # Limit to 3 bivectors
                    basis = bivector_bases[i]
                    # Simple digit pair encoding
                    components[basis] = float((d1*10 + d2) / 100.0)  # Normalize to [0,1]
        
        # Trivector part: overall scale
        components["e123"] = float(dec_value / Decimal(str(10 ** len(base10_digits))))
        
        # Create and return the multivector
        return Multivector(components)

    def _extract_hyperbolic_vector(self, multivector):
        """Extract the hyperbolic vector component (H) from a multivector."""
        # Phase 2 implementation of H extraction
        # H is derived primarily from scalar and the first two vector components
        
        # Initialize the H vector
        H = np.zeros(3, dtype=np.float64)
        
        # Extract scalar component
        scalar = multivector.scalar_part()
        H[0] = scalar
        
        # Extract vector components for H
        vector_part = multivector.vector_part()
        if "e1" in vector_part.components:
            H[1] = vector_part.components["e1"]
        if "e2" in vector_part.components:
            H[2] = vector_part.components["e2"]
        
        # Ensure the vector is valid in hyperbolic space (norm < 1)
        norm = np.linalg.norm(H)
        if norm >= 1.0:
            H = H * 0.99 / norm  # Scale to ensure it's within the Poincaré disk
        
        return H

    def _extract_elliptical_vector(self, multivector):
        """Extract the elliptical vector component (E) from a multivector."""
        # Phase 2 implementation of E extraction
        # E is derived primarily from bivector components
        
        # Initialize the E vector
        E = np.zeros(3, dtype=np.float64)
        
        # Extract bivector components for E
        bivector_part = multivector.bivector_part()
        
        if "e12" in bivector_part.components:
            E[0] = bivector_part.components["e12"]
        if "e23" in bivector_part.components:
            E[1] = bivector_part.components["e23"]
        if "e31" in bivector_part.components:
            E[2] = bivector_part.components["e31"]
        
        # Normalize to ensure it's on the unit sphere (norm = 1)
        norm = np.linalg.norm(E)
        if norm > 0:
            E = E / norm
        else:
            # Default to a standard unit vector if all zeros
            E[0] = 1.0
        
        return E

    def _extract_euclidean_vector(self, multivector):
        """Extract the Euclidean vector component (U) from a multivector."""
        # Phase 2 implementation of U extraction
        # U is derived from the second and third vector components, and the trivector
        
        # Initialize the U vector
        U = np.zeros(3, dtype=np.float64)
        
        # Extract vector components for U
        vector_part = multivector.vector_part()
        if "e2" in vector_part.components:
            U[0] = vector_part.components["e2"]
        if "e3" in vector_part.components:
            U[1] = vector_part.components["e3"]
        
        # Extract trivector component for U[2]
        trivector_part = multivector.trivector_part()
        if "e123" in trivector_part.components:
            U[2] = trivector_part.components["e123"]
        
        return U

    def _extract_zpa(self, multivector):
        """Extract the complete ZPA from a multivector."""
        # Extract the three components
        H = self._extract_hyperbolic_vector(multivector)
        E = self._extract_elliptical_vector(multivector)
        U = self._extract_euclidean_vector(multivector)
        
        return H, E, U

    def _apply_rotation_to_multivector(self, multivector, angle):
        """Apply a rotation to a multivector."""
        # Create a copy of the components dict
        components = multivector.components.copy()
        
        # Create rotation matrix for the xy-plane
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to vector components
        vector_part = multivector.vector_part()
        vector = np.zeros(3)
        for i in range(3):
            basis = f"e{i+1}"
            if basis in vector_part.components:
                vector[i] = vector_part.components[basis]
        
        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        
        # Update vector components
        for i in range(3):
            basis = f"e{i+1}"
            components[basis] = rotated_vector[i]
        
        # Apply rotation to bivector components
        bivector_part = multivector.bivector_part()
        bivector = np.zeros(3)
        bases = ["e12", "e23", "e31"]
        for i, basis in enumerate(bases):
            if basis in bivector_part.components:
                bivector[i] = bivector_part.components[basis]
        
        # Rotate the bivector
        rotated_bivector = np.dot(rotation_matrix, bivector)
        
        # Update bivector components
        for i, basis in enumerate(bases):
            components[basis] = rotated_bivector[i]
        
        # Create a new multivector with rotated components
        return Multivector(components)

    def _reconstruct_multivector_from_zpa(self, H_normalized, E_normalized, U_normalized, 
                                          H_invariants, E_invariants, U_invariants):
        """Reconstruct a multivector from a normalized ZPA with invariants."""
        # Apply invariants to get original vectors
        H = denormalize_with_invariants(H_normalized, H_invariants, "hyperbolic")
        E = denormalize_with_invariants(E_normalized, E_invariants, "elliptical")
        U = denormalize_with_invariants(U_normalized, U_invariants, "euclidean")
        
        # Create components for the multivector
        components = {}
        
        # Scalar part from H[0]
        components["1"] = H[0]
        
        # Vector parts from H and U
        components["e1"] = H[1]  # First vector component from H[1]
        components["e2"] = H[2]  # Second vector component from H[2]
        components["e3"] = U[1]  # Third vector component from U[1]
        
        # Bivector parts from E
        components["e12"] = E[0]
        components["e23"] = E[1]
        components["e31"] = E[2]
        
        # Trivector part from U[2]
        components["e123"] = U[2]
        
        # Create and return the multivector
        return Multivector(components)

    def _calculate_zpa_distance(self, zpa1, zpa2):
        """Calculate distance between two ZPAs."""
        H1, E1, U1 = zpa1
        H2, E2, U2 = zpa2
        
        # Calculate distances for each component
        H_distance = np.linalg.norm(H1 - H2)
        E_distance = np.linalg.norm(E1 - E2)
        U_distance = np.linalg.norm(U1 - U2)
        
        # Combine distances with weights
        weights = [0.4, 0.3, 0.3]  # Equal importance for now
        total_distance = weights[0] * H_distance + weights[1] * E_distance + weights[2] * U_distance
        
        return total_distance

    def _calculate_multivector_distance(self, mv1, mv2):
        """Calculate distance between two multivectors."""
        # Extract components by grade
        scalar1 = mv1.scalar_part()
        vector1 = mv1.vector_part()
        bivector1 = mv1.bivector_part()
        trivector1 = mv1.trivector_part()
        
        scalar2 = mv2.scalar_part()
        vector2 = mv2.vector_part()
        bivector2 = mv2.bivector_part()
        trivector2 = mv2.trivector_part()
        
        # Distance in scalar part
        scalar_distance = abs(scalar1 - scalar2)
        
        # Distance in vector part
        vector_distance = 0
        for i in range(1, 4):
            basis = f"e{i}"
            v1 = vector1.components.get(basis, 0)
            v2 = vector2.components.get(basis, 0)
            vector_distance += (v1 - v2) ** 2
        vector_distance = np.sqrt(vector_distance)
        
        # Distance in bivector part
        bivector_distance = 0
        for basis in ["e12", "e23", "e31"]:
            b1 = bivector1.components.get(basis, 0)
            b2 = bivector2.components.get(basis, 0)
            bivector_distance += (b1 - b2) ** 2
        bivector_distance = np.sqrt(bivector_distance)
        
        # Distance in trivector part
        trivector_distance = 0
        t1 = trivector1.components.get("e123", 0)
        t2 = trivector2.components.get("e123", 0)
        trivector_distance = abs(t1 - t2)
        
        # Combine distances with weights
        weights = [0.4, 0.3, 0.2, 0.1]  # Prioritize lower grades
        total_distance = (
            weights[0] * scalar_distance +
            weights[1] * vector_distance +
            weights[2] * bivector_distance +
            weights[3] * trivector_distance
        )
        
        return total_distance

    def _number_to_base(self, n, base):
        """Convert a number to a list of digits in a given base."""
        if n == 0:
            return [0]
            
        digits = []
        while n:
            digits.append(int(n % base))
            n //= base
            
        return list(reversed(digits))


if __name__ == '__main__':
    unittest.main()