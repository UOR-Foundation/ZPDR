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
    calculate_multibase_coherence,
    COHERENCE_THRESHOLD,
    PRECISION_TOLERANCE
)

class TestZPDRCoherence(unittest.TestCase):
    """
    Test suite for validating the coherence calculations in ZPDR Phase 2.
    
    This test suite focuses on the mathematical properties of coherence measures
    used to validate the integrity and consistency of ZPDR representations.
    """
    
    def setUp(self):
        """Set up test fixtures and constants needed for coherence testing."""
        # Define precision tolerance
        self.precision_tolerance = Decimal('1e-15')
        
        # Define coherence threshold
        self.coherence_threshold = COHERENCE_THRESHOLD
        
        # Define test vectors
        self.test_H = np.array([0.125, 0.25, 0.5], dtype=np.float64)
        self.test_E = np.array([0.7071, 0.7071, 0], dtype=np.float64)
        self.test_U = np.array([0.333, 0.333, 0.333], dtype=np.float64)
        
        # Define incoherent vectors for testing
        self.incoherent_H = np.array([0.9, 0.01, 0.01], dtype=np.float64)
        self.incoherent_E = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.incoherent_U = np.array([0.99, 0.01, 0.0], dtype=np.float64)
        
        # Define test base representations
        self.test_base2 = [1, 0, 1, 0, 1, 0]  # Binary 42
        self.test_base10 = [4, 2]             # Decimal 42

    def test_internal_coherence_calculation(self):
        """
        Test the algorithm for calculating internal coherence of a vector.
        
        Internal coherence measures the consistency of a vector's components.
        """
        # Calculate internal coherence for test vectors
        H_coherence = calculate_internal_coherence(self.test_H)
        E_coherence = calculate_internal_coherence(self.test_E)
        U_coherence = calculate_internal_coherence(self.test_U)
        
        # Verify results are valid Decimals in [0,1] range
        self.assertIsInstance(H_coherence, Decimal)
        self.assertIsInstance(E_coherence, Decimal)
        self.assertIsInstance(U_coherence, Decimal)
        
        self.assertGreaterEqual(H_coherence, Decimal('0'))
        self.assertLessEqual(H_coherence, Decimal('1'))
        self.assertGreaterEqual(E_coherence, Decimal('0'))
        self.assertLessEqual(E_coherence, Decimal('1'))
        self.assertGreaterEqual(U_coherence, Decimal('0'))
        self.assertLessEqual(U_coherence, Decimal('1'))
        
        # Verify test vectors have high coherence (they are designed to be coherent)
        self.assertGreaterEqual(H_coherence, self.coherence_threshold)
        self.assertGreaterEqual(E_coherence, self.coherence_threshold)
        self.assertGreaterEqual(U_coherence, self.coherence_threshold)
        
        # Calculate internal coherence for incoherent vectors
        incoherent_H_coherence = calculate_internal_coherence(self.incoherent_H)
        incoherent_E_coherence = calculate_internal_coherence(self.incoherent_E)
        incoherent_U_coherence = calculate_internal_coherence(self.incoherent_U)
        
        # At least one incoherent vector should have coherence below threshold
        # This verifies that the function properly detects incoherence
        below_threshold = (
            incoherent_H_coherence < self.coherence_threshold or
            incoherent_E_coherence < self.coherence_threshold or
            incoherent_U_coherence < self.coherence_threshold
        )
        self.assertTrue(below_threshold, 
                      "At least one incoherent vector should have coherence below threshold")

    def test_cross_coherence_calculation(self):
        """
        Test the algorithm for calculating cross-coherence between vector pairs.
        
        Cross-coherence measures the relationship between vectors from different spaces.
        """
        # Calculate cross-coherence for test vector pairs
        HE_coherence = calculate_cross_coherence(self.test_H, self.test_E)
        EU_coherence = calculate_cross_coherence(self.test_E, self.test_U)
        HU_coherence = calculate_cross_coherence(self.test_H, self.test_U)
        
        # Verify results are valid Decimals in [0,1] range
        self.assertIsInstance(HE_coherence, Decimal)
        self.assertIsInstance(EU_coherence, Decimal)
        self.assertIsInstance(HU_coherence, Decimal)
        
        self.assertGreaterEqual(HE_coherence, Decimal('0'))
        self.assertLessEqual(HE_coherence, Decimal('1'))
        self.assertGreaterEqual(EU_coherence, Decimal('0'))
        self.assertLessEqual(EU_coherence, Decimal('1'))
        self.assertGreaterEqual(HU_coherence, Decimal('0'))
        self.assertLessEqual(HU_coherence, Decimal('1'))
        
        # Test vectors should have high cross-coherence
        self.assertGreaterEqual(HE_coherence, self.coherence_threshold)
        self.assertGreaterEqual(EU_coherence, self.coherence_threshold)
        self.assertGreaterEqual(HU_coherence, self.coherence_threshold)
        
        # Calculate cross-coherence for incoherent vector pairs
        incoherent_HE = calculate_cross_coherence(self.incoherent_H, self.incoherent_E)
        incoherent_EU = calculate_cross_coherence(self.incoherent_E, self.incoherent_U)
        incoherent_HU = calculate_cross_coherence(self.incoherent_H, self.incoherent_U)
        
        # At least one incoherent pair should have coherence below threshold
        below_threshold = (
            incoherent_HE < self.coherence_threshold or
            incoherent_EU < self.coherence_threshold or
            incoherent_HU < self.coherence_threshold
        )
        self.assertTrue(below_threshold, 
                      "At least one incoherent vector pair should have coherence below threshold")
        
        # Verify symmetry property: cross_coherence(a,b) = cross_coherence(b,a)
        HE_reverse = calculate_cross_coherence(self.test_E, self.test_H)
        self.assertAlmostEqual(float(HE_coherence), float(HE_reverse), delta=float(self.precision_tolerance),
                            msg="Cross-coherence should be symmetric")

    def test_global_coherence_calculation(self):
        """
        Test the algorithm for calculating global coherence from individual measures.
        
        Global coherence combines internal and cross-coherence measures.
        """
        # Internal coherence values
        H_coherence = calculate_internal_coherence(self.test_H)
        E_coherence = calculate_internal_coherence(self.test_E)
        U_coherence = calculate_internal_coherence(self.test_U)
        
        # Cross-coherence values
        HE_coherence = calculate_cross_coherence(self.test_H, self.test_E)
        EU_coherence = calculate_cross_coherence(self.test_E, self.test_U)
        HU_coherence = calculate_cross_coherence(self.test_H, self.test_U)
        
        # Calculate global coherence
        global_coherence = calculate_global_coherence(
            [H_coherence, E_coherence, U_coherence],
            [HE_coherence, EU_coherence, HU_coherence]
        )
        
        # Verify result is a valid Decimal in [0,1] range
        self.assertIsInstance(global_coherence, Decimal)
        self.assertGreaterEqual(global_coherence, Decimal('0'))
        self.assertLessEqual(global_coherence, Decimal('1'))
        
        # Test vectors should have high global coherence
        self.assertGreaterEqual(global_coherence, self.coherence_threshold)
        
        # Verify global coherence behaves correctly with low input values
        # Create a mix of high and low coherence inputs
        mixed_internal = [H_coherence, E_coherence, Decimal('0.5')]
        mixed_cross = [HE_coherence, Decimal('0.6'), HU_coherence]
        
        mixed_global = calculate_global_coherence(mixed_internal, mixed_cross)
        
        # Mixed inputs should result in lower global coherence
        self.assertLess(mixed_global, global_coherence)
        
        # If all inputs are below threshold, global coherence should be below threshold
        low_internal = [Decimal('0.5'), Decimal('0.6'), Decimal('0.7')]
        low_cross = [Decimal('0.6'), Decimal('0.5'), Decimal('0.7')]
        
        low_global = calculate_global_coherence(low_internal, low_cross)
        self.assertLess(low_global, self.coherence_threshold)

    def test_trilateral_coherence_validation(self):
        """
        Test the validation of coherence for a complete trilateral vector (H, E, U).
        
        This tests the overall validity assessment of a ZPA triple.
        """
        # Test with coherent vectors
        is_valid, coherence = validate_trilateral_coherence((self.test_H, self.test_E, self.test_U))
        
        # Verify validation succeeds with high coherence
        self.assertTrue(is_valid, "Coherent vectors should validate successfully")
        self.assertGreaterEqual(coherence, self.coherence_threshold)
        
        # Test with incoherent vectors
        is_valid, coherence = validate_trilateral_coherence(
            (self.incoherent_H, self.incoherent_E, self.incoherent_U)
        )
        
        # Verification should fail for incoherent vectors
        self.assertFalse(is_valid, "Incoherent vectors should fail validation")
        self.assertLess(coherence, self.coherence_threshold)
        
        # Test with mixed vectors (should still be coherent)
        is_valid, coherence = validate_trilateral_coherence((self.test_H, self.test_E, self.incoherent_U))
        
        # Outcome depends on actual implementation, but coherence should be affected
        self.assertLess(coherence, Decimal('1.0'), 
                      "Mixed vectors should have reduced coherence")

    def test_multibase_coherence_calculation(self):
        """
        Test the calculation of coherence between multiple base representations.
        
        This tests the base-independence property of ZPDR encodings.
        """
        # Calculate coherence between base representations
        base_coherence = calculate_multibase_coherence(self.test_base2, self.test_base10)
        
        # Verify result is a valid Decimal in [0,1] range
        self.assertIsInstance(base_coherence, Decimal)
        self.assertGreaterEqual(base_coherence, Decimal('0'))
        self.assertLessEqual(base_coherence, Decimal('1'))
        
        # These representations of 42 should have high coherence
        self.assertGreaterEqual(base_coherence, self.coherence_threshold)
        
        # Create a different but coherent base representation
        base8 = [5, 2]  # 42 in octal
        
        # Calculate coherence with octal representation
        base2_8_coherence = calculate_multibase_coherence(self.test_base2, base8)
        base10_8_coherence = calculate_multibase_coherence(self.test_base10, base8)
        
        # These should all be coherent
        self.assertGreaterEqual(base2_8_coherence, self.coherence_threshold)
        self.assertGreaterEqual(base10_8_coherence, self.coherence_threshold)
        
        # Test with incoherent base representation
        incoherent_base = [9, 9, 9]  # Not a valid representation of 42
        
        incoherent_coherence = calculate_multibase_coherence(self.test_base10, incoherent_base)
        
        # Coherence should be lower
        self.assertLess(incoherent_coherence, self.coherence_threshold)

    def test_coherence_mathematical_properties(self):
        """
        Test mathematical properties of the coherence measures.
        
        This ensures coherence calculations behave as expected mathematically.
        """
        # Test internal coherence with mathematically-derived patterns
        
        # Uniform vector should have high coherence
        uniform = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
        uniform_coherence = calculate_internal_coherence(uniform)
        self.assertGreaterEqual(uniform_coherence, self.coherence_threshold)
        
        # Vector with one dominant component should have lower coherence
        skewed = np.array([0.99, 0.01, 0.0])
        skewed_coherence = calculate_internal_coherence(skewed)
        self.assertLess(skewed_coherence, uniform_coherence)
        
        # Test cross-coherence with mathematically-derived patterns
        
        # Parallel vectors should have high cross-coherence
        v1 = np.array([0.6, 0.8, 0.0])
        v2 = np.array([0.3, 0.4, 0.0])  # Parallel to v1
        parallel_coherence = calculate_cross_coherence(v1, v2)
        self.assertGreaterEqual(parallel_coherence, self.coherence_threshold)
        
        # Orthogonal vectors should have low cross-coherence
        v3 = np.array([0.0, 0.0, 1.0])  # Orthogonal to v1 and v2
        orthogonal_coherence = calculate_cross_coherence(v1, v3)
        self.assertLess(orthogonal_coherence, parallel_coherence)
        
        # Test global coherence with mathematically-derived patterns
        
        # Perfectly aligned system should have perfect coherence
        internal_perfect = [Decimal('1.0'), Decimal('1.0'), Decimal('1.0')]
        cross_perfect = [Decimal('1.0'), Decimal('1.0'), Decimal('1.0')]
        perfect_global = calculate_global_coherence(internal_perfect, cross_perfect)
        self.assertAlmostEqual(float(perfect_global), 1.0, delta=float(self.precision_tolerance))
        
        # If one of the internal or cross coherences is zero, global coherence should be reduced
        internal_mixed = [Decimal('1.0'), Decimal('1.0'), Decimal('0.0')]
        cross_mixed = [Decimal('1.0'), Decimal('1.0'), Decimal('1.0')]
        mixed_global = calculate_global_coherence(internal_mixed, cross_mixed)
        self.assertLess(mixed_global, perfect_global)

    def test_coherence_numerical_stability(self):
        """
        Test the numerical stability of coherence calculations.
        
        This ensures coherence measures handle extreme cases correctly.
        """
        # Test with zero vectors
        zero_vector = np.zeros(3)
        
        # Internal coherence of zero vector
        zero_coherence = calculate_internal_coherence(zero_vector)
        self.assertIsInstance(zero_coherence, Decimal, "Zero vector coherence should return a Decimal")
        self.assertGreaterEqual(zero_coherence, Decimal('0'))
        self.assertLessEqual(zero_coherence, Decimal('1'))
        
        # Cross-coherence with zero vector
        cross_zero = calculate_cross_coherence(self.test_H, zero_vector)
        self.assertIsInstance(cross_zero, Decimal, "Cross-coherence with zero should return a Decimal")
        self.assertGreaterEqual(cross_zero, Decimal('0'))
        self.assertLessEqual(cross_zero, Decimal('1'))
        
        # Test with very small vectors
        tiny_vector = np.array([1e-14, 1e-15, 1e-16])
        
        # Internal coherence of tiny vector
        tiny_coherence = calculate_internal_coherence(tiny_vector)
        self.assertIsInstance(tiny_coherence, Decimal, "Tiny vector coherence should return a Decimal")
        self.assertGreaterEqual(tiny_coherence, Decimal('0'))
        self.assertLessEqual(tiny_coherence, Decimal('1'))
        
        # Test with NaN or Inf values - should not crash
        nan_vector = np.array([np.nan, 0, 0])
        
        # These should not raise exceptions
        try:
            calculate_internal_coherence(nan_vector)
            calculate_cross_coherence(nan_vector, self.test_H)
        except:
            self.fail("Coherence calculations should handle NaN values gracefully")
        
        # Test global coherence with extreme inputs
        extreme_internal = [Decimal('1.0'), Decimal('0.0'), Decimal('0.0')]
        extreme_cross = [Decimal('0.0'), Decimal('0.0'), Decimal('1.0')]
        
        extreme_global = calculate_global_coherence(extreme_internal, extreme_cross)
        self.assertIsInstance(extreme_global, Decimal)
        self.assertGreaterEqual(extreme_global, Decimal('0'))
        self.assertLessEqual(extreme_global, Decimal('1'))

    def test_coherence_threshold_significance(self):
        """
        Test the significance of the coherence threshold.
        
        This ensures the threshold properly separates coherent from incoherent systems.
        """
        # Create a series of increasingly coherent vectors
        coherence_levels = []
        
        # Generate vectors with increasing coherence
        for i in range(1, 11):
            level = i / 10.0  # Coherence level from 0.1 to 1.0
            
            # Generate internal and cross coherence values at this level
            internal = [Decimal(str(level))] * 3
            cross = [Decimal(str(level))] * 3
            
            # Calculate global coherence
            global_coherence = calculate_global_coherence(internal, cross)
            coherence_levels.append((level, float(global_coherence)))
        
        # Verify monotonicity - higher input should yield higher output
        for i in range(1, len(coherence_levels)):
            prev_level, prev_coherence = coherence_levels[i-1]
            curr_level, curr_coherence = coherence_levels[i]
            
            self.assertGreater(curr_coherence, prev_coherence,
                             "Higher coherence inputs should yield higher global coherence")
        
        # Find the point where the global coherence crosses the threshold
        threshold_level = None
        for level, global_coherence in coherence_levels:
            if global_coherence >= float(self.coherence_threshold):
                threshold_level = level
                break
        
        # There should be a threshold level
        self.assertIsNotNone(threshold_level, 
                           "There should be an input level that produces global coherence above threshold")
        
        # Check that threshold is in a reasonable range for ZPDR system
        self.assertGreaterEqual(threshold_level, 0.7,
                              "Coherence threshold should require fairly high input coherence")

    def test_zero_point_coherence_invariants(self):
        """
        Test coherence invariants under zero-point normalization.
        
        This ensures coherence is preserved through normalization.
        """
        # Create hyperbolic vectors with different orientations but same structure
        H1 = HyperbolicVector(self.test_H)
        
        # Create a rotated version
        H2_components = np.array([-self.test_H[0], -self.test_H[1], self.test_H[2]])
        H2 = HyperbolicVector(H2_components)
        
        # Calculate internal coherence for both
        H1_coherence = calculate_internal_coherence(H1.components)
        H2_coherence = calculate_internal_coherence(H2.components)
        
        # Verify rotations don't affect internal coherence
        self.assertAlmostEqual(float(H1_coherence), float(H2_coherence), delta=0.05,
                            msg="Internal coherence should be similar for differently oriented vectors")
        
        # Similar test for elliptical vectors
        E1 = EllipticalVector(self.test_E)
        
        # Create a rotated version
        E2_components = np.array([self.test_E[1], -self.test_E[0], self.test_E[2]])
        E2 = EllipticalVector(E2_components)
        
        # Calculate internal coherence for both
        E1_coherence = calculate_internal_coherence(E1.components)
        E2_coherence = calculate_internal_coherence(E2.components)
        
        # Verify rotations don't affect internal coherence
        self.assertAlmostEqual(float(E1_coherence), float(E2_coherence), delta=0.05,
                            msg="Internal coherence should be similar for differently oriented vectors")
        
        # Create a coherent triple and calculate global coherence
        triple1 = (H1.components, E1.components, self.test_U)
        is_valid1, coherence1 = validate_trilateral_coherence(triple1)
        
        # Create a triple with the same vectors but different orientations
        triple2 = (H2.components, E2.components, self.test_U)
        is_valid2, coherence2 = validate_trilateral_coherence(triple2)
        
        # While actual coherence values may differ due to cross-coherence changes,
        # both should be either valid or invalid
        self.assertEqual(is_valid1, is_valid2, 
                      "Validation outcome should be consistent for differently oriented vectors")


if __name__ == '__main__':
    unittest.main()