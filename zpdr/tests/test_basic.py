"""
Basic tests for the ZPDR framework.

Tests the core components:
- Multivector class
- Geometric spaces
- Space transformations
"""

import unittest
import numpy as np
from zpdr.core.multivector import Multivector
from zpdr.core.geometric_spaces import (
    SpaceType, 
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector, 
    SpaceTransformer
)


class TestMultivector(unittest.TestCase):
    """Test the Multivector class."""
    
    def test_creation(self):
        """Test creating a multivector."""
        # Create a scalar
        scalar = Multivector({"1": 5.0})
        self.assertEqual(scalar.scalar_part(), 5.0)
        
        # Create a vector
        vector = Multivector({"e1": 1.0, "e2": 2.0, "e3": 3.0})
        self.assertEqual(vector.grade(1).components, {"e1": 1.0, "e2": 2.0, "e3": 3.0})
        
        # Create a bivector
        bivector = Multivector({"e12": 1.0, "e23": 2.0, "e31": 3.0})
        self.assertEqual(bivector.grade(2).components, {"e12": 1.0, "e23": 2.0, "e31": 3.0})
    
    def test_addition(self):
        """Test adding multivectors."""
        mv1 = Multivector({"1": 1.0, "e1": 2.0, "e12": 3.0})
        mv2 = Multivector({"1": 2.0, "e1": 1.0, "e23": 4.0})
        
        result = mv1 + mv2
        expected = Multivector({"1": 3.0, "e1": 3.0, "e12": 3.0, "e23": 4.0})
        
        # Compare components
        self.assertEqual(result.components, expected.components)
    
    def test_geometric_product(self):
        """Test the geometric product."""
        # e1 * e1 = 1 (in Euclidean space)
        e1 = Multivector({"e1": 1.0})
        result = e1 * e1
        self.assertAlmostEqual(result.scalar_part(), 1.0)
        
        # e1 * e2 = e12 (in Euclidean space)
        e1 = Multivector({"e1": 1.0})
        e2 = Multivector({"e2": 1.0})
        result = e1 * e2
        self.assertEqual(result.components, {"e12": 1.0})
        
        # e2 * e1 = -e12 (in Euclidean space)
        result = e2 * e1
        self.assertEqual(result.components, {"e12": -1.0})
    
    def test_grade_extraction(self):
        """Test extracting grades."""
        mv = Multivector({
            "1": 1.0,       # scalar
            "e1": 2.0,      # vector
            "e12": 3.0,     # bivector
            "e123": 4.0     # trivector
        })
        
        scalar = mv.grade(0)
        vector = mv.grade(1)
        bivector = mv.grade(2)
        trivector = mv.grade(3)
        
        self.assertEqual(scalar.components, {"1": 1.0})
        self.assertEqual(vector.components, {"e1": 2.0})
        self.assertEqual(bivector.components, {"e12": 3.0})
        self.assertEqual(trivector.components, {"e123": 4.0})
    
    def test_norm(self):
        """Test norm calculation."""
        # Scalar norm
        scalar = Multivector({"1": 3.0})
        self.assertAlmostEqual(scalar.norm(), 3.0)
        
        # Vector norm (e1^2 + e2^2 + e3^2)^0.5
        vector = Multivector({"e1": 1.0, "e2": 2.0, "e3": 2.0})
        self.assertAlmostEqual(vector.norm(), 3.0)


class TestGeometricSpaces(unittest.TestCase):
    """Test the geometric spaces."""
    
    def test_hyperbolic_vector(self):
        """Test HyperbolicVector."""
        # Create a valid hyperbolic vector
        v = HyperbolicVector([0.3, 0.4])
        self.assertTrue(v.is_valid())
        self.assertAlmostEqual(np.linalg.norm(v.components), 0.5)
        
        # Create a vector outside the disk and verify it's projected back
        v = HyperbolicVector([2.0, 0.0])
        self.assertTrue(v.is_valid())
        self.assertLess(np.linalg.norm(v.components), 1.0)
    
    def test_elliptical_vector(self):
        """Test EllipticalVector."""
        # Create a vector and verify it's normalized to the sphere
        v = EllipticalVector([3.0, 4.0])
        self.assertTrue(v.is_valid())
        self.assertAlmostEqual(np.linalg.norm(v.components), 1.0)
    
    def test_euclidean_vector(self):
        """Test EuclideanVector."""
        v = EuclideanVector([1.0, 2.0, 3.0])
        self.assertTrue(v.is_valid())
        self.assertAlmostEqual(np.linalg.norm(v.components), np.sqrt(14))
    
    def test_hyperbolic_addition(self):
        """Test hyperbolic addition."""
        v1 = HyperbolicVector([0.1, 0.2])
        v2 = HyperbolicVector([0.2, 0.1])
        result = v1 + v2
        
        # Hyperbolic addition should result in a valid hyperbolic vector
        self.assertTrue(result.is_valid())
    
    def test_elliptical_addition(self):
        """Test elliptical addition."""
        v1 = EllipticalVector([1.0, 0.0])
        v2 = EllipticalVector([0.0, 1.0])
        result = v1 + v2
        
        # Elliptical addition should result in a unit vector
        self.assertTrue(result.is_valid())
        self.assertAlmostEqual(np.linalg.norm(result.components), 1.0)
    
    def test_euclidean_addition(self):
        """Test euclidean addition."""
        v1 = EuclideanVector([1.0, 2.0])
        v2 = EuclideanVector([3.0, 4.0])
        result = v1 + v2
        
        # Check result
        np.testing.assert_array_equal(result.components, np.array([4.0, 6.0]))
    
    def test_inner_products(self):
        """Test inner products in different spaces."""
        # Euclidean inner product
        v1 = EuclideanVector([1.0, 2.0, 3.0])
        v2 = EuclideanVector([4.0, 5.0, 6.0])
        self.assertAlmostEqual(v1.inner_product(v2), 32.0)
        
        # Elliptical inner product (dot product of unit vectors)
        v1 = EllipticalVector([1.0, 0.0])
        v2 = EllipticalVector([0.0, 1.0])
        self.assertAlmostEqual(v1.inner_product(v2), 0.0)
        
        # Hyperbolic inner product
        v1 = HyperbolicVector([0.1, 0.0])
        v2 = HyperbolicVector([0.1, 0.0])
        self.assertGreater(v1.inner_product(v2), 0.0)


class TestSpaceTransformations(unittest.TestCase):
    """Test transformations between spaces."""
    
    def test_euclidean_to_hyperbolic(self):
        """Test transforming Euclidean to hyperbolic."""
        v = EuclideanVector([0.5, 0.5])
        h = SpaceTransformer.euclidean_to_hyperbolic(v)
        
        # Check the vector is in hyperbolic space
        self.assertIsInstance(h, HyperbolicVector)
        self.assertTrue(h.is_valid())
        
        # Large Euclidean vectors should be scaled to fit in hyperbolic space
        v = EuclideanVector([100.0, 100.0])
        h = SpaceTransformer.euclidean_to_hyperbolic(v)
        self.assertTrue(h.is_valid())
        self.assertLess(np.linalg.norm(h.components), 1.0)
    
    def test_euclidean_to_elliptical(self):
        """Test transforming Euclidean to elliptical."""
        v = EuclideanVector([3.0, 4.0])
        e = SpaceTransformer.euclidean_to_elliptical(v)
        
        # Check the vector is in elliptical space
        self.assertIsInstance(e, EllipticalVector)
        self.assertTrue(e.is_valid())
        
        # Should be normalized to unit length
        self.assertAlmostEqual(np.linalg.norm(e.components), 1.0)
    
    def test_hyperbolic_to_euclidean(self):
        """Test transforming hyperbolic to Euclidean."""
        h = HyperbolicVector([0.3, 0.4])
        v = SpaceTransformer.hyperbolic_to_euclidean(h)
        
        # Check the vector is in Euclidean space
        self.assertIsInstance(v, EuclideanVector)
        self.assertTrue(v.is_valid())
        
        # Components should be preserved
        np.testing.assert_array_almost_equal(v.components, np.array([0.3, 0.4]))
    
    def test_elliptical_to_euclidean(self):
        """Test transforming elliptical to Euclidean."""
        e = EllipticalVector([3.0, 4.0])  # Will be normalized internally
        v = SpaceTransformer.elliptical_to_euclidean(e)
        
        # Check the vector is in Euclidean space
        self.assertIsInstance(v, EuclideanVector)
        self.assertTrue(v.is_valid())
        
        # Should maintain the unit length
        self.assertAlmostEqual(np.linalg.norm(v.components), 1.0)
    
    def test_round_trip_transformations(self):
        """Test round-trip transformations between spaces."""
        # Euclidean to hyperbolic and back
        orig = EuclideanVector([0.3, 0.4])
        h = SpaceTransformer.euclidean_to_hyperbolic(orig)
        back = SpaceTransformer.hyperbolic_to_euclidean(h)
        np.testing.assert_array_almost_equal(orig.components, back.components)
        
        # Euclidean to elliptical and back (note: elliptical normalizes)
        orig = EuclideanVector([3.0, 4.0])
        e = SpaceTransformer.euclidean_to_elliptical(orig)
        back = SpaceTransformer.elliptical_to_euclidean(e)
        # Can't check exact equality due to normalization
        self.assertAlmostEqual(np.linalg.norm(back.components), 1.0)
        
        # Hyperbolic to elliptical and back
        orig = HyperbolicVector([0.3, 0.4])
        e = SpaceTransformer.hyperbolic_to_elliptical(orig)
        back = SpaceTransformer.elliptical_to_hyperbolic(e)
        # Can't check exact equality due to normalization in elliptical space
        self.assertTrue(back.is_valid())


if __name__ == "__main__":
    unittest.main()