import unittest
import numpy as np
import json
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
    PRECISION_TOLERANCE
)

class TestZPDRSerialization(unittest.TestCase):
    """
    Test suite for validating serialization format in ZPDR Phase 2.
    
    This test suite focuses on the serialization and deserialization of
    ZPDR structures, particularly the ZPA (Zero-Point Address) manifest.
    """
    
    def setUp(self):
        """Set up test fixtures and constants needed for serialization testing."""
        # Define test vectors
        self.test_H = np.array([0.125, 0.25, 0.5], dtype=np.float64)
        self.test_E = np.array([0.7071, 0.7071, 0], dtype=np.float64)
        self.test_U = np.array([0.333, 0.333, 0.333], dtype=np.float64)
        
        # Define test invariants
        self.test_H_invariants = {
            'rotation_angle': 0.0,
            'scale_factor': 1.0
        }
        self.test_E_invariants = {
            'rotation_angle': np.pi/4,
            'radius': 1.0
        }
        self.test_U_invariants = {
            'rotation_angle': 0.0,
            'magnitude': 0.577
        }
        
        # Define coherence values
        self.test_coherence = 0.98
        
        # Define a complete test ZPA
        self.test_zpa = {
            'version': '2.0',
            'hyperbolic': {
                'data': self.test_H.tolist(),
                'invariants': self.test_H_invariants,
                'coherence': 0.99
            },
            'elliptical': {
                'data': self.test_E.tolist(),
                'invariants': self.test_E_invariants,
                'coherence': 0.98
            },
            'euclidean': {
                'data': self.test_U.tolist(),
                'invariants': self.test_U_invariants,
                'coherence': 0.97
            },
            'cross_coherence': {
                'HE': 0.96,
                'EU': 0.95,
                'HU': 0.94
            },
            'global_coherence': self.test_coherence,
            'metadata': {
                'created': '2023-07-01T12:00:00Z',
                'description': 'Test ZPA',
                'origin': 'ZPDR Test Suite'
            }
        }

    def test_basic_serialization_format(self):
        """
        Test the basic serialization format for ZPA.
        
        This ensures the ZPA can be properly serialized to a string format.
        """
        # Serialize the test ZPA to JSON
        serialized = self._serialize_zpa(self.test_zpa)
        
        # Verify it's a valid JSON string
        self.assertIsInstance(serialized, str)
        
        # Verify it can be parsed back to a dictionary
        try:
            parsed = json.loads(serialized)
            self.assertIsInstance(parsed, dict)
        except json.JSONDecodeError:
            self.fail("Serialized ZPA should be valid JSON")
        
        # Verify structure is preserved
        self.assertIn('version', parsed)
        self.assertIn('hyperbolic', parsed)
        self.assertIn('elliptical', parsed)
        self.assertIn('euclidean', parsed)
        self.assertIn('cross_coherence', parsed)
        self.assertIn('global_coherence', parsed)
        
        # Check version
        self.assertEqual(parsed['version'], '2.0')

    def test_vector_serialization(self):
        """
        Test serialization of geometric vectors.
        
        This ensures vector data is properly serialized and deserialized.
        """
        # Create test vectors
        h_vector = HyperbolicVector(self.test_H)
        e_vector = EllipticalVector(self.test_E)
        u_vector = EuclideanVector(self.test_U)
        
        # Serialize each vector
        h_serialized = self._serialize_vector(h_vector)
        e_serialized = self._serialize_vector(e_vector)
        u_serialized = self._serialize_vector(u_vector)
        
        # Verify serialized forms are strings
        self.assertIsInstance(h_serialized, str)
        self.assertIsInstance(e_serialized, str)
        self.assertIsInstance(u_serialized, str)
        
        # Deserialize back to vectors
        h_deserialized = self._deserialize_vector(h_serialized, "hyperbolic")
        e_deserialized = self._deserialize_vector(e_serialized, "elliptical")
        u_deserialized = self._deserialize_vector(u_serialized, "euclidean")
        
        # Verify they match the originals
        self.assertTrue(np.allclose(h_vector.components, h_deserialized.components, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(e_vector.components, e_deserialized.components, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(u_vector.components, u_deserialized.components, rtol=1e-14, atol=1e-14))
        
        # Verify they maintain their space-specific properties
        self.assertTrue(h_deserialized.is_valid())
        self.assertTrue(e_deserialized.is_valid())
        self.assertTrue(u_deserialized.is_valid())

    def test_invariants_serialization(self):
        """
        Test serialization of vector invariants.
        
        This ensures invariants are properly serialized and deserialized with high precision.
        """
        # Serialize each set of invariants
        h_inv_serialized = self._serialize_invariants(self.test_H_invariants)
        e_inv_serialized = self._serialize_invariants(self.test_E_invariants)
        u_inv_serialized = self._serialize_invariants(self.test_U_invariants)
        
        # Verify serialized forms are strings
        self.assertIsInstance(h_inv_serialized, str)
        self.assertIsInstance(e_inv_serialized, str)
        self.assertIsInstance(u_inv_serialized, str)
        
        # Deserialize back to dictionaries
        h_inv_deserialized = self._deserialize_invariants(h_inv_serialized)
        e_inv_deserialized = self._deserialize_invariants(e_inv_serialized)
        u_inv_deserialized = self._deserialize_invariants(u_inv_serialized)
        
        # Verify keys are preserved
        self.assertIn('rotation_angle', h_inv_deserialized)
        self.assertIn('scale_factor', h_inv_deserialized)
        self.assertIn('rotation_angle', e_inv_deserialized)
        self.assertIn('radius', e_inv_deserialized)
        self.assertIn('rotation_angle', u_inv_deserialized)
        self.assertIn('magnitude', u_inv_deserialized)
        
        # Verify values match with high precision
        self.assertAlmostEqual(h_inv_deserialized['rotation_angle'], self.test_H_invariants['rotation_angle'], delta=1e-14)
        self.assertAlmostEqual(h_inv_deserialized['scale_factor'], self.test_H_invariants['scale_factor'], delta=1e-14)
        self.assertAlmostEqual(e_inv_deserialized['rotation_angle'], self.test_E_invariants['rotation_angle'], delta=1e-14)
        self.assertAlmostEqual(e_inv_deserialized['radius'], self.test_E_invariants['radius'], delta=1e-14)
        self.assertAlmostEqual(u_inv_deserialized['rotation_angle'], self.test_U_invariants['rotation_angle'], delta=1e-14)
        self.assertAlmostEqual(u_inv_deserialized['magnitude'], self.test_U_invariants['magnitude'], delta=1e-14)

    def test_complete_zpa_serialization(self):
        """
        Test serialization and deserialization of a complete ZPA manifest.
        
        This ensures the entire ZPA structure can be correctly serialized and deserialized.
        """
        # Serialize the test ZPA
        serialized = self._serialize_zpa(self.test_zpa)
        
        # Deserialize back to a dictionary
        deserialized = self._deserialize_zpa(serialized)
        
        # Verify structure matches
        self.assertEqual(deserialized['version'], self.test_zpa['version'])
        self.assertEqual(deserialized['global_coherence'], self.test_zpa['global_coherence'])
        
        # Check vector data
        h_data = deserialized['hyperbolic']['data']
        e_data = deserialized['elliptical']['data']
        u_data = deserialized['euclidean']['data']
        
        self.assertTrue(np.allclose(h_data, self.test_H, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(e_data, self.test_E, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(u_data, self.test_U, rtol=1e-14, atol=1e-14))
        
        # Check coherence values
        self.assertEqual(deserialized['hyperbolic']['coherence'], self.test_zpa['hyperbolic']['coherence'])
        self.assertEqual(deserialized['elliptical']['coherence'], self.test_zpa['elliptical']['coherence'])
        self.assertEqual(deserialized['euclidean']['coherence'], self.test_zpa['euclidean']['coherence'])
        self.assertEqual(deserialized['cross_coherence']['HE'], self.test_zpa['cross_coherence']['HE'])
        
        # Check metadata
        self.assertEqual(deserialized['metadata']['description'], self.test_zpa['metadata']['description'])

    def test_zpa_reconstruction_from_manifest(self):
        """
        Test reconstruction of ZPA vectors from a manifest.
        
        This ensures that a serialized ZPA can be correctly reconstructed into
        usable geometric vectors with proper invariants.
        """
        # Serialize the test ZPA
        serialized = self._serialize_zpa(self.test_zpa)
        
        # Deserialize back to a dictionary
        deserialized = self._deserialize_zpa(serialized)
        
        # Extract the vector data and invariants
        h_data = np.array(deserialized['hyperbolic']['data'])
        e_data = np.array(deserialized['elliptical']['data'])
        u_data = np.array(deserialized['euclidean']['data'])
        
        h_invariants = deserialized['hyperbolic']['invariants']
        e_invariants = deserialized['elliptical']['invariants']
        u_invariants = deserialized['euclidean']['invariants']
        
        # Create geometric vectors
        h_vector = HyperbolicVector(h_data)
        e_vector = EllipticalVector(e_data)
        u_vector = EuclideanVector(u_data)
        
        # Verify vectors are valid in their respective spaces
        self.assertTrue(h_vector.is_valid())
        self.assertTrue(e_vector.is_valid())
        self.assertTrue(u_vector.is_valid())
        
        # Apply invariants to denormalize
        h_denorm = denormalize_with_invariants(h_data, h_invariants, "hyperbolic")
        e_denorm = denormalize_with_invariants(e_data, e_invariants, "elliptical")
        u_denorm = denormalize_with_invariants(u_data, u_invariants, "euclidean")
        
        # Verify we can re-extract invariants that match the original
        h_norm, h_inv = normalize_with_invariants(h_denorm, "hyperbolic")
        e_norm, e_inv = normalize_with_invariants(e_denorm, "elliptical")
        u_norm, u_inv = normalize_with_invariants(u_denorm, "euclidean")
        
        # Verify normalization produces the canonical form
        self.assertTrue(np.allclose(h_norm, h_data, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(e_norm, e_data, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(u_norm, u_data, rtol=1e-14, atol=1e-14))
        
        # Verify invariants are preserved with high precision
        self.assertAlmostEqual(h_inv['rotation_angle'], h_invariants['rotation_angle'], delta=1e-14)
        self.assertAlmostEqual(e_inv['rotation_angle'], e_invariants['rotation_angle'], delta=1e-14)
        self.assertAlmostEqual(u_inv['rotation_angle'], u_invariants['rotation_angle'], delta=1e-14)

    def test_serialization_size_efficiency(self):
        """
        Test the efficiency of the serialization format.
        
        This ensures the serialized ZPA is compact enough for practical use.
        """
        # Serialize the test ZPA
        serialized = self._serialize_zpa(self.test_zpa)
        
        # Check serialized size
        size_bytes = len(serialized.encode('utf-8'))
        
        # A basic ZPA with 3D vectors shouldn't be unreasonably large
        # Size will vary based on exact format, but should be reasonable
        self.assertLess(size_bytes, 2000, "Serialized ZPA should be reasonably compact")
        
        # Test serialization with reduced precision
        compact_zpa = self._serialize_zpa_compact(self.test_zpa)
        compact_size = len(compact_zpa.encode('utf-8'))
        
        # Compact format should be smaller
        self.assertLess(compact_size, size_bytes, "Compact serialization should be smaller")
        
        # But still deserializable and usable
        deserialized = self._deserialize_zpa(compact_zpa)
        self.assertIn('hyperbolic', deserialized)
        self.assertIn('data', deserialized['hyperbolic'])

    def test_metadata_serialization(self):
        """
        Test serialization of ZPA metadata.
        
        This ensures additional metadata can be properly included in the manifest.
        """
        # Create a ZPA with extended metadata
        extended_zpa = self.test_zpa.copy()
        extended_zpa['metadata'] = {
            'created': '2023-07-01T12:00:00Z',
            'description': 'Test ZPA',
            'origin': 'ZPDR Test Suite',
            'content_type': 'text/plain',
            'content_size': 1024,
            'hash': 'sha256:abc123',
            'tags': ['test', 'zpdr', 'phase2'],
            'custom': {
                'application': 'ZPDR-Test',
                'version': '2.0.0',
                'settings': {
                    'precision': 'high',
                    'mode': 'standard'
                }
            }
        }
        
        # Serialize the extended ZPA
        serialized = self._serialize_zpa(extended_zpa)
        
        # Deserialize back to a dictionary
        deserialized = self._deserialize_zpa(serialized)
        
        # Verify metadata is preserved
        self.assertEqual(deserialized['metadata']['description'], extended_zpa['metadata']['description'])
        self.assertEqual(deserialized['metadata']['content_type'], extended_zpa['metadata']['content_type'])
        self.assertEqual(deserialized['metadata']['hash'], extended_zpa['metadata']['hash'])
        self.assertEqual(len(deserialized['metadata']['tags']), len(extended_zpa['metadata']['tags']))
        self.assertEqual(deserialized['metadata']['custom']['application'], 
                      extended_zpa['metadata']['custom']['application'])
        self.assertEqual(deserialized['metadata']['custom']['settings']['precision'],
                      extended_zpa['metadata']['custom']['settings']['precision'])

    def test_version_compatibility(self):
        """
        Test compatibility between different manifest versions.
        
        This ensures the serialization format handles version differences correctly.
        """
        # Create a v1.0 format ZPA (simplified format)
        v1_zpa = {
            'version': '1.0',
            'H': self.test_H.tolist(),
            'E': self.test_E.tolist(),
            'U': self.test_U.tolist(),
            'coherence': self.test_coherence,
            'metadata': {
                'description': 'Legacy ZPA'
            }
        }
        
        # Serialize the v1.0 ZPA
        v1_serialized = json.dumps(v1_zpa)
        
        # Try to deserialize with v2.0 deserializer
        # This should apply version compatibility logic
        v1_deserialized = self._deserialize_zpa(v1_serialized)
        
        # Verify we can still extract the essential components
        self.assertIn('hyperbolic', v1_deserialized)
        self.assertIn('elliptical', v1_deserialized)
        self.assertIn('euclidean', v1_deserialized)
        
        # Compare vector data
        h_data = np.array(v1_deserialized['hyperbolic']['data'])
        e_data = np.array(v1_deserialized['elliptical']['data'])
        u_data = np.array(v1_deserialized['euclidean']['data'])
        
        self.assertTrue(np.allclose(h_data, self.test_H, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(e_data, self.test_E, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(u_data, self.test_U, rtol=1e-14, atol=1e-14))
        
        # Now create a future version with extra fields
        v3_zpa = {
            'version': '3.0',
            'hyperbolic': {
                'data': self.test_H.tolist(),
                'invariants': self.test_H_invariants,
                'coherence': 0.99,
                'future_field': 'new value'
            },
            'elliptical': {
                'data': self.test_E.tolist(),
                'invariants': self.test_E_invariants,
                'coherence': 0.98
            },
            'euclidean': {
                'data': self.test_U.tolist(),
                'invariants': self.test_U_invariants,
                'coherence': 0.97
            },
            'cross_coherence': {
                'HE': 0.96,
                'EU': 0.95,
                'HU': 0.94
            },
            'global_coherence': self.test_coherence,
            'future_component': {
                'data': [0.1, 0.2, 0.3],
                'type': 'quaternion'
            },
            'metadata': {
                'created': '2023-07-01T12:00:00Z',
                'description': 'Future ZPA'
            }
        }
        
        # Serialize the v3.0 ZPA
        v3_serialized = json.dumps(v3_zpa)
        
        # Try to deserialize with v2.0 deserializer
        v3_deserialized = self._deserialize_zpa(v3_serialized)
        
        # Verify we can still extract the essential components
        # even if we ignore future fields
        self.assertIn('hyperbolic', v3_deserialized)
        self.assertIn('elliptical', v3_deserialized)
        self.assertIn('euclidean', v3_deserialized)
        
        # Compare vector data
        h_data = np.array(v3_deserialized['hyperbolic']['data'])
        e_data = np.array(v3_deserialized['elliptical']['data'])
        u_data = np.array(v3_deserialized['euclidean']['data'])
        
        self.assertTrue(np.allclose(h_data, self.test_H, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(e_data, self.test_E, rtol=1e-14, atol=1e-14))
        self.assertTrue(np.allclose(u_data, self.test_U, rtol=1e-14, atol=1e-14))

    # Helper methods for the tests
    def _serialize_zpa(self, zpa):
        """Serialize a ZPA to JSON."""
        # Use json.dumps with high precision
        serialized = json.dumps(zpa, sort_keys=True, indent=2)
        return serialized

    def _deserialize_zpa(self, serialized):
        """Deserialize a ZPA from JSON."""
        # Use json.loads
        deserialized = json.loads(serialized)
        
        # Handle version compatibility
        if 'version' not in deserialized:
            # Very old format - try to apply heuristics
            deserialized['version'] = '1.0'
        
        # Convert v1.0 format to v2.0 format if needed
        if deserialized['version'] == '1.0':
            v2_zpa = {
                'version': '2.0',
                'hyperbolic': {
                    'data': deserialized.get('H', [0, 0, 0]),
                    'invariants': {
                        'rotation_angle': 0.0,
                        'scale_factor': 1.0
                    },
                    'coherence': 0.99
                },
                'elliptical': {
                    'data': deserialized.get('E', [1, 0, 0]),
                    'invariants': {
                        'rotation_angle': 0.0,
                        'radius': 1.0
                    },
                    'coherence': 0.99
                },
                'euclidean': {
                    'data': deserialized.get('U', [0, 0, 0]),
                    'invariants': {
                        'rotation_angle': 0.0,
                        'magnitude': 1.0
                    },
                    'coherence': 0.99
                },
                'cross_coherence': {
                    'HE': 0.99,
                    'EU': 0.99,
                    'HU': 0.99
                },
                'global_coherence': deserialized.get('coherence', 0.99),
                'metadata': deserialized.get('metadata', {})
            }
            return v2_zpa
        else:
            return deserialized

    def _serialize_vector(self, vector):
        """Serialize a geometric vector."""
        # Convert vector components to list
        components = vector.components.tolist()
        
        # Create a dictionary with space type
        vector_data = {
            'space_type': vector.space_type.value,
            'components': components
        }
        
        # Serialize to JSON
        return json.dumps(vector_data)

    def _deserialize_vector(self, serialized, space_type):
        """Deserialize a geometric vector."""
        # Parse JSON
        vector_data = json.loads(serialized)
        
        # Extract components
        components = vector_data['components']
        
        # Create appropriate vector type
        if space_type == "hyperbolic":
            return HyperbolicVector(components)
        elif space_type == "elliptical":
            return EllipticalVector(components)
        else:  # Euclidean
            return EuclideanVector(components)

    def _serialize_invariants(self, invariants):
        """Serialize vector invariants."""
        # Use JSON with high precision
        return json.dumps(invariants, sort_keys=True)

    def _deserialize_invariants(self, serialized):
        """Deserialize vector invariants."""
        # Parse JSON
        return json.loads(serialized)

    def _serialize_zpa_compact(self, zpa):
        """Serialize a ZPA in compact format."""
        # Create a reduced precision copy
        compact_zpa = {
            'version': zpa['version'],
            'hyperbolic': {
                'data': [round(x, 6) for x in zpa['hyperbolic']['data']],
                'invariants': {
                    k: round(v, 6) if isinstance(v, (int, float)) else v
                    for k, v in zpa['hyperbolic']['invariants'].items()
                },
                'coherence': round(zpa['hyperbolic']['coherence'], 3)
            },
            'elliptical': {
                'data': [round(x, 6) for x in zpa['elliptical']['data']],
                'invariants': {
                    k: round(v, 6) if isinstance(v, (int, float)) else v
                    for k, v in zpa['elliptical']['invariants'].items()
                },
                'coherence': round(zpa['elliptical']['coherence'], 3)
            },
            'euclidean': {
                'data': [round(x, 6) for x in zpa['euclidean']['data']],
                'invariants': {
                    k: round(v, 6) if isinstance(v, (int, float)) else v
                    for k, v in zpa['euclidean']['invariants'].items()
                },
                'coherence': round(zpa['euclidean']['coherence'], 3)
            },
            'cross_coherence': {
                k: round(v, 3) for k, v in zpa['cross_coherence'].items()
            },
            'global_coherence': round(zpa['global_coherence'], 3),
            'metadata': {
                'description': zpa['metadata'].get('description', '')
            }
        }
        
        # Serialize with minimal whitespace
        return json.dumps(compact_zpa, separators=(',', ':'))


if __name__ == '__main__':
    unittest.main()