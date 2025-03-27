"""
Verifier Module for ZPDR

This module implements verification mechanisms for Zero-Point Data Resolution,
ensuring the integrity, coherence, and mathematical consistency of ZPA manifests.
"""

import numpy as np
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from decimal import Decimal, getcontext

from .multivector import Multivector
from .geometric_spaces import (
    HyperbolicVector,
    EllipticalVector,
    EuclideanVector,
    SpaceTransformer
)
from .zpa_manifest import ZPAManifest

from ..utils import (
    to_decimal_array,
    calculate_internal_coherence,
    calculate_cross_coherence,
    calculate_global_coherence,
    validate_trilateral_coherence,
    normalize_with_invariants,
    denormalize_with_invariants,
    COHERENCE_THRESHOLD,
    PRECISION_TOLERANCE
)

class Verifier:
    """
    Verification system for ZPDR.
    
    This class provides methods to verify the integrity, coherence, 
    and mathematical consistency of ZPA manifests at various levels of rigor.
    """
    
    def __init__(self, coherence_threshold: float = float(COHERENCE_THRESHOLD),
                 precision_tolerance: float = float(PRECISION_TOLERANCE)):
        """
        Initialize the verifier.
        
        Args:
            coherence_threshold: Threshold for valid coherence
            precision_tolerance: Tolerance for numerical precision
        """
        self.coherence_threshold = coherence_threshold
        self.precision_tolerance = precision_tolerance
        
        # Define verification levels
        self.verification_levels = {
            'minimal': self._verify_minimal,
            'standard': self._verify_standard,
            'strict': self._verify_strict
        }
        
        # Performance metrics
        self._metrics = {
            'verifications_performed': 0,
            'verification_failures': 0,
            'verification_time_ms': 0.0
        }
    
    def verify_manifest(self, manifest: ZPAManifest, 
                       level: str = 'standard') -> Tuple[bool, Dict[str, Any]]:
        """
        Verify a ZPA manifest according to the specified level.
        
        Args:
            manifest: ZPA manifest to verify
            level: Verification level ('minimal', 'standard', 'strict')
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Track metrics
        start_time = time.time()
        self._metrics['verifications_performed'] += 1
        
        # Validate the level
        if level not in self.verification_levels:
            level = 'standard'
        
        # Run the appropriate verification function
        is_valid, results = self.verification_levels[level](manifest)
        
        # Update metrics
        if not is_valid:
            self._metrics['verification_failures'] += 1
        
        # Track timing
        end_time = time.time()
        self._metrics['verification_time_ms'] += (end_time - start_time) * 1000
        
        return is_valid, results
    
    def verify_reconstruction(self, original_data: Any, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that a manifest can correctly reconstruct the original data.
        
        Args:
            original_data: Original data to compare against
            manifest: ZPA manifest to verify
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # First check that the manifest itself is valid
        is_valid, base_results = self.verify_manifest(manifest)
        
        if not is_valid:
            base_results['reconstruction_valid'] = False
            base_results['reason'] = 'Invalid manifest'
            return False, base_results
        
        # Reconstruct data from manifest based on data type
        try:
            from .zpdr_processor import ZPDRProcessor
            processor = ZPDRProcessor()
            reconstructed_data = processor.decode(manifest, verify=False)
            
            # Compare with original based on data type
            if isinstance(original_data, (int, float, str)):
                reconstruction_valid = (reconstructed_data == original_data)
            elif isinstance(original_data, bytes):
                # For bytes, convert if needed
                if isinstance(reconstructed_data, int):
                    byte_length = (reconstructed_data.bit_length() + 7) // 8
                    reconstructed_bytes = reconstructed_data.to_bytes(
                        max(1, byte_length), byteorder='big'
                    )
                    reconstruction_valid = (reconstructed_bytes == original_data)
                else:
                    reconstruction_valid = (reconstructed_data == original_data)
            else:
                # Unsupported type
                reconstruction_valid = False
            
            base_results['reconstruction_valid'] = reconstruction_valid
            
            if not reconstruction_valid:
                base_results['reason'] = 'Reconstruction mismatch'
                self._metrics['verification_failures'] += 1
                return False, base_results
            
            return True, base_results
            
        except Exception as e:
            base_results['reconstruction_valid'] = False
            base_results['reason'] = f'Reconstruction error: {str(e)}'
            self._metrics['verification_failures'] += 1
            return False, base_results
    
    def verify_tamper_resistance(self, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that a manifest has not been tampered with.
        
        This performs a series of checks to detect potential tampering:
        1. Verify internal consistency between vectors and coherence values
        2. Check geometric properties of vectors
        3. Verify invariant consistency
        
        Args:
            manifest: ZPA manifest to verify
            
        Returns:
            Tuple of (is_tamper_free, verification_results)
        """
        # Start with basic verification
        is_valid, base_results = self.verify_manifest(manifest, level='strict')
        
        # Initialize tamper detection results
        tamper_results = {
            'tamper_free': is_valid,
            'issues': []
        }
        
        # If basic verification failed, return that result
        if not is_valid:
            tamper_results['issues'].append('Basic verification failed')
            tamper_results.update(base_results)
            return False, tamper_results
        
        # Calculate hashes for additional integrity checking
        vector_hash = self._calculate_vector_hash(manifest.H, manifest.E, manifest.U)
        tamper_results['vector_hash'] = vector_hash
        
        # Check for inconsistencies between vectors and coherence values
        calculated_H_coherence = float(calculate_internal_coherence(manifest.H))
        if abs(calculated_H_coherence - manifest.H_coherence) > 0.01:
            tamper_results['tamper_free'] = False
            tamper_results['issues'].append('H coherence mismatch')
        
        calculated_E_coherence = float(calculate_internal_coherence(manifest.E))
        if abs(calculated_E_coherence - manifest.E_coherence) > 0.01:
            tamper_results['tamper_free'] = False
            tamper_results['issues'].append('E coherence mismatch')
        
        calculated_U_coherence = float(calculate_internal_coherence(manifest.U))
        if abs(calculated_U_coherence - manifest.U_coherence) > 0.01:
            tamper_results['tamper_free'] = False
            tamper_results['issues'].append('U coherence mismatch')
        
        # Check cross-coherence consistency
        calculated_HE_coherence = float(calculate_cross_coherence(manifest.H, manifest.E))
        if abs(calculated_HE_coherence - manifest.HE_coherence) > 0.01:
            tamper_results['tamper_free'] = False
            tamper_results['issues'].append('HE coherence mismatch')
        
        calculated_EU_coherence = float(calculate_cross_coherence(manifest.E, manifest.U))
        if abs(calculated_EU_coherence - manifest.EU_coherence) > 0.01:
            tamper_results['tamper_free'] = False
            tamper_results['issues'].append('EU coherence mismatch')
        
        calculated_HU_coherence = float(calculate_cross_coherence(manifest.H, manifest.U))
        if abs(calculated_HU_coherence - manifest.HU_coherence) > 0.01:
            tamper_results['tamper_free'] = False
            tamper_results['issues'].append('HU coherence mismatch')
        
        # If any issues were found, mark as tampered
        if tamper_results['issues']:
            self._metrics['verification_failures'] += 1
        
        return tamper_results['tamper_free'], tamper_results
    
    def verify_invariants(self, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify the invariants in a ZPA manifest.
        
        This checks both weak invariants (value ranges) and strong invariants
        (mathematical consistency between vectors and their invariants).
        
        Args:
            manifest: ZPA manifest to verify
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Initialize results
        results = {
            'invariants_valid': True,
            'issues': []
        }
        
        # Check weak invariants (value ranges)
        if not self._verify_weak_invariants(manifest.H_invariants, 'hyperbolic'):
            results['invariants_valid'] = False
            results['issues'].append('Invalid H invariants')
        
        if not self._verify_weak_invariants(manifest.E_invariants, 'elliptical'):
            results['invariants_valid'] = False
            results['issues'].append('Invalid E invariants')
        
        if not self._verify_weak_invariants(manifest.U_invariants, 'euclidean'):
            results['invariants_valid'] = False
            results['issues'].append('Invalid U invariants')
        
        # Check strong invariants (mathematical consistency)
        if not self._verify_strong_invariants(manifest):
            results['invariants_valid'] = False
            results['issues'].append('Inconsistent invariants')
        
        # If any issues were found, mark as invalid
        if results['issues']:
            self._metrics['verification_failures'] += 1
        
        return results['invariants_valid'], results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification performance metrics."""
        metrics = self._metrics.copy()
        
        # Calculate failure rate
        if metrics['verifications_performed'] > 0:
            metrics['failure_rate'] = (metrics['verification_failures'] / 
                                     metrics['verifications_performed'])
            
            metrics['avg_verification_time_ms'] = (metrics['verification_time_ms'] / 
                                                metrics['verifications_performed'])
        else:
            metrics['failure_rate'] = 0.0
            metrics['avg_verification_time_ms'] = 0.0
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset verification metrics."""
        self._metrics = {
            'verifications_performed': 0,
            'verification_failures': 0,
            'verification_time_ms': 0.0
        }
    
    def _verify_minimal(self, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform minimal verification of a ZPA manifest.
        
        This only checks basic coherence and geometric properties.
        
        Args:
            manifest: ZPA manifest to verify
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Check that coherence is above threshold
        is_coherent = manifest.global_coherence >= self.coherence_threshold
        
        # Check basic geometric properties
        h_norm = np.linalg.norm(manifest.H)
        e_norm = np.linalg.norm(manifest.E)
        
        is_valid_h = h_norm < 1.0 and np.all(np.isfinite(manifest.H))
        is_valid_e = abs(e_norm - 1.0) < self.precision_tolerance and np.all(np.isfinite(manifest.E))
        is_valid_u = np.all(np.isfinite(manifest.U))
        
        # Compile results
        results = {
            'level': 'minimal',
            'coherence': manifest.global_coherence,
            'coherence_threshold': self.coherence_threshold,
            'coherence_valid': is_coherent,
            'vectors_valid': {
                'H': is_valid_h,
                'E': is_valid_e,
                'U': is_valid_u
            }
        }
        
        # Overall validity
        is_valid = is_coherent and is_valid_h and is_valid_e and is_valid_u
        
        return is_valid, results
    
    def _verify_standard(self, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform standard verification of a ZPA manifest.
        
        This checks coherence, geometric properties, and basic invariant consistency.
        
        Args:
            manifest: ZPA manifest to verify
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Start with minimal verification
        is_valid, results = self._verify_minimal(manifest)
        results['level'] = 'standard'
        
        # If minimal verification failed, no need to continue
        if not is_valid:
            return is_valid, results
        
        # Check invariant consistency
        invariants_valid, invariant_results = self.verify_invariants(manifest)
        results['invariants_valid'] = invariants_valid
        
        if not invariants_valid:
            results['invariant_issues'] = invariant_results['issues']
            is_valid = False
        
        # Check all component coherences
        component_coherences_valid = (
            manifest.H_coherence >= 0.5 * self.coherence_threshold and
            manifest.E_coherence >= 0.5 * self.coherence_threshold and
            manifest.U_coherence >= 0.5 * self.coherence_threshold
        )
        
        results['component_coherences_valid'] = component_coherences_valid
        
        if not component_coherences_valid:
            is_valid = False
        
        return is_valid, results
    
    def _verify_strict(self, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform strict verification of a ZPA manifest.
        
        This applies the most rigorous checks, including tamper detection,
        strong invariant verification, and detailed coherence analysis.
        
        Args:
            manifest: ZPA manifest to verify
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Start with standard verification
        is_valid, results = self._verify_standard(manifest)
        results['level'] = 'strict'
        
        # If standard verification failed, no need to continue
        if not is_valid:
            return is_valid, results
        
        # Perform tamper detection
        tamper_free, tamper_results = self.verify_tamper_resistance(manifest)
        results['tamper_free'] = tamper_free
        
        if not tamper_free:
            results['tamper_issues'] = tamper_results['issues']
            is_valid = False
        
        # Check cross-coherence values
        cross_coherences_valid = (
            manifest.HE_coherence >= 0.5 * self.coherence_threshold and
            manifest.EU_coherence >= 0.5 * self.coherence_threshold and
            manifest.HU_coherence >= 0.5 * self.coherence_threshold
        )
        
        results['cross_coherences_valid'] = cross_coherences_valid
        
        if not cross_coherences_valid:
            is_valid = False
        
        # Add hash for integrity verification
        results['integrity_hash'] = self._calculate_manifest_hash(manifest)
        
        return is_valid, results
    
    def _verify_weak_invariants(self, invariants: Dict[str, float], space_type: str) -> bool:
        """
        Verify weak invariants (value ranges) for a specific space.
        
        Args:
            invariants: Invariants dictionary
            space_type: Type of space ('hyperbolic', 'elliptical', 'euclidean')
            
        Returns:
            True if invariants are valid, False otherwise
        """
        # Check rotation angle is in valid range
        if 'rotation_angle' in invariants:
            angle = invariants['rotation_angle']
            if angle < -np.pi or angle > np.pi or not np.isfinite(angle):
                return False
        
        # Space-specific checks
        if space_type == 'hyperbolic':
            # Check scale factor is positive and finite
            if 'scale_factor' in invariants:
                scale = invariants['scale_factor']
                if scale <= 0 or not np.isfinite(scale):
                    return False
                    
        elif space_type == 'elliptical':
            # Check radius is 1.0 (or very close)
            if 'radius' in invariants:
                radius = invariants['radius']
                if abs(radius - 1.0) > self.precision_tolerance or not np.isfinite(radius):
                    return False
                    
        elif space_type == 'euclidean':
            # Check magnitude is positive and finite
            if 'magnitude' in invariants:
                magnitude = invariants['magnitude']
                if magnitude <= 0 or not np.isfinite(magnitude):
                    return False
        
        # All checks passed
        return True
    
    def _verify_strong_invariants(self, manifest: ZPAManifest) -> bool:
        """
        Verify strong invariants (mathematical consistency).
        
        This checks that the invariants correctly relate to their vectors
        and that applying invariants properly transforms the vectors.
        
        Args:
            manifest: ZPA manifest to verify
            
        Returns:
            True if invariants are valid, False otherwise
        """
        try:
            # Normalize vectors to extract computed invariants
            H_normalized, H_computed = normalize_with_invariants(manifest.H, "hyperbolic")
            E_normalized, E_computed = normalize_with_invariants(manifest.E, "elliptical")
            U_normalized, U_computed = normalize_with_invariants(manifest.U, "euclidean")
            
            # Compare rotation angles (most important invariant)
            if 'rotation_angle' in H_computed and 'rotation_angle' in manifest.H_invariants:
                # Rotation angles can differ by multiples of 2π
                h_angle_diff = abs(H_computed['rotation_angle'] - manifest.H_invariants['rotation_angle'])
                h_angle_diff = min(h_angle_diff, abs(h_angle_diff - 2 * np.pi), abs(h_angle_diff + 2 * np.pi))
                if h_angle_diff > 0.1:  # Allow some tolerance
                    return False
            
            if 'rotation_angle' in E_computed and 'rotation_angle' in manifest.E_invariants:
                e_angle_diff = abs(E_computed['rotation_angle'] - manifest.E_invariants['rotation_angle'])
                e_angle_diff = min(e_angle_diff, abs(e_angle_diff - 2 * np.pi), abs(e_angle_diff + 2 * np.pi))
                if e_angle_diff > 0.1:
                    return False
            
            if 'rotation_angle' in U_computed and 'rotation_angle' in manifest.U_invariants:
                u_angle_diff = abs(U_computed['rotation_angle'] - manifest.U_invariants['rotation_angle'])
                u_angle_diff = min(u_angle_diff, abs(u_angle_diff - 2 * np.pi), abs(u_angle_diff + 2 * np.pi))
                if u_angle_diff > 0.1:
                    return False
            
            # Verify that applying invariants correctly transforms vectors
            H_denormalized = denormalize_with_invariants(H_normalized, manifest.H_invariants, "hyperbolic")
            E_denormalized = denormalize_with_invariants(E_normalized, manifest.E_invariants, "elliptical")
            U_denormalized = denormalize_with_invariants(U_normalized, manifest.U_invariants, "euclidean")
            
            # Check that denormalized vectors are close to original
            if not np.allclose(H_denormalized, manifest.H, rtol=1e-3, atol=1e-3):
                return False
                
            if not np.allclose(E_denormalized, manifest.E, rtol=1e-3, atol=1e-3):
                return False
                
            if not np.allclose(U_denormalized, manifest.U, rtol=1e-3, atol=1e-3):
                return False
            
            # All checks passed
            return True
            
        except Exception:
            # Any exception during verification indicates an issue
            return False
    
    def _calculate_vector_hash(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> str:
        """Calculate a hash of the vector data for integrity checking."""
        # Convert vectors to bytes with fixed precision
        h_bytes = np.array([round(x, 10) for x in H]).tobytes()
        e_bytes = np.array([round(x, 10) for x in E]).tobytes()
        u_bytes = np.array([round(x, 10) for x in U]).tobytes()
        
        # Combine and hash
        combined = h_bytes + e_bytes + u_bytes
        return hashlib.sha256(combined).hexdigest()
    
    def _calculate_manifest_hash(self, manifest: ZPAManifest) -> str:
        """Calculate a hash of the entire manifest for integrity checking."""
        # Convert manifest to canonical form
        canonical_dict = manifest.to_dict()
        
        # Remove potentially variable metadata
        if 'metadata' in canonical_dict:
            metadata_copy = canonical_dict['metadata'].copy()
            # Remove volatile fields
            for field in ['created', 'modified', 'timestamp']:
                if field in metadata_copy:
                    del metadata_copy[field]
            canonical_dict['metadata'] = metadata_copy
        
        # Convert to stable string representation
        import json
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        
        # Hash the canonical representation
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

class IntegrityVerifier:
    """
    Advanced integrity verification system for ZPDR.
    
    This class provides methods to verify the integrity of ZPA manifests,
    with a focus on tamper resistance and cryptographic verification.
    """
    
    def __init__(self, base_verifier: Verifier = None):
        """
        Initialize the integrity verifier.
        
        Args:
            base_verifier: Optional base Verifier instance
        """
        self.base_verifier = base_verifier or Verifier()
    
    def create_integrity_proof(self, manifest: ZPAManifest) -> Dict[str, str]:
        """
        Create an integrity proof for a ZPA manifest.
        
        This creates a set of cryptographic hashes and checksums that can be
        used to verify the manifest has not been tampered with.
        
        Args:
            manifest: ZPA manifest to create proof for
            
        Returns:
            Dictionary of integrity proof elements
        """
        proof = {
            'version': '1.0',
            'vector_hash': self._calculate_vector_hash(manifest),
            'coherence_hash': self._calculate_coherence_hash(manifest),
            'invariant_hash': self._calculate_invariant_hash(manifest),
            'manifest_hash': self._calculate_full_hash(manifest),
            'timestamp': time.time()
        }
        
        return proof
    
    def verify_integrity_proof(self, manifest: ZPAManifest, 
                              proof: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify an integrity proof against a ZPA manifest.
        
        Args:
            manifest: ZPA manifest to verify
            proof: Integrity proof to check
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Check supported version
        if proof.get('version', '1.0') != '1.0':
            return False, {'error': 'Unsupported proof version'}
        
        # Calculate current hashes
        current_proof = self.create_integrity_proof(manifest)
        
        # Initialize results
        results = {
            'valid': True,
            'mismatches': []
        }
        
        # Check each hash
        for key in ['vector_hash', 'coherence_hash', 'invariant_hash', 'manifest_hash']:
            if key in proof and key in current_proof:
                if proof[key] != current_proof[key]:
                    results['valid'] = False
                    results['mismatches'].append(key)
        
        # Add information about when the proof was created
        if 'timestamp' in proof:
            results['age_seconds'] = time.time() - proof['timestamp']
        
        return results['valid'], results
    
    def _calculate_vector_hash(self, manifest: ZPAManifest) -> str:
        """Calculate a hash of the vector data."""
        # Convert vectors to bytes with fixed precision
        h_bytes = np.array([round(x, 10) for x in manifest.H]).tobytes()
        e_bytes = np.array([round(x, 10) for x in manifest.E]).tobytes()
        u_bytes = np.array([round(x, 10) for x in manifest.U]).tobytes()
        
        # Combine and hash
        combined = h_bytes + e_bytes + u_bytes
        return hashlib.sha256(combined).hexdigest()
    
    def _calculate_coherence_hash(self, manifest: ZPAManifest) -> str:
        """Calculate a hash of the coherence values."""
        # Create string with fixed precision
        coherence_str = (
            f"{manifest.H_coherence:.6f},{manifest.E_coherence:.6f},{manifest.U_coherence:.6f},"
            f"{manifest.HE_coherence:.6f},{manifest.EU_coherence:.6f},{manifest.HU_coherence:.6f},"
            f"{manifest.global_coherence:.6f}"
        )
        
        return hashlib.sha256(coherence_str.encode('utf-8')).hexdigest()
    
    def _calculate_invariant_hash(self, manifest: ZPAManifest) -> str:
        """Calculate a hash of the invariants."""
        import json
        
        # Convert invariants to canonical form with fixed precision
        def process_dict(d):
            return {k: round(v, 10) if isinstance(v, float) else v for k, v in d.items()}
        
        h_invariants = process_dict(manifest.H_invariants)
        e_invariants = process_dict(manifest.E_invariants)
        u_invariants = process_dict(manifest.U_invariants)
        
        # Convert to canonical json
        invariants_json = json.dumps({
            'H': h_invariants,
            'E': e_invariants,
            'U': u_invariants
        }, sort_keys=True, separators=(',', ':'))
        
        return hashlib.sha256(invariants_json.encode('utf-8')).hexdigest()
    
    def _calculate_full_hash(self, manifest: ZPAManifest) -> str:
        """Calculate a hash of the full manifest excluding volatile metadata."""
        # Convert manifest to canonical form
        canonical_dict = manifest.to_dict()
        
        # Remove potentially variable metadata
        if 'metadata' in canonical_dict:
            metadata_copy = canonical_dict['metadata'].copy()
            # Remove volatile fields
            for field in ['created', 'modified', 'timestamp']:
                if field in metadata_copy:
                    del metadata_copy[field]
            canonical_dict['metadata'] = metadata_copy
        
        # Convert to stable string representation
        import json
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        
        # Hash the canonical representation
        return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()