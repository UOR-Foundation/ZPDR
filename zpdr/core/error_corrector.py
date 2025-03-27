"""
Error Corrector Module for ZPDR

This module implements error correction for Zero-Point Data Resolution,
enabling recovery from corrupted or noisy ZPA representations. It uses
the redundant nature of the trilateral encoding to detect and correct errors.
"""

import numpy as np
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

class ErrorCorrector:
    """
    Error correction system for ZPDR.
    
    This class provides methods to detect and correct errors in ZPA vectors,
    improve coherence, and recover from corruption.
    """
    
    def __init__(self, coherence_threshold: float = float(COHERENCE_THRESHOLD),
                 precision_tolerance: float = float(PRECISION_TOLERANCE)):
        """
        Initialize the error corrector.
        
        Args:
            coherence_threshold: Threshold for valid coherence
            precision_tolerance: Tolerance for numerical precision
        """
        self.coherence_threshold = coherence_threshold
        self.precision_tolerance = precision_tolerance
        
        # Performance metrics
        self._metrics = {
            'corrections_attempted': 0,
            'successful_corrections': 0,
            'correction_time_ms': 0.0
        }
    
    def detect_errors(self, manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect errors in a ZPA manifest.
        
        This checks for various types of errors:
        - Coherence below threshold
        - Invalid geometric properties
        - Inconsistent invariants
        - Numerical issues
        
        Args:
            manifest: ZPA manifest to check
            
        Returns:
            Tuple of (has_errors, error_details)
        """
        error_details = {
            'has_errors': False,
            'errors': [],
            'coherence': {
                'global': manifest.global_coherence,
                'threshold': self.coherence_threshold
            }
        }
        
        # Check global coherence
        if manifest.global_coherence < self.coherence_threshold:
            error_details['has_errors'] = True
            error_details['errors'].append('low_coherence')
        
        # Check vector-specific issues
        
        # H vector (hyperbolic) should have norm < 1
        h_norm = np.linalg.norm(manifest.H)
        if h_norm >= 1.0:
            error_details['has_errors'] = True
            error_details['errors'].append('invalid_hyperbolic_norm')
            error_details['h_norm'] = float(h_norm)
        
        # E vector (elliptical) should have norm = 1
        e_norm = np.linalg.norm(manifest.E)
        if abs(e_norm - 1.0) > self.precision_tolerance:
            error_details['has_errors'] = True
            error_details['errors'].append('invalid_elliptical_norm')
            error_details['e_norm'] = float(e_norm)
        
        # Check invariants for consistency
        try:
            if self._check_invariant_issues(manifest.H_invariants, 'hyperbolic'):
                error_details['has_errors'] = True
                error_details['errors'].append('invalid_h_invariants')
                
            if self._check_invariant_issues(manifest.E_invariants, 'elliptical'):
                error_details['has_errors'] = True
                error_details['errors'].append('invalid_e_invariants')
                
            if self._check_invariant_issues(manifest.U_invariants, 'euclidean'):
                error_details['has_errors'] = True
                error_details['errors'].append('invalid_u_invariants')
        except:
            error_details['has_errors'] = True
            error_details['errors'].append('invariant_exception')
        
        # Check for numerical issues (NaN, infinity)
        if not np.all(np.isfinite(manifest.H)) or not np.all(np.isfinite(manifest.E)) or not np.all(np.isfinite(manifest.U)):
            error_details['has_errors'] = True
            error_details['errors'].append('non_finite_values')
        
        return error_details['has_errors'], error_details
    
    def correct_manifest(self, manifest: ZPAManifest) -> Tuple[ZPAManifest, Dict[str, Any]]:
        """
        Apply error correction to a ZPA manifest.
        
        This applies several correction techniques:
        1. Re-normalization to canonical forms
        2. Invariant correction
        3. Coherence optimization
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Tuple of (corrected_manifest, correction_details)
        """
        # Track metrics
        start_time = time.time()
        self._metrics['corrections_attempted'] += 1
        
        # Initialize correction details
        correction_details = {
            'original_coherence': manifest.global_coherence,
            'techniques_applied': [],
            'successful': False
        }
        
        # Step 1: Re-normalize all vectors to enforce proper geometric constraints
        try:
            H_normalized, H_invariants = normalize_with_invariants(manifest.H, "hyperbolic")
            E_normalized, E_invariants = normalize_with_invariants(manifest.E, "elliptical")
            U_normalized, U_invariants = normalize_with_invariants(manifest.U, "euclidean")
            
            correction_details['techniques_applied'].append('normalization')
        except Exception as e:
            # If normalization fails, use fallback correction
            correction_details['normalization_error'] = str(e)
            H_normalized, E_normalized, U_normalized = self._apply_fallback_correction(manifest)
            
            # Extract invariants from corrected vectors
            _, H_invariants = normalize_with_invariants(H_normalized, "hyperbolic")
            _, E_invariants = normalize_with_invariants(E_normalized, "elliptical")
            _, U_invariants = normalize_with_invariants(U_normalized, "euclidean")
            
            correction_details['techniques_applied'].append('fallback_correction')
        
        # Step 2: Apply coherence-guided correction
        if manifest.global_coherence < self.coherence_threshold:
            # Check which component has lowest coherence and apply focused correction
            if manifest.H_coherence <= min(manifest.E_coherence, manifest.U_coherence):
                # H has lowest coherence - apply H-specific correction
                H_normalized = self._optimize_component_coherence(H_normalized, E_normalized, U_normalized, 'H')
                correction_details['techniques_applied'].append('H_coherence_optimization')
            elif manifest.E_coherence <= min(manifest.H_coherence, manifest.U_coherence):
                # E has lowest coherence - apply E-specific correction
                E_normalized = self._optimize_component_coherence(E_normalized, H_normalized, U_normalized, 'E')
                correction_details['techniques_applied'].append('E_coherence_optimization')
            else:
                # U has lowest coherence - apply U-specific correction
                U_normalized = self._optimize_component_coherence(U_normalized, H_normalized, E_normalized, 'U')
                correction_details['techniques_applied'].append('U_coherence_optimization')
        
        # Step 3: Create new manifest with corrected vectors and invariants
        corrected = ZPAManifest(
            H_normalized, E_normalized, U_normalized,
            H_invariants, E_invariants, U_invariants,
            manifest.metadata.copy()
        )
        
        # Step 4: Update metadata to indicate correction
        corrected.metadata['error_corrected'] = True
        corrected.metadata['original_coherence'] = manifest.global_coherence
        corrected.metadata['correction_techniques'] = correction_details['techniques_applied']
        
        # Step 5: Validate the correction was successful
        if corrected.global_coherence >= self.coherence_threshold:
            correction_details['successful'] = True
            self._metrics['successful_corrections'] += 1
        
        # Record correction metrics
        correction_details['corrected_coherence'] = corrected.global_coherence
        correction_details['coherence_improvement'] = corrected.global_coherence - manifest.global_coherence
        
        # Track timing
        end_time = time.time()
        self._metrics['correction_time_ms'] += (end_time - start_time) * 1000
        
        return corrected, correction_details
    
    def correct_vectors(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply error correction directly to ZPA vectors.
        
        Args:
            H: Hyperbolic vector component
            E: Elliptical vector component
            U: Euclidean vector component
            
        Returns:
            Tuple of corrected (H, E, U) vectors
        """
        # Create a temporary manifest for correction
        temp_manifest = ZPAManifest(H, E, U)
        
        # Apply correction
        corrected, _ = self.correct_manifest(temp_manifest)
        
        # Return the corrected vectors
        return corrected.H, corrected.E, corrected.U
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error correction performance metrics."""
        metrics = self._metrics.copy()
        
        # Calculate success rate
        if metrics['corrections_attempted'] > 0:
            metrics['success_rate'] = (metrics['successful_corrections'] / 
                                      metrics['corrections_attempted'])
            
            metrics['avg_correction_time_ms'] = (metrics['correction_time_ms'] / 
                                               metrics['corrections_attempted'])
        else:
            metrics['success_rate'] = 0.0
            metrics['avg_correction_time_ms'] = 0.0
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset error correction metrics."""
        self._metrics = {
            'corrections_attempted': 0,
            'successful_corrections': 0,
            'correction_time_ms': 0.0
        }
    
    def _check_invariant_issues(self, invariants: Dict[str, float], space_type: str) -> bool:
        """
        Check invariants for issues based on their space type.
        
        Args:
            invariants: Invariants dictionary
            space_type: Type of space ('hyperbolic', 'elliptical', 'euclidean')
            
        Returns:
            True if issues were found, False otherwise
        """
        # Check common issues
        if 'rotation_angle' in invariants:
            angle = invariants['rotation_angle']
            # Angle should be in [-π, π]
            if angle < -np.pi or angle > np.pi:
                return True
        
        # Space-specific checks
        if space_type == 'hyperbolic':
            if 'scale_factor' in invariants and invariants['scale_factor'] <= 0:
                return True
                
        elif space_type == 'elliptical':
            if 'radius' in invariants and abs(invariants['radius'] - 1.0) > self.precision_tolerance:
                return True
                
        elif space_type == 'euclidean':
            if 'magnitude' in invariants and invariants['magnitude'] <= 0:
                return True
        
        # No issues found
        return False
    
    def _apply_fallback_correction(self, manifest: ZPAManifest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply fallback correction when main correction fails.
        
        This makes minimal assumptions about the vectors and uses more
        aggressive techniques to recover a valid state.
        
        Args:
            manifest: Manifest to correct
            
        Returns:
            Tuple of corrected (H, E, U) vectors
        """
        # Create fallback vectors
        H_fallback = np.copy(manifest.H)
        E_fallback = np.copy(manifest.E)
        U_fallback = np.copy(manifest.U)
        
        # Fix hyperbolic vector (must have norm < 1)
        h_norm = np.linalg.norm(H_fallback)
        if h_norm >= 1.0 or not np.all(np.isfinite(H_fallback)):
            # Replace with a valid hyperbolic vector with similar direction
            if h_norm > 0 and np.all(np.isfinite(H_fallback)):
                H_fallback = H_fallback * 0.5 / h_norm  # Scale to norm 0.5
            else:
                # Complete replacement with default vector
                H_fallback = np.array([0.5, 0.0, 0.0])
        
        # Fix elliptical vector (must have norm = 1)
        e_norm = np.linalg.norm(E_fallback)
        if abs(e_norm - 1.0) > self.precision_tolerance or not np.all(np.isfinite(E_fallback)):
            # Replace with a valid elliptical vector with similar direction
            if e_norm > 0 and np.all(np.isfinite(E_fallback)):
                E_fallback = E_fallback / e_norm  # Normalize to unit sphere
            else:
                # Complete replacement with default vector
                E_fallback = np.array([1.0, 0.0, 0.0])
        
        # Fix euclidean vector (must be finite)
        if not np.all(np.isfinite(U_fallback)):
            # Replace non-finite values with zeros
            U_fallback = np.where(np.isfinite(U_fallback), U_fallback, 0.0)
            
            # If all zeros, use a default vector
            if np.all(U_fallback == 0.0):
                U_fallback = np.array([0.0, 0.0, 1.0])
        
        return H_fallback, E_fallback, U_fallback
    
    def _optimize_component_coherence(self, target: np.ndarray, 
                                     ref1: np.ndarray, 
                                     ref2: np.ndarray,
                                     component: str) -> np.ndarray:
        """
        Optimize the coherence of a specific vector component.
        
        Args:
            target: Vector to optimize
            ref1: First reference vector
            ref2: Second reference vector
            component: Component name ('H', 'E', or 'U')
            
        Returns:
            Optimized vector
        """
        # Create a vector of candidates with small variations
        candidates = []
        
        # Add original vector
        candidates.append(target)
        
        # Add variations
        for scale in [0.98, 0.99, 1.01, 1.02]:
            # Scale the vector (except for E which must maintain unit norm)
            if component == 'E':
                # Apply rotation instead of scaling
                angle = 0.01 * scale
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                candidate = np.dot(rotation_matrix, target)
                # Ensure unit norm
                candidate = candidate / np.linalg.norm(candidate)
                candidates.append(candidate)
            else:
                # Apply scaling
                candidate = target * scale
                candidates.append(candidate)
        
        # Add noise-reduced variants
        for noise_scale in [0.1, 0.01]:
            # Create a variant that interpolates toward a "canonical" form
            if component == 'H':
                canonical = np.array([0.5, 0.0, 0.0])
            elif component == 'E':
                canonical = np.array([1.0, 0.0, 0.0])
            else:  # 'U'
                canonical = np.array([0.0, 0.0, 1.0])
                
            # Interpolate
            candidate = target * (1.0 - noise_scale) + canonical * noise_scale
            
            # Normalize E to unit sphere
            if component == 'E':
                candidate = candidate / np.linalg.norm(candidate)
                
            candidates.append(candidate)
        
        # Add variants that align more with the other components
        for align_scale in [0.1, 0.2]:
            # Align with ref1
            candidate1 = target * (1.0 - align_scale) + ref1 * align_scale
            if component == 'E':
                candidate1 = candidate1 / np.linalg.norm(candidate1)
            candidates.append(candidate1)
            
            # Align with ref2
            candidate2 = target * (1.0 - align_scale) + ref2 * align_scale
            if component == 'E':
                candidate2 = candidate2 / np.linalg.norm(candidate2)
            candidates.append(candidate2)
        
        # Evaluate all candidates
        best_score = -1
        best_candidate = target
        
        for candidate in candidates:
            # Construct the triple for coherence testing
            if component == 'H':
                triple = (candidate, ref1, ref2)
            elif component == 'E':
                triple = (ref1, candidate, ref2)
            else:  # 'U'
                triple = (ref1, ref2, candidate)
            
            # Calculate coherence
            _, coherence = validate_trilateral_coherence(triple)
            
            # Update best if improvement found
            if float(coherence) > best_score:
                best_score = float(coherence)
                best_candidate = candidate
        
        return best_candidate
  

class ProgressiveErrorCorrector:
    """
    Advanced error correction that progressively applies correction techniques.
    
    This applies increasingly aggressive correction techniques until coherence
    improves sufficiently, preserving as much of the original data as possible.
    """
    
    def __init__(self, base_corrector: ErrorCorrector = None):
        """
        Initialize the progressive error corrector.
        
        Args:
            base_corrector: Optional base ErrorCorrector instance
        """
        self.base_corrector = base_corrector or ErrorCorrector()
        self.coherence_threshold = self.base_corrector.coherence_threshold
        
        # Define correction levels with increasing aggressiveness
        self.correction_levels = [
            self._level1_correction,  # Mild normalization
            self._level2_correction,  # Component-specific optimization
            self._level3_correction,  # Cross-component alignment
            self._level4_correction,  # Aggressive reconstruction
            self._level5_correction   # Complete replacement
        ]
    
    def correct_manifest(self, manifest: ZPAManifest) -> Tuple[ZPAManifest, Dict[str, Any]]:
        """
        Progressively apply correction techniques until coherence improves.
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Tuple of (corrected_manifest, correction_details)
        """
        # Check if correction is needed
        has_errors, error_details = self.base_corrector.detect_errors(manifest)
        
        if not has_errors:
            # No correction needed
            return manifest, {'needed': False, 'level': 0, 'coherence': manifest.global_coherence}
        
        # Try each correction level until coherence improves sufficiently
        best_manifest = manifest
        best_coherence = manifest.global_coherence
        best_level = 0
        
        correction_details = {
            'original_coherence': manifest.global_coherence,
            'corrections_applied': [],
            'final_level': 0
        }
        
        for level, correction_func in enumerate(self.correction_levels, 1):
            # Apply this level of correction
            try:
                corrected = correction_func(manifest)
                correction_details['corrections_applied'].append(f"level{level}")
            except Exception as e:
                # If this level fails, skip to next level
                correction_details['corrections_applied'].append(f"level{level}_failed: {str(e)}")
                continue
            
            # Check if coherence improved
            if corrected.global_coherence > best_coherence:
                best_manifest = corrected
                best_coherence = corrected.global_coherence
                best_level = level
            
            # If we've reached sufficient coherence, stop
            if best_coherence >= self.coherence_threshold:
                break
        
        # Update correction details
        correction_details['final_level'] = best_level
        correction_details['coherence_improvement'] = best_coherence - manifest.global_coherence
        correction_details['successful'] = best_coherence >= self.coherence_threshold
        
        # Update manifest metadata
        best_manifest.metadata['error_corrected'] = True
        best_manifest.metadata['correction_level'] = best_level
        best_manifest.metadata['original_coherence'] = manifest.global_coherence
        
        return best_manifest, correction_details
    
    def _level1_correction(self, manifest: ZPAManifest) -> ZPAManifest:
        """
        Level 1 correction: Mild normalization.
        
        This applies basic normalization to ensure proper geometric properties.
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Corrected ZPA manifest
        """
        # Re-normalize all vectors
        H_normalized, H_invariants = normalize_with_invariants(manifest.H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(manifest.E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(manifest.U, "euclidean")
        
        # Create new manifest with normalized vectors
        corrected = ZPAManifest(
            H_normalized, E_normalized, U_normalized,
            H_invariants, E_invariants, U_invariants,
            manifest.metadata.copy()
        )
        
        return corrected
    
    def _level2_correction(self, manifest: ZPAManifest) -> ZPAManifest:
        """
        Level 2 correction: Component-specific optimization.
        
        This focuses on improving the component with lowest coherence.
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Corrected ZPA manifest
        """
        # Re-normalize all vectors first
        H_normalized, H_invariants = normalize_with_invariants(manifest.H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(manifest.E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(manifest.U, "euclidean")
        
        # Find component with lowest coherence
        lowest_coherence = min(manifest.H_coherence, manifest.E_coherence, manifest.U_coherence)
        
        if manifest.H_coherence == lowest_coherence:
            # Optimize H component
            H_normalized = self.base_corrector._optimize_component_coherence(
                H_normalized, E_normalized, U_normalized, 'H'
            )
        elif manifest.E_coherence == lowest_coherence:
            # Optimize E component
            E_normalized = self.base_corrector._optimize_component_coherence(
                E_normalized, H_normalized, U_normalized, 'E'
            )
        else:
            # Optimize U component
            U_normalized = self.base_corrector._optimize_component_coherence(
                U_normalized, H_normalized, E_normalized, 'U'
            )
        
        # Create new manifest with optimized vectors
        corrected = ZPAManifest(
            H_normalized, E_normalized, U_normalized,
            H_invariants, E_invariants, U_invariants,
            manifest.metadata.copy()
        )
        
        return corrected
    
    def _level3_correction(self, manifest: ZPAManifest) -> ZPAManifest:
        """
        Level 3 correction: Cross-component alignment.
        
        This aligns all components to improve cross-coherence.
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Corrected ZPA manifest
        """
        # Re-normalize all vectors first
        H_normalized, H_invariants = normalize_with_invariants(manifest.H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(manifest.E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(manifest.U, "euclidean")
        
        # Find the component with highest internal coherence to use as anchor
        highest_coherence = max(manifest.H_coherence, manifest.E_coherence, manifest.U_coherence)
        
        # Align other components to the anchor
        if manifest.H_coherence == highest_coherence:
            # H is anchor - align E and U to H
            E_normalized = self._align_vectors(E_normalized, H_normalized, 0.2, 'E')
            U_normalized = self._align_vectors(U_normalized, H_normalized, 0.2, 'U')
        elif manifest.E_coherence == highest_coherence:
            # E is anchor - align H and U to E
            H_normalized = self._align_vectors(H_normalized, E_normalized, 0.2, 'H')
            U_normalized = self._align_vectors(U_normalized, E_normalized, 0.2, 'U')
        else:
            # U is anchor - align H and E to U
            H_normalized = self._align_vectors(H_normalized, U_normalized, 0.2, 'H')
            E_normalized = self._align_vectors(E_normalized, U_normalized, 0.2, 'E')
        
        # Create new manifest with aligned vectors
        corrected = ZPAManifest(
            H_normalized, E_normalized, U_normalized,
            H_invariants, E_invariants, U_invariants,
            manifest.metadata.copy()
        )
        
        return corrected
    
    def _level4_correction(self, manifest: ZPAManifest) -> ZPAManifest:
        """
        Level 4 correction: Aggressive reconstruction.
        
        This partially reconstructs vectors based on canonical patterns.
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Corrected ZPA manifest
        """
        # Apply fallback correction to get minimally valid vectors
        H_fixed, E_fixed, U_fixed = self.base_corrector._apply_fallback_correction(manifest)
        
        # Create canonical reference vectors
        H_canonical = np.array([0.5, 0.0, 0.0])
        E_canonical = np.array([1.0, 0.0, 0.0])
        U_canonical = np.array([0.0, 0.0, 1.0])
        
        # Blend fixed vectors with canonical vectors
        blend_factor = 0.5
        H_blend = H_fixed * (1 - blend_factor) + H_canonical * blend_factor
        # For E, we need to maintain unit norm
        E_blend = E_fixed * (1 - blend_factor) + E_canonical * blend_factor
        E_blend = E_blend / np.linalg.norm(E_blend)
        U_blend = U_fixed * (1 - blend_factor) + U_canonical * blend_factor
        
        # Get invariants from the blended vectors
        _, H_invariants = normalize_with_invariants(H_blend, "hyperbolic")
        _, E_invariants = normalize_with_invariants(E_blend, "elliptical")
        _, U_invariants = normalize_with_invariants(U_blend, "euclidean")
        
        # Create new manifest with blended vectors
        corrected = ZPAManifest(
            H_blend, E_blend, U_blend,
            H_invariants, E_invariants, U_invariants,
            manifest.metadata.copy()
        )
        
        return corrected
    
    def _level5_correction(self, manifest: ZPAManifest) -> ZPAManifest:
        """
        Level 5 correction: Complete replacement.
        
        This completely replaces the vectors with canonical ones.
        
        Args:
            manifest: ZPA manifest to correct
            
        Returns:
            Corrected ZPA manifest with canonical vectors
        """
        # Create canonical vectors with guaranteed high coherence
        H_canonical = np.array([0.5, 0.0, 0.0])
        E_canonical = np.array([1.0, 0.0, 0.0])
        U_canonical = np.array([0.0, 0.0, 1.0])
        
        # Get invariants from the canonical vectors
        _, H_invariants = normalize_with_invariants(H_canonical, "hyperbolic")
        _, E_invariants = normalize_with_invariants(E_canonical, "elliptical")
        _, U_invariants = normalize_with_invariants(U_canonical, "euclidean")
        
        # Create new manifest with canonical vectors
        corrected = ZPAManifest(
            H_canonical, E_canonical, U_canonical,
            H_invariants, E_invariants, U_invariants,
            manifest.metadata.copy()
        )
        
        # Mark as completely replaced
        corrected.metadata['completely_replaced'] = True
        
        return corrected
    
    def _align_vectors(self, target: np.ndarray, reference: np.ndarray, 
                       blend_factor: float, component: str) -> np.ndarray:
        """
        Align target vector with reference vector.
        
        Args:
            target: Vector to align
            reference: Reference vector to align with
            blend_factor: Factor for blending (0-1)
            component: Component type ('H', 'E', or 'U')
            
        Returns:
            Aligned vector
        """
        # Simple linear interpolation with normalization if needed
        result = target * (1 - blend_factor) + reference * blend_factor
        
        # For elliptical vectors, ensure unit norm
        if component == 'E':
            result = result / np.linalg.norm(result)
        
        # For hyperbolic vectors, ensure norm < 1
        if component == 'H' and np.linalg.norm(result) >= 1.0:
            result = result * 0.9 / np.linalg.norm(result)
        
        return result