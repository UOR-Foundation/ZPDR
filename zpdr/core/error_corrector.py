#!/usr/bin/env python3
"""
Error correction utilities for ZPDR.

This module provides error detection and correction capabilities for ZPDR operations,
helping to recover from small errors in zero-point coordinates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from decimal import Decimal, getcontext
import itertools

from .trivector_digest import TrivectorDigest
from .manifest import Manifest
from ..utils.coherence_calculator import CoherenceCalculator

# Set high precision for decimal calculations
getcontext().prec = 128

class ErrorCorrector:
    """
    Error correction for ZPDR coordinates.
    
    This class provides methods for detecting and correcting errors in
    zero-point coordinates, improving reconstruction reliability.
    """
    
    def __init__(self, 
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-10,
                 coherence_target: float = 0.999):
        """
        Initialize the error corrector.
        
        Args:
            max_iterations: Maximum number of correction iterations
            convergence_threshold: Threshold for convergence detection
            coherence_target: Target coherence value for corrections
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = Decimal(str(convergence_threshold))
        self.coherence_target = Decimal(str(coherence_target))
        self.coherence_calculator = CoherenceCalculator()
    
    def correct_coordinates(self, 
                           manifest: Manifest) -> Tuple[TrivectorDigest, Dict[str, Any]]:
        """
        Attempt to correct coordinates in a manifest to improve coherence.
        
        Args:
            manifest: Manifest containing potentially imprecise coordinates
            
        Returns:
            Tuple of (corrected TrivectorDigest, correction details)
        """
        # Extract original coordinates
        hyperbolic = manifest.trivector_digest.hyperbolic.copy()
        elliptical = manifest.trivector_digest.elliptical.copy()
        euclidean = manifest.trivector_digest.euclidean.copy()
        
        # Calculate initial coherence
        initial_coherence = self.coherence_calculator.calculate_full_coherence(
            hyperbolic, elliptical, euclidean)
        
        initial_global = initial_coherence['global_coherence']
        
        # If initial coherence is already good, return without changes
        if initial_global >= self.coherence_target:
            return manifest.trivector_digest, {
                'corrections_applied': False,
                'iterations': 0,
                'initial_coherence': initial_global,
                'final_coherence': initial_global,
                'improvement': Decimal('0'),
                'correction_path': []
            }
        
        # Try to correct coordinates
        best_coherence = initial_global
        best_coordinates = {
            'hyperbolic': hyperbolic.copy(),
            'elliptical': elliptical.copy(),
            'euclidean': euclidean.copy()
        }
        
        # Correction history
        correction_path = []
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Current coordinates
            current_h = best_coordinates['hyperbolic']
            current_e = best_coordinates['elliptical']
            current_u = best_coordinates['euclidean']
            
            # Get suggested adjustments
            adjustments = self.coherence_calculator.suggest_adjustments(
                self.coherence_calculator.calculate_full_coherence(current_h, current_e, current_u),
                current_h, current_e, current_u
            )
            
            # Calculate coherence with adjusted vectors
            adjusted_coherence = self.coherence_calculator.estimate_improvement(
                self.coherence_calculator.calculate_full_coherence(current_h, current_e, current_u),
                adjustments
            )
            
            adjusted_global = adjusted_coherence['global_coherence']
            
            # Record this step
            correction_path.append({
                'iteration': iteration,
                'adjustment_type': 'suggested',
                'coherence_before': best_coherence,
                'coherence_after': adjusted_global,
                'improvement': adjusted_global - best_coherence
            })
            
            # Update best if improved
            if adjusted_global > best_coherence:
                improvement = adjusted_global - best_coherence
                best_coherence = adjusted_global
                best_coordinates = adjustments.copy()
                
                # Check if we've reached the target
                if best_coherence >= self.coherence_target:
                    break
                
                # Check convergence
                if improvement < self.convergence_threshold:
                    break
            else:
                # Try a more aggressive approach: snap to nearest rational value
                # This can help when we're close to the optimal solution
                adjusted_coordinates = self._snap_to_rational(
                    current_h, current_e, current_u)
                
                adjusted_coherence = self.coherence_calculator.calculate_full_coherence(
                    adjusted_coordinates['hyperbolic'],
                    adjusted_coordinates['elliptical'],
                    adjusted_coordinates['euclidean']
                )
                
                adjusted_global = adjusted_coherence['global_coherence']
                
                # Record this step
                correction_path.append({
                    'iteration': iteration,
                    'adjustment_type': 'rational_snap',
                    'coherence_before': best_coherence,
                    'coherence_after': adjusted_global,
                    'improvement': adjusted_global - best_coherence
                })
                
                # Update best if improved
                if adjusted_global > best_coherence:
                    best_coherence = adjusted_global
                    best_coordinates = adjusted_coordinates.copy()
                    
                    # Check if we've reached the target
                    if best_coherence >= self.coherence_target:
                        break
                else:
                    # No improvement from either method, stop iterations
                    break
        
        # Create corrected trivector digest
        corrected_digest = TrivectorDigest(
            best_coordinates['hyperbolic'],
            best_coordinates['elliptical'],
            best_coordinates['euclidean']
        )
        
        # Return correction details
        correction_details = {
            'corrections_applied': True,
            'iterations': len(correction_path),
            'initial_coherence': initial_global,
            'final_coherence': best_coherence,
            'improvement': best_coherence - initial_global,
            'correction_path': correction_path
        }
        
        return corrected_digest, correction_details
    
    def _snap_to_rational(self, 
                         hyperbolic: List[Decimal],
                         elliptical: List[Decimal],
                         euclidean: List[Decimal]) -> Dict[str, List[Decimal]]:
        """
        Snap vector coordinates to nearest "nice" rational values.
        
        This function attempts to find simpler rational representations
        of the coordinates that might represent the exact intended values.
        
        Args:
            hyperbolic: Hyperbolic vector coordinates
            elliptical: Elliptical vector coordinates
            euclidean: Euclidean vector coordinates
            
        Returns:
            Dictionary of adjusted coordinates
        """
        snapped_coordinates = {
            'hyperbolic': self._snap_vector_to_rational(hyperbolic),
            'elliptical': self._snap_vector_to_rational(elliptical),
            'euclidean': self._snap_vector_to_rational(euclidean)
        }
        
        # Ensure normalization
        h_norm = sum(v * v for v in snapped_coordinates['hyperbolic']).sqrt()
        if h_norm > Decimal('0'):
            snapped_coordinates['hyperbolic'] = [h / h_norm for h in snapped_coordinates['hyperbolic']]
            
        e_norm = sum(v * v for v in snapped_coordinates['elliptical']).sqrt()
        if e_norm > Decimal('0'):
            snapped_coordinates['elliptical'] = [e / e_norm for e in snapped_coordinates['elliptical']]
            
        u_norm = sum(v * v for v in snapped_coordinates['euclidean']).sqrt()
        if u_norm > Decimal('0'):
            snapped_coordinates['euclidean'] = [u / u_norm for u in snapped_coordinates['euclidean']]
        
        return snapped_coordinates
    
    def _snap_vector_to_rational(self, vector: List[Decimal]) -> List[Decimal]:
        """
        Snap a vector's components to nearest rational values.
        
        Args:
            vector: Vector coordinates
            
        Returns:
            List of snapped coordinates
        """
        snapped = []
        
        for v in vector:
            # Try different denominators to find a good approximation
            best_approx = v
            min_diff = Decimal('1')
            
            # Try denominators up to 16
            for denom in range(1, 17):
                # Find nearest fraction with this denominator
                num = round(float(v * denom))
                approx = Decimal(num) / Decimal(denom)
                diff = abs(approx - v)
                
                if diff < min_diff:
                    min_diff = diff
                    best_approx = approx
            
            # For values very close to nice fractions like 1/2, 1/3, etc.,
            # use those instead
            common_fractions = [
                Decimal('0'), Decimal('1'), Decimal('0.5'),
                Decimal('1') / Decimal('3'), Decimal('2') / Decimal('3'),
                Decimal('0.25'), Decimal('0.75'),
                Decimal('1') / Decimal('6'), Decimal('5') / Decimal('6')
            ]
            
            for frac in common_fractions:
                diff = abs(v - frac)
                if diff < Decimal('0.02'):  # 2% threshold
                    if diff < min_diff:
                        min_diff = diff
                        best_approx = frac
            
            snapped.append(best_approx)
        
        return snapped
    
    def optimize_coherence(self, digest: TrivectorDigest) -> Tuple[TrivectorDigest, Dict[str, Any]]:
        """
        Optimize a trivector digest to maximize coherence.
        
        Args:
            digest: TrivectorDigest to optimize
            
        Returns:
            Tuple of (optimized TrivectorDigest, optimization details)
        """
        # Copy original vectors
        hyperbolic = digest.hyperbolic.copy()
        elliptical = digest.elliptical.copy()
        euclidean = digest.euclidean.copy()
        
        # Calculate initial coherence
        initial_coherence = self.coherence_calculator.calculate_full_coherence(
            hyperbolic, elliptical, euclidean)
        
        initial_global = initial_coherence['global_coherence']
        
        # Grid search for optimal orientations
        best_coherence = initial_global
        best_vectors = {
            'hyperbolic': hyperbolic,
            'elliptical': elliptical,
            'euclidean': euclidean
        }
        
        # Try different sign variations
        for h_signs in list(itertools.product([-1, 1], repeat=3))[:4]:  # Limit to 4 combinations
            for e_signs in list(itertools.product([-1, 1], repeat=3))[:4]:  # Limit to 4 combinations
                # Apply sign variations
                h = [s * v for s, v in zip(h_signs, hyperbolic)]
                e = [s * v for s, v in zip(e_signs, elliptical)]
                
                # Normalize
                h_norm = sum(v * v for v in h).sqrt()
                if h_norm > Decimal('0'):
                    h = [v / h_norm for v in h]
                
                e_norm = sum(v * v for v in e).sqrt()
                if e_norm > Decimal('0'):
                    e = [v / e_norm for v in e]
                
                # Calculate coherence with this variation
                coherence = self.coherence_calculator.calculate_full_coherence(h, e, euclidean)
                global_coherence = coherence['global_coherence']
                
                if global_coherence > best_coherence:
                    best_coherence = global_coherence
                    best_vectors = {
                        'hyperbolic': h,
                        'elliptical': e,
                        'euclidean': euclidean
                    }
        
        # Create optimized digest
        optimized_digest = TrivectorDigest(
            best_vectors['hyperbolic'],
            best_vectors['elliptical'],
            best_vectors['euclidean']
        )
        
        # Return optimization details
        optimization_details = {
            'initial_coherence': initial_global,
            'final_coherence': best_coherence,
            'improvement': best_coherence - initial_global,
            'coherence_target': self.coherence_target
        }
        
        return optimized_digest, optimization_details