"""
ZPDR Processor Module

This module implements the full ZPDR encoding/decoding pipeline, including verification
mechanisms, error correction, and performance optimization features. It serves as the
main processing component for Phase 3 of the ZPDR implementation.

The processor handles the complete workflow:
1. Data encoding to multivector representation
2. Multivector to ZPA triple extraction
3. ZPA normalization, coherence calculation, and manifest generation
4. Verification of ZPA integrity
5. Error detection and correction
6. Reconstruction from ZPA back to original data
"""

import numpy as np
import threading
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set, TypeVar, Generic, Callable
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext

from .multivector import Multivector
from .geometric_spaces import (
    HyperbolicVector,
    EllipticalVector,
    EuclideanVector,
    SpaceTransformer,
    SpaceType
)
from .zpa_manifest import (
    ZPAManifest,
    create_zpa_manifest,
    serialize_zpa,
    deserialize_zpa
)

from ..utils import (
    to_decimal_array,
    calculate_internal_coherence,
    calculate_cross_coherence,
    calculate_global_coherence,
    validate_trilateral_coherence,
    normalize_with_invariants,
    denormalize_with_invariants,
    encode_to_zpdr_triple,
    reconstruct_from_zpdr_triple,
    embed_in_base,
    reconstruct_from_base,
    calculate_multibase_coherence,
    COHERENCE_THRESHOLD,
    PRECISION_TOLERANCE
)

# Type for values that can be encoded
DataValue = TypeVar('DataValue', int, str, bytes, float)

# Configuration constants
DEFAULT_CONFIG = {
    'coherence_threshold': float(COHERENCE_THRESHOLD),
    'precision_tolerance': float(PRECISION_TOLERANCE),
    'error_correction_enabled': True,
    'verification_level': 'standard',  # 'minimal', 'standard', 'strict'
    'multithreaded': False,
    'max_workers': 4,
    'optimization_level': 1,  # 0-3 (higher = more aggressive optimization)
    'serialization_format': 'full'  # 'full', 'compact', 'binary'
}

class ZPDRProcessor:
    """
    Main ZPDR Processor class that handles the full encoding/decoding pipeline.
    
    This class provides methods for encoding data to ZPA, decoding ZPA back to 
    data, verification, error correction, and optimization features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a ZPDR processor with configuration options.
        
        Args:
            config: Optional configuration dictionary overriding defaults
        """
        # Initialize configuration with defaults
        self.config = DEFAULT_CONFIG.copy()
        
        # Update with provided config if any
        if config:
            self.config.update(config)
        
        # Set up thread locks for multithreaded processing
        self._lock = threading.RLock()
        
        # Performance metrics tracking
        self._metrics = {
            'encode_calls': 0,
            'decode_calls': 0,
            'encode_time_ms': 0.0,
            'decode_time_ms': 0.0,
            'error_corrections': 0,
            'verification_failures': 0
        }
        
        # Cache for optimized processing
        self._cache = {}
    
    def encode(self, data: DataValue, description: str = None) -> ZPAManifest:
        """
        Encode data to a ZPA manifest.
        
        This method implements the full encoding pipeline:
        1. Convert data to multivector representation
        2. Extract ZPA triple (H, E, U)
        3. Normalize and calculate coherence
        4. Create and return manifest
        
        Args:
            data: Data value to encode (int, str, bytes, float)
            description: Optional description for the manifest
            
        Returns:
            ZPAManifest object containing the encoded data
        """
        # Track metrics
        start_time = time.time()
        self._metrics['encode_calls'] += 1
        
        try:
            # Step 1: Convert data to multivector
            multivector = self._data_to_multivector(data)
            
            # Step 2: Extract ZPA triple
            H, E, U = self._extract_trilateral_vector(multivector)
            
            # Step 3: Create manifest with normalization and coherence calculation
            manifest = create_zpa_manifest(H, E, U, description)
            
            # Step 4: Add metadata about the original data
            manifest.metadata['data_type'] = type(data).__name__
            if isinstance(data, (int, float)):
                manifest.metadata['data_value'] = str(data)
            elif isinstance(data, str) and len(data) < 100:
                manifest.metadata['data_value'] = data
            
            # Step 5: Verify the ZPA integrity
            if self.config['verification_level'] != 'minimal':
                self._verify_zpa_integrity(manifest.H, manifest.E, manifest.U)
            
            # Track performance metrics
            end_time = time.time()
            self._metrics['encode_time_ms'] += (end_time - start_time) * 1000
            
            return manifest
            
        except Exception as e:
            # Log the error and re-raise
            print(f"Error in ZPDR encoding: {e}")
            raise
    
    def decode(self, manifest: Union[ZPAManifest, str], verify: bool = True) -> DataValue:
        """
        Decode data from a ZPA manifest.
        
        This method implements the full decoding pipeline:
        1. Parse manifest if provided as string
        2. Verify ZPA integrity
        3. Extract ZPA triple
        4. Apply error correction if needed
        5. Reconstruct original data
        
        Args:
            manifest: ZPA manifest object or serialized string
            verify: Whether to verify ZPA integrity before decoding
            
        Returns:
            Original data value (int, str, etc.)
        """
        # Track metrics
        start_time = time.time()
        self._metrics['decode_calls'] += 1
        
        try:
            # Step 1: Parse manifest if it's a string
            if isinstance(manifest, str):
                manifest = ZPAManifest.deserialize(manifest)
            
            # Step 2: Verify ZPA integrity if requested
            if verify and self.config['verification_level'] != 'minimal':
                if not self._verify_zpa_integrity(manifest.H, manifest.E, manifest.U):
                    # Apply error correction if enabled
                    if self.config['error_correction_enabled']:
                        self._metrics['error_corrections'] += 1
                        manifest = self._apply_error_correction(manifest)
                    else:
                        raise ValueError("ZPA integrity verification failed")
            
            # Step 3: Reconstruct data from ZPA
            data_type = manifest.metadata.get('data_type', 'int')
            if data_type == 'int':
                result = reconstruct_from_zpdr_triple(manifest.H, manifest.E, manifest.U)
            elif data_type == 'str':
                result = self._reconstruct_string_from_zpa(manifest.H, manifest.E, manifest.U)
            elif data_type == 'float':
                result = self._reconstruct_float_from_zpa(manifest.H, manifest.E, manifest.U)
            elif data_type == 'bytes':
                result = self._reconstruct_bytes_from_zpa(manifest.H, manifest.E, manifest.U)
            else:
                # Default to int reconstruction
                result = reconstruct_from_zpdr_triple(manifest.H, manifest.E, manifest.U)
            
            # Track performance metrics
            end_time = time.time()
            self._metrics['decode_time_ms'] += (end_time - start_time) * 1000
            
            return result
            
        except Exception as e:
            # Log the error and re-raise
            print(f"Error in ZPDR decoding: {e}")
            raise
    
    def process_batch(self, data_values: List[DataValue]) -> List[ZPAManifest]:
        """
        Process a batch of data values in parallel.
        
        Args:
            data_values: List of data values to encode
            
        Returns:
            List of ZPA manifests corresponding to input values
        """
        # If multithreaded mode is enabled, use thread pool
        if self.config['multithreaded']:
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # Map the encode function to all values
                results = list(executor.map(self.encode, data_values))
                return results
        else:
            # Process sequentially
            return [self.encode(value) for value in data_values]
    
    def verify_manifest(self, manifest: Union[ZPAManifest, str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform comprehensive verification of a ZPA manifest.
        
        Args:
            manifest: ZPA manifest object or serialized string
            
        Returns:
            Tuple of (is_valid, verification_results)
        """
        # Parse manifest if it's a string
        if isinstance(manifest, str):
            try:
                manifest = ZPAManifest.deserialize(manifest)
            except Exception as e:
                return False, {'error': f"Failed to parse manifest: {e}"}
        
        # Initialize verification results
        results = {
            'coherence': {
                'global': manifest.global_coherence,
                'H': manifest.H_coherence,
                'E': manifest.E_coherence,
                'U': manifest.U_coherence,
                'HE': manifest.HE_coherence,
                'EU': manifest.EU_coherence,
                'HU': manifest.HU_coherence
            },
            'invariants': {
                'H': manifest.H_invariants,
                'E': manifest.E_invariants,
                'U': manifest.U_invariants
            },
            'vectors': {
                'H_valid': self._is_valid_hyperbolic(manifest.H),
                'E_valid': self._is_valid_elliptical(manifest.E),
                'U_valid': self._is_valid_euclidean(manifest.U)
            },
            'metadata': manifest.metadata
        }
        
        # Check global coherence threshold
        results['coherence_valid'] = manifest.global_coherence >= self.config['coherence_threshold']
        
        # Check invariant consistency
        results['invariants_valid'] = self._verify_invariant_consistency(
            manifest.H_invariants,
            manifest.E_invariants,
            manifest.U_invariants
        )
        
        # Overall validity
        is_valid = (results['coherence_valid'] and
                   results['invariants_valid'] and
                   results['vectors']['H_valid'] and
                   results['vectors']['E_valid'] and
                   results['vectors']['U_valid'])
        
        # Additional tamper detection for strict verification
        if self.config['verification_level'] == 'strict':
            tamper_check = self._detect_tampering(manifest)
            results['tamper_detection'] = tamper_check
            is_valid = is_valid and not tamper_check['tampered']
        
        # Track verification failures
        if not is_valid:
            self._metrics['verification_failures'] += 1
        
        return is_valid, results
    
    def correct_errors(self, manifest: Union[ZPAManifest, str]) -> ZPAManifest:
        """
        Apply error correction to a ZPA manifest.
        
        Args:
            manifest: ZPA manifest object or serialized string
            
        Returns:
            Corrected ZPA manifest
        """
        # Parse manifest if it's a string
        if isinstance(manifest, str):
            manifest = ZPAManifest.deserialize(manifest)
        
        # Apply error correction
        corrected_manifest = self._apply_error_correction(manifest)
        
        # Track error correction
        self._metrics['error_corrections'] += 1
        
        return corrected_manifest
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the processor.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate average times if there are enough calls
        metrics = self._metrics.copy()
        
        if metrics['encode_calls'] > 0:
            metrics['avg_encode_time_ms'] = metrics['encode_time_ms'] / metrics['encode_calls']
        
        if metrics['decode_calls'] > 0:
            metrics['avg_decode_time_ms'] = metrics['decode_time_ms'] / metrics['decode_calls']
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self._metrics = {
            'encode_calls': 0,
            'decode_calls': 0,
            'encode_time_ms': 0.0,
            'decode_time_ms': 0.0,
            'error_corrections': 0,
            'verification_failures': 0
        }
    
    def optimize(self, level: Optional[int] = None) -> None:
        """
        Set optimization level and clear caches.
        
        Args:
            level: Optimization level (0-3, higher = more aggressive)
        """
        if level is not None:
            self.config['optimization_level'] = max(0, min(3, level))
        
        # Clear caches to ensure they're rebuilt with new optimization settings
        self._cache = {}
    
    # ===== Private helper methods =====
    
    def _data_to_multivector(self, data: DataValue) -> Multivector:
        """Convert data to a multivector representation."""
        if isinstance(data, int):
            # For integers, use the existing utilities
            # (they already handle multivector creation)
            return self._create_multivector_for_value(data)
        
        elif isinstance(data, str):
            # Convert string to integer values and then to multivector
            # First, convert to bytes using UTF-8 encoding
            bytes_data = data.encode('utf-8')
            
            # Then treat as bytes
            return self._bytes_to_multivector(bytes_data)
        
        elif isinstance(data, bytes):
            # Convert bytes directly to multivector
            return self._bytes_to_multivector(data)
        
        elif isinstance(data, float):
            # Convert float to multivector using a specialized approach
            return self._float_to_multivector(data)
        
        else:
            # Unsupported type
            raise TypeError(f"Unsupported data type for ZPDR encoding: {type(data)}")
    
    def _create_multivector_for_value(self, value: int) -> Multivector:
        """Create a multivector representation from an integer value."""
        # Convert to Decimal for high precision
        dec_value = Decimal(str(value))
        
        # Generate representations in multiple bases for multidimensional embedding
        base2_digits = embed_in_base(value, 2)   # Binary
        base3_digits = embed_in_base(value, 3)   # Ternary
        base10_digits = embed_in_base(value, 10) # Decimal
        
        # Initialize components dictionary
        components = {}
        
        # Calculate bit length to determine the scale factor for normalization
        bit_length = value.bit_length() or 1  # Ensure at least 1 to avoid division by zero
        
        # Scalar component from value magnitude (normalized)
        scalar_factor = Decimal(str(1.0)) / Decimal(str(max(1, bit_length)))
        components["1"] = float(scalar_factor * dec_value)
        
        # Vector components from base-2 and base-3 representations
        vector_scale = min(1.0, 3.0 / max(1, len(base2_digits)))
        
        # Set vector components based on binary representation
        for i, digit in enumerate(base2_digits[:3]):  # Use first 3 digits
            if i < 3:  # Limit to 3D vector
                basis = f"e{i+1}"
                components[basis] = float(digit * vector_scale)
        
        # Ensure we have all three vector components
        for i in range(3):
            basis = f"e{i+1}"
            if basis not in components:
                components[basis] = 0.0
        
        # Bivector components from digit patterns
        # e12 from binary digit patterns
        if len(base2_digits) >= 2:
            e12_value = sum(d1 ^ d2 for d1, d2 in zip(base2_digits[:-1], base2_digits[1:]))
            components["e12"] = float(e12_value / max(1, len(base2_digits) - 1))
        else:
            components["e12"] = 0.0
            
        # e23 from ternary digit patterns
        if len(base3_digits) >= 2:
            e23_value = sum((d1 + d2) % 3 for d1, d2 in zip(base3_digits[:-1], base3_digits[1:]))
            components["e23"] = float(e23_value / max(1, len(base3_digits) - 1))
        else:
            components["e23"] = 0.0
            
        # e31 from decimal digit patterns
        if len(base10_digits) >= 2:
            e31_value = sum((d1 * d2) % 10 for d1, d2 in zip(base10_digits[:-1], base10_digits[1:]))
            components["e31"] = float(e31_value / max(1, 9 * (len(base10_digits) - 1)))
        else:
            components["e31"] = 0.0
        
        # Trivector component from overall normalized value
        components["e123"] = float(dec_value / Decimal(str(10 ** max(1, len(base10_digits)))))
        
        # Create and return the multivector
        return Multivector(components)
    
    def _bytes_to_multivector(self, bytes_data: bytes) -> Multivector:
        """Convert bytes to a multivector representation."""
        # For simplicity in Phase 3, we'll convert bytes to an integer first
        # More sophisticated implementations can map byte patterns directly
        
        # Convert bytes to a large integer (big-endian)
        value = int.from_bytes(bytes_data, byteorder='big')
        
        # Use the integer-to-multivector conversion but add type information
        multivector = self._create_multivector_for_value(value)
        
        # Add a special component to mark this as bytes-sourced
        components = multivector.components.copy()
        components["bytes_length"] = len(bytes_data)
        
        return Multivector(components)
    
    def _float_to_multivector(self, value: float) -> Multivector:
        """Convert float to a multivector representation."""
        # Extract mantissa and exponent from float
        import struct
        import math
        
        # Handle special cases
        if math.isnan(value):
            # Encode NaN as a special marker
            return Multivector({"1": 0.0, "e1": 0.0, "e2": 0.0, "e3": 0.0, 
                               "e12": 0.0, "e23": 0.0, "e31": 0.0, "e123": 0.0,
                               "float_nan": 1.0})
        
        if math.isinf(value):
            # Encode +/- infinity
            sign = 1.0 if value > 0 else -1.0
            return Multivector({"1": sign, "e1": 0.0, "e2": 0.0, "e3": 0.0, 
                               "e12": 0.0, "e23": 0.0, "e31": 0.0, "e123": 0.0,
                               "float_inf": 1.0})
        
        # Decompose float into sign, mantissa, and exponent
        # IEEE 754 format: sign * mantissa * 2^exponent
        bits = struct.unpack('!Q', struct.pack('!d', value))[0]
        sign = -1.0 if (bits >> 63) & 1 else 1.0
        exponent = ((bits >> 52) & 0x7FF) - 1023  # Bias is 1023 for double precision
        mantissa = 1.0 + float((bits & 0xFFFFFFFFFFFFF) / 2**52)
        
        # Create base multivector from the mantissa
        int_mantissa = int(mantissa * 2**52)  # Convert to integer with precision
        base_multivector = self._create_multivector_for_value(int_mantissa)
        
        # Adjust components with sign and exponent
        components = base_multivector.components.copy()
        
        # Add float-specific markers
        components["float_sign"] = sign
        components["float_exponent"] = float(exponent)
        
        # Create final multivector
        return Multivector(components)
    
    def _extract_trilateral_vector(self, multivector: Multivector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract the trilateral vector (H, E, U) from a multivector."""
        from ..utils import _extract_trilateral_vectors
        
        # Use the existing extraction function from utils
        H, E, U = _extract_trilateral_vectors(multivector)
        
        # Apply optimization if enabled
        if self.config['optimization_level'] >= 2:
            # Advanced optimizations can be applied here in the future
            pass
        
        return H, E, U
    
    def _verify_zpa_integrity(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> bool:
        """Verify the integrity of a ZPA triple."""
        # Use the validation utility to check coherence
        is_valid, coherence = validate_trilateral_coherence((H, E, U))
        
        # Additional verification for more strict levels
        if self.config['verification_level'] == 'strict':
            # Also verify that H, E, U have the correct properties
            h_valid = self._is_valid_hyperbolic(H)
            e_valid = self._is_valid_elliptical(E)
            u_valid = self._is_valid_euclidean(U)
            
            # Check invariant consistency if vectors are normalized
            try:
                _, H_invariants = normalize_with_invariants(H, "hyperbolic")
                _, E_invariants = normalize_with_invariants(E, "elliptical")
                _, U_invariants = normalize_with_invariants(U, "euclidean")
                
                invariants_valid = self._verify_invariant_consistency(
                    H_invariants, E_invariants, U_invariants
                )
            except:
                invariants_valid = False
            
            return is_valid and h_valid and e_valid and u_valid and invariants_valid
        
        return is_valid
    
    def _is_valid_hyperbolic(self, vector: np.ndarray) -> bool:
        """Check if a vector is valid in hyperbolic space."""
        # In the Poincaré disk model, the norm must be < 1
        return np.linalg.norm(vector) < 1.0
    
    def _is_valid_elliptical(self, vector: np.ndarray) -> bool:
        """Check if a vector is valid in elliptical space."""
        # On the unit sphere, the norm should be very close to 1
        return abs(np.linalg.norm(vector) - 1.0) < 1e-10
    
    def _is_valid_euclidean(self, vector: np.ndarray) -> bool:
        """Check if a vector is valid in euclidean space."""
        # In euclidean space, all finite vectors are valid
        return np.all(np.isfinite(vector))
    
    def _verify_invariant_consistency(self, 
                                     H_invariants: Dict[str, float],
                                     E_invariants: Dict[str, float],
                                     U_invariants: Dict[str, float]) -> bool:
        """Verify that invariants are mathematically consistent."""
        # Check rotation angles are in a valid range
        for invariants in [H_invariants, E_invariants, U_invariants]:
            if 'rotation_angle' in invariants:
                angle = invariants['rotation_angle']
                if not (-np.pi <= angle <= np.pi):
                    return False
        
        # Check scale factors are positive
        if 'scale_factor' in H_invariants and H_invariants['scale_factor'] <= 0:
            return False
            
        # In elliptical space, radius should be 1.0 or very close
        if 'radius' in E_invariants and abs(E_invariants['radius'] - 1.0) > 1e-10:
            return False
            
        # Check euclidean magnitude is positive
        if 'magnitude' in U_invariants and U_invariants['magnitude'] <= 0:
            return False
        
        return True
    
    def _detect_tampering(self, manifest: ZPAManifest) -> Dict[str, Any]:
        """Detect potential tampering in a ZPA manifest."""
        result = {
            'tampered': False,
            'issues': []
        }
        
        # Check for inconsistencies between vectors and coherence values
        calculated_coherence = float(calculate_internal_coherence(manifest.H))
        if abs(calculated_coherence - manifest.H_coherence) > 0.01:
            result['tampered'] = True
            result['issues'].append('H_coherence mismatch')
        
        calculated_coherence = float(calculate_internal_coherence(manifest.E))
        if abs(calculated_coherence - manifest.E_coherence) > 0.01:
            result['tampered'] = True
            result['issues'].append('E_coherence mismatch')
        
        calculated_coherence = float(calculate_internal_coherence(manifest.U))
        if abs(calculated_coherence - manifest.U_coherence) > 0.01:
            result['tampered'] = True
            result['issues'].append('U_coherence mismatch')
        
        # Check for cross-coherence inconsistencies
        calculated_coherence = float(calculate_cross_coherence(manifest.H, manifest.E))
        if abs(calculated_coherence - manifest.HE_coherence) > 0.01:
            result['tampered'] = True
            result['issues'].append('HE_coherence mismatch')
        
        # Check that E vector is properly normalized (unit sphere)
        if abs(np.linalg.norm(manifest.E) - 1.0) > 1e-10:
            result['tampered'] = True
            result['issues'].append('E vector not normalized')
        
        return result
    
    def _apply_error_correction(self, manifest: ZPAManifest) -> ZPAManifest:
        """Apply error correction to a ZPA manifest."""
        # Step 1: Re-normalize all vectors to canonical form
        H_normalized, H_invariants = normalize_with_invariants(manifest.H, "hyperbolic")
        E_normalized, E_invariants = normalize_with_invariants(manifest.E, "elliptical")
        U_normalized, U_invariants = normalize_with_invariants(manifest.U, "euclidean")
        
        # Step 2: Apply denormalization with extracted invariants
        H_corrected = denormalize_with_invariants(H_normalized, H_invariants, "hyperbolic")
        E_corrected = denormalize_with_invariants(E_normalized, E_invariants, "elliptical")
        U_corrected = denormalize_with_invariants(U_normalized, U_invariants, "euclidean")
        
        # Step 3: Create new manifest with corrected vectors
        corrected_manifest = create_zpa_manifest(
            H_corrected, E_corrected, U_corrected,
            manifest.metadata.get('description', 'Error corrected ZPA')
        )
        
        # Step 4: Preserve metadata from original manifest
        corrected_manifest.metadata.update(manifest.metadata)
        corrected_manifest.metadata['error_corrected'] = True
        
        return corrected_manifest
    
    def _reconstruct_string_from_zpa(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> str:
        """Reconstruct a string from ZPA vectors."""
        # Reconstruct the bytes value first
        bytes_value = self._reconstruct_bytes_from_zpa(H, E, U)
        
        # Decode bytes as UTF-8
        try:
            return bytes_value.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to a simpler encoding if UTF-8 fails
            return bytes_value.decode('latin-1')
    
    def _reconstruct_bytes_from_zpa(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> bytes:
        """Reconstruct bytes from ZPA vectors."""
        # First reconstruct the multivector
        multivector = self._reconstruct_multivector_from_zpa(H, E, U)
        
        # Extract the integer value
        value = reconstruct_from_zpdr_triple(H, E, U)
        
        # Get the bytes length if available
        bytes_length = int(multivector.components.get("bytes_length", 0))
        
        # Convert integer back to bytes
        if bytes_length > 0:
            # We know the exact length
            return value.to_bytes(bytes_length, byteorder='big')
        else:
            # Guess the minimum length needed
            length = (value.bit_length() + 7) // 8
            return value.to_bytes(max(1, length), byteorder='big')
    
    def _reconstruct_float_from_zpa(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> float:
        """Reconstruct a float from ZPA vectors."""
        # First reconstruct the multivector
        multivector = self._reconstruct_multivector_from_zpa(H, E, U)
        
        # Check for special values
        if "float_nan" in multivector.components and multivector.components["float_nan"] > 0.5:
            return float('nan')
            
        if "float_inf" in multivector.components and multivector.components["float_inf"] > 0.5:
            sign = multivector.components.get("1", 1.0)
            return float('inf') if sign > 0 else float('-inf')
        
        # Extract float components
        sign = multivector.components.get("float_sign", 1.0)
        exponent = multivector.components.get("float_exponent", 0.0)
        
        # Reconstruct mantissa from the integer value
        int_value = reconstruct_from_zpdr_triple(H, E, U)
        mantissa = int_value / 2**52
        
        # Combine components
        result = sign * mantissa * (2.0 ** exponent)
        
        return result
    
    def _reconstruct_multivector_from_zpa(self, H: np.ndarray, E: np.ndarray, U: np.ndarray) -> Multivector:
        """Reconstruct a multivector from ZPA vectors."""
        from ..utils import _reconstruct_multivector_from_trilateral
        return _reconstruct_multivector_from_trilateral(H, E, U)


class StreamingZPDRProcessor:
    """
    Streaming version of the ZPDR processor for handling large data streams.
    
    This class provides methods for processing data in chunks, maintaining
    consistent state across chunks, and enabling efficient streaming operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a streaming ZPDR processor.
        
        Args:
            config: Optional configuration dictionary
        """
        # Set up core processor with the same config
        self.processor = ZPDRProcessor(config)
        
        # Additional streaming-specific configuration
        self.config = self.processor.config.copy()
        self.config.update({
            'chunk_size': 1024,  # Default chunk size in bytes
            'buffer_size': 8192  # Internal buffer size
        })
        
        # Update with provided config if any
        if config:
            self.config.update(config)
        
        # Initialize state for streaming
        self._buffer = bytearray()
        self._state = {}
        self._manifests = []
    
    def encode_stream(self, data_stream, chunk_size: Optional[int] = None) -> List[ZPAManifest]:
        """
        Encode a data stream into a series of ZPA manifests.
        
        Args:
            data_stream: Stream-like object supporting read()
            chunk_size: Optional custom chunk size
            
        Returns:
            List of ZPA manifests representing the stream
        """
        # Use default chunk size if not specified
        if chunk_size is None:
            chunk_size = self.config['chunk_size']
        
        # Clear previous state
        self._buffer = bytearray()
        self._manifests = []
        
        # Process the stream in chunks
        while True:
            # Read a chunk
            chunk = data_stream.read(chunk_size)
            
            # If we've reached the end of the stream
            if not chunk:
                # Process any remaining data in the buffer
                if self._buffer:
                    manifest = self.processor.encode(bytes(self._buffer))
                    self._manifests.append(manifest)
                    self._buffer = bytearray()
                break
            
            # Add chunk to buffer
            self._buffer.extend(chunk)
            
            # Process buffer if it has at least one chunk
            while len(self._buffer) >= chunk_size:
                # Extract a chunk from the buffer
                process_chunk = bytes(self._buffer[:chunk_size])
                self._buffer = self._buffer[chunk_size:]
                
                # Encode the chunk
                manifest = self.processor.encode(process_chunk)
                self._manifests.append(manifest)
        
        return self._manifests
    
    def decode_stream(self, manifests: List[Union[ZPAManifest, str]]) -> bytes:
        """
        Decode a series of ZPA manifests back to a byte stream.
        
        Args:
            manifests: List of ZPA manifests or serialized strings
            
        Returns:
            Reconstructed data as bytes
        """
        # Decode each manifest and concatenate results
        result = bytearray()
        
        for manifest in manifests:
            # Decode manifest
            chunk_data = self.processor.decode(manifest)
            
            # If the result is not bytes, convert it
            if isinstance(chunk_data, str):
                chunk_data = chunk_data.encode('utf-8')
            elif isinstance(chunk_data, int):
                # Determine the minimum number of bytes needed
                byte_length = (chunk_data.bit_length() + 7) // 8
                chunk_data = chunk_data.to_bytes(max(1, byte_length), byteorder='big')
            
            # Add to result
            result.extend(chunk_data)
        
        return bytes(result)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the processor.
        
        Returns:
            Dictionary of performance metrics
        """
        # Get core processor metrics
        metrics = self.processor.get_performance_metrics()
        
        # Add streaming-specific metrics
        metrics.update({
            'chunks_processed': len(self._manifests),
            'buffer_size': len(self._buffer)
        })
        
        return metrics