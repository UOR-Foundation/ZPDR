"""
Zero-Point Address (ZPA) Manifest Module for ZPDR

This module implements serialization and deserialization of ZPA manifests,
which are the standard format for storing and transmitting Zero-Point Addresses.
A ZPA manifest contains the three geometric vectors (H, E, U) that form the
Zero-Point Address, along with their invariants, coherence measures, and metadata.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from decimal import Decimal
from datetime import datetime

from .geometric_spaces import (
    HyperbolicVector,
    EllipticalVector,
    EuclideanVector,
    SpaceType
)

from ..utils import (
    to_decimal_array,
    calculate_internal_coherence,
    calculate_cross_coherence,
    calculate_global_coherence,
    validate_trilateral_coherence,
    normalize_with_invariants,
    denormalize_with_invariants,
    COHERENCE_THRESHOLD
)

# Current manifest version
MANIFEST_VERSION = "2.0"

class ZPAManifest:
    """
    Zero-Point Address Manifest Class

    Encapsulates the ZPA triple (H, E, U vectors), their invariants,
    coherence measures, and metadata in a serializable format.
    """
    
    def __init__(self, 
                 H_vector: Optional[np.ndarray] = None,
                 E_vector: Optional[np.ndarray] = None,
                 U_vector: Optional[np.ndarray] = None,
                 H_invariants: Optional[Dict[str, float]] = None,
                 E_invariants: Optional[Dict[str, float]] = None,
                 U_invariants: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a ZPA manifest with vector data and metadata.

        Args:
            H_vector: Hyperbolic vector component
            E_vector: Elliptical vector component
            U_vector: Euclidean vector component
            H_invariants: Invariants for the hyperbolic vector
            E_invariants: Invariants for the elliptical vector
            U_invariants: Invariants for the euclidean vector
            metadata: Additional metadata about the ZPA
        """
        # Initialize vectors
        self.H = H_vector if H_vector is not None else np.zeros(3)
        self.E = E_vector if E_vector is not None else np.array([1.0, 0.0, 0.0])
        self.U = U_vector if U_vector is not None else np.zeros(3)
        
        # Convert to proper vector types if they're not already
        self.H_vector = self.H if isinstance(self.H, HyperbolicVector) else HyperbolicVector(self.H)
        self.E_vector = self.E if isinstance(self.E, EllipticalVector) else EllipticalVector(self.E)
        self.U_vector = self.U if isinstance(self.U, EuclideanVector) else EuclideanVector(self.U)
        
        # Get the numpy arrays for easier manipulation
        self.H = self.H_vector.components
        self.E = self.E_vector.components
        self.U = self.U_vector.components
        
        # Initialize invariants
        self.H_invariants = H_invariants.copy() if H_invariants is not None else {}
        self.E_invariants = E_invariants.copy() if E_invariants is not None else {}
        self.U_invariants = U_invariants.copy() if U_invariants is not None else {}
        
        # Calculate coherence values
        self._calculate_coherence()
        
        # Initialize metadata
        self.metadata = metadata.copy() if metadata is not None else {}
        
        # Set default metadata if none provided
        if not self.metadata:
            self.metadata = {
                'created': datetime.utcnow().isoformat() + 'Z',
                'description': 'ZPDR Zero-Point Address',
                'version': MANIFEST_VERSION
            }
        
        # Set manifest version
        self.version = MANIFEST_VERSION
        
    def _calculate_coherence(self) -> None:
        """Calculate all coherence measures for the ZPA vectors."""
        # Calculate internal coherence for each vector
        self.H_coherence = float(calculate_internal_coherence(self.H))
        self.E_coherence = float(calculate_internal_coherence(self.E))
        self.U_coherence = float(calculate_internal_coherence(self.U))
        
        # Calculate cross-coherence between vector pairs
        self.HE_coherence = float(calculate_cross_coherence(self.H, self.E))
        self.EU_coherence = float(calculate_cross_coherence(self.E, self.U))
        self.HU_coherence = float(calculate_cross_coherence(self.H, self.U))
        
        # Calculate global coherence
        internal_coherences = [
            Decimal(str(self.H_coherence)),
            Decimal(str(self.E_coherence)),
            Decimal(str(self.U_coherence))
        ]
        cross_coherences = [
            Decimal(str(self.HE_coherence)),
            Decimal(str(self.EU_coherence)),
            Decimal(str(self.HU_coherence))
        ]
        self.global_coherence = float(calculate_global_coherence(
            internal_coherences, cross_coherences
        ))
        
        # Store validation result
        self.is_valid = self.global_coherence >= float(COHERENCE_THRESHOLD)
    
    def normalize_vectors(self) -> None:
        """Normalize all vectors to canonical orientation and extract invariants."""
        # Normalize each vector and extract invariants
        H_normalized, self.H_invariants = normalize_with_invariants(self.H, "hyperbolic")
        E_normalized, self.E_invariants = normalize_with_invariants(self.E, "elliptical")
        U_normalized, self.U_invariants = normalize_with_invariants(self.U, "euclidean")
        
        # Update the vectors with normalized versions
        self.H = H_normalized
        self.E = E_normalized
        self.U = U_normalized
        
        # Update vector objects
        self.H_vector = HyperbolicVector(self.H)
        self.E_vector = EllipticalVector(self.E)
        self.U_vector = EuclideanVector(self.U)
        
        # Recalculate coherence with normalized vectors
        self._calculate_coherence()
    
    def denormalize_vectors(self) -> None:
        """Apply invariants to denormalize vectors to original orientation."""
        # Only apply if we have invariants
        if self.H_invariants and self.E_invariants and self.U_invariants:
            # Denormalize each vector using its invariants
            H_denormalized = denormalize_with_invariants(self.H, self.H_invariants, "hyperbolic")
            E_denormalized = denormalize_with_invariants(self.E, self.E_invariants, "elliptical")
            U_denormalized = denormalize_with_invariants(self.U, self.U_invariants, "euclidean")
            
            # Update the vectors
            self.H = H_denormalized
            self.E = E_denormalized
            self.U = U_denormalized
            
            # Update vector objects
            self.H_vector = HyperbolicVector(self.H)
            self.E_vector = EllipticalVector(self.E)
            self.U_vector = EuclideanVector(self.U)
            
            # Recalculate coherence with denormalized vectors
            self._calculate_coherence()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the manifest to a dictionary for serialization."""
        manifest = {
            'version': self.version,
            'hyperbolic': {
                'data': self.H.tolist(),
                'invariants': self.H_invariants,
                'coherence': self.H_coherence
            },
            'elliptical': {
                'data': self.E.tolist(),
                'invariants': self.E_invariants,
                'coherence': self.E_coherence
            },
            'euclidean': {
                'data': self.U.tolist(),
                'invariants': self.U_invariants,
                'coherence': self.U_coherence
            },
            'cross_coherence': {
                'HE': self.HE_coherence,
                'EU': self.EU_coherence,
                'HU': self.HU_coherence
            },
            'global_coherence': self.global_coherence,
            'metadata': self.metadata
        }
        
        return manifest
    
    @classmethod
    def from_dict(cls, manifest_dict: Dict[str, Any]) -> 'ZPAManifest':
        """Create a ZPAManifest from a dictionary representation."""
        # Handle version compatibility
        version = manifest_dict.get('version', '1.0')
        
        # Extract vectors based on version
        if version == '1.0':
            # Legacy format with direct vectors
            H = np.array(manifest_dict.get('H', [0, 0, 0]))
            E = np.array(manifest_dict.get('E', [1, 0, 0]))
            U = np.array(manifest_dict.get('U', [0, 0, 0]))
            
            # Default invariants
            H_invariants = {'rotation_angle': 0.0, 'scale_factor': 1.0}
            E_invariants = {'rotation_angle': 0.0, 'radius': 1.0}
            U_invariants = {'rotation_angle': 0.0, 'magnitude': 1.0}
        else:
            # Current format with nested structure
            H = np.array(manifest_dict.get('hyperbolic', {}).get('data', [0, 0, 0]))
            E = np.array(manifest_dict.get('elliptical', {}).get('data', [1, 0, 0]))
            U = np.array(manifest_dict.get('euclidean', {}).get('data', [0, 0, 0]))
            
            # Extract invariants
            H_invariants = manifest_dict.get('hyperbolic', {}).get('invariants', 
                                                              {'rotation_angle': 0.0, 'scale_factor': 1.0})
            E_invariants = manifest_dict.get('elliptical', {}).get('invariants',
                                                              {'rotation_angle': 0.0, 'radius': 1.0})
            U_invariants = manifest_dict.get('euclidean', {}).get('invariants',
                                                             {'rotation_angle': 0.0, 'magnitude': 1.0})
        
        # Extract metadata
        metadata = manifest_dict.get('metadata', {})
        
        # Create manifest object
        manifest = cls(H, E, U, H_invariants, E_invariants, U_invariants, metadata)
        
        return manifest
    
    def serialize(self, compact: bool = False) -> str:
        """
        Serialize the manifest to a JSON string.

        Args:
            compact: If True, use compact format with reduced precision

        Returns:
            JSON string representation of the manifest
        """
        manifest_dict = self.to_dict()
        
        if compact:
            # Reduce precision for compact format
            for space in ['hyperbolic', 'elliptical', 'euclidean']:
                manifest_dict[space]['data'] = [round(x, 6) for x in manifest_dict[space]['data']]
                manifest_dict[space]['coherence'] = round(manifest_dict[space]['coherence'], 3)
                for k, v in manifest_dict[space]['invariants'].items():
                    if isinstance(v, (int, float)):
                        manifest_dict[space]['invariants'][k] = round(v, 6)
            
            # Reduce precision for cross_coherence
            for k, v in manifest_dict['cross_coherence'].items():
                manifest_dict['cross_coherence'][k] = round(v, 3)
            
            # Reduce precision for global_coherence
            manifest_dict['global_coherence'] = round(manifest_dict['global_coherence'], 3)
            
            # Reduce metadata
            manifest_dict['metadata'] = {
                'description': manifest_dict['metadata'].get('description', '')
            }
            
            # Serialize with minimal whitespace
            return json.dumps(manifest_dict, separators=(',', ':'))
        else:
            # Full precision serialization
            return json.dumps(manifest_dict, indent=2, sort_keys=True)
    
    @classmethod
    def deserialize(cls, json_str: str) -> 'ZPAManifest':
        """
        Deserialize a JSON string to a ZPAManifest object.

        Args:
            json_str: JSON string representation of the manifest

        Returns:
            ZPAManifest object
        """
        try:
            manifest_dict = json.loads(json_str)
            return cls.from_dict(manifest_dict)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for ZPA manifest")

def create_zpa_manifest(H: np.ndarray, E: np.ndarray, U: np.ndarray, 
                       description: str = None) -> ZPAManifest:
    """
    Create a ZPA manifest from vector data.

    This is a convenience function to create a ZPA manifest from raw vector data.
    It normalizes the vectors and extracts invariants automatically.

    Args:
        H: Hyperbolic vector component
        E: Elliptical vector component
        U: Euclidean vector component
        description: Optional description for the manifest

    Returns:
        ZPAManifest object with normalized vectors and extracted invariants
    """
    # Create basic manifest
    manifest = ZPAManifest(H, E, U)
    
    # Normalize vectors and extract invariants
    manifest.normalize_vectors()
    
    # Set description if provided
    if description:
        manifest.metadata['description'] = description
    
    return manifest

def serialize_zpa(H: np.ndarray, E: np.ndarray, U: np.ndarray,
                 H_invariants: Dict[str, float] = None,
                 E_invariants: Dict[str, float] = None,
                 U_invariants: Dict[str, float] = None,
                 metadata: Dict[str, Any] = None,
                 compact: bool = False) -> str:
    """
    Serialize ZPA vector data to a JSON string manifest.

    This is a convenience function to directly create and serialize a ZPA manifest.

    Args:
        H: Hyperbolic vector component
        E: Elliptical vector component
        U: Euclidean vector component
        H_invariants: Invariants for the hyperbolic vector
        E_invariants: Invariants for the elliptical vector
        U_invariants: Invariants for the euclidean vector
        metadata: Additional metadata about the ZPA
        compact: If True, use compact format with reduced precision

    Returns:
        JSON string representation of the ZPA manifest
    """
    # Create manifest
    manifest = ZPAManifest(H, E, U, H_invariants, E_invariants, U_invariants, metadata)
    
    # Serialize
    return manifest.serialize(compact)

def deserialize_zpa(json_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Deserialize a JSON string manifest to ZPA vector data.

    This is a convenience function to directly extract ZPA vector data from a manifest.

    Args:
        json_str: JSON string representation of the manifest

    Returns:
        Tuple of (H, E, U, metadata) from the manifest
    """
    # Deserialize to manifest object
    manifest = ZPAManifest.deserialize(json_str)
    
    # Return vector data and metadata
    return manifest.H, manifest.E, manifest.U, manifest.metadata