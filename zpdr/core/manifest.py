#!/usr/bin/env python3
"""
Manifest implementation for ZPDR.

This module provides functionality for creating, reading, and manipulating
ZPDR manifests, which contain the zero-point coordinates of a data object
along with metadata and verification information.
"""

import json
import hashlib
import os
import base64
import datetime
import copy
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal
from pathlib import Path

from .trivector_digest import TrivectorDigest

class Manifest:
    """
    Represents a ZPDR manifest containing zero-point coordinates and metadata.
    """
    
    def __init__(self,
                 trivector_digest: TrivectorDigest,
                 original_filename: str,
                 file_size: int,
                 checksum: str,
                 base_fiber_id: str = "Cl_3,0",
                 structure_id: str = "standard_trivector",
                 coherence_threshold: float = 0.99,
                 additional_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a ZPDR manifest.
        
        Args:
            trivector_digest: TrivectorDigest object containing zero-point coordinates
            original_filename: Name of the original file
            file_size: Size of the original file in bytes
            checksum: Checksum of the original file (format: algorithm:hash)
            base_fiber_id: Identifier for the base fiber (Clifford algebra)
            structure_id: Identifier for the structure used
            coherence_threshold: Minimum coherence threshold for verification
            additional_metadata: Optional additional metadata
        """
        self.trivector_digest = trivector_digest
        self.original_filename = original_filename
        self.file_size = file_size
        self.checksum = checksum
        self.base_fiber_id = base_fiber_id
        self.structure_id = structure_id
        self.coherence_threshold = Decimal(str(coherence_threshold))
        self.additional_metadata = additional_metadata or {}
        
        # Add creation timestamp if not present
        if 'creation_time' not in self.additional_metadata:
            self.additional_metadata['creation_time'] = datetime.datetime.now().isoformat()
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path], trivector_digest: TrivectorDigest,
                 base_fiber_id: str = "Cl_3,0", structure_id: str = "standard_trivector",
                 coherence_threshold: float = 0.99) -> 'Manifest':
        """
        Create a manifest from a file.
        
        Args:
            file_path: Path to the original file
            trivector_digest: TrivectorDigest object containing zero-point coordinates
            base_fiber_id: Identifier for the base fiber (Clifford algebra)
            structure_id: Identifier for the structure used
            coherence_threshold: Minimum coherence threshold for verification
            
        Returns:
            A new Manifest instance
        """
        # Convert to Path object
        path = Path(file_path)
        
        # Get file metadata
        original_filename = path.name
        file_size = path.stat().st_size
        
        # Calculate checksum
        checksum = cls._calculate_checksum(file_path)
        
        # Create and return the manifest
        return cls(
            trivector_digest=trivector_digest,
            original_filename=original_filename,
            file_size=file_size,
            checksum=checksum,
            base_fiber_id=base_fiber_id,
            structure_id=structure_id,
            coherence_threshold=coherence_threshold
        )
    
    @staticmethod
    def _calculate_checksum(file_path: Union[str, Path]) -> str:
        """
        Calculate the SHA-256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Checksum string in format "sha256:hash"
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks to avoid loading large files into memory
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return f"sha256:{sha256_hash.hexdigest()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the manifest to a dictionary representation.
        
        Returns:
            Dictionary representation of the manifest
        """
        return {
            "version": "1.0",
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "base_fiber_id": self.base_fiber_id,
            "structure_id": self.structure_id,
            "coherence": {
                "global_coherence": str(self.trivector_digest.get_global_coherence()),
                "coherence_threshold": str(self.coherence_threshold)
            },
            "coordinates": {
                "hyperbolic": {
                    "vector": [str(v) for v in self.trivector_digest.hyperbolic],
                    "invariants": [str(v) for v in self.trivector_digest.invariants['hyperbolic']]
                },
                "elliptical": {
                    "vector": [str(v) for v in self.trivector_digest.elliptical],
                    "invariants": [str(v) for v in self.trivector_digest.invariants['elliptical']]
                },
                "euclidean": {
                    "vector": [str(v) for v in self.trivector_digest.euclidean],
                    "invariants": [str(v) for v in self.trivector_digest.invariants['euclidean']]
                }
            },
            "additional_metadata": self.additional_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Manifest':
        """
        Create a manifest from a dictionary representation.
        
        Args:
            data: Dictionary representation of a manifest
            
        Returns:
            A new Manifest instance
        """
        # Extract coordinates
        hyperbolic = [Decimal(v) for v in data['coordinates']['hyperbolic']['vector']]
        elliptical = [Decimal(v) for v in data['coordinates']['elliptical']['vector']]
        euclidean = [Decimal(v) for v in data['coordinates']['euclidean']['vector']]
        
        # Extract invariants
        invariants = {
            'hyperbolic': [Decimal(v) for v in data['coordinates']['hyperbolic']['invariants']],
            'elliptical': [Decimal(v) for v in data['coordinates']['elliptical']['invariants']],
            'euclidean': [Decimal(v) for v in data['coordinates']['euclidean']['invariants']]
        }
        
        # Create trivector digest
        trivector_digest = TrivectorDigest(
            hyperbolic=hyperbolic,
            elliptical=elliptical,
            euclidean=euclidean,
            invariants=invariants
        )
        
        # Create and return the manifest
        return cls(
            trivector_digest=trivector_digest,
            original_filename=data['original_filename'],
            file_size=data['file_size'],
            checksum=data['checksum'],
            base_fiber_id=data['base_fiber_id'],
            structure_id=data['structure_id'],
            coherence_threshold=Decimal(data['coherence']['coherence_threshold']),
            additional_metadata=data.get('additional_metadata', {})
        )
    
    def to_json(self) -> str:
        """
        Convert the manifest to a JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Manifest':
        """
        Create a manifest from a JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            A new Manifest instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save the manifest to a file.
        
        Args:
            file_path: Path to save the manifest to
        """
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'Manifest':
        """
        Load a manifest from a file.
        
        Args:
            file_path: Path to the manifest file
            
        Returns:
            Loaded Manifest instance
        """
        with open(file_path, 'r') as f:
            json_str = f.read()
        
        return cls.from_json(json_str)
    
    def verify_coherence(self) -> bool:
        """
        Verify that the manifest's coherence meets the threshold.
        
        Returns:
            True if coherence is acceptable, False otherwise
        """
        global_coherence = self.trivector_digest.get_global_coherence()
        return global_coherence >= self.coherence_threshold
    
    def __str__(self) -> str:
        """
        String representation of the manifest.
        
        Returns:
            String representation
        """
        return (f"ZPDR Manifest for '{self.original_filename}' "
                f"(size: {self.file_size} bytes, "
                f"coherence: {self.trivector_digest.get_global_coherence()}, "
                f"checksum: {self.checksum})")
    
    def copy(self) -> 'Manifest':
        """
        Create a deep copy of this manifest.
        
        Returns:
            A new Manifest instance with the same attributes
        """
        # Create a copy of the trivector digest
        trivector_copy = TrivectorDigest(
            hyperbolic=self.trivector_digest.hyperbolic.copy(),
            elliptical=self.trivector_digest.elliptical.copy(),
            euclidean=self.trivector_digest.euclidean.copy(),
            invariants={
                'hyperbolic': self.trivector_digest.invariants['hyperbolic'].copy(),
                'elliptical': self.trivector_digest.invariants['elliptical'].copy(),
                'euclidean': self.trivector_digest.invariants['euclidean'].copy()
            }
        )
        
        # Create a copy of the manifest
        return Manifest(
            trivector_digest=trivector_copy,
            original_filename=self.original_filename,
            file_size=self.file_size,
            checksum=self.checksum,
            base_fiber_id=self.base_fiber_id,
            structure_id=self.structure_id,
            coherence_threshold=float(self.coherence_threshold),
            additional_metadata=copy.deepcopy(self.additional_metadata)
        )