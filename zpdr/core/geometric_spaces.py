"""
Geometric Spaces for Zero-Point Data Resolution (ZPDR)

This module implements the three essential geometric spaces used in ZPDR:
- Hyperbolic space (negative curvature)
- Elliptical space (positive curvature)
- Euclidean space (flat/zero curvature)

It also provides transformations between these spaces, forming the foundation
for the ZPA (Zero-Point Address) extraction and reconstruction.

Mathematical Foundation:
The ZPDR framework is built on the Prime Framework's principle of representing
data across three complementary geometric spaces, each with different curvature
properties. This trilateral representation enables robust data encoding with
inherent error detection and correction capabilities.

Key Components:
1. Geometric Vectors - Space-specific vector implementations that respect the
   geometric properties of their respective spaces:
   - HyperbolicVector: Uses the Poincaré disk model to represent vectors in 
     negative-curvature space. In ZPDR, hyperbolic vectors capture base 
     transformation systems.
   - EllipticalVector: Uses a spherical model to represent vectors in 
     positive-curvature space. In ZPDR, elliptical vectors capture 
     transformation spans.
   - EuclideanVector: Uses standard vector representation in flat space.
     In ZPDR, Euclidean vectors capture the transformed object itself.

2. SpaceTransformer - Provides methods to convert vectors between the three
   geometric spaces while preserving their essential properties, enabling the
   construction of coherent ZPA triples.

Each vector implementation maintains high-precision calculations using Python's
Decimal type to ensure numerical stability and accuracy, especially important
for operations near the boundaries of the geometric spaces.

The geometric spaces together implement the fiber structure described in the
Prime Framework, with each space acting as a fiber bundle over the reference
manifold, connected through the coherence inner product relationship.
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from enum import Enum
from decimal import Decimal

# Import utilities for high-precision calculations
from ..utils import (
    to_decimal_array, 
    calculate_internal_coherence,
    calculate_cross_coherence,
    calculate_global_coherence,
    PRECISION_TOLERANCE
)


class SpaceType(Enum):
    """Enumeration of geometric space types used in ZPDR."""
    HYPERBOLIC = "hyperbolic"
    ELLIPTICAL = "elliptical"
    EUCLIDEAN = "euclidean"


class GeometricVector:
    """
    Base class for vectors in different geometric spaces.
    
    Provides common functionality for all geometric vectors regardless
    of the space they inhabit. Specific space implementations inherit
    from this base class.
    
    The GeometricVector class defines the interface and shared functionality for
    vectors in different geometric spaces (hyperbolic, elliptical, and Euclidean).
    Each space has its own unique properties related to curvature, distance metrics, 
    and inner products, but they all share common vector operations.
    
    In the ZPDR framework, geometric vectors form the basis of the trilateral
    representation system. Each type of geometric vector encodes different aspects
    of the data:
    
    1. Hyperbolic vectors (H): Encode the base transformation system in negative-
       curvature space. These vectors capture fundamental transformational properties
       and provide a robust basis for error correction.
       
    2. Elliptical vectors (E): Encode the transformation span in positive-curvature
       space. These vectors describe how transformations extend and interact across
       the geometric domain.
       
    3. Euclidean vectors (U): Encode the transformed object in flat space. These
       vectors represent the direct manifestation of the data in conventional space.
    
    All geometric vectors implement high-precision arithmetic using Python's Decimal
    type to ensure numerical stability, particularly important for operations near
    the boundaries of non-Euclidean spaces.
    
    Key mathematical properties:
    - Each vector maintains both standard floating-point and high-precision
      representations
    - Vectors support operations like addition, scalar multiplication, inner product,
      normalization, etc.
    - Each vector type defines its own validity conditions based on the geometric
      constraints of its space
    - Each vector type implements invariant extraction for zero-point normalization
    
    These properties enable the ZPDR framework to maintain coherence across different
    geometric spaces and ensure robust data encoding and reconstruction.
    """
    
    def __init__(self, components: List[float], space_type: SpaceType):
        """
        Initialize a geometric vector.
        
        Args:
            components: List of vector components
            space_type: Type of geometric space this vector belongs to
        """
        self.components = np.array(components, dtype=float)
        self.space_type = space_type
        self.dimension = len(components)
        
        # The _dec_components attribute will be set by subclasses for high precision
        # This is just a placeholder to make type checking happy
        self._dec_components = []
    
    def __str__(self) -> str:
        """String representation of the vector."""
        return f"{self.space_type.value.capitalize()}Vector({self.components})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.space_type.value.capitalize()}Vector(components={list(self.components)})"
    
    def __eq__(self, other) -> bool:
        """Check if two vectors are equal with high precision."""
        if not isinstance(other, GeometricVector):
            return False
        if self.space_type != other.space_type:
            return False
            
        # Use more strict tolerance for high-precision comparison
        return np.allclose(self.components, other.components, rtol=1e-14, atol=1e-14)
    
    def __add__(self, other) -> 'GeometricVector':
        """Vector addition (only valid for vectors in the same space)."""
        if not isinstance(other, GeometricVector):
            return NotImplemented
        if self.space_type != other.space_type:
            raise ValueError(f"Cannot add vectors from different spaces: {self.space_type} and {other.space_type}")
        
        # The specific space implementations will override this method
        # to handle addition according to the space's geometry
        return GeometricVector(list(self.components + other.components), self.space_type)
    
    def __mul__(self, scalar: float) -> 'GeometricVector':
        """Scalar multiplication."""
        # This is a base implementation - subclasses will provide high-precision versions
        return GeometricVector(list(self.components * scalar), self.space_type)
    
    def __rmul__(self, scalar: float) -> 'GeometricVector':
        """Right scalar multiplication."""
        return self.__mul__(scalar)
    
    def inner_product(self, other: 'GeometricVector') -> float:
        """
        Compute the inner product with another vector in the same space.
        
        Must be implemented by specific space subclasses with high precision.
        
        Args:
            other: Vector to compute inner product with (must be in the same space)
            
        Returns:
            Scalar inner product value
        """
        raise NotImplementedError("Inner product must be implemented by specific space classes")
    
    def norm(self) -> float:
        """
        Compute the norm of the vector with high precision.
        
        Returns:
            Scalar norm value
        """
        # This uses the inner_product method which is implemented with high precision
        # in the specific space subclasses
        return np.sqrt(abs(self.inner_product(self)))
    
    def normalize(self) -> 'GeometricVector':
        """
        Return a normalized (unit) vector in the same direction with high precision.
        
        Must be implemented by specific space subclasses with high precision.
        
        Returns:
            Normalized vector
        """
        # This is a base implementation - subclasses will provide space-specific versions
        norm_value = self.norm()
        if norm_value < 1e-14:  # Use tighter tolerance for high precision
            return self  # Avoid division by zero for near-zero vectors
        return self * (1.0 / norm_value)
    
    def get_invariants(self) -> Dict[str, float]:
        """
        Extract geometric invariants from the vector.
        
        Must be implemented by specific space subclasses with high precision.
        
        Returns:
            Dictionary of invariants specific to the space type
        """
        raise NotImplementedError("get_invariants must be implemented by specific space classes")


class HyperbolicVector(GeometricVector):
    """
    Vector in hyperbolic space (negative curvature).
    
    Implements the Poincaré disk model of hyperbolic geometry.
    Hyperbolic vectors in ZPDR are used to capture base transformation systems.
    
    The HyperbolicVector class implements vectors in hyperbolic space using the
    Poincaré disk model, where points are represented within the unit disk in
    Euclidean space, but with a non-Euclidean metric that gives the space its
    negative curvature properties.
    
    In ZPDR, hyperbolic vectors (H component of the ZPA triple) encode the base
    transformation system of the data. This negative-curvature space provides
    several advantages for data representation:
    
    1. Exponential Expansion: Hyperbolic space allows exponentially more "room"
       as one moves toward the boundary of the disk, enabling efficient representation
       of hierarchical structures and transformational relationships.
       
    2. Robust Error Boundaries: The boundary of the Poincaré disk serves as a natural
       limit that helps contain and detect errors in the encoding process.
       
    3. Invariant Properties: Hyperbolic transformations preserve important geometric
       invariants that can be extracted and used for zero-point normalization.
       
    4. Stable Zero-Point: The center of the disk provides a natural, stable zero-point
       for normalization operations.
    
    Mathematical properties of the Poincaré disk model:
    - Points lie within the unit disk: |v| < 1
    - The metric becomes singular at the boundary |v| = 1
    - Geodesics are either diameters of the disk or arcs of circles orthogonal to the boundary
    - The distance between points increases exponentially as they approach the boundary
    - Möbius transformations act as isometries (distance-preserving maps)
    
    This implementation ensures vectors remain within the unit disk by projecting any
    out-of-bounds points back to the valid region, uses high-precision arithmetic for
    boundary calculations, and implements the proper hyperbolic operations (addition,
    inner product, distance, etc.) according to the mathematical principles of hyperbolic
    geometry.
    """
    
    def __init__(self, components: List[float]):
        """
        Initialize a hyperbolic vector.
        
        Args:
            components: List of vector components in the Poincaré disk model
        """
        super().__init__(components, SpaceType.HYPERBOLIC)
        
        # Convert to high-precision Decimal calculations
        dec_components = to_decimal_array(components)
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in dec_components)
        norm = norm_squared.sqrt()
        
        # Ensure the vector is valid (lies within the Poincaré disk)
        if norm < Decimal('1.0'):
            self.components = np.array(components, dtype=float)
        else:
            # Project back into the unit disk if point is outside
            # Use high precision for the projection
            scale_factor = Decimal('0.99') / norm
            projected = [x * scale_factor for x in dec_components]
            # Convert back to numpy array
            self.components = np.array([float(x) for x in projected], dtype=float)
        
        # Store high-precision components for later calculations
        self._dec_components = to_decimal_array(self.components)
    
    def is_valid(self) -> bool:
        """
        Check if the vector is valid in hyperbolic space.
        
        In the Poincaré disk model, points must lie within the unit disk.
        
        Returns:
            True if the vector is valid, False otherwise
        """
        # Use high-precision Decimal calculations
        norm_squared = sum(x * x for x in self._dec_components)
        
        # Points in the Poincaré disk model must satisfy |v| < 1
        return norm_squared < Decimal('1.0')
    
    def __add__(self, other) -> 'HyperbolicVector':
        """
        Hyperbolic addition in the Poincaré disk model.
        
        Args:
            other: Another hyperbolic vector
            
        Returns:
            Resulting hyperbolic vector
        """
        if not isinstance(other, HyperbolicVector):
            return NotImplemented
        
        # Implement Möbius addition for the Poincaré disk model
        # using high precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Calculate norms and dot product with high precision
        a_norm_sq = sum(x * x for x in a)
        b_norm_sq = sum(x * x for x in b)
        a_dot_b = sum(x * y for x, y in zip(a, b))
        
        # Formula: (a + b(1 + 2<a,b> + |a|²)) / (1 + 2<a,b> + |a|²|b|²)
        numerator = []
        for i in range(len(a)):
            term = a[i] + b[i] * (Decimal('1.0') + Decimal('2.0') * a_dot_b + a_norm_sq)
            numerator.append(term)
        
        denominator = Decimal('1.0') + Decimal('2.0') * a_dot_b + a_norm_sq * b_norm_sq
        
        # Division with high precision
        result = [x / denominator for x in numerator]
        
        # Convert back to float for GeometricVector compatibility
        return HyperbolicVector([float(x) for x in result])
    
    def inner_product(self, other: 'HyperbolicVector') -> float:
        """
        Hyperbolic inner product in the Poincaré disk model.
        
        Args:
            other: Another hyperbolic vector
            
        Returns:
            Scalar inner product value
        """
        if not isinstance(other, HyperbolicVector):
            raise ValueError("Inner product only defined between vectors of the same space")
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Calculate norms and dot product with high precision
        a_norm_sq = sum(x * x for x in a)
        b_norm_sq = sum(x * x for x in b)
        euclidean_inner = sum(x * y for x, y in zip(a, b))
        
        # Hyperbolic inner product formula for Poincaré disk
        denominator = (Decimal('1.0') - a_norm_sq) * (Decimal('1.0') - b_norm_sq)
        
        # Avoid division by zero
        if denominator < Decimal('1e-10'):
            return float('inf')
        
        # Complete calculation with high precision
        result = Decimal('4.0') * euclidean_inner / denominator
        
        # Convert back to float
        return float(result)
    
    def distance_to(self, other: 'HyperbolicVector') -> float:
        """
        Compute the hyperbolic distance to another point.
        
        Args:
            other: Another hyperbolic vector
            
        Returns:
            Scalar distance value
        """
        if not isinstance(other, HyperbolicVector):
            raise ValueError("Distance only defined between vectors of the same space")
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Compute squared Euclidean distance with high precision
        euclidean_dist_sq = sum((x - y)**2 for x, y in zip(a, b))
        
        # Calculate norms with high precision
        a_norm_sq = sum(x * x for x in a)
        b_norm_sq = sum(x * x for x in b)
        
        # Poincaré distance formula with high precision
        numerator = Decimal('2.0') * euclidean_dist_sq
        denominator = (Decimal('1.0') - a_norm_sq) * (Decimal('1.0') - b_norm_sq)
        
        # Avoid division by zero
        if denominator < Decimal('1e-10'):
            return float('inf')
        
        # Calculate the argument for arccosh
        arg = Decimal('1.0') + numerator / denominator
        
        # The distance is acosh(arg)
        if arg < Decimal('1.0'):
            return 0.0  # Numerical error, points are essentially the same
        
        # Convert back to float for numpy
        return float(np.arccosh(float(arg)))
    
    def normalize(self) -> 'HyperbolicVector':
        """
        Return a normalized hyperbolic vector with high precision.
        
        In hyperbolic space, normalization projects to a specific radius within the disk.
        
        Returns:
            Normalized hyperbolic vector
        """
        # Use high-precision Decimal calculations
        dec_components = self._dec_components
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in dec_components)
        
        # If already normalized or zero vector, return self
        if norm_squared <= Decimal('0') or abs(norm_squared - Decimal('1.0')) < PRECISION_TOLERANCE:
            return self
        
        # Normalize to project onto the preferred radius (0.5 for hyperbolic space)
        norm = norm_squared.sqrt()
        target_radius = Decimal('0.5')
        
        # Scale the vector to the target radius
        normalized = [x * (target_radius / norm) for x in dec_components]
        
        # Convert back to float for GeometricVector compatibility
        return HyperbolicVector([float(x) for x in normalized])
    
    def get_invariants(self) -> Dict[str, float]:
        """
        Extract rotation and scaling invariants from the hyperbolic vector.
        
        These invariants are essential for proper reconstruction.
        
        Returns:
            Dictionary of invariants (rotation_angle, scale_factor)
        """
        # Use high-precision calculations
        dec_components = self._dec_components
        
        # Extract rotation information (for hyperbolic space, this could be an angle)
        # This is a simplified approach for Phase 1
        if len(dec_components) >= 2 and (dec_components[0]**2 + dec_components[1]**2) > Decimal('0'):
            # Calculate hyperbolic angle
            rotation_angle = Decimal(str(np.arctan2(float(dec_components[1]), float(dec_components[0]))))
        else:
            rotation_angle = Decimal('0')
        
        # Extract scale information
        norm_squared = sum(x * x for x in dec_components)
        scale_factor = norm_squared.sqrt()
        
        # Return as dictionary of float values
        return {
            'rotation_angle': float(rotation_angle),
            'scale_factor': float(scale_factor)
        }


class EllipticalVector(GeometricVector):
    """
    Vector in elliptical space (positive curvature).
    
    Implements a spherical model of elliptical geometry.
    Elliptical vectors in ZPDR are used to capture transformation spans.
    
    The EllipticalVector class implements vectors in elliptical space using a
    spherical model, where points are represented on the surface of a unit sphere
    in Euclidean space. This representation gives the space its positive curvature
    properties.
    
    In ZPDR, elliptical vectors (E component of the ZPA triple) encode the
    transformation span of the data. This positive-curvature space provides
    several advantages for data representation:
    
    1. Bounded Domain: Elliptical space is naturally bounded, ensuring that all valid
       representations have finite extent and uniform coverage properties.
       
    2. Rotation Invariants: The spherical model permits natural representation of 
       rotational invariants, which are essential for stable zero-point normalization.
       
    3. Complementary Constraints: The positive curvature constraints complement the
       negative curvature of hyperbolic space, creating a balanced trilateral system
       that enhances error detection capabilities.
       
    4. Global Connectivity: Unlike hyperbolic space, elliptical space is globally
       connected, allowing representations to "wrap around" in a coherent manner.
    
    Mathematical properties of the spherical model:
    - Points lie on the unit sphere: |v| = 1
    - The shortest path between points (geodesic) is the great circle arc
    - The distance between points is the angle between the corresponding vectors
    - The total "area" of the space is finite
    - Rotations act as isometries (distance-preserving maps)
    
    This implementation ensures vectors remain on the unit sphere by normalizing them
    during construction, uses high-precision arithmetic for stable calculations, and
    implements the proper elliptical operations (addition, inner product, distance, etc.)
    according to the mathematical principles of elliptical geometry.
    """
    
    def __init__(self, components: List[float]):
        """
        Initialize an elliptical vector.
        
        Args:
            components: List of vector components
        """
        super().__init__(components, SpaceType.ELLIPTICAL)
        
        # Convert to high-precision Decimal calculations
        dec_components = to_decimal_array(components)
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in dec_components)
        norm = norm_squared.sqrt()
        
        # Normalize to ensure the vector lies on the unit sphere
        if norm > Decimal('0'):
            # Normalize with high precision
            normalized = [x / norm for x in dec_components]
            # Convert back to numpy array
            self.components = np.array([float(x) for x in normalized], dtype=float)
        else:
            self.components = np.array(components, dtype=float)
        
        # Store high-precision components for later calculations
        self._dec_components = to_decimal_array(self.components)
    
    def is_valid(self) -> bool:
        """
        Check if the vector is valid in elliptical space.
        
        In the spherical model, points must lie on the unit sphere.
        
        Returns:
            True if the vector is valid, False otherwise
        """
        # Use high-precision Decimal calculations
        norm_squared = sum(x * x for x in self._dec_components)
        
        # Points on a sphere have unit norm
        return abs(norm_squared - Decimal('1.0')) < PRECISION_TOLERANCE
    
    def __add__(self, other) -> 'EllipticalVector':
        """
        Elliptical addition on the sphere.
        
        This is approximated by doing Euclidean addition and projecting back to the sphere.
        
        Args:
            other: Another elliptical vector
            
        Returns:
            Resulting elliptical vector
        """
        if not isinstance(other, EllipticalVector):
            return NotImplemented
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Add the vectors in Euclidean space with high precision
        result = [x + y for x, y in zip(a, b)]
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in result)
        norm = norm_squared.sqrt()
        
        # Project back to the sphere with high precision
        if norm > Decimal('0'):
            normalized = [x / norm for x in result]
            # Convert back to float for GeometricVector compatibility
            return EllipticalVector([float(x) for x in normalized])
        else:
            # If result is zero vector, return a default unit vector
            default = [Decimal('0')] * len(result)
            default[0] = Decimal('1.0')
            return EllipticalVector([float(x) for x in default])
    
    def inner_product(self, other: 'EllipticalVector') -> float:
        """
        Elliptical inner product (equivalent to spherical dot product).
        
        Args:
            other: Another elliptical vector
            
        Returns:
            Scalar inner product value
        """
        if not isinstance(other, EllipticalVector):
            raise ValueError("Inner product only defined between vectors of the same space")
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Calculate dot product with high precision
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # Convert back to float
        return float(dot_product)
    
    def distance_to(self, other: 'EllipticalVector') -> float:
        """
        Compute the elliptical (great circle) distance to another point.
        
        Args:
            other: Another elliptical vector
            
        Returns:
            Scalar distance value (angle in radians)
        """
        if not isinstance(other, EllipticalVector):
            raise ValueError("Distance only defined between vectors of the same space")
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Compute the cosine of the angle between the vectors with high precision
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # Ensure the dot product is within valid range for arccos
        cos_angle = max(min(dot_product, Decimal('1.0')), Decimal('-1.0'))
        
        # Convert to float for numpy
        cos_angle_float = float(cos_angle)
        
        # Return the angle in radians
        return np.arccos(cos_angle_float)
    
    def normalize(self) -> 'EllipticalVector':
        """
        Return a normalized elliptical vector with high precision.
        
        In elliptical space, normalization projects onto the unit sphere.
        
        Returns:
            Normalized elliptical vector
        """
        # For elliptical vectors, they are already normalized to unit sphere
        if self.is_valid():
            return self
        
        # Use high-precision Decimal calculations
        dec_components = self._dec_components
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in dec_components)
        
        # If zero vector, return a default unit vector
        if norm_squared <= Decimal('0'):
            default = [Decimal('0')] * len(dec_components)
            default[0] = Decimal('1.0')
            return EllipticalVector([float(x) for x in default])
        
        # Normalize to unit length
        norm = norm_squared.sqrt()
        normalized = [x / norm for x in dec_components]
        
        # Convert back to float for GeometricVector compatibility
        return EllipticalVector([float(x) for x in normalized])
    
    def get_invariants(self) -> Dict[str, float]:
        """
        Extract rotation and radius invariants from the elliptical vector.
        
        These invariants are essential for proper reconstruction.
        
        Returns:
            Dictionary of invariants (rotation_angle, radius)
        """
        # Use high-precision calculations
        dec_components = self._dec_components
        
        # Extract rotation information (for elliptical space, this is the angle on the sphere)
        if len(dec_components) >= 2 and (dec_components[0]**2 + dec_components[1]**2) > Decimal('0'):
            # Calculate spherical angle
            rotation_angle = Decimal(str(np.arctan2(float(dec_components[1]), float(dec_components[0]))))
        else:
            rotation_angle = Decimal('0')
        
        # For elliptical vectors on the unit sphere, radius is always 1.0
        radius = Decimal('1.0')
        
        # Return as dictionary of float values
        return {
            'rotation_angle': float(rotation_angle),
            'radius': float(radius)
        }


class EuclideanVector(GeometricVector):
    """
    Vector in Euclidean space (zero curvature).
    
    Implements standard Euclidean geometry.
    Euclidean vectors in ZPDR are used to capture the transformed object itself.
    
    The EuclideanVector class implements vectors in standard Euclidean space,
    which has zero curvature (flat geometry). This is the most familiar vector
    space, with the standard dot product and distance metrics.
    
    In ZPDR, Euclidean vectors (U component of the ZPA triple) encode the transformed
    object itself, representing the direct manifestation of the data. This flat-space
    representation provides several advantages:
    
    1. Direct Interpretation: Euclidean space allows for straightforward interpretation
       of vector components, making it ideal for representing the final transformed data.
       
    2. Computational Efficiency: Operations in Euclidean space are computationally
       efficient and numerically stable, as they don't involve complex metric tensors.
       
    3. Complementary Balance: The zero-curvature properties provide a balanced middle
       ground between the negative curvature of hyperbolic space and the positive
       curvature of elliptical space, enhancing the overall coherence of the trilateral
       representation.
       
    4. Natural Embedding: Most conventional data already exists in Euclidean space,
       making this representation a natural choice for capturing the transformed object.
    
    Mathematical properties of Euclidean space:
    - The space extends infinitely in all directions
    - The shortest path between points is a straight line
    - The distance between points is given by the standard Euclidean metric
    - The inner product is the familiar dot product
    - Parallel lines never intersect
    - Translations and rotations act as isometries (distance-preserving maps)
    
    This implementation provides high-precision arithmetic for vector operations and
    implements the standard Euclidean operations (addition, inner product, distance, etc.)
    according to the mathematical principles of Euclidean geometry.
    """
    
    def __init__(self, components: List[float]):
        """
        Initialize a Euclidean vector.
        
        Args:
            components: List of vector components
        """
        super().__init__(components, SpaceType.EUCLIDEAN)
        
        # Store high-precision components for later calculations
        self._dec_components = to_decimal_array(self.components)
    
    def is_valid(self) -> bool:
        """
        Check if the vector is valid in Euclidean space.
        
        All finite vectors are valid in Euclidean space.
        
        Returns:
            True if the vector is valid, False otherwise
        """
        # All finite vectors are valid in Euclidean space
        return all(np.isfinite(self.components))
    
    def __add__(self, other) -> 'EuclideanVector':
        """
        Standard Euclidean vector addition with high precision.
        
        Args:
            other: Another Euclidean vector
            
        Returns:
            Resulting Euclidean vector
        """
        if not isinstance(other, EuclideanVector):
            return NotImplemented
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Add vectors with high precision
        result = [x + y for x, y in zip(a, b)]
        
        # Convert back to float for GeometricVector compatibility
        return EuclideanVector([float(x) for x in result])
    
    def inner_product(self, other: 'EuclideanVector') -> float:
        """
        Standard Euclidean inner product (dot product) with high precision.
        
        Args:
            other: Another Euclidean vector
            
        Returns:
            Scalar inner product value
        """
        if not isinstance(other, EuclideanVector):
            raise ValueError("Inner product only defined between vectors of the same space")
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Calculate dot product with high precision
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # Convert back to float
        return float(dot_product)
    
    def distance_to(self, other: 'EuclideanVector') -> float:
        """
        Compute the Euclidean distance to another point with high precision.
        
        Args:
            other: Another Euclidean vector
            
        Returns:
            Scalar distance value
        """
        if not isinstance(other, EuclideanVector):
            raise ValueError("Distance only defined between vectors of the same space")
        
        # Use high-precision Decimal calculations
        a = self._dec_components
        b = other._dec_components
        
        # Calculate squared distance with high precision
        squared_distance = sum((x - y)**2 for x, y in zip(a, b))
        
        # Take square root with high precision
        distance = squared_distance.sqrt()
        
        # Convert back to float
        return float(distance)
    
    def normalize(self) -> 'EuclideanVector':
        """
        Return a normalized Euclidean vector with high precision.
        
        Returns:
            Normalized Euclidean vector
        """
        # Use high-precision Decimal calculations
        dec_components = self._dec_components
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in dec_components)
        
        # If zero vector, return self
        if norm_squared <= Decimal('0'):
            return self
        
        # Normalize with high precision
        norm = norm_squared.sqrt()
        normalized = [x / norm for x in dec_components]
        
        # Convert back to float for GeometricVector compatibility
        return EuclideanVector([float(x) for x in normalized])
    
    def get_invariants(self) -> Dict[str, float]:
        """
        Extract phase and magnitude invariants from the Euclidean vector.
        
        These invariants are essential for proper reconstruction.
        
        Returns:
            Dictionary of invariants (phase, magnitude)
        """
        # Use high-precision calculations
        dec_components = self._dec_components
        
        # Extract phase information (for Euclidean space, this could be an angle)
        if len(dec_components) >= 2 and (dec_components[0]**2 + dec_components[1]**2) > Decimal('0'):
            # Calculate Euclidean angle (phase)
            phase = Decimal(str(np.arctan2(float(dec_components[1]), float(dec_components[0]))))
        else:
            phase = Decimal('0')
        
        # Extract magnitude
        norm_squared = sum(x * x for x in dec_components)
        magnitude = norm_squared.sqrt()
        
        # Return as dictionary of float values
        return {
            'phase': float(phase),
            'magnitude': float(magnitude)
        }


class SpaceTransformer:
    """
    Transformer for converting vectors between different geometric spaces.
    
    This class provides methods to transform vectors between hyperbolic,
    elliptical, and Euclidean spaces, enabling the construction of the
    Zero-Point Address (ZPA) triple.
    
    The SpaceTransformer is a critical component of the ZPDR framework that
    implements the fiber bundle connections between the three geometric spaces.
    It provides methods to map vectors between spaces while preserving their
    essential geometric properties and maintaining coherence.
    
    In the mathematical foundation of the Prime Framework, these transformations
    represent the structure-preserving maps that connect the different fibers
    over the reference manifold. The consistent relationships maintained by these
    transformations are what give ZPDR its robust error-detection and error-correction
    capabilities.
    
    Key transformation types:
    
    1. Hyperbolic ↔ Euclidean: Maps between the Poincaré disk model and standard
       Euclidean space, preserving important invariants like relative positions and
       distances after accounting for the different metrics.
       
    2. Elliptical ↔ Euclidean: Maps between the unit sphere model and standard
       Euclidean space, typically through radial projection that preserves angular
       relationships.
       
    3. Hyperbolic ↔ Elliptical: Typically performed via Euclidean space as an
       intermediate step, these transformations connect the negative and positive
       curvature spaces in a coherent manner.
    
    The transformations implement the appropriate mathematical operations to ensure
    the transformed vectors remain valid in their destination spaces and maintain
    the geometric relationships required for coherent ZPDR triples.
    
    These transformations, combined with the invariant extraction and normalization
    procedures in each vector class, enable the construction of normalized ZPA triples
    that form the foundation of ZPDR's data encoding and reconstruction capabilities.
    """
    
    @staticmethod
    def hyperbolic_to_euclidean(v: HyperbolicVector) -> EuclideanVector:
        """
        Transform a hyperbolic vector to Euclidean space with high precision.
        
        Uses the Poincaré disk to Euclidean transformation.
        
        Args:
            v: Hyperbolic vector to transform
            
        Returns:
            Equivalent Euclidean vector
        """
        # Use high-precision calculations
        dec_components = v._dec_components
        
        # For the Poincaré disk model, a direct mapping preserves the coordinates
        # but changes the interpretation of the metric
        
        # Convert to float for EuclideanVector
        return EuclideanVector([float(x) for x in dec_components])
    
    @staticmethod
    def euclidean_to_hyperbolic(v: EuclideanVector) -> HyperbolicVector:
        """
        Transform a Euclidean vector to hyperbolic space with high precision.
        
        Projects into the Poincaré disk model.
        
        Args:
            v: Euclidean vector to transform
            
        Returns:
            Equivalent hyperbolic vector
        """
        # Use high-precision calculations
        dec_components = v._dec_components
        
        # Calculate norm with high precision
        norm_squared = sum(x * x for x in dec_components)
        norm = norm_squared.sqrt()
        
        # Project into the Poincaré disk if necessary
        if norm >= Decimal('1.0'):
            # Scale to fit within the unit disk with high precision
            scale_factor = Decimal('0.99') / norm
            projected = [x * scale_factor for x in dec_components]
            # Convert to float for HyperbolicVector
            return HyperbolicVector([float(x) for x in projected])
        else:
            # Already within the disk
            return HyperbolicVector([float(x) for x in dec_components])
    
    @staticmethod
    def elliptical_to_euclidean(v: EllipticalVector) -> EuclideanVector:
        """
        Transform an elliptical vector to Euclidean space with high precision.
        
        Uses radial projection from the unit sphere.
        
        Args:
            v: Elliptical vector to transform
            
        Returns:
            Equivalent Euclidean vector
        """
        # Use high-precision calculations
        dec_components = v._dec_components
        
        # Direct mapping - elliptical vectors are already normalized to unit sphere
        # Convert to float for EuclideanVector
        return EuclideanVector([float(x) for x in dec_components])
    
    @staticmethod
    def euclidean_to_elliptical(v: EuclideanVector) -> EllipticalVector:
        """
        Transform a Euclidean vector to elliptical space with high precision.
        
        Projects onto the unit sphere.
        
        Args:
            v: Euclidean vector to transform
            
        Returns:
            Equivalent elliptical vector
        """
        # Project onto the unit sphere (will be normalized by EllipticalVector)
        return EllipticalVector(list(v.components))
    
    @staticmethod
    def hyperbolic_to_elliptical(v: HyperbolicVector) -> EllipticalVector:
        """
        Transform a hyperbolic vector to elliptical space with high precision.
        
        Uses Euclidean space as an intermediate step.
        
        Args:
            v: Hyperbolic vector to transform
            
        Returns:
            Equivalent elliptical vector
        """
        # Convert to Euclidean first, then to elliptical
        euclidean = SpaceTransformer.hyperbolic_to_euclidean(v)
        return SpaceTransformer.euclidean_to_elliptical(euclidean)
    
    @staticmethod
    def elliptical_to_hyperbolic(v: EllipticalVector) -> HyperbolicVector:
        """
        Transform an elliptical vector to hyperbolic space with high precision.
        
        Uses Euclidean space as an intermediate step.
        
        Args:
            v: Elliptical vector to transform
            
        Returns:
            Equivalent hyperbolic vector
        """
        # Convert to Euclidean first, then to hyperbolic
        euclidean = SpaceTransformer.elliptical_to_euclidean(v)
        return SpaceTransformer.euclidean_to_hyperbolic(euclidean)


# Example creation function for testing
def create_vector(components: List[float], space_type: SpaceType) -> GeometricVector:
    """
    Create a vector in the specified geometric space.
    
    Args:
        components: List of vector components
        space_type: Type of geometric space
        
    Returns:
        A vector in the specified space
    """
    if space_type == SpaceType.HYPERBOLIC:
        return HyperbolicVector(components)
    elif space_type == SpaceType.ELLIPTICAL:
        return EllipticalVector(components)
    elif space_type == SpaceType.EUCLIDEAN:
        return EuclideanVector(components)
    else:
        raise ValueError(f"Unknown space type: {space_type}")