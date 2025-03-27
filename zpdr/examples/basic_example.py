"""
Basic example demonstrating the usage of ZPDR core components.

This example shows:
1. Creating and manipulating multivectors
2. Working with different geometric spaces
3. Converting between spaces
"""

import numpy as np
from zpdr.core.multivector import Multivector
from zpdr.core.geometric_spaces import (
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector, 
    SpaceTransformer
)


def multivector_demo():
    """Demonstrate basic multivector operations."""
    print("\n=== Multivector Demonstration ===")
    
    # Create a general multivector with components of different grades
    mv = Multivector({
        "1": 1.0,       # scalar part
        "e1": 2.0,      # vector part
        "e2": 3.0,
        "e12": 4.0,     # bivector part
        "e123": 5.0     # trivector part
    })
    
    print(f"Original multivector: {mv}")
    
    # Extract components by grade
    scalar = mv.grade(0)
    vector = mv.grade(1)
    bivector = mv.grade(2)
    trivector = mv.grade(3)
    
    print(f"Scalar part: {scalar}")
    print(f"Vector part: {vector}")
    print(f"Bivector part: {bivector}")
    print(f"Trivector part: {trivector}")
    
    # Demonstrate geometric product
    e1 = Multivector({"e1": 1.0})
    e2 = Multivector({"e2": 1.0})
    
    print("\nGeometric products:")
    print(f"e1 * e1 = {e1 * e1}")  # Should be 1 (scalar)
    print(f"e1 * e2 = {e1 * e2}")  # Should be e12 (bivector)
    print(f"e2 * e1 = {e2 * e1}")  # Should be -e12 (bivector)
    
    # Demonstrate inner and outer products
    print("\nInner and outer products:")
    mv1 = Multivector({"e1": 1.0, "e2": 2.0})
    mv2 = Multivector({"e1": 3.0, "e2": 4.0})
    
    print(f"mv1 = {mv1}")
    print(f"mv2 = {mv2}")
    print(f"Inner product: {mv1.inner_product(mv2)}")  # Should be a scalar
    print(f"Outer product: {mv1.outer_product(mv2)}")  # Should be a bivector


def geometric_spaces_demo():
    """Demonstrate operations in different geometric spaces."""
    print("\n=== Geometric Spaces Demonstration ===")
    
    # Create vectors in each space
    hyperbolic = HyperbolicVector([0.3, 0.4])
    elliptical = EllipticalVector([3.0, 4.0])
    euclidean = EuclideanVector([3.0, 4.0])
    
    print(f"Hyperbolic vector: {hyperbolic}")
    print(f"Elliptical vector: {elliptical}")
    print(f"Euclidean vector: {euclidean}")
    
    # Norms in each space
    print("\nNorms:")
    print(f"Hyperbolic norm: {hyperbolic.norm()}")
    print(f"Elliptical norm: {elliptical.norm()}")
    print(f"Euclidean norm: {euclidean.norm()}")
    
    # Addition in each space
    print("\nAddition in each space:")
    h1 = HyperbolicVector([0.1, 0.2])
    h2 = HyperbolicVector([0.2, 0.1])
    print(f"Hyperbolic: {h1} + {h2} = {h1 + h2}")
    
    e1 = EllipticalVector([1.0, 0.0])
    e2 = EllipticalVector([0.0, 1.0])
    print(f"Elliptical: {e1} + {e2} = {e1 + e2}")
    
    u1 = EuclideanVector([1.0, 2.0])
    u2 = EuclideanVector([3.0, 4.0])
    print(f"Euclidean: {u1} + {u2} = {u1 + u2}")


def space_transformations_demo():
    """Demonstrate transformations between geometric spaces."""
    print("\n=== Space Transformations Demonstration ===")
    
    # Create initial vectors
    euclidean = EuclideanVector([0.6, 0.8])
    print(f"Original Euclidean vector: {euclidean}")
    
    # Transform to other spaces
    hyperbolic = SpaceTransformer.euclidean_to_hyperbolic(euclidean)
    elliptical = SpaceTransformer.euclidean_to_elliptical(euclidean)
    
    print(f"Transformed to Hyperbolic: {hyperbolic}")
    print(f"Transformed to Elliptical: {elliptical}")
    
    # Transform back
    back_to_euclidean1 = SpaceTransformer.hyperbolic_to_euclidean(hyperbolic)
    back_to_euclidean2 = SpaceTransformer.elliptical_to_euclidean(elliptical)
    
    print(f"Back to Euclidean from Hyperbolic: {back_to_euclidean1}")
    print(f"Back to Euclidean from Elliptical: {back_to_euclidean2}")
    
    # Direct transformations between hyperbolic and elliptical
    h_to_e = SpaceTransformer.hyperbolic_to_elliptical(hyperbolic)
    e_to_h = SpaceTransformer.elliptical_to_hyperbolic(elliptical)
    
    print(f"Hyperbolic to Elliptical: {h_to_e}")
    print(f"Elliptical to Hyperbolic: {e_to_h}")


def main():
    """Run all demonstrations."""
    print("Zero-Point Data Resolution (ZPDR) Basic Example")
    print("===============================================")
    
    multivector_demo()
    geometric_spaces_demo()
    space_transformations_demo()


if __name__ == "__main__":
    main()