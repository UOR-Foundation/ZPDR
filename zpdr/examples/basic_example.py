"""
Basic example demonstrating the usage of ZPDR core components.

This example provides a practical introduction to the Zero-Point Data Resolution (ZPDR)
framework by demonstrating its core mathematical components and operations. The example
is structured to progressively introduce the fundamental concepts of ZPDR:

1. Multivectors and Clifford Algebra:
   - Creating multivectors with components of different grades
   - Extracting components by grade (scalar, vector, bivector, trivector)
   - Performing geometric, inner, and outer products
   - Understanding the algebraic structure of the fiber algebra

2. Geometric Spaces:
   - Working with hyperbolic space (negative curvature)
   - Working with elliptical space (positive curvature)
   - Working with Euclidean space (zero curvature)
   - Calculating norms and performing additions in each space

3. Space Transformations:
   - Converting vectors between different geometric spaces
   - Understanding how the transformations preserve mathematical relationships
   - Observing the effects of round-trip transformations

Mathematical Foundation:
This example demonstrates the practical implementation of the Prime Framework's
mathematical principles, showing how data can be represented and transformed across
different geometric spaces while maintaining coherent relationships. The trilateral
representation system (hyperbolic, elliptical, and Euclidean vectors) forms the
foundation of ZPDR's robust data encoding capabilities.

Usage:
Run this script directly to see step-by-step demonstrations of ZPDR components:

    python -m zpdr.examples.basic_example
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
    """
    Demonstrate basic multivector operations in the Clifford algebra framework.
    
    This function illustrates the fundamental operations of Clifford algebra that
    form the mathematical foundation of ZPDR's fiber algebra. It demonstrates:
    
    1. Creating multivectors with components of various grades:
       - Grade 0: Scalar (magnitude information)
       - Grade 1: Vector (directional information)
       - Grade 2: Bivector (area/rotational information)
       - Grade 3: Trivector (volume/orientation information)
       
    2. Grade projections for extracting specific components of a multivector
    
    3. Geometric product operations, showing how:
       - e1 * e1 = 1 (square of a basis vector is scalar)
       - e1 * e2 = e12 (product of different basis vectors creates a bivector)
       - e2 * e1 = -e12 (geometric product is anti-commutative for vector terms)
       
    4. Inner and outer products, demonstrating:
       - Inner product: Contracts grades and gives a lower-grade result
       - Outer product: Expands grades and gives a higher-grade result
       
    These operations demonstrate how the Clifford algebra structure enables
    ZPDR to represent mathematical objects across multiple geometric spaces
    simultaneously and perform coherent transformations between them.
    """
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
    """
    Demonstrate operations in the three geometric spaces used in ZPDR.
    
    This function illustrates how ZPDR operates in three complementary geometric
    spaces, each with different curvature properties:
    
    1. Hyperbolic Space (negative curvature):
       - Represented by the Poincaré disk model
       - Used in ZPDR to capture base transformation systems
       - Points must lie within the unit disk (|v| < 1)
       - Special addition operation using Möbius transformation
    
    2. Elliptical Space (positive curvature):
       - Represented by points on the unit sphere
       - Used in ZPDR to capture transformation spans
       - All points have unit norm (|v| = 1)
       - Addition involves Euclidean addition followed by normalization
    
    3. Euclidean Space (zero/flat curvature):
       - Conventional vector space with standard operations
       - Used in ZPDR to capture the transformed object itself
       - No constraints on vector magnitudes
       - Standard vector addition
    
    The demonstration shows how vectors in each space have different properties
    and behaviors, particularly with respect to norms and addition operations.
    These distinct properties enable ZPDR to represent mathematical objects
    in a robust trilateral system that provides inherent error detection and
    correction capabilities through the maintained relationships between spaces.
    """
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
    """
    Demonstrate transformations between the three geometric spaces in ZPDR.
    
    This function illustrates how vectors can be transformed between hyperbolic,
    elliptical, and Euclidean spaces while preserving their essential geometric
    properties. These transformations are a key component of ZPDR's ability to
    maintain coherent relationships across different spaces.
    
    The demonstration shows:
    
    1. Euclidean to Hyperbolic: 
       - Projects Euclidean vectors into the Poincaré disk model
       - Preserves direction while adjusting magnitude to fit within unit disk
       - Essential for creating the H component of the ZPA triple
       
    2. Euclidean to Elliptical:
       - Projects Euclidean vectors onto the unit sphere
       - Preserves direction while normalizing to unit length
       - Essential for creating the E component of the ZPA triple
       
    3. Round-Trip Transformations:
       - Hyperbolic → Euclidean → Hyperbolic
       - Elliptical → Euclidean → Elliptical
       - Shows how information is preserved through space transformations
       
    4. Direct Transformations:
       - Hyperbolic ↔ Elliptical (usually via Euclidean space)
       - Demonstrates the relationships between negative and positive curvature spaces
    
    These transformations implement the fiber bundle connections described in the
    Prime Framework, linking different geometric spaces in a coherent mathematical
    structure. The SpaceTransformer ensures these transformations maintain the
    geometric invariants necessary for zero-point normalization and data reconstruction.
    """
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
    """
    Run all ZPDR basic demonstrations in a structured sequence.
    
    This function executes the three main demonstrations of ZPDR components:
    
    1. Multivector operations (Clifford algebra foundation)
    2. Geometric spaces (hyperbolic, elliptical, and Euclidean)
    3. Space transformations (conversions between different spaces)
    
    The demonstrations progress from foundational mathematical concepts to
    more advanced operations, providing a comprehensive introduction to the
    core components of the ZPDR framework.
    
    This structured approach helps users understand:
    - The mathematical building blocks of ZPDR
    - How these components interact in a coherent system
    - The practical implementation of Prime Framework principles
    
    By running this example, users can observe the behavior of ZPDR components
    in action and gain insights into how the framework represents and transforms
    data across different geometric spaces.
    """
    print("Zero-Point Data Resolution (ZPDR) Basic Example")
    print("===============================================")
    
    multivector_demo()
    geometric_spaces_demo()
    space_transformations_demo()


if __name__ == "__main__":
    main()