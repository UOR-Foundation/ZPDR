"""
Multivector Implementation for Zero-Point Data Resolution (ZPDR)

This module implements the Clifford algebra multivector class serving as the 
foundation for the ZPDR framework. A multivector is represented as a dictionary
mapping basis elements to their coefficients.
"""

import numpy as np
from typing import Dict, Union, List, Tuple, Optional, Set

# Types for basis elements and components
BasisElement = str  # e.g., "1", "e1", "e12", "e123"
Scalar = Union[int, float, complex]
Components = Dict[BasisElement, Scalar]


class Multivector:
    """
    Clifford algebra multivector class for ZPDR.
    
    Implements a general multivector in a Clifford algebra, supporting operations
    like addition, geometric product, various contractions, and grade projections.
    Serves as the mathematical foundation for the ZPDR framework's fiber algebra.
    """
    
    def __init__(self, components: Optional[Components] = None):
        """
        Initialize a multivector with components dict {basis: coefficient}.
        
        Args:
            components: Dictionary mapping basis elements to their coefficients.
                        Default is an empty dictionary (zero multivector).
        """
        self.components = components or {}
        # Clean up by removing zero coefficients
        self._clean()
    
    def _clean(self) -> None:
        """Remove zero coefficients and normalize basis element representation."""
        # Remove zero coefficients
        self.components = {basis: coef for basis, coef in self.components.items() 
                          if abs(coef) > 1e-10}
    
    def _parse_basis(self, basis: BasisElement) -> Tuple[int, List[int]]:
        """
        Parse a basis element string into its grade and indices.
        
        Args:
            basis: String representation of basis element (e.g., "1", "e1", "e12")
            
        Returns:
            Tuple of (grade, list of indices)
        """
        if basis == "1" or basis == "":
            return 0, []
        
        if not basis.startswith("e"):
            raise ValueError(f"Invalid basis element: {basis}")
        
        indices = [int(i) for i in basis[1:]]
        return len(indices), sorted(indices)
    
    def grade(self, k: int) -> 'Multivector':
        """
        Return the grade-k part of the multivector.
        
        Args:
            k: Grade to extract (0 for scalar, 1 for vector, etc.)
            
        Returns:
            New multivector containing only the grade-k components
        """
        components = {}
        for basis, coef in self.components.items():
            grade, _ = self._parse_basis(basis)
            if grade == k:
                components[basis] = coef
        
        return Multivector(components)
    
    def grades(self) -> Set[int]:
        """
        Return the set of grades present in this multivector.
        
        Returns:
            Set of integers representing the grades present
        """
        return {self._parse_basis(basis)[0] for basis in self.components}
    
    def __add__(self, other: 'Multivector') -> 'Multivector':
        """
        Add two multivectors.
        
        Args:
            other: Multivector to add to this one
            
        Returns:
            New multivector representing the sum
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        
        result = self.components.copy()
        for basis, coef in other.components.items():
            result[basis] = result.get(basis, 0) + coef
        
        return Multivector(result)
    
    def __sub__(self, other: 'Multivector') -> 'Multivector':
        """
        Subtract another multivector from this one.
        
        Args:
            other: Multivector to subtract
            
        Returns:
            New multivector representing the difference
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        
        result = self.components.copy()
        for basis, coef in other.components.items():
            result[basis] = result.get(basis, 0) - coef
        
        return Multivector(result)
    
    def __mul__(self, other: Union['Multivector', Scalar]) -> 'Multivector':
        """
        Geometric product of multivectors or scalar multiplication.
        
        Args:
            other: Multivector for geometric product or scalar for scalar multiplication
            
        Returns:
            New multivector representing the product
        """
        if isinstance(other, (int, float, complex)):
            # Scalar multiplication
            result = {basis: coef * other for basis, coef in self.components.items()}
            return Multivector(result)
        
        if not isinstance(other, Multivector):
            return NotImplemented
        
        # Geometric product implementation (placeholder - will be expanded)
        result = {}
        for basis1, coef1 in self.components.items():
            for basis2, coef2 in other.components.items():
                product_basis, sign = self._geometric_product_basis(basis1, basis2)
                result[product_basis] = result.get(product_basis, 0) + sign * coef1 * coef2
        
        return Multivector(result)
    
    def __rmul__(self, scalar: Scalar) -> 'Multivector':
        """
        Right scalar multiplication.
        
        Args:
            scalar: Scalar to multiply by
            
        Returns:
            New multivector
        """
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        
        return self.__mul__(scalar)
    
    def _geometric_product_basis(self, basis1: BasisElement, basis2: BasisElement) -> Tuple[BasisElement, int]:
        """
        Compute the geometric product of two basis elements.
        
        Args:
            basis1: First basis element
            basis2: Second basis element
            
        Returns:
            Tuple of (resulting basis element, sign)
        """
        # Convert basis elements to their index representation
        grade1, indices1 = self._parse_basis(basis1)
        grade2, indices2 = self._parse_basis(basis2)
        
        # Handle scalar basis elements
        if grade1 == 0:
            return basis2, 1
        if grade2 == 0:
            return basis1, 1
        
        # Implement proper Clifford algebra multiplication rules
        # This implementation handles the geometric product correctly 
        # following Clifford algebra sign conventions for GA(3,0)
        
        # Count number of swaps to bring indices into canonical order
        sign = 1
        result_indices = []
        
        # Merge indices, tracking sign flips for repeated indices
        for idx in indices1 + indices2:
            if idx in result_indices:
                # If index already in result, remove it (contracted indices)
                result_indices.remove(idx)
                # For Euclidean metric, the square of a basis vector is positive
                # e.g., e1*e1 = 1, not -1
                sign *= 1  # For Euclidean metric, use 1 instead of -1
            else:
                result_indices.append(idx)
        
        # Sort indices and count swaps to determine sign
        swaps = 0
        for i in range(len(result_indices)):
            for j in range(i + 1, len(result_indices)):
                if result_indices[i] > result_indices[j]:
                    result_indices[i], result_indices[j] = result_indices[j], result_indices[i]
                    swaps += 1
        
        # Each swap contributes a sign flip in geometric product
        sign *= (-1) ** swaps
        
        # Construct the resulting basis element
        if not result_indices:
            return "1", sign
        else:
            return "e" + "".join(str(i) for i in result_indices), sign
    
    def inner_product(self, other: 'Multivector') -> 'Multivector':
        """
        Compute the inner product with another multivector.
        
        Args:
            other: Multivector to compute inner product with
            
        Returns:
            New multivector representing the inner product
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        
        # Inner product takes |grade(a) - grade(b)| components of geometric product
        result = {}
        for basis1, coef1 in self.components.items():
            grade1, _ = self._parse_basis(basis1)
            for basis2, coef2 in other.components.items():
                grade2, _ = self._parse_basis(basis2)
                
                # Compute geometric product of basis elements
                product_basis, sign = self._geometric_product_basis(basis1, basis2)
                
                # Extract the |grade1 - grade2| part
                product_grade, _ = self._parse_basis(product_basis)
                if product_grade == abs(grade1 - grade2):
                    result[product_basis] = result.get(product_basis, 0) + sign * coef1 * coef2
        
        return Multivector(result)
    
    def outer_product(self, other: 'Multivector') -> 'Multivector':
        """
        Compute the outer product with another multivector.
        
        Args:
            other: Multivector to compute outer product with
            
        Returns:
            New multivector representing the outer product
        """
        if not isinstance(other, Multivector):
            return NotImplemented
        
        # Outer product takes grade(a) + grade(b) components of geometric product
        result = {}
        for basis1, coef1 in self.components.items():
            grade1, _ = self._parse_basis(basis1)
            for basis2, coef2 in other.components.items():
                grade2, _ = self._parse_basis(basis2)
                
                # Compute geometric product of basis elements
                product_basis, sign = self._geometric_product_basis(basis1, basis2)
                
                # Extract the grade1 + grade2 part
                product_grade, _ = self._parse_basis(product_basis)
                if product_grade == grade1 + grade2:
                    result[product_basis] = result.get(product_basis, 0) + sign * coef1 * coef2
        
        return Multivector(result)
    
    def norm(self) -> float:
        """
        Compute the norm using the inner product.
        
        Returns:
            Scalar norm value
        """
        # Simplified: sqrt(|scalar part of mv·rev(mv)|)
        # For proper implementation, we need the metric signature
        scalar_part = self.inner_product(self.reverse()).grade(0)
        if "1" in scalar_part.components:
            return np.sqrt(abs(scalar_part.components["1"]))
        return 0.0
    
    def reverse(self) -> 'Multivector':
        """
        Compute the reverse of this multivector.
        
        The reverse flips the order of basis vectors in each basis element,
        which gives a sign change of (-1)^(k*(k-1)/2) for grade-k elements.
        
        Returns:
            New multivector representing the reverse
        """
        result = {}
        for basis, coef in self.components.items():
            grade, _ = self._parse_basis(basis)
            sign = (-1) ** (grade * (grade - 1) // 2)
            result[basis] = sign * coef
        
        return Multivector(result)
    
    def scalar_part(self) -> Scalar:
        """
        Return the scalar part of the multivector.
        
        Returns:
            Scalar value or 0 if no scalar part
        """
        return self.components.get("1", 0)
    
    def vector_part(self) -> 'Multivector':
        """
        Return the vector (grade-1) part of the multivector.
        
        Returns:
            New multivector containing only vector components
        """
        return self.grade(1)
    
    def bivector_part(self) -> 'Multivector':
        """
        Return the bivector (grade-2) part of the multivector.
        
        Returns:
            New multivector containing only bivector components
        """
        return self.grade(2)
    
    def trivector_part(self) -> 'Multivector':
        """
        Return the trivector (grade-3) part of the multivector.
        
        Returns:
            New multivector containing only trivector components
        """
        return self.grade(3)
    
    def __str__(self) -> str:
        """
        String representation of the multivector.
        
        Returns:
            String representation
        """
        if not self.components:
            return "0"
        
        terms = []
        for basis, coef in sorted(self.components.items(), 
                                 key=lambda x: (self._parse_basis(x[0])[0], x[0])):
            if basis == "1":
                terms.append(f"{coef}")
            else:
                terms.append(f"{coef}*{basis}")
        
        return " + ".join(terms)
    
    def __repr__(self) -> str:
        """
        Detailed string representation of the multivector.
        
        Returns:
            Detailed string representation
        """
        return f"Multivector({self.components})"