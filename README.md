# Zero-Point Data Resolution (ZPDR)

[![ZPDR Tests](https://github.com/zpdr/zpdr/actions/workflows/run-tests.yml/badge.svg)](https://github.com/zpdr/zpdr/actions/workflows/run-tests.yml)
[![ZPDR Basic Tests](https://github.com/zpdr/zpdr/actions/workflows/basic-tests.yml/badge.svg)](https://github.com/zpdr/zpdr/actions/workflows/basic-tests.yml)

A mathematical framework for universal data encoding and resolution based on the principles of the Prime Framework.

## Description

Zero-Point Data Resolution (ZPDR) is a novel approach to representing data using a geometric three-vector decomposition. ZPDR encodes data objects as coordinates in three complementary geometric vector spaces:

1. **Hyperbolic vector (H)** - Captures the base transformation system in a negative-curvature space
2. **Elliptical vector (E)** - Captures the transformation span in a positive-curvature space
3. **Euclidean vector (U)** - Captures the transformed object itself in a flat space

This approach provides a base-agnostic, universal way to encode and reconstruct data with high coherence.

## Structure

The library is organized into the following modules:

- `zpdr/core/` - Core mathematical structures
  - `multivector.py` - Clifford algebra implementation
  - `geometric_spaces.py` - Hyperbolic, elliptical, and Euclidean spaces
  - (Additional components to be added in future phases)
- `zpdr/utils/` - Utility functions
- `zpdr/examples/` - Example applications
- `zpdr/tests/` - Test suite

## Installation

```bash
# Clone the repository
git clone https://github.com/zpdr/zpdr.git
cd zpdr

# Install in development mode
pip install -e .
```

## Example Usage

Here's a simple example showing the creation and manipulation of multivectors:

```python
from zpdr.core.multivector import Multivector

# Create a multivector
mv = Multivector({
    "1": 1.0,       # scalar part
    "e1": 2.0,      # vector part
    "e2": 3.0,
    "e12": 4.0,     # bivector part
    "e123": 5.0     # trivector part
})

# Extract parts by grade
scalar = mv.grade(0)
vector = mv.grade(1)
bivector = mv.grade(2)
trivector = mv.grade(3)

# Perform operations
e1 = Multivector({"e1": 1.0})
e2 = Multivector({"e2": 1.0})
product = e1 * e2  # Geometric product
```

And working with geometric spaces:

```python
from zpdr.core.geometric_spaces import (
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector, 
    SpaceTransformer
)

# Create vectors in different spaces
h = HyperbolicVector([0.3, 0.4])
e = EllipticalVector([3.0, 4.0])
u = EuclideanVector([3.0, 4.0])

# Transform between spaces
h_to_e = SpaceTransformer.hyperbolic_to_elliptical(h)
e_to_u = SpaceTransformer.elliptical_to_euclidean(e)
```

For more examples, see the `zpdr/examples/` directory.

## Implementation Status

The project is being implemented in phases:

- [x] **Phase 1**: Core Mathematical Framework
  - Basic Clifford algebra multivector class
  - Mathematical spaces (hyperbolic, elliptical, Euclidean)
  - Basic transformations between spaces

- [x] **Phase 2**: Basic ZPDR Operations
  - Multivector data encoding
  - ZPA extraction algorithms
  - Basic coherence calculations
  - Simple serialization format

- [x] **Phase 3**: Full Pipeline
  - Complete encoding/decoding pipeline
  - Verification mechanisms
  - Error correction
  - Performance optimization

- [x] **Phase 4**: Examples and Documentation
  - Example applications
  - Comprehensive documentation
  - API reference documentation
  - Interactive Jupyter notebooks
  - Continuous integration testing

## Documentation

For a comprehensive reference of the ZPDR framework, see:

- [API Reference](API_REFERENCE.md) - Complete documentation of classes, methods, and functions
- [Pure-ZPDR.md](Pure-ZPDR.md) - Theoretical foundation of the ZPDR framework
- [Jupyter Notebooks](notebooks/) - Interactive examples and tutorials

## Testing

The ZPDR framework includes a comprehensive test suite that validates the mathematical principles and functionality of the implementation. To run the tests:

```bash
# Run all tests
pytest zpdr/tests/

# Run specific test modules
pytest zpdr/tests/test_basic.py
pytest zpdr/tests/test_zpdr_trilateral_extraction.py

# Run tests with coverage
pytest --cov=zpdr zpdr/tests/
```

The GitHub Actions workflows automatically run these tests on multiple Python versions to ensure compatibility and correctness.

## License

This project is licensed under the MIT License - see the LICENSE file for details.