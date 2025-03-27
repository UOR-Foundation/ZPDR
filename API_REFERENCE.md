# ZPDR API Reference

## Table of Contents

1. [Introduction](#introduction)
2. [Core Modules](#core-modules)
   - [Multivector](#multivector)
   - [Geometric Spaces](#geometric-spaces)
   - [ZPA Manifest](#zpa-manifest)
   - [ZPDR Processor](#zpdr-processor)
   - [Error Corrector](#error-corrector)
   - [Verifier](#verifier)
3. [Utility Functions](#utility-functions)
   - [High-Precision Arithmetic](#high-precision-arithmetic)
   - [Coherence Calculations](#coherence-calculations)
   - [Normalization Operations](#normalization-operations)
   - [Encoding/Decoding Helpers](#encodingdecoding-helpers)
4. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [File Processing](#file-processing)
   - [Error Correction](#error-correction)
5. [Mathematical Concepts](#mathematical-concepts)
   - [Trilateral Vector System](#trilateral-vector-system)
   - [Coherence Measures](#coherence-measures)
   - [Zero-Point Normalization](#zero-point-normalization)
   - [Error Detection and Correction](#error-detection-and-correction)

## Introduction

The Zero-Point Data Resolution (ZPDR) framework is a mathematical data representation system based on the Prime Framework. It represents data as a trilateral vector system across three geometric spaces: hyperbolic (negative curvature), elliptical (positive curvature), and Euclidean (flat/zero curvature).

This API reference documents the classes, methods, and functions provided by the ZPDR implementation, with explanations of the mathematical concepts and usage patterns.

## Core Modules

### Multivector

The `Multivector` class implements Clifford algebra operations, serving as the mathematical foundation for the ZPDR framework's fiber algebra.

#### Class: `Multivector`

**Description**: A general multivector in a Clifford algebra, supporting operations like addition, geometric product, various contractions, and grade projections.

**Import**:
```python
from zpdr.core.multivector import Multivector
```

**Constructor**:
```python
Multivector(components: Optional[Dict[str, Union[int, float, complex]]] = None)
```

**Parameters**:
- `components`: Dictionary mapping basis elements to their coefficients. Default is an empty dictionary (zero multivector).

**Key Methods**:

- `grade(k: int) -> Multivector`: Return the grade-k part of the multivector.
  - `k`: Grade to extract (0 for scalar, 1 for vector, etc.)
  - Returns: New multivector containing only the grade-k components

- `grades() -> Set[int]`: Return the set of grades present in the multivector.
  - Returns: Set of integers representing the grades present

- `__add__(other: Multivector) -> Multivector`: Add two multivectors.
  - `other`: Multivector to add to this one
  - Returns: New multivector representing the sum

- `__sub__(other: Multivector) -> Multivector`: Subtract another multivector from this one.
  - `other`: Multivector to subtract
  - Returns: New multivector representing the difference

- `__mul__(other: Union[Multivector, Scalar]) -> Multivector`: Geometric product of multivectors or scalar multiplication.
  - `other`: Multivector for geometric product or scalar for scalar multiplication
  - Returns: New multivector representing the product

- `inner_product(other: Multivector) -> Multivector`: Compute the inner product with another multivector.
  - `other`: Multivector to compute inner product with
  - Returns: New multivector representing the inner product

- `outer_product(other: Multivector) -> Multivector`: Compute the outer product with another multivector.
  - `other`: Multivector to compute outer product with
  - Returns: New multivector representing the outer product

- `norm() -> float`: Compute the norm using the inner product.
  - Returns: Scalar norm value

- `reverse() -> Multivector`: Compute the reverse of the multivector.
  - Returns: New multivector representing the reverse

- `scalar_part() -> Scalar`: Return the scalar part of the multivector.
  - Returns: Scalar value or 0 if no scalar part

- `vector_part() -> Multivector`: Return the vector (grade-1) part.
  - Returns: New multivector containing only vector components

- `bivector_part() -> Multivector`: Return the bivector (grade-2) part.
  - Returns: New multivector containing only bivector components

- `trivector_part() -> Multivector`: Return the trivector (grade-3) part.
  - Returns: New multivector containing only trivector components

**Example Usage**:

```python
# Create a multivector
mv = Multivector({
    "1": 1.0,       # scalar part
    "e1": 2.0,      # vector part
    "e2": 3.0,      # vector part
    "e12": 4.0      # bivector part
})

# Extract components by grade
scalar = mv.grade(0)    # grade-0 (scalar) components
vector = mv.grade(1)    # grade-1 (vector) components
bivector = mv.grade(2)  # grade-2 (bivector) components

# Perform operations
e1 = Multivector({"e1": 1.0})
e2 = Multivector({"e2": 1.0})
product = e1 * e2       # geometric product (results in "e12" bivector)
inner = e1.inner_product(e2)  # inner product
outer = e1.outer_product(e2)  # outer product (same as e1 ∧ e2)
```

### Geometric Spaces

The `geometric_spaces` module implements the three geometric spaces used in ZPDR (hyperbolic, elliptical, and Euclidean) and transformations between them.

#### Class: `GeometricVector` (Abstract Base Class)

**Description**: Base class for vectors in different geometric spaces, providing common functionality.

**Import**:
```python
from zpdr.core.geometric_spaces import GeometricVector, SpaceType
```

**Constructor**:
```python
GeometricVector(components: List[float], space_type: SpaceType)
```

**Parameters**:
- `components`: List of vector components
- `space_type`: Type of geometric space this vector belongs to (from SpaceType enum)

**Key Methods**:

- `inner_product(other: GeometricVector) -> float`: Compute inner product with another vector (abstract).
  - `other`: Vector to compute inner product with (must be in the same space)
  - Returns: Scalar inner product value

- `norm() -> float`: Compute the norm of the vector with high precision.
  - Returns: Scalar norm value

- `normalize() -> GeometricVector`: Return a normalized vector in the same direction (abstract).
  - Returns: Normalized vector

- `get_invariants() -> Dict[str, float]`: Extract geometric invariants from the vector (abstract).
  - Returns: Dictionary of invariants specific to the space type

#### Class: `HyperbolicVector`

**Description**: Vector in hyperbolic space (negative curvature) using the Poincaré disk model.

**Import**:
```python
from zpdr.core.geometric_spaces import HyperbolicVector
```

**Constructor**:
```python
HyperbolicVector(components: List[float])
```

**Parameters**:
- `components`: List of vector components in the Poincaré disk model

**Key Methods**:

- `is_valid() -> bool`: Check if the vector is valid in hyperbolic space (|v| < 1).
  - Returns: True if the vector is valid, False otherwise

- `__add__(other: HyperbolicVector) -> HyperbolicVector`: Hyperbolic addition using Möbius transformation.
  - `other`: Another hyperbolic vector
  - Returns: Resulting hyperbolic vector

- `inner_product(other: HyperbolicVector) -> float`: Hyperbolic inner product in the Poincaré disk model.
  - `other`: Another hyperbolic vector
  - Returns: Scalar inner product value

- `distance_to(other: HyperbolicVector) -> float`: Compute the hyperbolic distance to another point.
  - `other`: Another hyperbolic vector
  - Returns: Scalar distance value

- `normalize() -> HyperbolicVector`: Return a normalized hyperbolic vector with high precision.
  - Returns: Normalized hyperbolic vector

- `get_invariants() -> Dict[str, float]`: Extract rotation and scaling invariants.
  - Returns: Dictionary with 'rotation_angle' and 'scale_factor' keys

#### Class: `EllipticalVector`

**Description**: Vector in elliptical space (positive curvature) using a spherical model.

**Import**:
```python
from zpdr.core.geometric_spaces import EllipticalVector
```

**Constructor**:
```python
EllipticalVector(components: List[float])
```

**Parameters**:
- `components`: List of vector components (automatically normalized to the unit sphere)

**Key Methods**:

- `is_valid() -> bool`: Check if the vector is valid in elliptical space (|v| = 1).
  - Returns: True if the vector is valid, False otherwise

- `__add__(other: EllipticalVector) -> EllipticalVector`: Elliptical addition on the sphere.
  - `other`: Another elliptical vector
  - Returns: Resulting elliptical vector (normalized to unit sphere)

- `inner_product(other: EllipticalVector) -> float`: Elliptical inner product (equivalent to spherical dot product).
  - `other`: Another elliptical vector
  - Returns: Scalar inner product value

- `distance_to(other: EllipticalVector) -> float`: Compute the elliptical (great circle) distance.
  - `other`: Another elliptical vector
  - Returns: Scalar distance value (angle in radians)

- `normalize() -> EllipticalVector`: Return a normalized elliptical vector.
  - Returns: Normalized elliptical vector (on the unit sphere)

- `get_invariants() -> Dict[str, float]`: Extract rotation and radius invariants.
  - Returns: Dictionary with 'rotation_angle' and 'radius' keys

#### Class: `EuclideanVector`

**Description**: Vector in Euclidean space (zero/flat curvature) with standard operations.

**Import**:
```python
from zpdr.core.geometric_spaces import EuclideanVector
```

**Constructor**:
```python
EuclideanVector(components: List[float])
```

**Parameters**:
- `components`: List of vector components in Euclidean space

**Key Methods**:

- `is_valid() -> bool`: Check if the vector is valid in Euclidean space (always true for finite vectors).
  - Returns: True if the vector is valid, False otherwise

- `__add__(other: EuclideanVector) -> EuclideanVector`: Standard Euclidean vector addition.
  - `other`: Another Euclidean vector
  - Returns: Resulting Euclidean vector

- `inner_product(other: EuclideanVector) -> float`: Standard Euclidean inner product (dot product).
  - `other`: Another Euclidean vector
  - Returns: Scalar inner product value

- `distance_to(other: EuclideanVector) -> float`: Compute the Euclidean distance.
  - `other`: Another Euclidean vector
  - Returns: Scalar distance value

- `normalize() -> EuclideanVector`: Return a normalized Euclidean vector.
  - Returns: Normalized Euclidean vector (unit length)

- `get_invariants() -> Dict[str, float]`: Extract phase and magnitude invariants.
  - Returns: Dictionary with 'phase' and 'magnitude' keys

#### Class: `SpaceTransformer`

**Description**: Utility for converting vectors between the three geometric spaces while preserving their essential properties.

**Import**:
```python
from zpdr.core.geometric_spaces import SpaceTransformer
```

**Static Methods**:

- `hyperbolic_to_euclidean(v: HyperbolicVector) -> EuclideanVector`: Transform a hyperbolic vector to Euclidean space.
  - `v`: Hyperbolic vector to transform
  - Returns: Equivalent Euclidean vector

- `euclidean_to_hyperbolic(v: EuclideanVector) -> HyperbolicVector`: Transform a Euclidean vector to hyperbolic space.
  - `v`: Euclidean vector to transform
  - Returns: Equivalent hyperbolic vector

- `elliptical_to_euclidean(v: EllipticalVector) -> EuclideanVector`: Transform an elliptical vector to Euclidean space.
  - `v`: Elliptical vector to transform
  - Returns: Equivalent Euclidean vector

- `euclidean_to_elliptical(v: EuclideanVector) -> EllipticalVector`: Transform a Euclidean vector to elliptical space.
  - `v`: Euclidean vector to transform
  - Returns: Equivalent elliptical vector

- `hyperbolic_to_elliptical(v: HyperbolicVector) -> EllipticalVector`: Transform a hyperbolic vector to elliptical space.
  - `v`: Hyperbolic vector to transform
  - Returns: Equivalent elliptical vector

- `elliptical_to_hyperbolic(v: EllipticalVector) -> HyperbolicVector`: Transform an elliptical vector to hyperbolic space.
  - `v`: Elliptical vector to transform
  - Returns: Equivalent hyperbolic vector

**Example Usage**:

```python
# Create vectors in each space
hyp = HyperbolicVector([0.3, 0.4])
ellip = EllipticalVector([0.6, 0.8])
eucl = EuclideanVector([0.5, 0.5])

# Verify space-specific properties
print(f"Hyperbolic vector valid: {hyp.is_valid()}")  # Must be within unit disk
print(f"Elliptical vector valid: {ellip.is_valid()}")  # Must be on unit sphere
print(f"Euclidean vector norm: {eucl.norm()}")

# Transform between spaces
eucl_from_hyp = SpaceTransformer.hyperbolic_to_euclidean(hyp)
ellip_from_hyp = SpaceTransformer.hyperbolic_to_elliptical(hyp)
eucl_from_ellip = SpaceTransformer.elliptical_to_euclidean(ellip)
```

### ZPA Manifest

The `ZPAManifest` class provides serialization and deserialization for Zero-Point Address (ZPA) representations.

#### Class: `ZPAManifest`

**Description**: Container for ZPDR trilateral vectors (H, E, U) along with metadata and coherence information.

**Import**:
```python
from zpdr.core.zpa_manifest import ZPAManifest
```

**Constructor**:
```python
ZPAManifest(H: Optional[np.ndarray] = None, 
           E: Optional[np.ndarray] = None, 
           U: Optional[np.ndarray] = None, 
           coherence: Optional[float] = None,
           metadata: Optional[Dict[str, Any]] = None)
```

**Parameters**:
- `H`: Hyperbolic vector component (numpy array)
- `E`: Elliptical vector component (numpy array)
- `U`: Euclidean vector component (numpy array)
- `coherence`: Global coherence value (float between 0 and 1)
- `metadata`: Dictionary of additional metadata

**Key Methods**:

- `is_valid() -> bool`: Check if the manifest represents a valid ZPA (based on coherence).
  - Returns: True if the ZPA is valid, False otherwise

- `serialize(compact: bool = False) -> str`: Serialize the manifest to a JSON string.
  - `compact`: If True, use a more compact representation
  - Returns: JSON string representation of the manifest

- `deserialize(json_str: str) -> ZPAManifest`: Create a manifest from its JSON string representation (static method).
  - `json_str`: JSON string to deserialize
  - Returns: ZPAManifest object

- `from_dict(data: Dict[str, Any]) -> ZPAManifest`: Create a manifest from a dictionary representation (static method).
  - `data`: Dictionary containing the manifest data
  - Returns: ZPAManifest object

- `to_dict() -> Dict[str, Any]`: Convert the manifest to a dictionary representation.
  - Returns: Dictionary representation of the manifest

**Example Usage**:

```python
# Create a ZPA manifest
H = np.array([0.1, 0.2, 0.3])
E = np.array([0.6, 0.7, 0.3]) / np.linalg.norm([0.6, 0.7, 0.3])  # Normalize for elliptical
U = np.array([0.4, 0.5, 0.6])

manifest = ZPAManifest(H, E, U, coherence=0.98, metadata={"description": "Example ZPA"})

# Serialize to JSON
json_str = manifest.serialize()

# Deserialize from JSON
recovered_manifest = ZPAManifest.deserialize(json_str)

# Check validity
is_valid = manifest.is_valid()  # Based on coherence threshold
```

### ZPDR Processor

The `ZPDRProcessor` class implements the main encoding and decoding pipeline for the ZPDR framework.

#### Class: `ZPDRProcessor`

**Description**: Main processor for the ZPDR framework, handling encoding to and decoding from ZPA representations.

**Import**:
```python
from zpdr.core.zpdr_processor import ZPDRProcessor
```

**Constructor**:
```python
ZPDRProcessor(config: Optional[Dict[str, Any]] = None)
```

**Parameters**:
- `config`: Optional configuration dictionary with settings like coherence threshold, verification level, etc.

**Key Methods**:

- `encode(data: Union[bytes, str, int, List, Dict], description: Optional[str] = None) -> ZPAManifest`: Encode data to a ZPA manifest.
  - `data`: Data to encode (supports various types)
  - `description`: Optional description for the manifest metadata
  - Returns: ZPAManifest containing the encoded data

- `decode(manifest: ZPAManifest) -> Union[bytes, str, int, List, Dict]`: Decode a ZPA manifest back to its original data.
  - `manifest`: ZPAManifest to decode
  - Returns: Original data reconstructed from the manifest

- `verify(manifest: ZPAManifest) -> bool`: Verify that a ZPA manifest is valid and coherent.
  - `manifest`: ZPAManifest to verify
  - Returns: True if the manifest is valid, False otherwise

- `get_performance_metrics() -> Dict[str, Any]`: Get performance metrics from the processor.
  - Returns: Dictionary containing performance metrics

#### Class: `StreamingZPDRProcessor`

**Description**: Extension of ZPDRProcessor that handles large data streams by processing in chunks.

**Import**:
```python
from zpdr.core.zpdr_processor import StreamingZPDRProcessor
```

**Constructor**:
```python
StreamingZPDRProcessor(config: Optional[Dict[str, Any]] = None)
```

**Parameters**:
- `config`: Optional configuration dictionary with settings like chunk size, coherence threshold, etc.

**Key Methods**:

- `encode_stream(stream: BinaryIO, chunk_size: Optional[int] = None) -> List[ZPAManifest]`: Encode a binary stream in chunks.
  - `stream`: Binary stream to encode
  - `chunk_size`: Size of chunks to process (bytes)
  - Returns: List of ZPAManifest objects, one per chunk

- `decode_stream(manifests: List[ZPAManifest]) -> bytes`: Decode a list of ZPA manifests back to the original data stream.
  - `manifests`: List of ZPAManifest objects to decode
  - Returns: Combined binary data from all manifests

**Example Usage**:

```python
# Create a processor
processor = ZPDRProcessor(config={
    'coherence_threshold': 0.95,
    'verification_level': 'standard',
    'error_correction_enabled': True
})

# Encode some data
data = "Hello, ZPDR!"
manifest = processor.encode(data, description="Text example")

# Decode the manifest
decoded_data = processor.decode(manifest)

# Verify a manifest
is_valid = processor.verify(manifest)

# Streaming example for large files
with open("large_file.bin", "rb") as f:
    # Create a streaming processor
    streaming_processor = StreamingZPDRProcessor(config={'chunk_size': 1024 * 1024})
    
    # Encode the file in chunks
    manifests = streaming_processor.encode_stream(f)
    
    # Decode the chunks back to the original file
    decoded_data = streaming_processor.decode_stream(manifests)
```

### Error Corrector

The `ErrorCorrector` class provides error correction capabilities for ZPDR representations.

#### Class: `ErrorCorrector`

**Description**: Implements error detection and correction mechanisms for ZPDR trilateral vectors.

**Import**:
```python
from zpdr.core.error_corrector import ErrorCorrector
```

**Constructor**:
```python
ErrorCorrector(config: Optional[Dict[str, Any]] = None)
```

**Parameters**:
- `config`: Optional configuration dictionary with settings like correction levels, etc.

**Key Methods**:

- `detect_errors(manifest: ZPAManifest) -> Dict[str, Any]`: Detect errors in a ZPA manifest.
  - `manifest`: ZPAManifest to check for errors
  - Returns: Dictionary with error detection results

- `correct(manifest: ZPAManifest, level: str = 'auto') -> Tuple[ZPAManifest, Dict[str, Any]]`: Apply error correction to a ZPA manifest.
  - `manifest`: ZPAManifest to correct
  - `level`: Correction level ('basic', 'intermediate', 'advanced', or 'auto')
  - Returns: Tuple of (corrected_manifest, correction_report)

**Example Usage**:

```python
# Create an error corrector
error_corrector = ErrorCorrector()

# Detect errors in a manifest
error_report = error_corrector.detect_errors(manifest)

# Apply error correction
corrected_manifest, correction_report = error_corrector.correct(manifest)

# Check if correction improved coherence
print(f"Original coherence: {manifest.coherence}")
print(f"Corrected coherence: {corrected_manifest.coherence}")
```

### Verifier

The `Verifier` class provides verification mechanisms for ZPDR representations.

#### Class: `Verifier`

**Description**: Implements verification of ZPDR trilateral vectors based on coherence and other metrics.

**Import**:
```python
from zpdr.core.verifier import Verifier
```

**Constructor**:
```python
Verifier(config: Optional[Dict[str, Any]] = None)
```

**Parameters**:
- `config`: Optional configuration dictionary with settings like verification level, etc.

**Key Methods**:

- `verify(manifest: ZPAManifest) -> Tuple[bool, Dict[str, Any]]`: Verify a ZPA manifest.
  - `manifest`: ZPAManifest to verify
  - Returns: Tuple of (is_valid, verification_report)

- `verify_trilateral(H: np.ndarray, E: np.ndarray, U: np.ndarray) -> Tuple[bool, float]`: Verify a trilateral vector directly.
  - `H`: Hyperbolic vector component
  - `E`: Elliptical vector component
  - `U`: Euclidean vector component
  - Returns: Tuple of (is_valid, coherence_value)

**Example Usage**:

```python
# Create a verifier
verifier = Verifier(config={'verification_level': 'strict'})

# Verify a manifest
is_valid, verification_report = verifier.verify(manifest)

# Directly verify a trilateral vector
H = np.array([0.1, 0.2, 0.3])
E = np.array([0.6, 0.7, 0.3]) / np.linalg.norm([0.6, 0.7, 0.3])
U = np.array([0.4, 0.5, 0.6])

is_valid, coherence = verifier.verify_trilateral(H, E, U)
```

## Utility Functions

### High-Precision Arithmetic

These functions provide high-precision arithmetic operations using Python's Decimal type.

**Import**:
```python
from zpdr.utils import to_decimal_array, PRECISION_TOLERANCE
```

**Key Functions**:

- `to_decimal_array(vector: Union[List[float], np.ndarray]) -> List[Decimal]`: Convert a vector to high-precision Decimal representation.
  - `vector`: Vector to convert
  - Returns: List of Decimal objects with high precision

### Coherence Calculations

These functions calculate coherence measures for ZPDR trilateral vectors.

**Import**:
```python
from zpdr.utils import (
    calculate_internal_coherence, 
    calculate_cross_coherence, 
    calculate_global_coherence,
    validate_trilateral_coherence,
    COHERENCE_THRESHOLD
)
```

**Key Functions**:

- `calculate_internal_coherence(vector: VectorData) -> Decimal`: Calculate internal coherence of a vector.
  - `vector`: Vector to calculate coherence for
  - Returns: Decimal representing internal coherence (1.0 is perfect coherence)

- `calculate_cross_coherence(vector1: VectorData, vector2: VectorData) -> Decimal`: Calculate cross-coherence between two vectors.
  - `vector1`: First vector
  - `vector2`: Second vector
  - Returns: Decimal representing cross-coherence (1.0 is perfect coherence)

- `calculate_global_coherence(internal_coherences: List[Decimal], cross_coherences: List[Decimal]) -> Decimal`: Calculate global coherence from individual measures.
  - `internal_coherences`: List of internal coherence values [H, E, U]
  - `cross_coherences`: List of cross coherence values [HE, EU, HU]
  - Returns: Global coherence as a high-precision Decimal

- `validate_trilateral_coherence(triple: ZPDRTriple) -> Tuple[bool, Decimal]`: Validate trilateral coherence of a ZPDR triple.
  - `triple`: Tuple of (H, E, U) vectors
  - Returns: Tuple of (is_valid, coherence_score)

### Normalization Operations

These functions handle normalization and denormalization of vectors with invariant preservation.

**Import**:
```python
from zpdr.utils import normalize_with_invariants, denormalize_with_invariants
```

**Key Functions**:

- `normalize_with_invariants(vector: VectorData, space_type: str = "hyperbolic") -> Tuple[np.ndarray, Dict[str, Union[float, Decimal]]]`: Normalize a vector and extract invariants.
  - `vector`: Vector to normalize
  - `space_type`: Type of space ('hyperbolic', 'elliptical', 'euclidean')
  - Returns: Tuple of (normalized_vector, invariants_dict)

- `denormalize_with_invariants(normalized_vector: VectorData, invariants: Dict[str, Union[float, Decimal]], space_type: str = "hyperbolic") -> np.ndarray`: Apply invariants to a normalized vector.
  - `normalized_vector`: Normalized vector
  - `invariants`: Dictionary of invariants extracted during normalization
  - `space_type`: Type of space ('hyperbolic', 'elliptical', 'euclidean')
  - Returns: Original vector before normalization

### Encoding/Decoding Helpers

These functions assist with encoding and decoding operations.

**Import**:
```python
from zpdr.utils import encode_to_zpdr_triple, reconstruct_from_zpdr_triple
```

**Key Functions**:

- `encode_to_zpdr_triple(value: int) -> ZPDRTriple`: Encode a value to a ZPDR triple (H, E, U).
  - `value`: Integer value to encode
  - Returns: ZPDRTriple containing (H, E, U) vectors

- `reconstruct_from_zpdr_triple(H: VectorData, E: VectorData, U: VectorData) -> int`: Reconstruct a value from its ZPDR triple.
  - `H`: Hyperbolic vector component
  - `E`: Elliptical vector component
  - `U`: Euclidean vector component
  - Returns: Original integer value

## Examples

### Basic Usage

The basic example demonstrates the core components of ZPDR:

```python
from zpdr.core.multivector import Multivector
from zpdr.core.geometric_spaces import (
    HyperbolicVector, 
    EllipticalVector, 
    EuclideanVector, 
    SpaceTransformer
)

# Create a multivector
mv = Multivector({
    "1": 1.0,       # scalar part
    "e1": 2.0,      # vector part
    "e2": 3.0,
    "e12": 4.0,     # bivector part
    "e123": 5.0     # trivector part
})

# Extract components by grade
scalar = mv.grade(0)
vector = mv.grade(1)
bivector = mv.grade(2)
trivector = mv.grade(3)

# Create vectors in each space
hyperbolic = HyperbolicVector([0.3, 0.4])
elliptical = EllipticalVector([3.0, 4.0])
euclidean = EuclideanVector([3.0, 4.0])

# Transform between spaces
euclidean_from_hyperbolic = SpaceTransformer.hyperbolic_to_euclidean(hyperbolic)
elliptical_from_hyperbolic = SpaceTransformer.hyperbolic_to_elliptical(hyperbolic)
```

### File Processing

The file processing example demonstrates encoding and decoding files:

```python
from zpdr.core.zpdr_processor import ZPDRProcessor, StreamingZPDRProcessor

# Create a processor
processor = ZPDRProcessor(config={
    'coherence_threshold': 0.95,
    'verification_level': 'standard',
    'error_correction_enabled': True
})

# Encode a small file
with open("small_file.txt", "rb") as f:
    file_data = f.read()

manifest = processor.encode(file_data, description="Small file example")

# Save the manifest
with open("small_file.zpdr", "w") as f:
    f.write(manifest.serialize())

# Decode the manifest
with open("small_file.zpdr", "r") as f:
    manifest_data = f.read()
    
recovered_manifest = ZPAManifest.deserialize(manifest_data)
recovered_data = processor.decode(recovered_manifest)

# Process a large file with streaming
streaming_processor = StreamingZPDRProcessor(config={'chunk_size': 1024 * 1024})

with open("large_file.bin", "rb") as f:
    manifests = streaming_processor.encode_stream(f)

# Save the manifests
with open("large_file.zpdr", "w") as f:
    f.write("[\n" + ",\n".join(m.serialize() for m in manifests) + "\n]")
    
# Decode the manifests
with open("large_file.zpdr", "r") as f:
    import json
    manifest_list = json.loads(f.read())
    
manifests = [ZPAManifest.from_dict(m) for m in manifest_list]
recovered_data = streaming_processor.decode_stream(manifests)
```

### Error Correction

The error correction example demonstrates ZPDR's error correction capabilities:

```python
from zpdr.core.error_corrector import ErrorCorrector
import numpy as np

# Create a verifier and error corrector
from zpdr.core.verifier import Verifier
verifier = Verifier()
error_corrector = ErrorCorrector()

# Create a valid ZPA
H = np.array([0.2, 0.3, 0.1])
E = np.array([0.6, 0.7, 0.4])
E = E / np.linalg.norm(E)  # Normalize for elliptical space
U = np.array([0.5, 0.2, 0.8])

# Verify it's valid
is_valid, coherence = verifier.verify_trilateral(H, E, U)
print(f"Original coherence: {coherence:.4f}, valid: {is_valid}")

# Add noise to simulate corruption
H_noisy = H + np.random.normal(0, 0.1, H.shape)
E_noisy = E + np.random.normal(0, 0.1, E.shape)
E_noisy = E_noisy / np.linalg.norm(E_noisy)  # Re-normalize
U_noisy = U + np.random.normal(0, 0.1, U.shape)

# Check if the corrupted vectors are still valid
is_valid, coherence = verifier.verify_trilateral(H_noisy, E_noisy, U_noisy)
print(f"Corrupted coherence: {coherence:.4f}, valid: {is_valid}")

# Create a manifest and apply error correction
from zpdr.core.zpa_manifest import ZPAManifest
corrupt_manifest = ZPAManifest(H_noisy, E_noisy, U_noisy)

corrected_manifest, report = error_corrector.correct(corrupt_manifest)

# Check if correction worked
is_valid, coherence = verifier.verify_trilateral(
    corrected_manifest.H, corrected_manifest.E, corrected_manifest.U
)
print(f"Corrected coherence: {coherence:.4f}, valid: {is_valid}")
```

## Mathematical Concepts

### Trilateral Vector System

The ZPDR framework represents data using three vectors across different geometric spaces:

1. **Hyperbolic Vector (H)**: Captures the base transformation system in negative-curvature space.
   - Represented in the Poincaré disk model (|v| < 1)
   - Provides exponential growth properties for hierarchical structures

2. **Elliptical Vector (E)**: Captures the transformation span in positive-curvature space.
   - Represented on the unit sphere (|v| = 1)
   - Provides bounded representation with global connectivity

3. **Euclidean Vector (U)**: Captures the transformed object in flat space.
   - Standard vector representation with familiar properties
   - Provides direct interpretation of the encoded data

### Coherence Measures

Coherence in ZPDR measures the mathematical consistency of the representation:

1. **Internal Coherence**: Measures how well a vector's components adhere to the expected mathematical structure within its geometric space.

2. **Cross-Coherence**: Measures the alignment and relationship between vectors from different geometric spaces, ensuring consistent transformation properties.

3. **Global Coherence**: Combines internal and cross-coherence measures into a single validation metric. A ZPA is considered valid if its global coherence exceeds the threshold (default 0.95).

### Zero-Point Normalization

Zero-point normalization in ZPDR refers to orienting vectors to a canonical reference state while preserving their essential geometric properties:

1. **Invariant Extraction**: Each vector type has specific invariants (e.g., rotation angles, scale factors) that are preserved during normalization.

2. **Normalization Process**: Transforms vectors to their canonical orientation in each space, removing redundant transformational degrees of freedom.

3. **Denormalization**: Applies the extracted invariants to restore the original vectors from their normalized form.

### Error Detection and Correction

ZPDR provides built-in error detection and correction through its coherence-based validation:

1. **Error Detection**: Uses coherence measures to detect when a ZPDR triple has been corrupted.

2. **Progressive Correction**: Implements multiple levels of error correction:
   - Basic: Ensures space-specific constraints (e.g., |E| = 1)
   - Intermediate: Uses cross-space transformations to improve coherence
   - Advanced: Applies invariant-based reconstruction and iterative refinement

3. **Correction Verification**: Verifies that the corrected representation meets coherence thresholds.