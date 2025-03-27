# Core components of the ZPDR framework
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