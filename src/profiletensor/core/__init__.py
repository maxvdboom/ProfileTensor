"""
Core data structures for ProfileTensor.

This module provides the fundamental data types for representing
mass spectrometry data:

- Spectrum: A single mass spectrum with m/z-intensity data
- ScanMetadata: Comprehensive metadata for a scan
- PrecursorInfo: Precursor ion information for MSn spectra

Enums for categorical metadata:
- Polarity: Ion polarity (positive/negative)
- SpectrumType: Data mode (profile/centroid)
- ActivationType: Fragmentation method
"""

from .scan_metadata import (
    ActivationType,
    Polarity,
    PrecursorInfo,
    ScanMetadata,
    SpectrumType,
)
from .spectrum import Spectrum

__all__ = [
    # Main classes
    "Spectrum",
    "ScanMetadata",
    "PrecursorInfo",
    # Enums
    "Polarity",
    "SpectrumType", 
    "ActivationType",
]
