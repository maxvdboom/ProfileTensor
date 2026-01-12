"""
I/O module for reading and writing mass spectrometry data.

This module provides:

Readers:
- MzMLReader: Read mzML/mzXML files
- read_mzml(): Convenience function to load mzML to MSRun

Base classes:
- SpectrumReader: Abstract base class for all readers

Registry:
- ReaderRegistry: Auto-detection and reader selection
- detect_vendor(): Detect file vendor from path
- Vendor: Enum of supported vendors
"""

from .base import SpectrumReader
from .registry import ReaderRegistry, detect_vendor, Vendor
from .readers import MzMLReader, read_mzml

__all__ = [
    # Base
    "SpectrumReader",
    # Readers
    "MzMLReader",
    "read_mzml",
    # Registry
    "ReaderRegistry",
    "detect_vendor",
    "Vendor",
]
