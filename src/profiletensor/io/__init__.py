"""
I/O module for reading and writing mass spectrometry data.

This module provides:

Readers:
- MzMLReader: Read mzML/mzXML files
- ProteoWizardReader: Universal reader for vendor formats via Apptainer

Convenience functions:
- read_mzml(): Load mzML to MSRun
- read_with_proteowizard(): Load vendor file via ProteoWizard
- convert_files_batch(): Batch convert vendor files (parallel)
- convert_and_read_batch(): Batch convert and read vendor files

Base classes:
- SpectrumReader: Abstract base class for all readers

Registry:
- ReaderRegistry: Auto-detection and reader selection
- detect_vendor(): Detect file vendor from path
- Vendor: Enum of supported vendors

Conversion options:
- ConversionOptions: Options for msconvert
- ConversionResult: Result of a conversion operation
"""

from .base import SpectrumReader
from .registry import ReaderRegistry, detect_vendor, Vendor
from .readers import (
    MzMLReader,
    read_mzml,
    ProteoWizardReader,
    ConversionOptions,
    ConversionResult,
    read_with_proteowizard,
    convert_files_batch,
    convert_and_read_batch,
)

__all__ = [
    # Base
    "SpectrumReader",
    # Readers
    "MzMLReader",
    "ProteoWizardReader",
    # Convenience functions
    "read_mzml",
    "read_with_proteowizard",
    "convert_files_batch",
    "convert_and_read_batch",
    # Registry
    "ReaderRegistry",
    "detect_vendor",
    "Vendor",
    # Options/Results
    "ConversionOptions",
    "ConversionResult",
]
