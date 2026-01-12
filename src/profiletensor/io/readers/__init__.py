"""
Spectrum file readers.

This module provides readers for various mass spectrometry file formats:

Tier 1 (Open formats):
- MzMLReader: mzML and mzXML files (pyteomics)

Tier 2 (Vendor-specific, direct reading):
- ThermoReader: Thermo .raw files (planned)
- BrukerReader: Bruker .d folders/timsTOF (planned)

Tier 3 (Via ProteoWizard/Apptainer):
- ProteoWizardReader: Universal reader for all vendor formats

Convenience functions:
- read_mzml(): Load mzML file into MSRun
- read_with_proteowizard(): Load vendor file via ProteoWizard
- convert_files_batch(): Batch convert vendor files with parallelization
- convert_and_read_batch(): Batch convert and read vendor files
"""

from .mzml import MzMLReader, read_mzml
from .proteowizard import (
    ProteoWizardReader,
    ConversionOptions,
    ConversionResult,
    read_with_proteowizard,
    convert_files_batch,
    convert_and_read_batch,
)

__all__ = [
    # Readers
    "MzMLReader",
    "ProteoWizardReader",
    # Convenience functions
    "read_mzml",
    "read_with_proteowizard",
    "convert_files_batch",
    "convert_and_read_batch",
    # Options/Results
    "ConversionOptions",
    "ConversionResult",
]
