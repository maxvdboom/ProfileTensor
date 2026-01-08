from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from ..core.spectrum import Spectrum, ScanMetadata

class SpectrumReader(ABC):
    """
    Abstract base class for spectrum file readers.
    
    All vendor-specific readers must implement this interface.
    """
    
    # Class-level attributes
    vendor: ClassVar[str]  # e.g., "Thermo", "Agilent"
    supported_extensions: ClassVar[list[str]]  # e.g., [".raw"]
    
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate file/folder exists and has correct extension."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        suffix = self.path.suffix.lower()
        # Handle compound extensions like .wiff.scan
        if self.path.name.endswith('.wiff.scan'):
            suffix = '.wiff.scan'
        
        if suffix not in self.supported_extensions:
            raise ValueError(
                f"Unsupported extension {suffix} for {self.vendor} reader. "
                f"Expected: {self.supported_extensions}"
            )
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this reader's dependencies are available.
        
        Returns False if required libraries/tools are not installed.
        """
        ...
    
    @classmethod
    def get_installation_instructions(cls) -> str:
        """Return instructions for installing this reader's dependencies."""
        return "See documentation for installation instructions."
    
    @abstractmethod
    def __enter__(self) -> 'SpectrumReader':
        ...
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...
    
    @abstractmethod
    def __iter__(self) -> Iterator[Spectrum]:
        """Iterate over all spectra in the file."""
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        """Total number of spectra."""
        ...
    
    @abstractmethod
    def get_spectrum(self, scan_number: int) -> Spectrum:
        """Random access to a specific scan."""
        ...
    
    def iter_ms_level(self, ms_level: int) -> Iterator[Spectrum]:
        """Iterate over spectra of a specific MS level."""
        for spectrum in self:
            if spectrum.metadata.ms_level == ms_level:
                yield spectrum
    
    @abstractmethod
    def get_ms_level_counts(self) -> dict[int, int]:
        """Return count of spectra per MS level."""
        ...
    
    @property
    @abstractmethod
    def run_metadata(self) -> dict:
        """
        File-level metadata.
        
        Expected keys (when available):
        - instrument_model: str
        - instrument_serial: str
        - acquisition_date: datetime
        - software_version: str
        - source_file: str
        """
        ...
    
    @property
    def total_spectra(self) -> int:
        """Alias for __len__."""
        return len(self)