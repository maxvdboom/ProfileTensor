"""
MSRun: A collection of spectra from a single LC-MS/MS acquisition.

This module defines the MSRun class that represents a complete mass
spectrometry run, containing all spectra and run-level metadata such
as instrument information, acquisition parameters, and source file details.
"""

from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Optional, overload

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum
from .scan_metadata import Polarity, SpectrumType


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """
    Run-level metadata for an LC-MS/MS acquisition.
    
    Attributes:
        source_file: Path to the original source file.
        instrument_model: Instrument model name (e.g., "Q Exactive HF").
        instrument_vendor: Instrument vendor (e.g., "Thermo Scientific").
        instrument_serial: Instrument serial number.
        acquisition_date: Date and time of acquisition.
        software_name: Acquisition software name.
        software_version: Acquisition software version.
        run_duration: Total run duration in seconds.
        polarity: Acquisition polarity (if single polarity run).
        extras: Additional vendor-specific metadata.
    """
    source_file: Optional[Path] = None
    instrument_model: Optional[str] = None
    instrument_vendor: Optional[str] = None
    instrument_serial: Optional[str] = None
    acquisition_date: Optional[datetime] = None
    software_name: Optional[str] = None
    software_version: Optional[str] = None
    run_duration: Optional[float] = None  # seconds
    polarity: Optional[Polarity] = None
    extras: dict = field(default_factory=dict)
    
    @property
    def source_filename(self) -> Optional[str]:
        """Return just the filename from source_file."""
        return self.source_file.name if self.source_file else None


class MSRun(Sequence[Spectrum]):
    """
    A complete LC-MS/MS run containing spectra and run metadata.
    
    MSRun is the primary container for mass spectrometry data, holding
    all spectra from a single acquisition along with run-level metadata.
    It provides efficient access patterns for spectra by scan number,
    MS level, and retention time.
    
    The class implements the Sequence protocol, allowing indexing and
    iteration over spectra in acquisition order.
    
    Attributes:
        metadata: Run-level metadata.
        
    Example:
        >>> from profiletensor.core import MSRun, Spectrum, ScanMetadata
        >>> import numpy as np
        >>> 
        >>> # Create spectra
        >>> spectra = [
        ...     Spectrum(np.array([100.0]), np.array([1000.0]), 
        ...              ScanMetadata(scan_number=1, ms_level=1, retention_time=0.0)),
        ...     Spectrum(np.array([200.0]), np.array([500.0]),
        ...              ScanMetadata(scan_number=2, ms_level=2, retention_time=0.1)),
        ... ]
        >>> 
        >>> # Create run
        >>> run = MSRun(spectra)
        >>> print(f"Run has {len(run)} spectra")
        Run has 2 spectra
        >>> 
        >>> # Access by scan number
        >>> ms1 = run.get_by_scan(1)
        >>> 
        >>> # Iterate MS1 spectra only
        >>> for spec in run.iter_ms_level(1):
        ...     print(spec.scan_number)
    """
    
    def __init__(
        self,
        spectra: Optional[list[Spectrum]] = None,
        metadata: Optional[RunMetadata] = None,
    ):
        """
        Initialize an MSRun.
        
        Args:
            spectra: List of Spectrum objects (will be sorted by scan number).
            metadata: Run-level metadata.
        """
        self._spectra: list[Spectrum] = []
        self._scan_index: dict[int, int] = {}  # scan_number -> list index
        self.metadata = metadata or RunMetadata()
        
        if spectra:
            self._add_spectra(spectra)
    
    def _add_spectra(self, spectra: list[Spectrum]) -> None:
        """Add spectra and rebuild indices."""
        # Sort by scan number
        self._spectra = sorted(spectra, key=lambda s: s.scan_number)
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """Rebuild the scan number index."""
        self._scan_index = {
            spec.scan_number: idx 
            for idx, spec in enumerate(self._spectra)
        }
    
    def add_spectrum(self, spectrum: Spectrum) -> None:
        """
        Add a single spectrum to the run.
        
        The spectrum is inserted in the correct position to maintain
        scan number ordering.
        
        Args:
            spectrum: Spectrum to add.
            
        Raises:
            ValueError: If scan number already exists.
        """
        if spectrum.scan_number in self._scan_index:
            raise ValueError(
                f"Scan number {spectrum.scan_number} already exists in run"
            )
        
        # Find insertion point to maintain sorted order
        insert_idx = 0
        for i, spec in enumerate(self._spectra):
            if spec.scan_number > spectrum.scan_number:
                insert_idx = i
                break
            insert_idx = i + 1
        
        self._spectra.insert(insert_idx, spectrum)
        self._rebuild_index()
    
    # -------------------------------------------------------------------------
    # Sequence protocol implementation
    # -------------------------------------------------------------------------
    
    @overload
    def __getitem__(self, index: int) -> Spectrum: ...
    
    @overload
    def __getitem__(self, index: slice) -> list[Spectrum]: ...
    
    def __getitem__(self, index: int | slice) -> Spectrum | list[Spectrum]:
        """Get spectrum by index (acquisition order)."""
        return self._spectra[index]
    
    def __len__(self) -> int:
        """Total number of spectra in the run."""
        return len(self._spectra)
    
    def __iter__(self) -> Iterator[Spectrum]:
        """Iterate over all spectra in acquisition order."""
        return iter(self._spectra)
    
    def __contains__(self, item: object) -> bool:
        """Check if spectrum or scan number is in run."""
        if isinstance(item, int):
            return item in self._scan_index
        if isinstance(item, Spectrum):
            return item.scan_number in self._scan_index
        return False
    
    # -------------------------------------------------------------------------
    # Access methods
    # -------------------------------------------------------------------------
    
    def get_by_scan(self, scan_number: int) -> Spectrum:
        """
        Get spectrum by scan number.
        
        Args:
            scan_number: The scan number to retrieve.
            
        Returns:
            The spectrum with the given scan number.
            
        Raises:
            KeyError: If scan number not found.
        """
        if scan_number not in self._scan_index:
            raise KeyError(f"Scan number {scan_number} not found in run")
        return self._spectra[self._scan_index[scan_number]]
    
    def get_by_rt(
        self, 
        retention_time: float, 
        tolerance: float = 0.0
    ) -> list[Spectrum]:
        """
        Get spectra near a retention time.
        
        Args:
            retention_time: Target retention time in seconds.
            tolerance: Tolerance window in seconds (±).
            
        Returns:
            List of spectra within the RT window.
        """
        return [
            spec for spec in self._spectra
            if abs(spec.retention_time - retention_time) <= tolerance
        ]
    
    def get_rt_range(
        self, 
        rt_start: float, 
        rt_end: float
    ) -> list[Spectrum]:
        """
        Get spectra within a retention time range.
        
        Args:
            rt_start: Start retention time in seconds (inclusive).
            rt_end: End retention time in seconds (inclusive).
            
        Returns:
            List of spectra within the RT range.
        """
        return [
            spec for spec in self._spectra
            if rt_start <= spec.retention_time <= rt_end
        ]
    
    # -------------------------------------------------------------------------
    # Iteration by MS level
    # -------------------------------------------------------------------------
    
    def iter_ms_level(self, ms_level: int) -> Iterator[Spectrum]:
        """
        Iterate over spectra of a specific MS level.
        
        Args:
            ms_level: MS level to filter by (1, 2, etc.).
            
        Yields:
            Spectra with the specified MS level.
        """
        for spectrum in self._spectra:
            if spectrum.ms_level == ms_level:
                yield spectrum
    
    def iter_ms1(self) -> Iterator[Spectrum]:
        """Iterate over MS1 spectra."""
        return self.iter_ms_level(1)
    
    def iter_ms2(self) -> Iterator[Spectrum]:
        """Iterate over MS2 spectra."""
        return self.iter_ms_level(2)
    
    def get_ms_level(self, ms_level: int) -> list[Spectrum]:
        """
        Get all spectra of a specific MS level.
        
        Args:
            ms_level: MS level to filter by.
            
        Returns:
            List of spectra with the specified MS level.
        """
        return list(self.iter_ms_level(ms_level))
    
    # -------------------------------------------------------------------------
    # Properties and statistics
    # -------------------------------------------------------------------------
    
    @property
    def scan_numbers(self) -> list[int]:
        """List of all scan numbers in order."""
        return [spec.scan_number for spec in self._spectra]
    
    @property
    def retention_times(self) -> NDArray[np.float64]:
        """Array of all retention times in seconds."""
        return np.array([spec.retention_time for spec in self._spectra])
    
    @property
    def rt_range(self) -> tuple[float, float]:
        """
        Return (min_rt, max_rt) tuple in seconds.
        
        Raises:
            ValueError: If run is empty.
        """
        if not self._spectra:
            raise ValueError("Cannot get rt_range of empty run")
        rts = self.retention_times
        return float(rts.min()), float(rts.max())
    
    @property
    def duration(self) -> float:
        """
        Run duration in seconds (max RT - min RT).
        
        Returns 0.0 for empty runs or single-spectrum runs.
        """
        if len(self._spectra) < 2:
            return 0.0
        rt_min, rt_max = self.rt_range
        return rt_max - rt_min
    
    def get_ms_level_counts(self) -> dict[int, int]:
        """
        Count spectra per MS level.
        
        Returns:
            Dictionary mapping MS level to count.
        """
        counts: dict[int, int] = {}
        for spec in self._spectra:
            level = spec.ms_level
            counts[level] = counts.get(level, 0) + 1
        return counts
    
    @property
    def n_ms1(self) -> int:
        """Number of MS1 spectra."""
        return self.get_ms_level_counts().get(1, 0)
    
    @property
    def n_ms2(self) -> int:
        """Number of MS2 spectra."""
        return self.get_ms_level_counts().get(2, 0)
    
    def get_tic(self, ms_level: Optional[int] = None) -> NDArray[np.float64]:
        """
        Get total ion current array.
        
        Args:
            ms_level: Filter by MS level (None for all spectra).
            
        Returns:
            Array of TIC values in acquisition order.
        """
        if ms_level is None:
            spectra = self._spectra
        else:
            spectra = self.get_ms_level(ms_level)
        
        return np.array([
            spec.metadata.total_ion_current or spec.total_intensity
            for spec in spectra
        ])
    
    def get_bpc(self, ms_level: Optional[int] = None) -> NDArray[np.float64]:
        """
        Get base peak chromatogram intensities.
        
        Args:
            ms_level: Filter by MS level (None for all spectra).
            
        Returns:
            Array of base peak intensities in acquisition order.
        """
        if ms_level is None:
            spectra = self._spectra
        else:
            spectra = self.get_ms_level(ms_level)
        
        return np.array([
            spec.base_peak_intensity if not spec.is_empty else 0.0
            for spec in spectra
        ])
    
    def summary(self) -> dict:
        """
        Generate a summary of the run.
        
        Returns:
            Dictionary with run statistics.
        """
        ms_counts = self.get_ms_level_counts()
        
        summary = {
            'n_spectra': len(self),
            'ms_level_counts': ms_counts,
            'scan_range': (self.scan_numbers[0], self.scan_numbers[-1]) if self._spectra else None,
            'rt_range_seconds': self.rt_range if self._spectra else None,
            'duration_seconds': self.duration,
        }
        
        if self.metadata.source_file:
            summary['source_file'] = str(self.metadata.source_file)
        if self.metadata.instrument_model:
            summary['instrument'] = self.metadata.instrument_model
        if self.metadata.acquisition_date:
            summary['acquisition_date'] = self.metadata.acquisition_date.isoformat()
            
        return summary
    
    # -------------------------------------------------------------------------
    # Filtering and subsetting
    # -------------------------------------------------------------------------
    
    def filter(
        self,
        ms_level: Optional[int] = None,
        rt_range: Optional[tuple[float, float]] = None,
        polarity: Optional[Polarity] = None,
        spectrum_type: Optional[SpectrumType] = None,
    ) -> 'MSRun':
        """
        Create a new MSRun with filtered spectra.
        
        Args:
            ms_level: Keep only spectra with this MS level.
            rt_range: Keep spectra within (rt_min, rt_max) in seconds.
            polarity: Keep only spectra with this polarity.
            spectrum_type: Keep only spectra with this type (profile/centroid).
            
        Returns:
            New MSRun with filtered spectra (metadata is preserved).
        """
        filtered = self._spectra
        
        if ms_level is not None:
            filtered = [s for s in filtered if s.ms_level == ms_level]
        
        if rt_range is not None:
            rt_min, rt_max = rt_range
            filtered = [s for s in filtered if rt_min <= s.retention_time <= rt_max]
        
        if polarity is not None:
            filtered = [s for s in filtered if s.metadata.polarity == polarity]
        
        if spectrum_type is not None:
            filtered = [s for s in filtered if s.metadata.spectrum_type == spectrum_type]
        
        return MSRun(spectra=filtered, metadata=self.metadata)
    
    def subset(self, scan_numbers: list[int]) -> 'MSRun':
        """
        Create a new MSRun with only the specified scans.
        
        Args:
            scan_numbers: List of scan numbers to include.
            
        Returns:
            New MSRun with selected spectra.
        """
        spectra = [
            self.get_by_scan(scan) 
            for scan in scan_numbers 
            if scan in self._scan_index
        ]
        return MSRun(spectra=spectra, metadata=self.metadata)
    
    # -------------------------------------------------------------------------
    # String representations
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        """String representation."""
        ms_counts = self.get_ms_level_counts()
        ms_str = ", ".join(f"MS{k}:{v}" for k, v in sorted(ms_counts.items()))
        
        if self._spectra:
            rt_min, rt_max = self.rt_range
            rt_str = f"RT {rt_min:.1f}-{rt_max:.1f}s"
        else:
            rt_str = "empty"
        
        source = ""
        if self.metadata.source_filename:
            source = f", source={self.metadata.source_filename}"
        
        return f"MSRun({len(self)} spectra, {ms_str}, {rt_str}{source})"
