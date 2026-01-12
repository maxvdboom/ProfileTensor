"""
Core spectrum representation for ProfileTensor.

This module defines the Spectrum class, the fundamental data structure
representing a single mass spectrum with its associated metadata.
Spectra consist of m/z-intensity pairs and comprehensive scan metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .scan_metadata import ScanMetadata, SpectrumType

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class Spectrum:
    """
    A single mass spectrum with associated metadata.
    
    This class represents the fundamental unit of MS data: a collection of
    m/z values and their corresponding intensities, along with comprehensive
    metadata about the acquisition.
    
    The m/z and intensity arrays are stored as NumPy arrays for efficient
    processing. For profile mode data, these represent the continuous signal;
    for centroid data, they represent discrete peaks.
    
    Attributes:
        mz: Array of m/z values (sorted in ascending order).
        intensity: Array of intensity values corresponding to mz.
        metadata: Comprehensive scan metadata.
        
    Example:
        >>> import numpy as np
        >>> from profiletensor.core.scan_metadata import ScanMetadata
        >>> 
        >>> metadata = ScanMetadata(scan_number=1, ms_level=1, retention_time=60.5)
        >>> spectrum = Spectrum(
        ...     mz=np.array([100.0, 150.0, 200.0]),
        ...     intensity=np.array([1000.0, 5000.0, 2500.0]),
        ...     metadata=metadata
        ... )
        >>> spectrum.n_points
        3
        >>> spectrum.mz_range
        (100.0, 200.0)
    """
    mz: NDArray[np.float64]
    intensity: NDArray[np.float64]
    metadata: ScanMetadata
    
    def __post_init__(self) -> None:
        """Validate spectrum data consistency."""
        if self.mz.ndim != 1:
            raise ValueError(f"mz must be 1-dimensional, got shape {self.mz.shape}")
        if self.intensity.ndim != 1:
            raise ValueError(f"intensity must be 1-dimensional, got shape {self.intensity.shape}")
        if len(self.mz) != len(self.intensity):
            raise ValueError(
                f"mz and intensity must have same length, "
                f"got {len(self.mz)} and {len(self.intensity)}"
            )
        # Ensure arrays are the correct dtype
        if self.mz.dtype != np.float64:
            object.__setattr__(self, 'mz', self.mz.astype(np.float64))
        if self.intensity.dtype != np.float64:
            object.__setattr__(self, 'intensity', self.intensity.astype(np.float64))
    
    @property
    def n_points(self) -> int:
        """Number of data points in the spectrum."""
        return len(self.mz)
    
    @property
    def is_empty(self) -> bool:
        """Check if spectrum has no data points."""
        return self.n_points == 0
    
    @property
    def mz_range(self) -> tuple[float, float]:
        """
        Return (min_mz, max_mz) tuple.
        
        Raises:
            ValueError: If spectrum is empty.
        """
        if self.is_empty:
            raise ValueError("Cannot get mz_range of empty spectrum")
        return float(self.mz[0]), float(self.mz[-1])
    
    @property
    def total_intensity(self) -> float:
        """Sum of all intensities (equivalent to TIC if complete)."""
        return float(np.sum(self.intensity))
    
    @property
    def base_peak_index(self) -> int:
        """Index of the most intense peak."""
        if self.is_empty:
            raise ValueError("Cannot get base_peak_index of empty spectrum")
        return int(np.argmax(self.intensity))
    
    @property
    def base_peak_mz(self) -> float:
        """m/z of the most intense peak."""
        return float(self.mz[self.base_peak_index])
    
    @property
    def base_peak_intensity(self) -> float:
        """Intensity of the most intense peak."""
        return float(self.intensity[self.base_peak_index])
    
    @property
    def is_profile(self) -> bool:
        """Check if this is profile mode data."""
        return self.metadata.spectrum_type == SpectrumType.PROFILE
    
    @property
    def is_centroid(self) -> bool:
        """Check if this is centroid mode data."""
        return self.metadata.spectrum_type == SpectrumType.CENTROID
    
    @property
    def ms_level(self) -> int:
        """MS level from metadata."""
        return self.metadata.ms_level
    
    @property
    def retention_time(self) -> float:
        """Retention time in seconds from metadata."""
        return self.metadata.retention_time
    
    @property
    def scan_number(self) -> int:
        """Scan number from metadata."""
        return self.metadata.scan_number
    
    def copy(self) -> 'Spectrum':
        """Create a deep copy of this spectrum."""
        return Spectrum(
            mz=self.mz.copy(),
            intensity=self.intensity.copy(),
            metadata=self.metadata  # ScanMetadata is frozen/immutable
        )
    
    def slice_mz(self, mz_min: float, mz_max: float) -> 'Spectrum':
        """
        Return a new spectrum containing only peaks within the m/z range.
        
        Args:
            mz_min: Minimum m/z value (inclusive).
            mz_max: Maximum m/z value (inclusive).
            
        Returns:
            New Spectrum with filtered data.
        """
        mask = (self.mz >= mz_min) & (self.mz <= mz_max)
        return Spectrum(
            mz=self.mz[mask].copy(),
            intensity=self.intensity[mask].copy(),
            metadata=self.metadata
        )
    
    def filter_by_intensity(
        self, 
        min_intensity: Optional[float] = None,
        max_intensity: Optional[float] = None,
        relative: bool = False
    ) -> 'Spectrum':
        """
        Filter peaks by intensity threshold.
        
        Args:
            min_intensity: Minimum intensity (inclusive).
            max_intensity: Maximum intensity (inclusive).
            relative: If True, thresholds are relative to base peak (0-1).
            
        Returns:
            New Spectrum with filtered data.
        """
        if relative and not self.is_empty:
            base = self.base_peak_intensity
            if min_intensity is not None:
                min_intensity = min_intensity * base
            if max_intensity is not None:
                max_intensity = max_intensity * base
        
        mask = np.ones(self.n_points, dtype=bool)
        if min_intensity is not None:
            mask &= self.intensity >= min_intensity
        if max_intensity is not None:
            mask &= self.intensity <= max_intensity
            
        return Spectrum(
            mz=self.mz[mask].copy(),
            intensity=self.intensity[mask].copy(),
            metadata=self.metadata
        )
    
    def normalize(self, method: str = 'max') -> 'Spectrum':
        """
        Return a normalized copy of the spectrum.
        
        Args:
            method: Normalization method.
                - 'max': Scale so maximum intensity is 1.0
                - 'sum': Scale so sum of intensities is 1.0
                - 'l2': L2 normalization (unit vector)
                
        Returns:
            New Spectrum with normalized intensities.
        """
        if self.is_empty:
            return self.copy()
            
        if method == 'max':
            factor = self.base_peak_intensity
        elif method == 'sum':
            factor = self.total_intensity
        elif method == 'l2':
            factor = float(np.linalg.norm(self.intensity))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if factor == 0:
            return self.copy()
            
        return Spectrum(
            mz=self.mz.copy(),
            intensity=self.intensity / factor,
            metadata=self.metadata
        )
    
    def to_torch(
        self, 
        dtype: Optional['torch.dtype'] = None,
        device: Optional[str] = None
    ) -> tuple['torch.Tensor', 'torch.Tensor']:
        """
        Convert spectrum to PyTorch tensors.
        
        Args:
            dtype: PyTorch dtype (default: torch.float32).
            device: Device to place tensors on.
            
        Returns:
            Tuple of (mz_tensor, intensity_tensor).
        """
        import torch
        
        if dtype is None:
            dtype = torch.float32
            
        mz_tensor = torch.from_numpy(self.mz).to(dtype=dtype)
        intensity_tensor = torch.from_numpy(self.intensity).to(dtype=dtype)
        
        if device is not None:
            mz_tensor = mz_tensor.to(device)
            intensity_tensor = intensity_tensor.to(device)
            
        return mz_tensor, intensity_tensor
    
    @classmethod
    def from_torch(
        cls,
        mz: 'torch.Tensor',
        intensity: 'torch.Tensor',
        metadata: ScanMetadata
    ) -> 'Spectrum':
        """
        Create a Spectrum from PyTorch tensors.
        
        Args:
            mz: m/z values as tensor.
            intensity: Intensity values as tensor.
            metadata: Scan metadata.
            
        Returns:
            New Spectrum instance.
        """
        return cls(
            mz=mz.detach().cpu().numpy().astype(np.float64),
            intensity=intensity.detach().cpu().numpy().astype(np.float64),
            metadata=metadata
        )
    
    def __len__(self) -> int:
        """Return number of data points."""
        return self.n_points
    
    def __repr__(self) -> str:
        """String representation."""
        if self.is_empty:
            mz_range_str = "empty"
        else:
            mz_min, mz_max = self.mz_range
            mz_range_str = f"m/z {mz_min:.2f}-{mz_max:.2f}"
        
        return (
            f"Spectrum(scan={self.scan_number}, "
            f"MS{self.ms_level}, "
            f"RT={self.retention_time:.2f}s, "
            f"{self.n_points} points, "
            f"{mz_range_str})"
        )
