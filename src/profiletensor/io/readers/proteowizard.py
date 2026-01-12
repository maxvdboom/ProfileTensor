"""
ProteoWizard-based universal file reader using Apptainer.

This module provides a universal reader for vendor-specific mass spectrometry
file formats by using ProteoWizard's msconvert tool running in an Apptainer
container. This enables reading of:

- Thermo .raw files
- Bruker .d folders
- Sciex .wiff files
- Shimadzu .lcd files
- Agilent .d folders
- Waters .raw folders
- And other formats supported by ProteoWizard

The reader converts files to mzML format using msconvert, then reads the
mzML using the MzMLReader.
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Callable
import logging

from ..base import SpectrumReader
from .mzml import MzMLReader, read_mzml
from ...core import Spectrum, MSRun, RunMetadata
from ...utils.external import (
    ParallelMode,
    get_system_resources,
    find_apptainer,
    find_sif_file,
    validate_sif_file,
    check_apptainer_available,
)


logger = logging.getLogger(__name__)


# Vendor extensions supported by ProteoWizard
PROTEOWIZARD_EXTENSIONS: dict[str, str] = {
    '.raw': 'Thermo/Waters',
    '.d': 'Agilent/Bruker',
    '.wiff': 'Sciex',
    '.wiff.scan': 'Sciex',
    '.lcd': 'Shimadzu',
    '.yep': 'Agilent',
    '.baf': 'Bruker',
    '.tdf': 'Bruker timsTOF',
    '.mzml': 'mzML (passthrough)',
    '.mzxml': 'mzXML (passthrough)',
}


@dataclass
class ConversionOptions:
    """Options for msconvert file conversion."""
    
    # Output format options
    output_format: str = 'mzML'  # mzML or mzXML
    precision: int = 64  # 32 or 64 bit
    compression: str = 'zlib'  # zlib, none
    
    # MS level filtering
    ms_levels: Optional[list[int]] = None  # None = all levels
    
    # Peak picking (centroiding)
    peak_picking: bool = False  # If True, apply peak picking
    peak_picking_ms_levels: Optional[list[int]] = None  # Levels to centroid
    
    # Scan range
    scan_range: Optional[tuple[int, int]] = None  # (start, end) scan numbers
    
    # Retention time range (in seconds)
    rt_range: Optional[tuple[float, float]] = None
    
    # m/z range
    mz_range: Optional[tuple[float, float]] = None
    
    # Additional msconvert arguments
    extra_args: list[str] = field(default_factory=list)
    
    def to_msconvert_args(self) -> list[str]:
        """Convert options to msconvert command-line arguments."""
        args = []
        
        # Output format (msconvert uses camelCase: --mzML, --mzXML)
        if self.output_format.lower() == 'mzml':
            args.append('--mzML')
        elif self.output_format.lower() == 'mzxml':
            args.append('--mzXML')
        
        # Precision
        if self.precision == 32:
            args.extend(['--32'])
        else:
            args.extend(['--64'])
        
        # Compression
        if self.compression == 'zlib':
            args.extend(['--zlib'])
        elif self.compression == 'none':
            args.extend(['--nocompression'])
        
        # MS level filter
        if self.ms_levels:
            levels_str = ' '.join(str(l) for l in self.ms_levels)
            args.extend(['--filter', f'msLevel {levels_str}'])
        
        # Peak picking
        if self.peak_picking:
            if self.peak_picking_ms_levels:
                levels = '-'.join(str(l) for l in self.peak_picking_ms_levels)
                args.extend(['--filter', f'peakPicking vendor msLevel={levels}'])
            else:
                args.extend(['--filter', 'peakPicking vendor'])
        
        # Scan range
        if self.scan_range:
            args.extend(['--filter', f'scanNumber [{self.scan_range[0]},{self.scan_range[1]}]'])
        
        # RT range (convert to minutes for msconvert)
        if self.rt_range:
            rt_start_min = self.rt_range[0] / 60.0
            rt_end_min = self.rt_range[1] / 60.0
            args.extend(['--filter', f'scanTime [{rt_start_min},{rt_end_min}]'])
        
        # m/z range
        if self.mz_range:
            args.extend(['--filter', f'mzWindow [{self.mz_range[0]},{self.mz_range[1]}]'])
        
        # Extra arguments
        args.extend(self.extra_args)
        
        return args


@dataclass
class ConversionResult:
    """Result of a file conversion."""
    input_path: Path
    output_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None
    conversion_time_seconds: float = 0.0


def _convert_single_file(
    input_path: Path,
    output_dir: Path,
    sif_path: Path,
    apptainer_path: Path,
    options: ConversionOptions,
) -> ConversionResult:
    """
    Convert a single file using msconvert in Apptainer.
    
    This function is designed to be called in a separate process.
    """
    import time
    start_time = time.time()
    
    try:
        # Determine output filename
        stem = input_path.stem
        if input_path.name.endswith('.wiff.scan'):
            stem = input_path.name[:-10]  # Remove .wiff.scan
        elif input_path.suffix.lower() == '.d':
            stem = input_path.stem
        
        output_ext = '.mzML' if options.output_format.lower() == 'mzml' else '.mzXML'
        output_path = output_dir / f"{stem}{output_ext}"
        
        # Build msconvert command
        msconvert_args = options.to_msconvert_args()
        
        # Apptainer bind paths
        input_parent = input_path.parent.resolve()
        output_dir_resolved = output_dir.resolve()
        
        # Build the full command
        # We need to bind both input and output directories
        cmd = [
            str(apptainer_path),
            'exec',
            '--bind', f'{input_parent}:/input:ro',
            '--bind', f'{output_dir_resolved}:/output',
            str(sif_path),
            'wine', 'msconvert',
            f'/input/{input_path.name}',
            '-o', '/output',
            '--outfile', output_path.name,
        ]
        cmd.extend(msconvert_args)
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            return ConversionResult(
                input_path=input_path,
                output_path=None,
                success=False,
                error_message=f"msconvert failed: {result.stderr[:500]}",
                conversion_time_seconds=elapsed,
            )
        
        if not output_path.exists():
            return ConversionResult(
                input_path=input_path,
                output_path=None,
                success=False,
                error_message="Output file not created",
                conversion_time_seconds=elapsed,
            )
        
        return ConversionResult(
            input_path=input_path,
            output_path=output_path,
            success=True,
            conversion_time_seconds=elapsed,
        )
        
    except subprocess.TimeoutExpired:
        return ConversionResult(
            input_path=input_path,
            output_path=None,
            success=False,
            error_message="Conversion timed out after 1 hour",
            conversion_time_seconds=time.time() - start_time,
        )
    except Exception as e:
        return ConversionResult(
            input_path=input_path,
            output_path=None,
            success=False,
            error_message=str(e),
            conversion_time_seconds=time.time() - start_time,
        )


class ProteoWizardReader(SpectrumReader):
    """
    Universal reader for vendor MS files using ProteoWizard/Apptainer.
    
    This reader uses ProteoWizard's msconvert tool running in an Apptainer
    container to convert vendor-specific files to mzML, then reads the mzML.
    
    Supported formats:
    - Thermo .raw files
    - Bruker .d folders
    - Sciex .wiff files
    - Shimadzu .lcd files
    - Agilent .d folders
    - Waters .raw folders
    
    Example:
        >>> with ProteoWizardReader("sample.raw", sif_path="pwiz.sif") as reader:
        ...     for spectrum in reader:
        ...         print(spectrum.scan_number, spectrum.ms_level)
        
        >>> # Or convert and read in one step
        >>> run = read_with_proteowizard("sample.raw")
    """
    
    vendor: ClassVar[str] = "ProteoWizard"
    supported_extensions: ClassVar[list[str]] = list(PROTEOWIZARD_EXTENSIONS.keys())
    
    def __init__(
        self,
        path: Path | str,
        sif_path: Optional[Path | str] = None,
        options: Optional[ConversionOptions] = None,
        keep_converted: bool = False,
        output_dir: Optional[Path | str] = None,
    ):
        """
        Initialize the ProteoWizard reader.
        
        Args:
            path: Path to vendor file.
            sif_path: Path to ProteoWizard SIF container. If None, searches
                     common locations.
            options: Conversion options for msconvert.
            keep_converted: If True, keep the converted mzML file.
            output_dir: Directory for converted files. If None, uses temp dir.
        """
        # Don't call super().__init__ yet - need to handle extensions differently
        self.path = Path(path)
        
        # Validate path exists
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        # Find SIF file
        if sif_path is None:
            found_sif = find_sif_file()
            if found_sif is None:
                raise FileNotFoundError(
                    "ProteoWizard SIF file not found. Provide sif_path or place "
                    "the SIF file in the current directory."
                )
            self._sif_path: Path = found_sif
        else:
            self._sif_path = Path(sif_path)
        
        # Validate SIF
        valid, msg = validate_sif_file(self._sif_path)
        if not valid:
            raise ValueError(f"Invalid SIF file: {msg}")
        
        # Find Apptainer
        found_apptainer = find_apptainer()
        if found_apptainer is None:
            raise RuntimeError(
                "Apptainer/Singularity not found. Install with: apt install apptainer"
            )
        self._apptainer: Path = found_apptainer
        
        self._options = options or ConversionOptions()
        self._keep_converted = keep_converted
        self._output_dir = Path(output_dir) if output_dir else None
        
        # Internal state
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._converted_path: Optional[Path] = None
        self._mzml_reader: Optional[MzMLReader] = None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if ProteoWizard/Apptainer is available."""
        available, _ = check_apptainer_available()
        if not available:
            return False
        
        sif = find_sif_file()
        return sif is not None
    
    @classmethod
    def get_installation_instructions(cls) -> str:
        """Return installation instructions."""
        return (
            "ProteoWizard Reader Requirements:\n"
            "\n"
            "1. Install Apptainer:\n"
            "   sudo apt install apptainer\n"
            "\n"
            "2. Get ProteoWizard SIF container:\n"
            "   apptainer pull docker://proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses\n"
            "\n"
            "3. Place the .sif file in your project directory or specify its path.\n"
        )
    
    def _validate_path(self) -> None:
        """Override to handle all ProteoWizard-supported extensions."""
        # Check extension
        suffix = self.path.suffix.lower()
        if self.path.name.lower().endswith('.wiff.scan'):
            suffix = '.wiff.scan'
        
        if suffix not in PROTEOWIZARD_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension {suffix}. "
                f"Supported: {list(PROTEOWIZARD_EXTENSIONS.keys())}"
            )
    
    def __enter__(self) -> 'ProteoWizardReader':
        """Convert file and open for reading."""
        self._validate_path()
        
        # Check if already mzML/mzXML - no conversion needed
        suffix = self.path.suffix.lower()
        if suffix in ['.mzml', '.mzxml']:
            self._mzml_reader = MzMLReader(self.path)
            self._mzml_reader.__enter__()
            return self
        
        # Create output directory
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            working_dir = self._output_dir
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix='profiletensor_')
            working_dir = Path(self._temp_dir.name)
        
        # Convert file
        logger.info(f"Converting {self.path.name} using ProteoWizard...")
        result = _convert_single_file(
            input_path=self.path,
            output_dir=working_dir,
            sif_path=self._sif_path,
            apptainer_path=self._apptainer,
            options=self._options,
        )
        
        if not result.success:
            raise RuntimeError(f"Conversion failed: {result.error_message}")
        
        if result.output_path is None:
            raise RuntimeError("Conversion succeeded but output path is None")
        
        self._converted_path = result.output_path
        logger.info(f"Conversion complete in {result.conversion_time_seconds:.1f}s")
        
        # Open the converted file
        self._mzml_reader = MzMLReader(self._converted_path)
        self._mzml_reader.__enter__()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources."""
        # Close mzML reader
        if self._mzml_reader is not None:
            self._mzml_reader.__exit__(exc_type, exc_val, exc_tb)
            self._mzml_reader = None
        
        # Clean up temp directory (if not keeping converted files)
        if self._temp_dir is not None and not self._keep_converted:
            self._temp_dir.cleanup()
            self._temp_dir = None
    
    def __iter__(self) -> Iterator[Spectrum]:
        """Iterate over all spectra."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return iter(self._mzml_reader)
    
    def __len__(self) -> int:
        """Total number of spectra."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return len(self._mzml_reader)
    
    def get_spectrum(self, scan_number: int) -> Spectrum:
        """Get spectrum by scan number."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return self._mzml_reader.get_spectrum(scan_number)
    
    def get_spectrum_by_index(self, index: int) -> Spectrum:
        """Get spectrum by index."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return self._mzml_reader.get_spectrum_by_index(index)
    
    def iter_ms_level(self, ms_level: int) -> Iterator[Spectrum]:
        """Iterate over spectra of a specific MS level."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return self._mzml_reader.iter_ms_level(ms_level)
    
    def get_ms_level_counts(self) -> dict[int, int]:
        """Return count of spectra per MS level."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return self._mzml_reader.get_ms_level_counts()
    
    @property
    def run_metadata(self) -> dict:
        """File-level metadata."""
        if self._mzml_reader is None:
            return {'source_file': str(self.path)}
        meta = self._mzml_reader.run_metadata.copy()
        meta['original_source_file'] = str(self.path)
        return meta
    
    def to_run(self) -> MSRun:
        """Load the entire file into an MSRun object."""
        if self._mzml_reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        
        run = self._mzml_reader.to_run()
        
        # Update source file to original path
        run.metadata = RunMetadata(
            source_file=self.path,
            instrument_model=run.metadata.instrument_model,
            instrument_vendor=run.metadata.instrument_vendor,
            instrument_serial=run.metadata.instrument_serial,
            acquisition_date=run.metadata.acquisition_date,
            software_name=run.metadata.software_name,
            software_version=run.metadata.software_version,
            run_duration=run.metadata.run_duration,
            polarity=run.metadata.polarity,
            extras=run.metadata.extras,
        )
        
        return run
    
    @property
    def converted_path(self) -> Optional[Path]:
        """Path to the converted mzML file (if available)."""
        return self._converted_path


def convert_files_batch(
    input_paths: list[Path],
    output_dir: Path,
    sif_path: Optional[Path] = None,
    options: Optional[ConversionOptions] = None,
    parallel_mode: ParallelMode = ParallelMode.LIGHT,
    custom_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, ConversionResult], None]] = None,
) -> list[ConversionResult]:
    """
    Convert multiple files in parallel using ProteoWizard.
    
    Args:
        input_paths: List of paths to convert.
        output_dir: Output directory for converted files.
        sif_path: Path to ProteoWizard SIF file.
        options: Conversion options.
        parallel_mode: Parallelization intensity (LIGHT=25%, HEAVY=75%).
        custom_workers: Number of workers for CUSTOM mode.
        progress_callback: Callback function(completed, total, result) for progress.
        
    Returns:
        List of ConversionResult objects.
    """
    if not input_paths:
        return []
    
    # Find SIF and Apptainer
    if sif_path is None:
        sif_path = find_sif_file()
        if sif_path is None:
            raise FileNotFoundError("ProteoWizard SIF file not found")
    
    apptainer = find_apptainer()
    if apptainer is None:
        raise RuntimeError("Apptainer not found")
    
    options = options or ConversionOptions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    resources = get_system_resources()
    n_workers = resources.get_workers(parallel_mode, custom_workers)
    
    logger.info(
        f"Converting {len(input_paths)} files with {n_workers} workers "
        f"(mode: {parallel_mode.name})"
    )
    
    results: list[ConversionResult] = []
    
    if n_workers == 1:
        # Sequential processing
        for i, path in enumerate(input_paths):
            result = _convert_single_file(
                input_path=path,
                output_dir=output_dir,
                sif_path=sif_path,
                apptainer_path=apptainer,
                options=options,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(input_paths), result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(
                    _convert_single_file,
                    input_path=path,
                    output_dir=output_dir,
                    sif_path=sif_path,
                    apptainer_path=apptainer,
                    options=options,
                ): path
                for path in input_paths
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(input_paths), result)
    
    # Sort results by input path order
    path_to_result = {r.input_path: r for r in results}
    results = [path_to_result[p] for p in input_paths]
    
    # Log summary
    success_count = sum(1 for r in results if r.success)
    logger.info(f"Conversion complete: {success_count}/{len(results)} successful")
    
    return results


def read_with_proteowizard(
    path: Path | str,
    sif_path: Optional[Path | str] = None,
    options: Optional[ConversionOptions] = None,
    keep_converted: bool = False,
    output_dir: Optional[Path | str] = None,
) -> MSRun:
    """
    Convenience function to read a vendor file using ProteoWizard.
    
    Args:
        path: Path to vendor file.
        sif_path: Path to ProteoWizard SIF file.
        options: Conversion options.
        keep_converted: If True, keep the converted mzML file.
        output_dir: Directory for converted file (if keeping).
        
    Returns:
        MSRun containing all spectra.
        
    Example:
        >>> run = read_with_proteowizard("sample.raw")
        >>> print(f"Loaded {len(run)} spectra")
    """
    with ProteoWizardReader(
        path,
        sif_path=sif_path,
        options=options,
        keep_converted=keep_converted,
        output_dir=output_dir,
    ) as reader:
        return reader.to_run()


def convert_and_read_batch(
    input_paths: list[Path],
    output_dir: Path,
    sif_path: Optional[Path] = None,
    options: Optional[ConversionOptions] = None,
    parallel_mode: ParallelMode = ParallelMode.LIGHT,
    custom_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, ConversionResult], None]] = None,
) -> list[tuple[Path, Optional[MSRun]]]:
    """
    Convert and read multiple vendor files in parallel.
    
    Args:
        input_paths: List of paths to convert.
        output_dir: Output directory for converted files.
        sif_path: Path to ProteoWizard SIF file.
        options: Conversion options.
        parallel_mode: Parallelization intensity.
        custom_workers: Number of workers for CUSTOM mode.
        progress_callback: Callback for conversion progress.
        
    Returns:
        List of (input_path, MSRun or None) tuples.
    """
    # First, convert all files
    results = convert_files_batch(
        input_paths=input_paths,
        output_dir=output_dir,
        sif_path=sif_path,
        options=options,
        parallel_mode=parallel_mode,
        custom_workers=custom_workers,
        progress_callback=progress_callback,
    )
    
    # Then read each converted file
    runs: list[tuple[Path, Optional[MSRun]]] = []
    for result in results:
        if result.success and result.output_path:
            try:
                run = read_mzml(result.output_path)
                # Update source to original path
                run.metadata = RunMetadata(
                    source_file=result.input_path,
                    instrument_model=run.metadata.instrument_model,
                    instrument_vendor=run.metadata.instrument_vendor,
                    instrument_serial=run.metadata.instrument_serial,
                    acquisition_date=run.metadata.acquisition_date,
                    software_name=run.metadata.software_name,
                    software_version=run.metadata.software_version,
                    run_duration=run.metadata.run_duration,
                    polarity=run.metadata.polarity,
                    extras=run.metadata.extras,
                )
                runs.append((result.input_path, run))
            except Exception as e:
                logger.error(f"Failed to read {result.output_path}: {e}")
                runs.append((result.input_path, None))
        else:
            runs.append((result.input_path, None))
    
    return runs
