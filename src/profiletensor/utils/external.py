"""
System and external tool utilities for ProfileTensor.

This module provides utilities for:
- Detecting system resources (CPU, memory)
- Finding and validating external tools (Apptainer, ProteoWizard)
- Parallel execution helpers
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import multiprocessing


class ParallelMode(Enum):
    """Parallelization intensity modes."""
    NONE = auto()      # Single-threaded
    LIGHT = auto()     # 25% of available resources
    HEAVY = auto()     # 75% of available resources
    MAX = auto()       # 100% of available resources (all cores)
    CUSTOM = auto()    # User-specified number of workers


@dataclass(frozen=True)
class SystemResources:
    """System resource information."""
    cpu_count: int
    cpu_count_physical: int
    memory_total_gb: float
    memory_available_gb: float
    
    def get_workers(self, mode: ParallelMode, custom_workers: Optional[int] = None) -> int:
        """
        Calculate number of workers based on parallel mode.
        
        Args:
            mode: Parallelization mode.
            custom_workers: Number of workers for CUSTOM mode.
            
        Returns:
            Number of worker processes to use.
        """
        if mode == ParallelMode.NONE:
            return 1
        elif mode == ParallelMode.LIGHT:
            # 25% of physical cores, minimum 1
            return max(1, self.cpu_count_physical // 4)
        elif mode == ParallelMode.HEAVY:
            # 75% of physical cores, minimum 1
            return max(1, int(self.cpu_count_physical * 0.75))
        elif mode == ParallelMode.MAX:
            return self.cpu_count_physical
        elif mode == ParallelMode.CUSTOM:
            if custom_workers is None:
                raise ValueError("custom_workers must be specified for CUSTOM mode")
            return max(1, min(custom_workers, self.cpu_count_physical))
        else:
            return 1


def get_system_resources() -> SystemResources:
    """
    Detect available system resources.
    
    Returns:
        SystemResources with CPU and memory information.
    """
    # CPU count
    cpu_count = os.cpu_count() or 1
    
    # Try to get physical core count (excluding hyperthreading)
    try:
        # On Linux, parse /proc/cpuinfo
        if Path('/proc/cpuinfo').exists():
            with open('/proc/cpuinfo') as f:
                content = f.read()
            # Count unique physical CPU + core combinations
            physical_ids = set()
            core_ids = set()
            current_physical = None
            for line in content.split('\n'):
                if line.startswith('physical id'):
                    current_physical = line.split(':')[1].strip()
                elif line.startswith('core id') and current_physical is not None:
                    core_id = line.split(':')[1].strip()
                    physical_ids.add((current_physical, core_id))
            cpu_count_physical = len(physical_ids) if physical_ids else cpu_count
        else:
            cpu_count_physical = cpu_count
    except Exception:
        cpu_count_physical = cpu_count
    
    # Memory info
    memory_total_gb = 0.0
    memory_available_gb = 0.0
    try:
        if Path('/proc/meminfo').exists():
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        memory_total_gb = int(line.split()[1]) / (1024 * 1024)
                    elif line.startswith('MemAvailable:'):
                        memory_available_gb = int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    
    return SystemResources(
        cpu_count=cpu_count,
        cpu_count_physical=cpu_count_physical,
        memory_total_gb=memory_total_gb,
        memory_available_gb=memory_available_gb,
    )


def find_apptainer() -> Optional[Path]:
    """
    Find the Apptainer/Singularity executable.
    
    Returns:
        Path to apptainer/singularity executable, or None if not found.
    """
    # Try apptainer first (newer name), then singularity
    for cmd in ['apptainer', 'singularity']:
        path = shutil.which(cmd)
        if path:
            return Path(path)
    return None


def check_apptainer_available() -> tuple[bool, str]:
    """
    Check if Apptainer is available and working.
    
    Returns:
        Tuple of (is_available, message).
    """
    apptainer = find_apptainer()
    if apptainer is None:
        return False, "Apptainer/Singularity not found. Install with: apt install apptainer"
    
    try:
        result = subprocess.run(
            [str(apptainer), '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Found: {version}"
        else:
            return False, f"Apptainer error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Apptainer command timed out"
    except Exception as e:
        return False, f"Error checking Apptainer: {e}"


def find_sif_file(
    search_paths: Optional[list[Path]] = None,
    filename_pattern: str = "pwiz*.sif"
) -> Optional[Path]:
    """
    Find a ProteoWizard SIF file.
    
    Args:
        search_paths: Paths to search (default: current dir, home dir).
        filename_pattern: Glob pattern for SIF filename.
        
    Returns:
        Path to SIF file, or None if not found.
    """
    import fnmatch
    
    if search_paths is None:
        search_paths = [
            Path.cwd(),
            Path.home(),
            Path.home() / '.local' / 'share' / 'profiletensor',
            Path('/opt/proteowizard'),
        ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        # Direct file check
        for item in search_path.iterdir():
            if item.is_file() and fnmatch.fnmatch(item.name, filename_pattern):
                return item
        
        # Also check for exact common names
        common_names = [
            'pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif',
            'proteowizard.sif',
            'pwiz.sif',
        ]
        for name in common_names:
            path = search_path / name
            if path.exists():
                return path
    
    return None


def validate_sif_file(sif_path: Path) -> tuple[bool, str]:
    """
    Validate that a SIF file is a valid Apptainer container.
    
    Args:
        sif_path: Path to SIF file.
        
    Returns:
        Tuple of (is_valid, message).
    """
    if not sif_path.exists():
        return False, f"SIF file not found: {sif_path}"
    
    if not sif_path.is_file():
        return False, f"Not a file: {sif_path}"
    
    # Check file size (SIF files should be at least a few MB)
    size_mb = sif_path.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        return False, f"SIF file too small ({size_mb:.1f} MB), may be corrupted"
    
    # Try to inspect the container
    apptainer = find_apptainer()
    if apptainer:
        try:
            result = subprocess.run(
                [str(apptainer), 'inspect', str(sif_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return True, f"Valid SIF container ({size_mb:.0f} MB)"
            else:
                return False, f"SIF inspection failed: {result.stderr[:100]}"
        except subprocess.TimeoutExpired:
            return False, "SIF inspection timed out"
        except Exception as e:
            # If we can't inspect, at least check magic bytes
            pass
    
    # Fallback: check SIF magic bytes
    try:
        with open(sif_path, 'rb') as f:
            magic = f.read(10)
        # SIF files start with "SIF" or older Singularity format
        if magic[:3] == b'SIF' or b'singularity' in magic.lower():
            return True, f"SIF file detected ({size_mb:.0f} MB)"
    except Exception as e:
        return False, f"Error reading SIF file: {e}"
    
    return True, f"SIF file exists ({size_mb:.0f} MB), could not fully validate"
