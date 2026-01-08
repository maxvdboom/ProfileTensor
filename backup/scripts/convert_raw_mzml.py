import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict

def convert_raw_to_mzml_with_msconvert(raw_file_path: str, output_dir: str, container_path: str = "pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif") -> str:
    """
    Convert raw vendor file to mzML using msconvert in apptainer container.
    For Sciex .wiff files, ensures the corresponding .wiff.scan file is in the same directory.
    
    Args:
        raw_file_path: Path to the raw vendor file
        output_dir: Directory to save the converted mzML file
        container_path: Path to the apptainer container
        
    Returns:
        str: Path to converted mzML file, or empty string if failed
    """
    try:
        print(f"  🔄 Converting {os.path.basename(raw_file_path)} to mzML...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare the command
        # Mount the directory containing the raw file and output directory
        raw_dir = os.path.dirname(raw_file_path)
        raw_filename = os.path.basename(raw_file_path)
        
        # For Sciex .wiff files, verify that the .wiff.scan file is present
        if raw_filename.lower().endswith('.wiff'):
            base_name = raw_filename[:-5]  # Remove '.wiff'
            scan_filename = f"{base_name}.wiff.scan"
            scan_path = os.path.join(raw_dir, scan_filename)
            
            if not os.path.exists(scan_path):
                print(f"  ❌ Missing required companion file: {scan_filename}")
                return ""
            else:
                print(f"  ✅ Found companion file: {scan_filename}")
        
        cmd = [
            "apptainer", "exec",
            "--bind", f"{raw_dir}:/input",
            "--bind", f"{output_dir}:/output",
            container_path,
            "wine", "msconvert",
            f"/input/{raw_filename}",
            "-o", "/output"
        ]
        
        print(f"  🔧 Running: {' '.join(cmd)}")
        
        # Run the conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            # Find the generated mzML file
            base_name = os.path.splitext(raw_filename)[0]
            if raw_filename.lower().endswith('.d'):
                base_name = raw_filename[:-2]  # Remove .d extension
            
            mzml_path = os.path.join(output_dir, f"{base_name}.mzML")
            
            if os.path.exists(mzml_path):
                print(f"  ✅ Conversion successful: {mzml_path}")
                return mzml_path
            else:
                print(f"  ❌ Conversion completed but mzML file not found")
                return ""
        else:
            print(f"  ❌ Conversion failed: {result.stderr}")
            return ""
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Conversion timed out")
        return ""
    except Exception as e:
        print(f"  ❌ Error during conversion: {e}")
        return ""


def find_raw_files(input_path: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Find all raw files in a directory or return single file path.
    
    Args:
        input_path: Path to a file or directory
        extensions: List of file extensions to search for (e.g., ['.raw', '.wiff', '.d'])
        
    Returns:
        List of raw file paths
    """
    if extensions is None:
        # Common vendor file formats
        extensions = ['.raw', '.wiff', '.d', '.mzML', '.mzXML']
    
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        return [str(input_path_obj)]
    elif input_path_obj.is_dir():
        raw_files = []
        for ext in extensions:
            # For .d files (Agilent/Bruker directories)
            if ext == '.d':
                raw_files.extend([str(p) for p in input_path_obj.glob(f'**/*{ext}') if p.is_dir()])
            else:
                raw_files.extend([str(p) for p in input_path_obj.glob(f'**/*{ext}') if p.is_file()])
        return sorted(raw_files)
    else:
        raise ValueError(f"Path does not exist: {input_path}")


def convert_worker(args: Tuple[str, str, str]) -> Tuple[str, bool]:
    """
    Worker function for parallel conversion.
    
    Args:
        args: Tuple of (raw_file_path, output_dir, container_path)
        
    Returns:
        Tuple of (raw_file_path, success)
    """
    raw_file_path, output_dir, container_path = args
    result = convert_raw_to_mzml_with_msconvert(raw_file_path, output_dir, container_path)
    return (raw_file_path, bool(result))


def convert_batch(raw_files: List[str], output_dir: str, container_path: str, num_workers: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Convert multiple raw files in parallel.
    
    Args:
        raw_files: List of raw file paths to convert
        output_dir: Directory to save the converted mzML files
        container_path: Path to the apptainer container
        num_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        Dictionary with conversion results
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n{'='*60}")
    print(f"Starting batch conversion of {len(raw_files)} files")
    print(f"Using {num_workers} parallel workers")
    print(f"{'='*60}\n")
    
    # Prepare arguments for workers
    worker_args = [(raw_file, output_dir, container_path) for raw_file in raw_files]
    
    # Run conversions in parallel
    results = {"successful": [], "failed": []}
    
    if num_workers == 1:
        # Serial processing
        for args in worker_args:
            raw_file, success = convert_worker(args)
            if success:
                results["successful"].append(raw_file)
            else:
                results["failed"].append(raw_file)
    else:
        # Parallel processing
        with Pool(processes=num_workers) as pool:
            for raw_file, success in pool.imap_unordered(convert_worker, worker_args):
                if success:
                    results["successful"].append(raw_file)
                else:
                    results["failed"].append(raw_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  ✅ Successful: {len(results['successful'])}")
    print(f"  ❌ Failed: {len(results['failed'])}")
    print(f"{'='*60}")
    
    if results["failed"]:
        print("\nFailed conversions:")
        for file in results["failed"]:
            print(f"  - {file}")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert raw vendor files to mzML format using msconvert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python convert_raw_mzml.py input.raw -o output_dir

  # Convert all files in a directory
  python convert_raw_mzml.py data/raw_files/ -o output_dir

  # Use specific number of parallel workers
  python convert_raw_mzml.py data/ -o output_dir -j 4

  # Specify custom container path
  python convert_raw_mzml.py input.raw -o output_dir -c /path/to/container.sif
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to a raw file or directory containing raw files"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for converted mzML files"
    )
    
    parser.add_argument(
        "-c", "--container",
        type=str,
        default="pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif",
        help="Path to apptainer container (default: pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif)"
    )
    
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: number of CPU cores)"
    )
    
    parser.add_argument(
        "-e", "--extensions",
        type=str,
        nargs="+",
        default=[".raw", ".wiff", ".d"],
        help="File extensions to search for (default: .raw .wiff .d)"
    )
    
    args = parser.parse_args()
    
    # Find raw files
    try:
        raw_files = find_raw_files(args.input, args.extensions)
        
        if not raw_files:
            print(f"❌ No raw files found in {args.input}")
            return 1
        
        print(f"Found {len(raw_files)} raw file(s)")
        
        # Convert files
        if len(raw_files) == 1:
            # Single file conversion
            result = convert_raw_to_mzml_with_msconvert(
                raw_files[0],
                args.output,
                args.container
            )
            return 0 if result else 1
        else:
            # Batch conversion with parallelization
            results = convert_batch(
                raw_files,
                args.output,
                args.container,
                args.jobs
            )
            return 0 if not results["failed"] else 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())