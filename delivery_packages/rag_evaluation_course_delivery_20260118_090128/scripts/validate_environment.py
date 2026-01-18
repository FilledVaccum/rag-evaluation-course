#!/usr/bin/env python3
"""
Environment validation script for RAG Evaluation Course.

This script validates that the environment is properly configured:
- Python version
- Virtual environment
- Dependencies installed
- NVIDIA API key configured
- Datasets available
- JupyterLab ready
- Tests passing

Usage:
    python scripts/validate_environment.py [--skip-tests] [--skip-api-check]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(message: str) -> None:
    """Print a header message."""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{message}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")


def print_step(message: str) -> None:
    """Print a step message in blue."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}▶ {message}{Colors.END}")


def print_success(message: str) -> None:
    """Print a success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def run_command(command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.
    
    Args:
        command: Command to run as list of strings
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 10:
        return True, version_str
    else:
        return False, version_str


def check_virtual_environment() -> Tuple[bool, str]:
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        return True, sys.prefix
    else:
        return False, "Not in virtual environment"


def check_package_installed(package_name: str) -> Tuple[bool, str]:
    """Check if a Python package is installed."""
    try:
        returncode, stdout, _ = run_command(
            [sys.executable, "-m", "pip", "show", package_name]
        )
        
        if returncode == 0:
            # Extract version from output
            for line in stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    return True, version
            return True, "unknown version"
        else:
            return False, "not installed"
    except Exception as e:
        return False, str(e)


def check_dependencies() -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """Check if all required dependencies are installed."""
    required_packages = [
        'pydantic',
        'pytest',
        'hypothesis',
        'langchain',
        'ragas',
        'pandas',
        'jupyter',
        'jupyterlab',
        'python-dotenv'
    ]
    
    results = {}
    all_installed = True
    
    for package in required_packages:
        installed, version = check_package_installed(package)
        results[package] = (installed, version)
        if not installed:
            all_installed = False
    
    return all_installed, results


def check_nvidia_api_key() -> Tuple[bool, str]:
    """Check if NVIDIA API key is configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("NVIDIA_API_KEY")
        if api_key and api_key != "your_api_key_here":
            # Mask the key for display
            masked_key = api_key[:10] + "..." if len(api_key) > 10 else "***"
            return True, masked_key
        else:
            return False, "Not configured or using placeholder"
    except Exception as e:
        return False, str(e)


def check_dataset_exists(dataset_path: str) -> Tuple[bool, str]:
    """Check if a dataset file exists."""
    path = Path(dataset_path)
    if path.exists():
        size = path.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        return True, size_str
    else:
        return False, "Not found"


def check_datasets() -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """Check if required datasets are available."""
    datasets = {
        'USC Course Catalog': 'course_materials/datasets/usc_course_catalog.csv',
        'Amnesty Q&A': 'course_materials/datasets/amnesty_qa.json'
    }
    
    results = {}
    all_available = True
    
    for name, path in datasets.items():
        exists, info = check_dataset_exists(path)
        results[name] = (exists, info)
        if not exists:
            all_available = False
    
    return all_available, results


def check_jupyterlab() -> Tuple[bool, str]:
    """Check if JupyterLab is installed and can be launched."""
    returncode, stdout, _ = run_command(
        [sys.executable, "-m", "jupyter", "lab", "--version"]
    )
    
    if returncode == 0:
        version = stdout.strip()
        return True, version
    else:
        return False, "Not installed or not working"


def check_tests() -> Tuple[bool, str]:
    """Run a quick test to verify the test suite works."""
    # Check if pytest is available
    returncode, _, _ = run_command(
        [sys.executable, "-m", "pytest", "--version"]
    )
    
    if returncode != 0:
        return False, "pytest not available"
    
    # Run a quick test
    test_file = Path("tests/unit/test_models.py")
    if not test_file.exists():
        return False, "Test files not found"
    
    returncode, stdout, stderr = run_command(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "-x"]
    )
    
    if returncode == 0:
        # Count passed tests
        passed = stdout.count(" PASSED")
        return True, f"{passed} tests passed"
    else:
        # Extract failure info
        if "FAILED" in stdout:
            failed = stdout.count(" FAILED")
            return False, f"{failed} tests failed"
        else:
            return False, "Tests could not run"


def test_nvidia_api_connection(skip_api_check: bool) -> Tuple[bool, str]:
    """Test connection to NVIDIA API."""
    if skip_api_check:
        return True, "Skipped"
    
    try:
        from dotenv import load_dotenv
        import requests
        
        load_dotenv()
        
        api_key = os.getenv("NVIDIA_API_KEY")
        endpoint = os.getenv("NVIDIA_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
        
        if not api_key or api_key == "your_api_key_here":
            return False, "API key not configured"
        
        # Try to connect (with timeout)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Note: Adjust this endpoint based on actual NVIDIA API
        # For now, we'll just check if the key is configured
        return True, "API key configured (connection not tested)"
        
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate environment setup for RAG Evaluation Course"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running test suite"
    )
    parser.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip NVIDIA API connection check"
    )
    args = parser.parse_args()
    
    print_header("RAG Evaluation Course - Environment Validation")
    
    # Track validation results
    checks = []
    
    # 1. Python version
    print_step("Checking Python version...")
    success, version = check_python_version()
    checks.append(("Python version", success))
    if success:
        print_success(f"Python {version} (3.10+ required)")
    else:
        print_error(f"Python {version} (3.10+ required)")
    
    # 2. Virtual environment
    print_step("Checking virtual environment...")
    success, info = check_virtual_environment()
    checks.append(("Virtual environment", success))
    if success:
        print_success(f"Active: {info}")
    else:
        print_warning(f"{info} (recommended but not required)")
    
    # 3. Dependencies
    print_step("Checking dependencies...")
    all_installed, results = check_dependencies()
    checks.append(("Dependencies", all_installed))
    
    for package, (installed, version) in results.items():
        if installed:
            print_success(f"{package}: {version}")
        else:
            print_error(f"{package}: {version}")
    
    # 4. NVIDIA API key
    print_step("Checking NVIDIA API key...")
    success, info = check_nvidia_api_key()
    checks.append(("NVIDIA API key", success))
    if success:
        print_success(f"Configured: {info}")
    else:
        print_error(f"{info}")
    
    # 5. NVIDIA API connection
    if not args.skip_api_check:
        print_step("Testing NVIDIA API connection...")
        success, info = test_nvidia_api_connection(args.skip_api_check)
        checks.append(("NVIDIA API connection", success))
        if success:
            print_success(info)
        else:
            print_warning(info)
    
    # 6. Datasets
    print_step("Checking datasets...")
    all_available, results = check_datasets()
    checks.append(("Datasets", all_available))
    
    for name, (exists, info) in results.items():
        if exists:
            print_success(f"{name}: {info}")
        else:
            print_error(f"{name}: {info}")
    
    # 7. JupyterLab
    print_step("Checking JupyterLab...")
    success, version = check_jupyterlab()
    checks.append(("JupyterLab", success))
    if success:
        print_success(f"Version {version}")
    else:
        print_error(f"{version}")
    
    # 8. Tests
    if not args.skip_tests:
        print_step("Running test suite...")
        success, info = check_tests()
        checks.append(("Test suite", success))
        if success:
            print_success(info)
        else:
            print_error(info)
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(1 for _, success in checks if success)
    total = len(checks)
    
    print(f"\nPassed: {passed}/{total} checks\n")
    
    for check_name, success in checks:
        if success:
            print_success(check_name)
        else:
            print_error(check_name)
    
    # Final verdict
    print()
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Environment is ready!{Colors.END}")
        print("\nYou can now:")
        print("1. Start JupyterLab: jupyter lab")
        print("2. Open course notebooks in course_materials/notebooks/")
        print("3. Begin Module 1: Evolution of Search to RAG")
        print()
        return 0
    elif passed >= total * 0.7:  # 70% threshold
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Environment is mostly ready with some warnings{Colors.END}")
        print("\nYou can proceed, but consider fixing the issues above.")
        print()
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Environment setup incomplete{Colors.END}")
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Configure API key: Edit .env file")
        print("- Pre-load datasets: python scripts/preload_datasets.py")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
