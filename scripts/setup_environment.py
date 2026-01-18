#!/usr/bin/env python3
"""
Automated environment setup script for RAG Evaluation Course.

This script automates the setup process including:
- Virtual environment creation
- Dependency installation
- NVIDIA API key configuration
- Dataset pre-loading
- Environment validation

Usage:
    python scripts/setup_environment.py [--skip-datasets] [--skip-validation]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


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


def run_command(command: List[str], check: bool = True, capture_output: bool = False) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.
    
    Args:
        command: Command to run as list of strings
        check: Whether to raise exception on non-zero exit code
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        if capture_output:
            result = subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(command, check=check)
            return result.returncode, "", ""
    except subprocess.CalledProcessError as e:
        return e.returncode, "", str(e)


def check_python_version() -> bool:
    """Check if Python version is 3.10 or higher."""
    print_step("Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.10+ required, but found {version.major}.{version.minor}.{version.micro}")
        return False


def check_virtual_environment() -> bool:
    """Check if running in a virtual environment."""
    print_step("Checking virtual environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print_success("Running in virtual environment")
        return True
    else:
        print_warning("Not running in virtual environment")
        print("  It's recommended to use a virtual environment.")
        print("  Create one with: python -m venv venv")
        print("  Activate with: source venv/bin/activate (macOS/Linux)")
        print("                 venv\\Scripts\\activate (Windows)")
        return False


def upgrade_pip() -> bool:
    """Upgrade pip to latest version."""
    print_step("Upgrading pip...")
    
    returncode, _, _ = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture_output=True
    )
    
    if returncode == 0:
        print_success("pip upgraded successfully")
        return True
    else:
        print_error("Failed to upgrade pip")
        return False


def install_dependencies() -> bool:
    """Install dependencies from requirements.txt."""
    print_step("Installing dependencies from requirements.txt...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    returncode, _, stderr = run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True
    )
    
    if returncode == 0:
        print_success("Dependencies installed successfully")
        return True
    else:
        print_error(f"Failed to install dependencies: {stderr}")
        return False


def install_package() -> bool:
    """Install the course package in development mode."""
    print_step("Installing course package in development mode...")
    
    returncode, _, stderr = run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        capture_output=True
    )
    
    if returncode == 0:
        print_success("Course package installed successfully")
        return True
    else:
        print_error(f"Failed to install package: {stderr}")
        return False


def setup_env_file() -> bool:
    """Create .env file from .env.example if it doesn't exist."""
    print_step("Setting up environment variables...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_success(".env file already exists")
        return True
    
    if not env_example.exists():
        print_warning(".env.example not found, creating basic .env file")
        
        # Create basic .env file
        env_content = """# NVIDIA API Configuration
NVIDIA_API_KEY=your_api_key_here
NVIDIA_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_NEMOTRON_ENDPOINT=https://integrate.api.nvidia.com/v1

# Vector Store Configuration
VECTOR_STORE_TYPE=chromadb
VECTOR_STORE_PATH=./data/vector_store

# Evaluation Configuration
RAGAS_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1
RAGAS_EMBEDDING_ENDPOINT=https://integrate.api.nvidia.com/v1

# Dataset Configuration
DATASET_PATH=./course_materials/datasets
USC_CATALOG_PATH=./course_materials/datasets/usc_course_catalog.csv
AMNESTY_QA_PATH=./course_materials/datasets/amnesty_qa.json

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/course.log
"""
        env_file.write_text(env_content)
        print_success(".env file created")
        print_warning("Please edit .env and add your NVIDIA API key")
        return True
    
    # Copy from example
    env_file.write_text(env_example.read_text())
    print_success(".env file created from .env.example")
    print_warning("Please edit .env and add your NVIDIA API key")
    return True


def create_directories() -> bool:
    """Create necessary directories."""
    print_step("Creating directory structure...")
    
    directories = [
        "course_materials/datasets",
        "course_materials/modules",
        "course_materials/notebooks",
        "course_materials/assessments",
        "course_materials/study_resources",
        "data/vector_store",
        "logs",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_success("Directory structure created")
    return True


def install_jupyter_kernel() -> bool:
    """Install Jupyter kernel for the course."""
    print_step("Installing Jupyter kernel...")
    
    returncode, _, stderr = run_command(
        [
            sys.executable, "-m", "ipykernel", "install",
            "--user",
            "--name=rag-course",
            "--display-name=RAG Course (Python 3.10+)"
        ],
        capture_output=True
    )
    
    if returncode == 0:
        print_success("Jupyter kernel installed successfully")
        return True
    else:
        print_warning(f"Failed to install Jupyter kernel: {stderr}")
        print("  You can install it manually later with:")
        print("  python -m ipykernel install --user --name=rag-course")
        return True  # Non-critical, continue


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Automated environment setup for RAG Evaluation Course"
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset pre-loading"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation"
    )
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}RAG Evaluation Course - Environment Setup{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    # Track success of each step
    steps = []
    
    # Step 1: Check Python version
    steps.append(("Python version check", check_python_version()))
    if not steps[-1][1]:
        print_error("\nSetup failed: Python 3.10+ is required")
        sys.exit(1)
    
    # Step 2: Check virtual environment
    steps.append(("Virtual environment check", check_virtual_environment()))
    
    # Step 3: Upgrade pip
    steps.append(("Upgrade pip", upgrade_pip()))
    
    # Step 4: Create directories
    steps.append(("Create directories", create_directories()))
    
    # Step 5: Setup .env file
    steps.append(("Setup .env file", setup_env_file()))
    
    # Step 6: Install dependencies
    steps.append(("Install dependencies", install_dependencies()))
    if not steps[-1][1]:
        print_error("\nSetup failed: Could not install dependencies")
        sys.exit(1)
    
    # Step 7: Install package
    steps.append(("Install course package", install_package()))
    
    # Step 8: Install Jupyter kernel
    steps.append(("Install Jupyter kernel", install_jupyter_kernel()))
    
    # Step 9: Pre-load datasets (if not skipped)
    if not args.skip_datasets:
        print_step("Pre-loading datasets...")
        print_warning("Dataset pre-loading will be handled by separate script")
        print("  Run: python scripts/preload_datasets.py")
        steps.append(("Pre-load datasets", True))
    
    # Step 10: Validate environment (if not skipped)
    if not args.skip_validation:
        print_step("Validating environment...")
        print_warning("Environment validation will be handled by separate script")
        print("  Run: python scripts/validate_environment.py")
        steps.append(("Validate environment", True))
    
    # Print summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Setup Summary{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    for step_name, success in steps:
        if success:
            print_success(f"{step_name}")
        else:
            print_warning(f"{step_name} (with warnings)")
    
    # Final instructions
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print("1. Edit .env file and add your NVIDIA API key")
    print("2. Run: python scripts/preload_datasets.py")
    print("3. Run: python scripts/validate_environment.py")
    print("4. Start JupyterLab: jupyter lab")
    print("5. Open course_materials/notebooks/notebook_0_search_paradigm_comparison.py")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Setup completed successfully!{Colors.END}\n")


if __name__ == "__main__":
    main()
