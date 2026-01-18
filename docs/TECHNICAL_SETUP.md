# Technical Setup Guide

## Evaluating RAG and Semantic Search Systems Course

This guide provides comprehensive instructions for setting up your technical environment for the "Evaluating RAG and Semantic Search Systems" course. Follow these steps carefully to ensure a smooth learning experience.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [JupyterLab Environment Setup](#jupyterlab-environment-setup)
3. [NVIDIA API Key Configuration](#nvidia-api-key-configuration)
4. [Dependency Installation](#dependency-installation)
5. [Dataset Pre-loading](#dataset-pre-loading)
6. [Environment Validation](#environment-validation)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Additional Resources](#additional-resources)

---

## System Requirements

### Minimum Requirements

- **Operating System**: macOS 10.15+, Ubuntu 20.04+, Windows 10+
- **Python Version**: 3.10 or higher (3.11 recommended)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk Space**: 5 GB free space
- **Internet Connection**: Required for NVIDIA API access and package installation

### Recommended Setup

- **Python**: 3.11 or 3.12
- **RAM**: 16 GB or more
- **CPU**: Multi-core processor (4+ cores)
- **GPU**: Optional, but beneficial for local model inference

### Software Prerequisites

- **Python**: Download from [python.org](https://www.python.org/downloads/)
- **pip**: Included with Python 3.10+
- **Git**: Download from [git-scm.com](https://git-scm.com/)
- **Text Editor**: VS Code, PyCharm, or your preferred IDE

---

## JupyterLab Environment Setup

### Step 1: Create Virtual Environment

Creating a virtual environment isolates your course dependencies from other Python projects.

#### On macOS/Linux:

```bash
# Navigate to course directory
cd rag-evaluation-course

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
which python
```

#### On Windows:

```bash
# Navigate to course directory
cd rag-evaluation-course

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) in your prompt)
where python
```

### Step 2: Upgrade pip and setuptools

```bash
# Upgrade pip to latest version
pip install --upgrade pip setuptools wheel
```

### Step 3: Install JupyterLab

```bash
# Install JupyterLab and extensions
pip install jupyterlab>=4.0.0
pip install ipykernel>=6.25.0
pip install nbformat>=5.9.0

# Register kernel with Jupyter
python -m ipykernel install --user --name=rag-course --display-name="RAG Course (Python 3.10+)"
```

### Step 4: Configure JupyterLab

Create a JupyterLab configuration file:

```bash
# Generate default config
jupyter lab --generate-config

# The config file will be created at:
# ~/.jupyter/jupyter_lab_config.py (macOS/Linux)
# %USERPROFILE%\.jupyter\jupyter_lab_config.py (Windows)
```

Optional: Add these settings to your config file for better experience:

```python
# Enable autosave
c.FileContentsManager.autosave_interval = 60  # seconds

# Set default notebook directory
c.ServerApp.root_dir = '/path/to/rag-evaluation-course'

# Disable token authentication for local use (optional)
c.ServerApp.token = ''
c.ServerApp.password = ''
```

### Step 5: Launch JupyterLab

```bash
# Start JupyterLab
jupyter lab

# JupyterLab will open in your default browser at:
# http://localhost:8888
```

---

## NVIDIA API Key Configuration

### Step 1: Obtain NVIDIA API Key

1. **Create NVIDIA Account**:
   - Visit [NVIDIA Developer Portal](https://developer.nvidia.com/)
   - Click "Join" or "Sign In"
   - Complete registration process

2. **Access NVIDIA AI Catalog**:
   - Navigate to [NVIDIA AI Catalog](https://catalog.ngc.nvidia.com/)
   - Sign in with your NVIDIA account

3. **Generate API Key**:
   - Go to "API Keys" section in your account settings
   - Click "Generate API Key"
   - Copy the generated key (you won't be able to see it again!)
   - Store it securely

### Step 2: Configure Environment Variables

#### Method 1: Using .env File (Recommended)

Create a `.env` file in the project root:

```bash
# Copy example file
cp .env.example .env

# Edit .env file with your preferred editor
nano .env  # or vim, code, etc.
```

Add your NVIDIA API key to `.env`:

```env
# NVIDIA API Configuration
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
```

#### Method 2: System Environment Variables

##### On macOS/Linux:

Add to your `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export NVIDIA_NIM_ENDPOINT="https://integrate.api.nvidia.com/v1"
```

Then reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

##### On Windows:

Using Command Prompt:

```cmd
setx NVIDIA_API_KEY "nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
setx NVIDIA_NIM_ENDPOINT "https://integrate.api.nvidia.com/v1"
```

Using PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", "nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "User")
[Environment]::SetEnvironmentVariable("NVIDIA_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1", "User")
```

### Step 3: Verify API Key Configuration

Test your API key setup:

```python
# Run in Python or Jupyter notebook
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API key
api_key = os.getenv("NVIDIA_API_KEY")
if api_key:
    print(f"✓ API Key configured: {api_key[:10]}...")
else:
    print("✗ API Key not found. Please check your .env file.")
```

### Step 4: Test NVIDIA API Connection

```python
# Test API connectivity
import requests
import os

api_key = os.getenv("NVIDIA_API_KEY")
endpoint = os.getenv("NVIDIA_NIM_ENDPOINT")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test endpoint (adjust based on actual NVIDIA API)
response = requests.get(f"{endpoint}/models", headers=headers)

if response.status_code == 200:
    print("✓ Successfully connected to NVIDIA API")
    print(f"Available models: {len(response.json())}")
else:
    print(f"✗ Connection failed: {response.status_code}")
    print(f"Error: {response.text}")
```

---

## Dependency Installation

### Step 1: Install Core Dependencies

```bash
# Ensure virtual environment is activated
# You should see (venv) in your prompt

# Install all required packages
pip install -r requirements.txt

# This will install:
# - Pydantic for data validation
# - Pytest and Hypothesis for testing
# - LangChain for NVIDIA integration
# - Ragas for RAG evaluation
# - Vector stores (ChromaDB, Milvus, Pinecone)
# - Data processing libraries (pandas, numpy)
# - Jupyter and visualization tools
```

### Step 2: Install Package in Development Mode

```bash
# Install the course package
pip install -e .

# This allows you to:
# - Import course modules from anywhere
# - Make changes without reinstalling
# - Use command-line tools
```

### Step 3: Install Optional Dependencies

#### For Full Feature Set:

```bash
pip install -e ".[full]"

# Includes:
# - Additional vector stores (Milvus, Pinecone)
# - OpenAI integration
# - Sentence transformers
# - Advanced visualization tools
```

#### For Development:

```bash
pip install -e ".[dev]"

# Includes:
# - Code formatters (black, isort)
# - Linters (flake8)
# - Type checkers (mypy)
```

#### For Everything:

```bash
pip install -e ".[full,dev]"
```

### Step 4: Verify Installation

```bash
# Check installed packages
pip list | grep -E "(pydantic|pytest|ragas|langchain|jupyter)"

# Expected output should show:
# pydantic                 2.x.x
# pytest                   7.x.x
# ragas                    0.x.x
# langchain                0.x.x
# jupyterlab               4.x.x
```

### Step 5: Run Installation Tests

```bash
# Run quick validation tests
pytest tests/unit/test_models.py -v

# All tests should pass
# If you see failures, check the troubleshooting section
```

---

## Dataset Pre-loading

### Step 1: Create Dataset Directory

```bash
# Create datasets directory if it doesn't exist
mkdir -p course_materials/datasets

# Verify directory structure
ls -la course_materials/datasets/
```

### Step 2: Download Course Datasets

The course uses two primary datasets:

#### USC Course Catalog

```bash
# Download USC Course Catalog (example - adjust URL as needed)
# This dataset will be provided by the course
# For now, create a placeholder

# Option 1: Download from course repository
# wget https://course-repo.com/datasets/usc_course_catalog.csv -O course_materials/datasets/usc_course_catalog.csv

# Option 2: Use provided dataset
# cp /path/to/provided/usc_course_catalog.csv course_materials/datasets/
```

#### Amnesty Q&A Dataset

```bash
# Download Amnesty Q&A dataset
# This dataset will be provided by the course

# Option 1: Download from course repository
# wget https://course-repo.com/datasets/amnesty_qa.json -O course_materials/datasets/amnesty_qa.json

# Option 2: Use provided dataset
# cp /path/to/provided/amnesty_qa.json course_materials/datasets/
```

### Step 3: Verify Dataset Format

Run the dataset validation script:

```python
# In Python or Jupyter notebook
from src.utils.dataset_manager import DatasetManager

# Initialize dataset manager
manager = DatasetManager()

# Load and validate USC Course Catalog
try:
    usc_data = manager.load_dataset("usc_course_catalog")
    print(f"✓ USC Course Catalog loaded: {len(usc_data)} records")
except Exception as e:
    print(f"✗ Error loading USC dataset: {e}")

# Load and validate Amnesty Q&A
try:
    amnesty_data = manager.load_dataset("amnesty_qa")
    print(f"✓ Amnesty Q&A loaded: {len(amnesty_data)} records")
except Exception as e:
    print(f"✗ Error loading Amnesty dataset: {e}")
```

### Step 4: Pre-process Datasets (Optional)

```bash
# Run preprocessing script
python scripts/preprocess_datasets.py

# This will:
# - Validate dataset schemas
# - Create embeddings (if configured)
# - Generate dataset statistics
# - Create train/test splits
```

---

## Environment Validation

### Automated Validation Script

Run the comprehensive validation script:

```bash
# Run environment validation
python scripts/validate_environment.py

# Expected output:
# ✓ Python version: 3.10+
# ✓ Virtual environment: Active
# ✓ Dependencies: All installed
# ✓ NVIDIA API Key: Configured
# ✓ Datasets: Available
# ✓ JupyterLab: Ready
# ✓ Tests: Passing
```

### Manual Validation Checklist

#### 1. Python Version

```bash
python --version
# Should show: Python 3.10.x or higher
```

#### 2. Virtual Environment

```bash
which python  # macOS/Linux
where python  # Windows
# Should point to venv/bin/python or venv\Scripts\python
```

#### 3. Package Installation

```bash
python -c "import pydantic, pytest, ragas, langchain; print('✓ All core packages imported')"
```

#### 4. NVIDIA API Configuration

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓ API Key:', 'Configured' if os.getenv('NVIDIA_API_KEY') else 'Missing')"
```

#### 5. JupyterLab

```bash
jupyter lab --version
# Should show: 4.x.x or higher
```

#### 6. Dataset Availability

```bash
ls -lh course_materials/datasets/
# Should show usc_course_catalog.csv and amnesty_qa.json
```

#### 7. Test Suite

```bash
pytest tests/ -v --tb=short
# All tests should pass
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Python Version Mismatch

**Problem**: `python --version` shows Python 2.x or < 3.10

**Solution**:
```bash
# Install Python 3.10+ from python.org
# Then use python3 explicitly:
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

#### Issue 2: pip Installation Fails

**Problem**: `pip install` fails with permission errors

**Solution**:
```bash
# Don't use sudo! Instead, ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Then retry installation
pip install -r requirements.txt
```

#### Issue 3: NVIDIA API Key Not Found

**Problem**: API calls fail with authentication errors

**Solution**:
```bash
# Check if .env file exists
ls -la .env

# Verify .env content
cat .env | grep NVIDIA_API_KEY

# Ensure python-dotenv is installed
pip install python-dotenv

# Load .env in your code
from dotenv import load_dotenv
load_dotenv()
```

#### Issue 4: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Option 1: Install package in development mode
pip install -e .

# Option 2: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # macOS/Linux
set PYTHONPATH=%PYTHONPATH%;%CD%\src          # Windows
```

#### Issue 5: JupyterLab Kernel Not Found

**Problem**: Kernel "RAG Course" not available in JupyterLab

**Solution**:
```bash
# Reinstall kernel
python -m ipykernel install --user --name=rag-course --display-name="RAG Course (Python 3.10+)"

# Restart JupyterLab
jupyter lab
```

#### Issue 6: Dataset Loading Fails

**Problem**: `FileNotFoundError` when loading datasets

**Solution**:
```bash
# Check dataset paths in .env
cat .env | grep DATASET

# Verify files exist
ls -la course_materials/datasets/

# Update paths in .env if needed
# Use absolute paths if relative paths don't work
```

#### Issue 7: Memory Errors

**Problem**: Out of memory errors during notebook execution

**Solution**:
```python
# Reduce batch sizes in notebooks
# Clear outputs regularly
# Restart kernel between exercises

# In Jupyter:
# Kernel -> Restart Kernel and Clear All Outputs
```

#### Issue 8: Slow API Responses

**Problem**: NVIDIA API calls timeout or are very slow

**Solution**:
```python
# Increase timeout in API calls
import requests

response = requests.get(url, timeout=60)  # 60 seconds

# Check your internet connection
# Verify API endpoint is correct
# Check NVIDIA service status
```

#### Issue 9: Vector Store Connection Issues

**Problem**: Cannot connect to ChromaDB/Milvus/Pinecone

**Solution**:
```bash
# For ChromaDB (local):
pip install --upgrade chromadb

# For Milvus (requires Docker):
docker pull milvusdb/milvus:latest
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

# For Pinecone (requires API key):
# Add to .env:
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
```

#### Issue 10: Test Failures

**Problem**: Tests fail during validation

**Solution**:
```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run specific failing test
pytest tests/unit/test_models.py::test_specific_function -v

# Check for missing dependencies
pip install -r requirements.txt --upgrade

# Clear pytest cache
pytest --cache-clear
```

### Getting Additional Help

If you encounter issues not covered here:

1. **Check Course Forums**: Post your question with error details
2. **Review Documentation**: See `ARCHITECTURE.md` and `README.md`
3. **GitHub Issues**: Report bugs at the course repository
4. **Office Hours**: Attend instructor office hours
5. **Study Groups**: Connect with other students

### Diagnostic Information

When reporting issues, include:

```bash
# System information
python --version
pip --version
jupyter --version

# Package versions
pip list | grep -E "(pydantic|pytest|ragas|langchain)"

# Environment variables (sanitized)
env | grep -E "(NVIDIA|DATASET|VECTOR)" | sed 's/=.*/=***/'

# Error messages (full traceback)
# Copy the complete error output
```

---

## Additional Resources

### Documentation

- **Course Documentation**: `docs/` directory
- **Architecture Guide**: `ARCHITECTURE.md`
- **Quick Start**: `QUICKSTART.md`
- **API Reference**: `docs/api/`

### External Resources

- **NVIDIA Developer Portal**: [developer.nvidia.com](https://developer.nvidia.com/)
- **NVIDIA AI Catalog**: [catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/)
- **Ragas Documentation**: [docs.ragas.io](https://docs.ragas.io/)
- **LangChain Documentation**: [python.langchain.com](https://python.langchain.com/)
- **Pydantic Documentation**: [docs.pydantic.dev](https://docs.pydantic.dev/)

### Video Tutorials

- **Environment Setup**: [Link to video]
- **NVIDIA API Configuration**: [Link to video]
- **JupyterLab Basics**: [Link to video]
- **Troubleshooting Common Issues**: [Link to video]

### Community Support

- **Course Forum**: [Link to forum]
- **Discord Channel**: [Link to Discord]
- **Study Groups**: [Link to study group finder]
- **Office Hours**: Check course schedule

---

## Next Steps

Once your environment is set up and validated:

1. **Review Course Structure**: Read `README.md` and `ARCHITECTURE.md`
2. **Start Module 1**: Open `course_materials/notebooks/notebook_0_search_paradigm_comparison.py`
3. **Join Study Group**: Connect with other students
4. **Attend Orientation**: Join the course orientation session

---

## Checklist

Use this checklist to track your setup progress:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] JupyterLab installed and configured
- [ ] NVIDIA API key obtained
- [ ] Environment variables configured (.env file)
- [ ] All dependencies installed (requirements.txt)
- [ ] Package installed in development mode
- [ ] Datasets downloaded and validated
- [ ] Environment validation script passed
- [ ] Test suite runs successfully
- [ ] First notebook opens in JupyterLab
- [ ] NVIDIA API connection tested

---

**Congratulations!** Your environment is ready. You can now begin the course. Happy learning!

For questions or issues, refer to the [Troubleshooting Guide](#troubleshooting-guide) or contact course support.
