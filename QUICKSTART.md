# Quick Start Guide

## Getting Started with RAG Evaluation Course Development

This guide will help you get up and running with the RAG Evaluation Course system.

## Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git (for version control)

## Installation Steps

### 1. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your NVIDIA API credentials
# (Use your preferred text editor)
nano .env
```

### 4. Verify Installation

```bash
# Run tests to verify setup
PYTHONPATH=src pytest tests/unit/test_models.py -v

# You should see all tests passing
```

## Project Structure Overview

```
rag-evaluation-course/
├── src/models/              # Data models (✅ Complete)
├── src/evaluation/          # Evaluation framework (🚧 To be implemented)
├── src/synthetic_data/      # Synthetic data generation (🚧 To be implemented)
├── src/platform_integration/# NVIDIA integration (🚧 To be implemented)
├── course_materials/        # Course content (🚧 To be created)
└── tests/                   # Test suite (✅ Infrastructure ready)
```

## What's Been Completed

✅ **Task 1: Project Infrastructure**
- Directory structure created
- Python virtual environment setup
- Pytest and Hypothesis configured
- Core Pydantic data models implemented:
  - Course models (Module, LectureMaterial, Slide)
  - Notebook models (JupyterNotebook, NotebookCell)
  - Assessment models (Assessment, Question, Quiz)
  - Dataset models (Dataset, TestSet, CourseRecord)
  - Evaluation models (EvaluationResults, Metric)
  - RAG models (RAGPipeline, RAGComponent)
  - Certification models (ExamDomain, MockExam)
- Unit tests for all models
- Configuration files (pytest.ini, .hypothesis.ini, .gitignore)
- Documentation (README.md, ARCHITECTURE.md)

## Next Steps

### Task 2: Implement Core Data Models and Validation

This task will add:
- Time allocation validation properties
- Module quiz question count validation
- Student dataset support validation
- Comprehensive property-based tests

### Task 3: Implement Dataset Management

This task will add:
- Dataset loading and preprocessing
- USC Course Catalog integration
- Amnesty Q&A dataset integration
- Schema validation

## Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_models.py

# Run with verbose output
PYTHONPATH=src pytest -v

# Run tests with specific marker
PYTHONPATH=src pytest -m unit

# Run property-based tests
PYTHONPATH=src pytest -m property
```

## Development Workflow

1. **Check Current Task**: Review `.kiro/specs/rag-evaluation-course/tasks.md`
2. **Implement Feature**: Write code in appropriate `src/` directory
3. **Write Tests**: Add tests in `tests/` directory
4. **Run Tests**: Verify with `pytest`
5. **Update Documentation**: Update relevant docs
6. **Mark Complete**: Update task status

## Common Commands

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Run Jupyter Lab
jupyter lab
```

## Troubleshooting

### Import Errors

If you get import errors, make sure to set PYTHONPATH:

```bash
export PYTHONPATH=src
# Or on Windows:
set PYTHONPATH=src
```

### Pydantic Warnings

You may see deprecation warnings from Pydantic. These are expected and will be addressed in future updates.

### Test Collection Warnings

The warning about `TestSet` class is expected - it's a Pydantic model, not a pytest test class.

## Getting Help

- **Documentation**: See `ARCHITECTURE.md` for detailed architecture
- **Spec Files**: Check `.kiro/specs/rag-evaluation-course/` for requirements and design
- **Reference Materials**: See `Reference Material/` for course content guidance

## Contributing

1. Pick a task from `tasks.md`
2. Create a feature branch
3. Implement the feature with tests
4. Run full test suite
5. Submit for review

## Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)

---

**Ready to start?** Check the next task in `.kiro/specs/rag-evaluation-course/tasks.md` and begin implementation!
