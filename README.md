# RAG Evaluation Course System

A comprehensive course system for teaching "Evaluating RAG and Semantic Search Systems" designed to prepare candidates for the NVIDIA-Certified Professional: Agentic AI (NCP-AAI) certification exam.

## Overview

This course provides hands-on training in RAG (Retrieval-Augmented Generation) evaluation, synthetic test data generation, and semantic search system assessment. The curriculum focuses on practical skills needed for production RAG systems and aligns with the NCP-AAI certification exam requirements.

## Course Structure

The course consists of 7 modules:

1. **Evolution of Search to RAG** - Foundation concepts
2. **Embeddings and Vector Stores** - Technical fundamentals
3. **RAG Architecture and Component Analysis** - Core architecture
4. **Synthetic Test Data Generation** - Test-driven development
5. **RAG Evaluation Metrics and Frameworks** - Evaluation mastery
6. **Semantic Search System Evaluation** - Enterprise focus
7. **Production Deployment and Advanced Topics** - Production readiness

## Features

- **Hands-On Learning**: 50% of course time dedicated to practical exercises
- **Jupyter Notebooks**: Interactive notebooks with intentional bugs for debugging practice
- **Property-Based Testing**: Comprehensive test coverage using Hypothesis
- **NVIDIA Platform Integration**: Leverages NVIDIA NIM, Nemotron, Triton, and NeMo
- **Certification Alignment**: Explicit mapping to NCP-AAI exam domains
- **Type Safety**: Full Pydantic models for data validation

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/nvidia/rag-evaluation-course.git
cd rag-evaluation-course

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Full Installation (with all optional dependencies)

```bash
pip install -e ".[full,dev]"
```

## Project Structure

```
rag-evaluation-course/
├── course_materials/          # Course content
│   ├── modules/              # Module lecture materials
│   ├── notebooks/            # Jupyter notebooks
│   ├── datasets/             # Course datasets
│   ├── assessments/          # Quizzes and exams
│   └── study_resources/      # Study guides and references
├── src/                      # Source code
│   ├── models/              # Pydantic data models
│   ├── evaluation/          # Evaluation framework
│   ├── synthetic_data/      # Synthetic data generation
│   ├── platform_integration/# NVIDIA platform integration
│   └── utils/               # Utility functions
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── property/            # Property-based tests
│   └── integration/         # Integration tests
├── requirements.txt          # Python dependencies
├── pytest.ini               # Pytest configuration
├── .hypothesis.ini          # Hypothesis configuration
└── setup.py                 # Package setup
```

## Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests only
pytest tests/property/ -m property

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test profile
pytest --hypothesis-profile=dev  # Quick testing
pytest --hypothesis-profile=ci   # CI/CD testing
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Jupyter Notebooks

```bash
# Start JupyterLab
jupyter lab

# Navigate to course_materials/notebooks/
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# NVIDIA API Configuration
NVIDIA_API_KEY=your_api_key_here
NVIDIA_NIM_ENDPOINT=https://api.nvidia.com/nim

# Vector Store Configuration
VECTOR_STORE_TYPE=chromadb
VECTOR_STORE_PATH=./data/vector_store

# Evaluation Configuration
RAGAS_LLM_ENDPOINT=https://api.nvidia.com/llm
```

## Course Datasets

### USC Course Catalog
- **Format**: CSV
- **Size**: ~1000 courses
- **Use**: Chunking practice, tabular data handling

### Amnesty Q&A
- **Format**: JSON (pre-formatted for Ragas)
- **Use**: Metric computation, faithfulness evaluation

## Certification Alignment

This course provides primary coverage (⭐⭐⭐) for:
- **Evaluation and Tuning** (13% of NCP-AAI exam)
- **Knowledge Integration and Data Handling** (10% of exam)

Supporting coverage for:
- **Agent Development** (15% of exam)
- **Agent Architecture and Design** (15% of exam)

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/nvidia/rag-evaluation-course/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nvidia/rag-evaluation-course/discussions)

## Acknowledgments

- NVIDIA for platform support and tools
- Ragas framework for evaluation capabilities
- Course contributors and reviewers

## Citation

If you use this course material in your research or teaching, please cite:

```bibtex
@misc{rag-evaluation-course,
  title={Evaluating RAG and Semantic Search Systems: A Comprehensive Course},
  author={RAG Evaluation Course Team},
  year={2024},
  publisher={NVIDIA},
  url={https://github.com/nvidia/rag-evaluation-course}
}
```
