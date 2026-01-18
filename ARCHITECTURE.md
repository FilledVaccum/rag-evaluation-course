# RAG Evaluation Course - Architecture Documentation

## Overview

This document describes the architecture and design decisions for the RAG Evaluation Course system.

## Project Structure

```
rag-evaluation-course/
├── course_materials/          # Course content and deliverables
│   ├── modules/              # Lecture materials for each module
│   ├── notebooks/            # Jupyter notebooks for hands-on practice
│   ├── datasets/             # Course datasets (USC Catalog, Amnesty Q&A)
│   ├── assessments/          # Quizzes, exams, and evaluation materials
│   └── study_resources/      # Study guides, references, and additional materials
│
├── src/                      # Source code
│   ├── models/              # Pydantic data models for type safety
│   │   ├── course.py        # Module, LectureMaterial, Slide models
│   │   ├── notebook.py      # JupyterNotebook, NotebookCell models
│   │   ├── assessment.py    # Assessment, Question, Quiz models
│   │   ├── dataset.py       # Dataset, TestSet, CourseRecord models
│   │   ├── evaluation.py    # EvaluationResults, Metric models
│   │   ├── rag.py          # RAGPipeline, RAGComponent models
│   │   └── certification.py # ExamDomain, MockExam models
│   │
│   ├── evaluation/          # Evaluation framework implementations
│   ├── synthetic_data/      # Synthetic data generation
│   ├── platform_integration/# NVIDIA platform integrations
│   └── utils/               # Utility functions
│
├── tests/                    # Test suite
│   ├── unit/                # Unit tests for individual components
│   ├── property/            # Property-based tests using Hypothesis
│   └── integration/         # Integration tests
│
├── .kiro/                    # Kiro spec files
│   └── specs/
│       └── rag-evaluation-course/
│           ├── requirements.md
│           ├── design.md
│           └── tasks.md
│
└── Reference Material/       # Course reference materials
```

## Core Components

### 1. Data Models (src/models/)

All data structures use Pydantic for:
- Type safety and validation
- Automatic serialization/deserialization
- Clear API contracts
- Runtime validation

**Key Models:**
- **Module**: Course module with time allocation, learning objectives, exam mappings
- **JupyterNotebook**: Notebook structure with cells, bugs, datasets
- **Assessment**: Quizzes, exams, questions with rubrics
- **Dataset**: Dataset management with preprocessing configs
- **EvaluationResults**: Evaluation metrics and analysis
- **RAGPipeline**: RAG component architecture

### 2. Evaluation Framework (src/evaluation/)

Will contain:
- Ragas integration
- Custom metric implementations
- LLM-as-a-Judge evaluators
- Results analysis tools

### 3. Synthetic Data Generation (src/synthetic_data/)

Will contain:
- Nemotron-4-340B integration
- Prompt engineering utilities
- Quality validators
- Synthesizer mixers

### 4. Platform Integration (src/platform_integration/)

Will contain:
- NVIDIA NIM client
- Embedding model interfaces
- LLM endpoint management
- API retry logic and error handling

## Design Principles

### 1. Type Safety First

All data structures use Pydantic models with:
- Explicit field types
- Validation rules
- Clear documentation
- Example schemas

### 2. Modular Architecture

Components are loosely coupled:
- Clear interfaces between modules
- Dependency injection where appropriate
- Easy to test in isolation
- Extensible for new features

### 3. Test-Driven Development

Comprehensive testing strategy:
- **Unit tests**: Individual component functionality
- **Property tests**: Universal properties using Hypothesis
- **Integration tests**: Component interactions

### 4. Progressive Complexity

Course structure follows learning progression:
1. Foundation concepts (Modules 1-2)
2. Core technical skills (Modules 3-5)
3. Advanced topics (Modules 6-7)

### 5. Certification Alignment

Every component maps to NCP-AAI exam domains:
- Explicit domain mappings in modules
- Coverage level indicators (⭐⭐⭐, ⭐⭐, ⭐)
- Practice questions aligned with exam format

## Data Flow

### Course Delivery Flow

```
Module Definition → Lecture Materials → Jupyter Notebooks → Assessments → Certification
```

### Evaluation Flow

```
Dataset → Preprocessing → RAG Pipeline → Evaluation Metrics → Analysis Report
```

### Synthetic Data Flow

```
Source Dataset → Prompt Engineering → LLM Generation → Quality Validation → Test Set
```

## Configuration Management

### Environment Variables (.env)

- NVIDIA API credentials
- Vector store configuration
- Model endpoints
- Testing profiles

### Pytest Configuration (pytest.ini)

- Test discovery patterns
- Markers for test categorization
- Coverage thresholds

### Hypothesis Configuration (.hypothesis.ini)

- Property test profiles (default, dev, ci, thorough)
- Example counts per profile
- Deadline settings

## Testing Strategy

### Unit Tests

Focus on:
- Model validation
- Individual function correctness
- Edge cases and error conditions

### Property-Based Tests

Focus on:
- Universal properties (time allocation, question counts)
- Invariants across all inputs
- Comprehensive input coverage

### Integration Tests

Focus on:
- Component interactions
- End-to-end workflows
- API integrations

## Future Extensions

### Planned Features

1. **Module Content Generation**
   - Automated slide generation
   - Diagram creation tools
   - Case study templates

2. **Advanced Evaluation**
   - Custom metric builder UI
   - Real-time evaluation dashboard
   - Comparative analysis tools

3. **Student Progress Tracking**
   - Performance analytics
   - Readiness assessment
   - Personalized study plans

4. **Production Deployment**
   - Docker containerization
   - CI/CD pipelines
   - Cloud deployment guides

## Dependencies

### Core Dependencies

- **pydantic**: Data validation and settings management
- **pytest**: Testing framework
- **hypothesis**: Property-based testing
- **langchain**: LLM orchestration
- **ragas**: RAG evaluation framework

### NVIDIA Platform

- **langchain-nvidia-ai-endpoints**: NVIDIA NIM integration
- NVIDIA Nemotron models
- NVIDIA embedding models

### Data Processing

- **pandas**: Tabular data handling
- **numpy**: Numerical operations
- **datasets**: Dataset management

### Notebooks

- **jupyter**: Notebook environment
- **jupyterlab**: Enhanced notebook interface
- **ipykernel**: Python kernel

## Development Workflow

1. **Setup**: Create virtual environment, install dependencies
2. **Development**: Write code with type hints and docstrings
3. **Testing**: Write tests before or alongside implementation
4. **Validation**: Run pytest and hypothesis tests
5. **Documentation**: Update docs and examples
6. **Review**: Code review and quality checks

## Quality Standards

- **Code Coverage**: Target 80%+ coverage
- **Type Safety**: All functions have type hints
- **Documentation**: All public APIs documented
- **Testing**: All features have unit and property tests
- **Style**: Follow PEP 8 with Black formatting

## Contributing Guidelines

1. Follow existing code structure and patterns
2. Write tests for all new features
3. Update documentation for API changes
4. Use type hints consistently
5. Run tests before committing
6. Keep commits focused and atomic

## Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Ragas Documentation](https://docs.ragas.io/)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NCP-AAI Certification Guide](https://www.nvidia.com/en-us/training/certification/)
