"""
Pytest configuration and fixtures for RAG Evaluation Course tests.
"""

import pytest
from hypothesis import settings, Verbosity
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# Hypothesis profiles configuration
settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=5000)
settings.register_profile("thorough", max_examples=1000, deadline=None)

# Load profile from environment or use default
settings.load_profile("default")


@pytest.fixture
def sample_module_data():
    """Sample module data for testing."""
    return {
        "module_number": 1,
        "title": "Evolution of Search to RAG",
        "duration_minutes": 45,
        "learning_objectives": [
            "Understand search evolution",
            "Compare BM25 vs semantic search"
        ],
        "exam_domain_mapping": {
            "Agent Architecture": 15.0,
            "Knowledge Integration": 10.0
        },
        "lecture_time_minutes": 18,
        "hands_on_time_minutes": 22,
        "discussion_time_minutes": 5
    }


@pytest.fixture
def sample_question_data():
    """Sample question data for testing."""
    return {
        "question_id": "q1_1",
        "question_text": "What is the primary advantage of RAG over fine-tuning?",
        "question_type": "multiple_choice",
        "options": [
            "Lower computational cost",
            "Dynamic knowledge updates without retraining",
            "Better performance on all tasks",
            "Simpler implementation"
        ],
        "correct_answer": "Dynamic knowledge updates without retraining",
        "explanation": "RAG allows updating knowledge by changing the retrieval corpus without retraining the model.",
        "exam_domain": "Agent Architecture",
        "difficulty": "intermediate",
        "points": 1
    }


@pytest.fixture
def sample_notebook_data():
    """Sample notebook data for testing."""
    return {
        "notebook_id": "notebook_0",
        "module_number": 1,
        "title": "Evolution from Classic Search to RAG",
        "learning_objectives": [
            "Compare search paradigms",
            "Understand RAG architecture"
        ],
        "cells": [],
        "intentional_bugs": [],
        "datasets": [],
        "metadata": {"kernel": "python3"}
    }


@pytest.fixture
def sample_test_set_data():
    """Sample test set data for testing."""
    return {
        "test_set_id": "test_001",
        "name": "Sample Test Set",
        "questions": ["What is RAG?", "How does retrieval work?"],
        "contexts": [
            "RAG is Retrieval-Augmented Generation...",
            "Retrieval uses embeddings..."
        ],
        "responses": [
            "RAG combines retrieval with generation...",
            "Retrieval works by finding similar embeddings..."
        ],
        "ground_truths": [
            "RAG is a technique that...",
            "Retrieval finds relevant documents..."
        ],
        "metadata": {"source": "test"}
    }


@pytest.fixture
def sample_course_record_data():
    """Sample course record data for testing."""
    return {
        "course_name": "CSCI 567",
        "units": 4,
        "catalog_description": "Machine Learning fundamentals and applications",
        "schedule_time": "MW 2:00-3:20 PM",
        "instructor": "Dr. Smith",
        "prerequisites": ["CSCI 270", "MATH 225"]
    }


@pytest.fixture
def sample_evaluation_results_data():
    """Sample evaluation results data for testing."""
    return {
        "evaluation_id": "eval_001",
        "metrics": {
            "faithfulness": 0.92,
            "relevancy": 0.85,
            "context_precision": 0.88
        },
        "detailed_results": [],
        "summary": "Overall good performance with high faithfulness",
        "recommendations": [
            "Improve context retrieval for better relevancy"
        ],
        "metadata": {
            "framework": "ragas",
            "timestamp": "2024-01-15T10:00:00Z"
        }
    }


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests using Hypothesis"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to run"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests that require NVIDIA API access"
    )
    config.addinivalue_line(
        "markers", "notebook: Tests for Jupyter notebook execution"
    )
