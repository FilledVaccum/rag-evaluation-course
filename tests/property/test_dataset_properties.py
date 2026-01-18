"""
Property-based tests for dataset management utilities.

Feature: rag-evaluation-course
Tests properties related to dataset management, preprocessing, and student dataset support.
"""

import pytest
from hypothesis import given, strategies as st, assume
import pandas as pd
import tempfile
from pathlib import Path
import json

from src.models.dataset import (
    Dataset,
    DatasetFormat,
    PreprocessConfig
)
from src.utils.dataset_manager import DatasetManager


# Custom strategies for generating test data
@st.composite
def dataset_with_preprocessing_config(draw):
    """Generate a dataset and preprocessing config for testing."""
    # Generate number of records
    num_records = draw(st.integers(min_value=1, max_value=50))
    
    # Generate course data
    courses = []
    for i in range(num_records):
        course = {
            "course_name": draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
            "units": draw(st.integers(min_value=1, max_value=8)),
            "catalog_description": draw(st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z')))),
            "schedule_time": draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z')))),
            "instructor": draw(st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Z')))),
        }
        courses.append(course)
    
    # Generate preprocessing config
    include_labels = draw(st.booleans())
    columns_to_include = draw(st.sampled_from([
        [],  # All columns
        ["course_name", "catalog_description", "units"],
        ["course_name", "units", "schedule_time"],
    ]))
    
    config = PreprocessConfig(
        include_labels=include_labels,
        columns_to_include=columns_to_include,
        chunking_strategy="row_based"
    )
    
    return courses, config


@st.composite
def student_dataset_strategy(draw):
    """Generate a student-provided dataset in various formats."""
    # Choose format
    format_type = draw(st.sampled_from([DatasetFormat.CSV, DatasetFormat.JSON, DatasetFormat.PARQUET]))
    
    # Generate number of records
    num_records = draw(st.integers(min_value=1, max_value=20))
    
    # Generate dataset with required fields for RAG evaluation
    # Student datasets should have: question/query, context, response, ground_truth
    data = {
        "question": [f"Question {i}" for i in range(num_records)],
        "context": [f"Context for question {i}" for i in range(num_records)],
        "response": [f"Response to question {i}" for i in range(num_records)],
        "ground_truth": [f"Ground truth for question {i}" for i in range(num_records)]
    }
    
    return format_type, data, num_records


# Feature: rag-evaluation-course, Property 6: Student Dataset Support
@given(student_dataset_strategy())
@pytest.mark.property
def test_property_student_dataset_support(dataset_info):
    """
    Property 6: Student Dataset Support
    
    For any student-provided domain-specific dataset that conforms to the expected format,
    the course system should be able to process it using the same pipelines as the
    provided datasets.
    
    This property verifies that:
    1. Student datasets in supported formats (CSV, JSON, Parquet) can be loaded
    2. The same preprocessing pipelines work on student data
    3. Student datasets can be formatted for Ragas evaluation
    4. The system handles various dataset sizes and structures
    
    Students should be able to apply course techniques to their own domains,
    not just provided examples.
    
    Validates: Requirements 11.3
    """
    format_type, data, num_records = dataset_info
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dataset file in the specified format
        if format_type == DatasetFormat.CSV:
            file_path = Path(tmp_dir) / "student_dataset.csv"
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
        elif format_type == DatasetFormat.JSON:
            file_path = Path(tmp_dir) / "student_dataset.json"
            with open(file_path, 'w') as f:
                json.dump(data, f)
                
        elif format_type == DatasetFormat.PARQUET:
            file_path = Path(tmp_dir) / "student_dataset.parquet"
            df = pd.DataFrame(data)
            df.to_parquet(file_path, index=False)
        
        # Initialize DatasetManager
        manager = DatasetManager(datasets_dir=tmp_dir)
        
        # Test 1: Load student dataset
        dataset = manager.load_dataset("student_dataset", dataset_path=str(file_path))
        
        # Verify dataset loaded successfully
        assert dataset is not None, "Student dataset should load successfully"
        assert dataset.num_records == num_records, (
            f"Dataset should have {num_records} records, got {dataset.num_records}"
        )
        assert dataset.format == format_type, (
            f"Dataset format should be {format_type}, got {dataset.format}"
        )
        
        # Test 2: Format for Ragas evaluation (same pipeline as provided datasets)
        if format_type == DatasetFormat.JSON:
            # JSON format with standard field names
            ragas_data = manager.format_for_ragas("student_dataset")
        else:
            # CSV/Parquet with explicit field mapping
            ragas_data = manager.format_for_ragas(
                "student_dataset",
                question_field="question",
                context_field="context",
                response_field="response",
                ground_truth_field="ground_truth"
            )
        
        # Verify Ragas formatting works
        assert ragas_data is not None, "Student dataset should format for Ragas"
        assert len(ragas_data.user_inputs) == num_records, (
            f"Ragas data should have {num_records} inputs, got {len(ragas_data.user_inputs)}"
        )
        assert len(ragas_data.retrieved_contexts) == num_records, (
            f"Ragas data should have {num_records} contexts, got {len(ragas_data.retrieved_contexts)}"
        )
        assert len(ragas_data.responses) == num_records, (
            f"Ragas data should have {num_records} responses, got {len(ragas_data.responses)}"
        )
        
        # Verify all contexts are lists (Ragas requirement)
        assert all(isinstance(ctx, list) for ctx in ragas_data.retrieved_contexts), (
            "All contexts should be lists for Ragas compatibility"
        )
        
        # Verify all fields are non-empty
        assert all(len(q) > 0 for q in ragas_data.user_inputs), (
            "All questions should be non-empty"
        )
        assert all(len(r) > 0 for r in ragas_data.responses), (
            "All responses should be non-empty"
        )
        
        # Test 3: Verify the same preprocessing utilities work
        # For tabular student data, preprocessing should work
        if format_type in [DatasetFormat.CSV, DatasetFormat.PARQUET]:
            config = PreprocessConfig(
                include_labels=True,
                columns_to_include=["question", "context"],
                chunking_strategy="row_based"
            )
            
            # This should work without errors
            processed = manager.preprocess_tabular("student_dataset", config)
            
            assert processed is not None, "Student dataset should preprocess successfully"
            assert len(processed.processed_records) > 0, (
                "Preprocessing should produce records"
            )
            assert all(isinstance(record, str) for record in processed.processed_records), (
                "All processed records should be strings"
            )


# Feature: rag-evaluation-course, Property 7: Dataset Preprocessing Utilities
@given(dataset_with_preprocessing_config())
@pytest.mark.property
def test_property_dataset_preprocessing_utilities(data_and_config):
    """
    Property 7: Dataset Preprocessing Utilities
    
    For any dataset provided by the course system, the system should include
    corresponding preprocessing scripts and data loading utilities.
    
    This property verifies that:
    1. Any valid tabular dataset can be loaded
    2. Preprocessing produces valid output
    3. Preprocessing utilities handle various configurations
    4. Output format is consistent and usable
    
    Validates: Requirements 11.4
    """
    courses, config = data_and_config
    
    # Create temporary CSV file
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_courses.csv"
        df = pd.DataFrame(courses)
        df.to_csv(csv_path, index=False)
        
        # Initialize DatasetManager
        manager = DatasetManager(datasets_dir=tmp_dir)
        
        # Load dataset
        dataset = manager.load_dataset("test_courses", dataset_path=str(csv_path))
        
        # Verify dataset loaded successfully
        assert dataset is not None
        assert dataset.num_records == len(courses)
        assert dataset.format == DatasetFormat.CSV
        
        # Preprocess dataset
        processed = manager.preprocess_tabular("test_courses", config)
        
        # Verify preprocessing produced valid output
        assert processed is not None
        assert len(processed.processed_records) > 0
        assert len(processed.processed_records) <= len(courses)  # May skip malformed rows
        
        # Verify all processed records are strings
        assert all(isinstance(record, str) for record in processed.processed_records)
        
        # Verify all processed records are non-empty
        assert all(len(record) > 0 for record in processed.processed_records)
        
        # Verify label inclusion is respected
        if config.include_labels:
            # Should contain label markers like "Class name:" or column names
            sample_record = processed.processed_records[0]
            # At least one record should have some structure indicating labels
            assert any(char in sample_record for char in [":", "."])
        
        # Verify metadata is present
        assert processed.metadata is not None
        assert "preprocessing_config" in processed.metadata
        assert "num_records" in processed.metadata


@given(
    num_records=st.integers(min_value=1, max_value=20),
    include_labels=st.booleans()
)
@pytest.mark.property
def test_property_preprocessing_preserves_record_count(num_records, include_labels):
    """
    Property: Preprocessing should preserve record count for valid data.
    
    For any valid dataset, preprocessing should produce the same number of
    records as the input (assuming no malformed data).
    
    Validates: Requirements 11.4
    """
    # Generate valid course records
    courses = []
    for i in range(num_records):
        course = {
            "course_name": f"CSCI {100 + i}",
            "units": 4,
            "catalog_description": f"Course {i} description",
            "schedule_time": "MW 2-3",
            "instructor": f"Dr. Instructor{i}",
            "prerequisites": []
        }
        courses.append(course)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "courses.csv"
        df = pd.DataFrame(courses)
        df.to_csv(csv_path, index=False)
        
        manager = DatasetManager(datasets_dir=tmp_dir)
        manager.load_dataset("courses", dataset_path=str(csv_path))
        
        config = PreprocessConfig(include_labels=include_labels)
        processed = manager.preprocess_tabular("courses", config)
        
        # Should preserve record count for valid data
        assert len(processed.processed_records) == num_records


@given(
    format_type=st.sampled_from([DatasetFormat.JSON, DatasetFormat.CSV])
)
@pytest.mark.property
def test_property_format_for_ragas_produces_valid_output(format_type):
    """
    Property: format_for_ragas should produce valid Ragas-compatible output.
    
    For any dataset in supported format, format_for_ragas should produce
    output with consistent field lengths and proper structure.
    
    Validates: Requirements 11.4
    """
    # Generate test data
    num_samples = 5
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        if format_type == DatasetFormat.JSON:
            import json
            json_path = Path(tmp_dir) / "test.json"
            data = {
                "user_input": [f"Question {i}" for i in range(num_samples)],
                "retrieved_contexts": [[f"Context {i}"] for i in range(num_samples)],
                "response": [f"Answer {i}" for i in range(num_samples)],
                "reference": [f"Truth {i}" for i in range(num_samples)]
            }
            with open(json_path, 'w') as f:
                json.dump(data, f)
            
            manager = DatasetManager(datasets_dir=tmp_dir)
            manager.load_dataset("test", dataset_path=str(json_path))
            
        else:  # CSV
            csv_path = Path(tmp_dir) / "test.csv"
            df = pd.DataFrame({
                "question": [f"Question {i}" for i in range(num_samples)],
                "context": [f"Context {i}" for i in range(num_samples)],
                "response": [f"Answer {i}" for i in range(num_samples)],
                "ground_truth": [f"Truth {i}" for i in range(num_samples)]
            })
            df.to_csv(csv_path, index=False)
            
            manager = DatasetManager(datasets_dir=tmp_dir)
            manager.load_dataset("test", dataset_path=str(csv_path))
        
        # Format for Ragas
        if format_type == DatasetFormat.JSON:
            ragas_data = manager.format_for_ragas("test")
        else:
            ragas_data = manager.format_for_ragas(
                "test",
                question_field="question",
                context_field="context",
                response_field="response",
                ground_truth_field="ground_truth"
            )
        
        # Verify valid Ragas output
        assert len(ragas_data.user_inputs) == num_samples
        assert len(ragas_data.retrieved_contexts) == num_samples
        assert len(ragas_data.responses) == num_samples
        
        # Verify all contexts are lists
        assert all(isinstance(ctx, list) for ctx in ragas_data.retrieved_contexts)
        
        # Verify all fields are non-empty
        assert all(len(q) > 0 for q in ragas_data.user_inputs)
        assert all(len(r) > 0 for r in ragas_data.responses)


@given(
    columns_to_include=st.lists(
        st.sampled_from(["course_name", "units", "catalog_description", "schedule_time"]),
        min_size=1,
        max_size=4,
        unique=True
    )
)
@pytest.mark.property
def test_property_column_filtering_works(columns_to_include):
    """
    Property: Column filtering should only include specified columns.
    
    For any subset of columns, preprocessing should only include data
    from those columns in the output.
    
    Validates: Requirements 11.4
    """
    # Create test data
    courses = [{
        "course_name": "CSCI 567",
        "units": 4,
        "catalog_description": "Machine Learning",
        "schedule_time": "MW 2-3",
        "instructor": "Dr. Smith",
        "prerequisites": []
    }]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "courses.csv"
        df = pd.DataFrame(courses)
        df.to_csv(csv_path, index=False)
        
        manager = DatasetManager(datasets_dir=tmp_dir)
        manager.load_dataset("courses", dataset_path=str(csv_path))
        
        config = PreprocessConfig(
            include_labels=True,
            columns_to_include=columns_to_include
        )
        processed = manager.preprocess_tabular("courses", config)
        
        # Verify output contains data from specified columns
        record = processed.processed_records[0]
        
        # Check that specified columns appear in output
        for col in columns_to_include:
            if col == "course_name":
                assert "CSCI 567" in record or "course_name" in record.lower()
            elif col == "units":
                assert "4" in record or "units" in record.lower()
            elif col == "catalog_description":
                assert "Machine Learning" in record or "description" in record.lower()
            elif col == "schedule_time":
                assert "MW 2-3" in record or "schedule" in record.lower()


@given(
    num_records=st.integers(min_value=1, max_value=10)
)
@pytest.mark.property
def test_property_preprocessing_is_deterministic(num_records):
    """
    Property: Preprocessing should be deterministic.
    
    For any dataset, running preprocessing twice with the same config
    should produce identical results.
    
    Validates: Requirements 11.4
    """
    # Generate test data
    courses = []
    for i in range(num_records):
        course = {
            "course_name": f"CSCI {100 + i}",
            "units": 4,
            "catalog_description": f"Course {i}",
            "schedule_time": "MW 2-3",
            "instructor": "Dr. A",
            "prerequisites": []
        }
        courses.append(course)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "courses.csv"
        df = pd.DataFrame(courses)
        df.to_csv(csv_path, index=False)
        
        manager = DatasetManager(datasets_dir=tmp_dir)
        manager.load_dataset("courses", dataset_path=str(csv_path))
        
        config = PreprocessConfig(include_labels=True)
        
        # Process twice
        processed1 = manager.preprocess_tabular("courses", config)
        processed2 = manager.preprocess_tabular("courses", config)
        
        # Results should be identical
        assert len(processed1.processed_records) == len(processed2.processed_records)
        assert processed1.processed_records == processed2.processed_records
