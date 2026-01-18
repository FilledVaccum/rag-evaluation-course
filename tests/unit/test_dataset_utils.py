"""
Unit tests for dataset management utilities.

Tests DatasetManager and DatasetValidator classes for edge cases and error conditions.
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path

from src.models.dataset import (
    Dataset,
    DatasetFormat,
    PreprocessConfig,
    CourseRecord
)
from src.utils.dataset_manager import DatasetManager
from src.utils.dataset_validator import DatasetValidator


class TestDatasetManager:
    """Unit tests for DatasetManager class."""
    
    def test_load_dataset_csv(self, tmp_path):
        """Test loading CSV dataset."""
        # Create temporary CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "course_name": ["CSCI 567", "CSCI 570"],
            "units": [4, 4],
            "catalog_description": ["ML course", "Algorithms course"],
            "schedule_time": ["MW 2-3", "TTh 10-11"],
            "instructor": ["Dr. A", "Dr. B"],
            "prerequisites": [[], []]
        })
        df.to_csv(csv_path, index=False)
        
        # Load dataset
        manager = DatasetManager(datasets_dir=str(tmp_path))
        dataset = manager.load_dataset("test", dataset_path=str(csv_path))
        
        assert dataset.dataset_id == "test"
        assert dataset.format == DatasetFormat.CSV
        assert dataset.num_records == 2
        assert "course_name" in dataset.schema
    
    def test_load_dataset_json(self, tmp_path):
        """Test loading JSON dataset."""
        # Create temporary JSON file
        json_path = tmp_path / "test.json"
        data = {
            "user_input": ["Question 1", "Question 2"],
            "retrieved_contexts": [["Context 1"], ["Context 2"]],
            "response": ["Answer 1", "Answer 2"],
            "reference": ["Truth 1", "Truth 2"]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        # Load dataset
        manager = DatasetManager(datasets_dir=str(tmp_path))
        dataset = manager.load_dataset("test", dataset_path=str(json_path))
        
        assert dataset.dataset_id == "test"
        assert dataset.format == DatasetFormat.JSON
        assert dataset.num_records == 2
    
    def test_load_dataset_file_not_found(self, tmp_path):
        """Test loading non-existent dataset raises FileNotFoundError."""
        manager = DatasetManager(datasets_dir=str(tmp_path))
        
        with pytest.raises(FileNotFoundError):
            manager.load_dataset("nonexistent", dataset_path=str(tmp_path / "missing.csv"))
    
    def test_load_dataset_unsupported_format(self, tmp_path):
        """Test loading unsupported format raises ValueError."""
        # Create file with unsupported extension
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("some data")
        
        manager = DatasetManager(datasets_dir=str(tmp_path))
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            manager.load_dataset("test", dataset_path=str(txt_path))
    
    def test_preprocess_tabular_with_labels(self, tmp_path):
        """Test preprocessing tabular data with labels."""
        # Create and load CSV
        csv_path = tmp_path / "courses.csv"
        df = pd.DataFrame({
            "course_name": ["CSCI 567"],
            "units": [4],
            "catalog_description": ["Machine Learning"],
            "schedule_time": ["MW 2-3"],
            "instructor": ["Dr. Smith"],
            "prerequisites": [[]]
        })
        df.to_csv(csv_path, index=False)
        
        manager = DatasetManager(datasets_dir=str(tmp_path))
        manager.load_dataset("courses", dataset_path=str(csv_path))
        
        # Preprocess
        config = PreprocessConfig(include_labels=True)
        processed = manager.preprocess_tabular("courses", config)
        
        assert len(processed.processed_records) == 1
        assert "Class name: CSCI 567" in processed.processed_records[0]
        assert "Units: 4" in processed.processed_records[0]
    
    def test_preprocess_tabular_without_labels(self, tmp_path):
        """Test preprocessing tabular data without labels."""
        # Create and load CSV
        csv_path = tmp_path / "courses.csv"
        df = pd.DataFrame({
            "course_name": ["CSCI 567"],
            "units": [4],
            "catalog_description": ["Machine Learning"],
            "schedule_time": ["MW 2-3"],
            "instructor": ["Dr. Smith"],
            "prerequisites": [[]]
        })
        df.to_csv(csv_path, index=False)
        
        manager = DatasetManager(datasets_dir=str(tmp_path))
        manager.load_dataset("courses", dataset_path=str(csv_path))
        
        # Preprocess
        config = PreprocessConfig(include_labels=False)
        processed = manager.preprocess_tabular("courses", config)
        
        assert len(processed.processed_records) == 1
        # Should not have labels
        assert "Class name:" not in processed.processed_records[0]
        assert "CSCI 567" in processed.processed_records[0]
    
    def test_preprocess_tabular_not_loaded(self):
        """Test preprocessing unloaded dataset raises ValueError."""
        manager = DatasetManager()
        config = PreprocessConfig()
        
        with pytest.raises(ValueError, match="not loaded"):
            manager.preprocess_tabular("nonexistent", config)
    
    def test_preprocess_tabular_not_dataframe(self, tmp_path):
        """Test preprocessing non-tabular data raises ValueError."""
        # Create and load JSON (not tabular)
        json_path = tmp_path / "test.json"
        data = {"key": "value"}
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        manager = DatasetManager(datasets_dir=str(tmp_path))
        manager.load_dataset("test", dataset_path=str(json_path))
        
        config = PreprocessConfig()
        with pytest.raises(ValueError, match="not tabular"):
            manager.preprocess_tabular("test", config)
    
    def test_format_for_ragas_json(self, tmp_path):
        """Test formatting JSON data for Ragas."""
        # Create JSON dataset
        json_path = tmp_path / "qa.json"
        data = {
            "user_input": ["Q1", "Q2"],
            "retrieved_contexts": [["C1"], ["C2"]],
            "response": ["A1", "A2"],
            "reference": ["T1", "T2"]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        manager = DatasetManager(datasets_dir=str(tmp_path))
        manager.load_dataset("qa", dataset_path=str(json_path))
        
        # Format for Ragas
        ragas_data = manager.format_for_ragas("qa")
        
        assert len(ragas_data.user_inputs) == 2
        assert ragas_data.user_inputs[0] == "Q1"
        assert ragas_data.retrieved_contexts[0] == ["C1"]
        assert ragas_data.responses[0] == "A1"
        assert ragas_data.ground_truths[0] == "T1"
    
    def test_format_for_ragas_dataframe(self, tmp_path):
        """Test formatting DataFrame for Ragas."""
        # Create CSV dataset
        csv_path = tmp_path / "qa.csv"
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "context": ["C1", "C2"],
            "response": ["A1", "A2"],
            "ground_truth": ["T1", "T2"]
        })
        df.to_csv(csv_path, index=False)
        
        manager = DatasetManager(datasets_dir=str(tmp_path))
        manager.load_dataset("qa", dataset_path=str(csv_path))
        
        # Format for Ragas with custom field names
        ragas_data = manager.format_for_ragas(
            "qa",
            question_field="question",
            context_field="context",
            response_field="response",
            ground_truth_field="ground_truth"
        )
        
        assert len(ragas_data.user_inputs) == 2
        assert ragas_data.user_inputs[0] == "Q1"
        assert ragas_data.retrieved_contexts[0] == ["C1"]
    
    def test_format_for_ragas_not_loaded(self):
        """Test formatting unloaded dataset raises ValueError."""
        manager = DatasetManager()
        
        with pytest.raises(ValueError, match="not loaded"):
            manager.format_for_ragas("nonexistent")


class TestDatasetValidator:
    """Unit tests for DatasetValidator class."""
    
    def test_validate_empty_dataset(self, tmp_path):
        """Test validation of empty dataset."""
        # Create CSV with headers but no data
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame(columns=["course_name", "units"])
        df.to_csv(csv_path, index=False)
        
        dataset = Dataset(
            dataset_id="empty",
            name="Empty Dataset",
            description="Test",
            format=DatasetFormat.CSV,
            path=str(csv_path),
            schema={}
        )
        
        validator = DatasetValidator()
        report = validator.validate_schema(dataset)
        
        assert not report.is_valid
        assert any("empty" in error.lower() for error in report.errors)
        assert len(report.suggestions) > 0
    
    def test_validate_missing_required_fields(self, tmp_path):
        """Test validation with missing required fields."""
        # Create CSV missing required fields
        csv_path = tmp_path / "incomplete.csv"
        df = pd.DataFrame({
            "course_name": ["CSCI 567"],
            "units": [4]
            # Missing: catalog_description, schedule_time
        })
        df.to_csv(csv_path, index=False)
        
        dataset = Dataset(
            dataset_id="incomplete",
            name="Incomplete Dataset",
            description="Test",
            format=DatasetFormat.CSV,
            path=str(csv_path),
            schema={"course_name": "object", "units": "int64"}
        )
        
        validator = DatasetValidator()
        required_fields = {"course_name", "units", "catalog_description", "schedule_time"}
        report = validator.validate_schema(dataset, required_fields=required_fields)
        
        assert not report.is_valid
        assert any("catalog_description" in error for error in report.errors)
        assert any("schedule_time" in error for error in report.errors)
        assert len(report.suggestions) > 0
    
    def test_validate_null_values(self, tmp_path):
        """Test validation with null values in required fields."""
        # Create CSV with null values
        csv_path = tmp_path / "nulls.csv"
        df = pd.DataFrame({
            "course_name": ["CSCI 567", None],
            "units": [4, 4],
            "catalog_description": ["ML", "AI"],
            "schedule_time": ["MW 2-3", "TTh 10-11"]
        })
        df.to_csv(csv_path, index=False)
        
        dataset = Dataset(
            dataset_id="nulls",
            name="Nulls Dataset",
            description="Test",
            format=DatasetFormat.CSV,
            path=str(csv_path),
            schema={}
        )
        
        validator = DatasetValidator()
        required_fields = {"course_name", "units", "catalog_description", "schedule_time"}
        report = validator.validate_schema(dataset, required_fields=required_fields)
        
        # Should have warnings about null values
        assert any("null" in warning.lower() for warning in report.warnings)
        assert any("Fill" in suggestion or "remove" in suggestion.lower() for suggestion in report.suggestions)
    
    def test_validate_json_format(self, tmp_path):
        """Test validation of JSON format."""
        # Create valid JSON
        json_path = tmp_path / "valid.json"
        data = {
            "user_input": ["Q1", "Q2"],
            "retrieved_contexts": [["C1"], ["C2"]],
            "response": ["A1", "A2"]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        dataset = Dataset(
            dataset_id="valid",
            name="Valid Dataset",
            description="Test",
            format=DatasetFormat.JSON,
            path=str(json_path),
            schema={}
        )
        
        validator = DatasetValidator()
        required_fields = {"user_input", "retrieved_contexts", "response"}
        report = validator.validate_schema(dataset, required_fields=required_fields)
        
        assert report.is_valid
        assert len(report.errors) == 0
    
    def test_validate_json_mismatched_lengths(self, tmp_path):
        """Test validation of JSON with mismatched list lengths."""
        # Create JSON with mismatched lengths
        json_path = tmp_path / "mismatch.json"
        data = {
            "user_input": ["Q1", "Q2"],
            "retrieved_contexts": [["C1"]],  # Only 1 item
            "response": ["A1", "A2"]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        dataset = Dataset(
            dataset_id="mismatch",
            name="Mismatch Dataset",
            description="Test",
            format=DatasetFormat.JSON,
            path=str(json_path),
            schema={}
        )
        
        validator = DatasetValidator()
        required_fields = {"user_input", "retrieved_contexts", "response"}
        report = validator.validate_schema(dataset, required_fields=required_fields)
        
        # Should have schema issues about mismatched lengths
        assert any("length" in issue.lower() for issue in report.schema_issues)
    
    def test_validate_jsonl_format(self, tmp_path):
        """Test validation of JSONL format."""
        # Create valid JSONL
        jsonl_path = tmp_path / "valid.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write(json.dumps({"user_input": "Q1", "response": "A1"}) + "\n")
            f.write(json.dumps({"user_input": "Q2", "response": "A2"}) + "\n")
        
        dataset = Dataset(
            dataset_id="valid",
            name="Valid Dataset",
            description="Test",
            format=DatasetFormat.JSONL,
            path=str(jsonl_path),
            schema={}
        )
        
        validator = DatasetValidator()
        required_fields = {"user_input", "response"}
        report = validator.validate_schema(dataset, required_fields=required_fields)
        
        assert report.is_valid
    
    def test_validate_parquet_format(self, tmp_path):
        """Test validation of Parquet format."""
        # Create valid Parquet
        parquet_path = tmp_path / "valid.parquet"
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "answer": ["A1", "A2"]
        })
        df.to_parquet(parquet_path)
        
        dataset = Dataset(
            dataset_id="valid",
            name="Valid Dataset",
            description="Test",
            format=DatasetFormat.PARQUET,
            path=str(parquet_path),
            schema={}
        )
        
        validator = DatasetValidator()
        required_fields = {"question", "answer"}
        report = validator.validate_schema(dataset, required_fields=required_fields)
        
        assert report.is_valid
    
    def test_generate_fix_suggestions_missing_column(self):
        """Test generating fix suggestions for missing columns."""
        validator = DatasetValidator()
        errors = ["Missing required column: 'question'"]
        
        suggestions = validator.generate_fix_suggestions(errors)
        
        assert any("Add 'question'" in s for s in suggestions)
        assert any("case-sensitive" in s for s in suggestions)
    
    def test_generate_fix_suggestions_null_values(self):
        """Test generating fix suggestions for null values."""
        validator = DatasetValidator()
        errors = ["Column 'context' has null values"]
        
        suggestions = validator.generate_fix_suggestions(errors)
        
        assert any("Fill null values" in s for s in suggestions)
        assert any("Remove rows" in s for s in suggestions)
    
    def test_generate_fix_suggestions_type_mismatch(self):
        """Test generating fix suggestions for type mismatches."""
        validator = DatasetValidator()
        errors = ["Expected type 'string' for 'question', got 'int'"]
        
        suggestions = validator.generate_fix_suggestions(errors)
        
        assert any("data types" in s.lower() for s in suggestions)
        assert any("Convert" in s for s in suggestions)
    
    def test_generate_fix_suggestions_empty_dataset(self):
        """Test generating fix suggestions for empty dataset."""
        validator = DatasetValidator()
        errors = ["Dataset is empty (no records found)"]
        
        suggestions = validator.generate_fix_suggestions(errors)
        
        assert any("contains data" in s.lower() for s in suggestions)
    
    def test_validate_auto_detect_ragas_format(self, tmp_path):
        """Test auto-detection of Ragas format."""
        # Create Ragas-formatted JSON
        json_path = tmp_path / "ragas.json"
        data = {
            "user_input": ["Q1"],
            "retrieved_contexts": [["C1"]],
            "response": ["A1"]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        dataset = Dataset(
            dataset_id="ragas",
            name="Ragas Dataset",
            description="Test",
            format=DatasetFormat.JSON,
            path=str(json_path),
            schema={}
        )
        
        validator = DatasetValidator()
        # Don't specify required_fields - should auto-detect
        report = validator.validate_schema(dataset)
        
        assert report.is_valid
    
    def test_validate_auto_detect_usc_catalog_format(self, tmp_path):
        """Test auto-detection of USC Catalog format."""
        # Create USC Catalog-formatted CSV
        csv_path = tmp_path / "usc.csv"
        df = pd.DataFrame({
            "course_name": ["CSCI 567"],
            "units": [4],
            "catalog_description": ["ML"],
            "schedule_time": ["MW 2-3"],
            "instructor": ["Dr. A"]
        })
        df.to_csv(csv_path, index=False)
        
        dataset = Dataset(
            dataset_id="usc",
            name="USC Dataset",
            description="Test",
            format=DatasetFormat.CSV,
            path=str(csv_path),
            schema={}
        )
        
        validator = DatasetValidator()
        # Don't specify required_fields - should auto-detect
        report = validator.validate_schema(dataset)
        
        assert report.is_valid


class TestCourseRecord:
    """Unit tests for CourseRecord model."""
    
    def test_to_embedding_string_with_labels(self):
        """Test converting course record to embedding string with labels."""
        record = CourseRecord(
            course_name="CSCI 567",
            units=4,
            catalog_description="Machine Learning fundamentals",
            schedule_time="MW 2:00-3:20 PM",
            instructor="Dr. Smith",
            prerequisites=["CSCI 270"]
        )
        
        result = record.to_embedding_string(include_labels=True)
        
        assert "Class name: CSCI 567" in result
        assert "Units: 4" in result
        assert "Machine Learning fundamentals" in result
        assert "Schedule: MW 2:00-3:20 PM" in result
    
    def test_to_embedding_string_without_labels(self):
        """Test converting course record to embedding string without labels."""
        record = CourseRecord(
            course_name="CSCI 567",
            units=4,
            catalog_description="Machine Learning fundamentals",
            schedule_time="MW 2:00-3:20 PM",
            instructor="Dr. Smith",
            prerequisites=[]
        )
        
        result = record.to_embedding_string(include_labels=False)
        
        assert "Class name:" not in result
        assert "Units:" not in result
        assert "CSCI 567" in result
        assert "4" in result
        assert "Machine Learning fundamentals" in result
