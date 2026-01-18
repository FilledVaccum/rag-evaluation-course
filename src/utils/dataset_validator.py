"""
Dataset validation utilities for schema validation and error reporting.

Provides DatasetValidator class for validating dataset schemas and formats.
"""

from typing import List, Dict, Optional, Any, Set
import pandas as pd
import json
from pathlib import Path

from src.models.dataset import (
    Dataset,
    DatasetFormat,
    ValidationReport
)


class DatasetValidator:
    """
    Validates dataset schemas and provides fix suggestions.
    
    Supports CSV, JSON, JSONL, and Parquet formats with clear error messages
    and actionable fix suggestions for common format issues.
    """
    
    # Required fields for different dataset types
    RAGAS_REQUIRED_FIELDS = {"user_input", "retrieved_contexts", "response"}
    RAGAS_OPTIONAL_FIELDS = {"reference", "ground_truth"}
    
    USC_CATALOG_REQUIRED_FIELDS = {"course_name", "units", "catalog_description", "schedule_time"}
    USC_CATALOG_OPTIONAL_FIELDS = {"instructor", "prerequisites"}
    
    def __init__(self):
        """Initialize DatasetValidator."""
        self.validation_cache: Dict[str, ValidationReport] = {}
    
    def validate_schema(
        self,
        dataset: Dataset,
        required_fields: Optional[Set[str]] = None,
        optional_fields: Optional[Set[str]] = None,
        data: Optional[Any] = None
    ) -> ValidationReport:
        """
        Validate dataset schema with clear error messages.
        
        Args:
            dataset: Dataset object to validate
            required_fields: Set of required field names (auto-detected if None)
            optional_fields: Set of optional field names
            data: Loaded data (if None, will load from dataset.path)
            
        Returns:
            ValidationReport with errors, warnings, and suggestions
            
        Example:
            >>> validator = DatasetValidator()
            >>> dataset = Dataset(dataset_id="test", ...)
            >>> report = validator.validate_schema(dataset)
            >>> if not report.is_valid:
            ...     print(f"Errors: {report.errors}")
            ...     print(f"Suggestions: {report.suggestions}")
        """
        errors: List[str] = []
        warnings: List[str] = []
        schema_issues: List[str] = []
        suggestions: List[str] = []
        
        # Load data if not provided
        if data is None:
            try:
                data = self._load_data(dataset)
            except Exception as e:
                errors.append(f"Failed to load dataset: {str(e)}")
                return ValidationReport(
                    dataset_id=dataset.dataset_id,
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    schema_issues=schema_issues,
                    suggestions=["Check file path and format", "Ensure file is not corrupted"]
                )
        
        # Auto-detect required fields if not provided
        if required_fields is None:
            required_fields, optional_fields = self._detect_dataset_type(data)
        
        # Check for empty dataset BEFORE format-specific validation
        if self._is_empty(data):
            errors.append("Dataset is empty (no records found)")
            suggestions.append("Ensure dataset file contains data")
            # Return early - no point in further validation
            return ValidationReport(
                dataset_id=dataset.dataset_id,
                is_valid=False,
                errors=errors,
                warnings=warnings,
                schema_issues=schema_issues,
                suggestions=suggestions
            )
        
        # Validate based on format
        if dataset.format == DatasetFormat.CSV or dataset.format == DatasetFormat.PARQUET:
            self._validate_tabular(data, required_fields, optional_fields, errors, warnings, schema_issues, suggestions)
        elif dataset.format == DatasetFormat.JSON:
            self._validate_json(data, required_fields, optional_fields, errors, warnings, schema_issues, suggestions)
        elif dataset.format == DatasetFormat.JSONL:
            self._validate_jsonl(data, required_fields, optional_fields, errors, warnings, schema_issues, suggestions)
        
        # Create validation report
        is_valid = len(errors) == 0
        
        report = ValidationReport(
            dataset_id=dataset.dataset_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            schema_issues=schema_issues,
            suggestions=suggestions
        )
        
        # Cache result
        self.validation_cache[dataset.dataset_id] = report
        
        return report
    
    def generate_fix_suggestions(self, errors: List[str]) -> List[str]:
        """
        Generate actionable fix suggestions for common format issues.
        
        Args:
            errors: List of validation errors
            
        Returns:
            List of fix suggestions
            
        Example:
            >>> errors = ["Missing required column: 'question'"]
            >>> suggestions = validator.generate_fix_suggestions(errors)
            >>> print(suggestions[0])
            "Add 'question' column to your dataset"
        """
        suggestions = []
        
        for error in errors:
            error_lower = error.lower()
            
            # Missing column/field
            if "missing" in error_lower and ("column" in error_lower or "field" in error_lower):
                field_name = self._extract_field_name(error)
                if field_name:
                    suggestions.append(f"Add '{field_name}' column to your dataset")
                    suggestions.append(f"Ensure column name is exactly '{field_name}' (case-sensitive)")
            
            # Null values
            elif "null" in error_lower or "missing values" in error_lower:
                field_name = self._extract_field_name(error)
                if field_name:
                    suggestions.append(f"Fill null values in '{field_name}' column")
                    suggestions.append(f"Remove rows with null values in '{field_name}'")
            
            # Type mismatch
            elif "type" in error_lower or "expected" in error_lower:
                suggestions.append("Check data types match expected schema")
                suggestions.append("Convert columns to correct types (e.g., string, int, list)")
            
            # Format issues
            elif "format" in error_lower:
                suggestions.append("Verify file format matches extension (.csv, .json, .jsonl, .parquet)")
                suggestions.append("Check file encoding (UTF-8 recommended)")
            
            # Empty dataset
            elif "empty" in error_lower:
                suggestions.append("Ensure dataset file contains data")
                suggestions.append("Check if file was properly saved")
        
        # Add general suggestions if no specific ones
        if not suggestions:
            suggestions.append("Review dataset schema requirements")
            suggestions.append("Check example datasets for correct format")
        
        return suggestions
    
    def _load_data(self, dataset: Dataset) -> Any:
        """Load data from dataset path."""
        path = Path(dataset.path)
        
        if dataset.format == DatasetFormat.CSV:
            return pd.read_csv(path)
        elif dataset.format == DatasetFormat.PARQUET:
            return pd.read_parquet(path)
        elif dataset.format == DatasetFormat.JSON:
            with open(path, 'r') as f:
                return json.load(f)
        elif dataset.format == DatasetFormat.JSONL:
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported format: {dataset.format}")
    
    def _detect_dataset_type(self, data: Any) -> tuple[Set[str], Set[str]]:
        """Auto-detect dataset type and return required/optional fields."""
        # Get available fields
        if isinstance(data, pd.DataFrame):
            available_fields = set(data.columns)
        elif isinstance(data, dict):
            available_fields = set(data.keys())
        elif isinstance(data, list) and data:
            available_fields = set(data[0].keys())
        else:
            return set(), set()
        
        # Check for Ragas format
        if self.RAGAS_REQUIRED_FIELDS.issubset(available_fields):
            return self.RAGAS_REQUIRED_FIELDS, self.RAGAS_OPTIONAL_FIELDS
        
        # Check for USC Catalog format
        if self.USC_CATALOG_REQUIRED_FIELDS.issubset(available_fields):
            return self.USC_CATALOG_REQUIRED_FIELDS, self.USC_CATALOG_OPTIONAL_FIELDS
        
        # Default: no specific requirements
        return set(), set()
    
    def _validate_tabular(
        self,
        df: pd.DataFrame,
        required_fields: Set[str],
        optional_fields: Optional[Set[str]],
        errors: List[str],
        warnings: List[str],
        schema_issues: List[str],
        suggestions: List[str]
    ) -> None:
        """Validate tabular data (DataFrame)."""
        # Check if DataFrame is empty
        if len(df) == 0:
            return  # Empty check is handled in main validate_schema method
        
        columns = set(df.columns)
        
        # Check required fields
        for field in required_fields:
            if field not in columns:
                errors.append(f"Missing required column: '{field}'")
                suggestions.append(f"Add '{field}' column to your CSV/Parquet file")
        
        # Check for null values in required fields
        for field in required_fields:
            if field in columns and df[field].isnull().any():
                null_count = df[field].isnull().sum()
                warnings.append(f"Column '{field}' has {null_count} null values")
                suggestions.append(f"Fill or remove null values in '{field}'")
        
        # Check data types
        for field in required_fields:
            if field in columns:
                dtype = df[field].dtype
                # Check for object type (could be string or mixed)
                if dtype == 'object':
                    # Check if all values are strings
                    non_string = df[field].apply(lambda x: not isinstance(x, str) if pd.notna(x) else False).any()
                    if non_string:
                        schema_issues.append(f"Column '{field}' contains non-string values")
    
    def _validate_json(
        self,
        data: Dict,
        required_fields: Set[str],
        optional_fields: Optional[Set[str]],
        errors: List[str],
        warnings: List[str],
        schema_issues: List[str],
        suggestions: List[str]
    ) -> None:
        """Validate JSON data."""
        available_fields = set(data.keys())
        
        # Check required fields
        for field in required_fields:
            if field not in available_fields:
                errors.append(f"Missing required field: '{field}'")
                suggestions.append(f"Add '{field}' field to your JSON file")
        
        # Check if fields are lists (for Ragas format)
        for field in required_fields:
            if field in data:
                if not isinstance(data[field], list):
                    schema_issues.append(f"Field '{field}' should be a list")
                    suggestions.append(f"Convert '{field}' to a list format")
                elif len(data[field]) == 0:
                    warnings.append(f"Field '{field}' is an empty list")
        
        # Check list lengths match
        if required_fields:
            lengths = {field: len(data[field]) for field in required_fields if field in data and isinstance(data[field], list)}
            if lengths and len(set(lengths.values())) > 1:
                schema_issues.append(f"Field lengths don't match: {lengths}")
                suggestions.append("Ensure all fields have the same number of items")
    
    def _validate_jsonl(
        self,
        data: List[Dict],
        required_fields: Set[str],
        optional_fields: Optional[Set[str]],
        errors: List[str],
        warnings: List[str],
        schema_issues: List[str],
        suggestions: List[str]
    ) -> None:
        """Validate JSONL data."""
        if not data:
            return
        
        # Check first record for required fields
        first_record = data[0]
        available_fields = set(first_record.keys())
        
        for field in required_fields:
            if field not in available_fields:
                errors.append(f"Missing required field: '{field}' in records")
                suggestions.append(f"Add '{field}' field to each record in JSONL file")
        
        # Check all records have consistent schema
        for i, record in enumerate(data):
            record_fields = set(record.keys())
            if record_fields != available_fields:
                warnings.append(f"Record {i} has inconsistent schema")
                if i < 5:  # Only suggest for first few
                    suggestions.append(f"Ensure all records have the same fields")
                break
    
    def _is_empty(self, data: Any) -> bool:
        """Check if dataset is empty."""
        if isinstance(data, pd.DataFrame):
            return len(data) == 0
        elif isinstance(data, (list, dict)):
            return len(data) == 0
        return False
    
    def _extract_field_name(self, error: str) -> Optional[str]:
        """Extract field name from error message."""
        # Look for quoted field name
        import re
        match = re.search(r"'([^']+)'", error)
        if match:
            return match.group(1)
        match = re.search(r'"([^"]+)"', error)
        if match:
            return match.group(1)
        return None
