"""
Dataset management utilities for loading and preprocessing datasets.

Provides DatasetManager class for handling USC Course Catalog and Amnesty Q&A datasets.
"""

from typing import List, Dict, Optional, Any
import pandas as pd
import json
from pathlib import Path

from src.models.dataset import (
    Dataset,
    DatasetFormat,
    CourseRecord,
    PreprocessConfig,
    ProcessedDataset,
    RagasDataset,
    TestSet
)


class DatasetManager:
    """
    Manages dataset loading and preprocessing for course materials.
    
    Supports USC Course Catalog (tabular) and Amnesty Q&A (pre-formatted) datasets.
    """
    
    def __init__(self, datasets_dir: str = "course_materials/datasets"):
        """
        Initialize DatasetManager.
        
        Args:
            datasets_dir: Directory containing dataset files
        """
        self.datasets_dir = Path(datasets_dir)
        self.loaded_datasets: Dict[str, Any] = {}
    
    def load_dataset(self, dataset_id: str, dataset_path: Optional[str] = None) -> Dataset:
        """
        Load dataset by ID or path.
        
        Supports USC Course Catalog (CSV) and Amnesty Q&A (JSON/JSONL) formats.
        
        Args:
            dataset_id: Dataset identifier ('usc_catalog' or 'amnesty_qa')
            dataset_path: Optional custom path to dataset file
            
        Returns:
            Dataset object with loaded data
            
        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If dataset format not supported
            
        Example:
            >>> manager = DatasetManager()
            >>> dataset = manager.load_dataset('usc_catalog')
            >>> print(f"Loaded {dataset.num_records} records")
        """
        # Use predefined paths for known datasets
        if dataset_path is None:
            if dataset_id == "usc_catalog":
                dataset_path = self.datasets_dir / "usc_courses.csv"
                format = DatasetFormat.CSV
                name = "USC Course Catalog"
                description = "University course catalog for chunking practice"
            elif dataset_id == "amnesty_qa":
                dataset_path = self.datasets_dir / "amnesty_qa.json"
                format = DatasetFormat.JSON
                name = "Amnesty Q&A"
                description = "Pre-formatted Q&A dataset for Ragas evaluation"
            else:
                raise ValueError(f"Unknown dataset_id: {dataset_id}. Provide dataset_path.")
        else:
            dataset_path = Path(dataset_path)
            # Infer format from extension
            ext = dataset_path.suffix.lower()
            if ext == ".csv":
                format = DatasetFormat.CSV
            elif ext == ".json":
                format = DatasetFormat.JSON
            elif ext == ".jsonl":
                format = DatasetFormat.JSONL
            elif ext == ".parquet":
                format = DatasetFormat.PARQUET
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            name = dataset_path.stem
            description = f"Custom dataset from {dataset_path.name}"
        
        # Check if file exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load data based on format
        if format == DatasetFormat.CSV:
            df = pd.read_csv(dataset_path)
            num_records = len(df)
            schema = {col: str(df[col].dtype) for col in df.columns}
            self.loaded_datasets[dataset_id] = df
        elif format == DatasetFormat.JSON:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                num_records = len(data)
            elif isinstance(data, dict):
                # Assume dict with lists
                num_records = len(next(iter(data.values())))
            else:
                num_records = 1
            schema = self._infer_schema_from_json(data)
            self.loaded_datasets[dataset_id] = data
        elif format == DatasetFormat.JSONL:
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            num_records = len(data)
            schema = self._infer_schema_from_json(data[0]) if data else {}
            self.loaded_datasets[dataset_id] = data
        elif format == DatasetFormat.PARQUET:
            df = pd.read_parquet(dataset_path)
            num_records = len(df)
            schema = {col: str(df[col].dtype) for col in df.columns}
            self.loaded_datasets[dataset_id] = df
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Get file size
        size_bytes = dataset_path.stat().st_size
        
        # Create Dataset object
        dataset = Dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            format=format,
            path=str(dataset_path),
            size_bytes=size_bytes,
            num_records=num_records,
            schema=schema,
            metadata={"loaded": True}
        )
        
        return dataset
    
    def preprocess_tabular(
        self,
        dataset_id: str,
        config: PreprocessConfig
    ) -> ProcessedDataset:
        """
        Preprocess tabular data (e.g., USC Course Catalog) for embedding.
        
        Converts DataFrame rows to self-descriptive strings suitable for embedding.
        
        Args:
            dataset_id: ID of loaded dataset
            config: Preprocessing configuration
            
        Returns:
            ProcessedDataset with text records ready for embedding
            
        Raises:
            ValueError: If dataset not loaded or not tabular
            
        Example:
            >>> config = PreprocessConfig(
            ...     include_labels=True,
            ...     columns_to_include=['course_name', 'catalog_description', 'units']
            ... )
            >>> processed = manager.preprocess_tabular('usc_catalog', config)
            >>> print(processed.processed_records[0])
            "Class name: CSCI 567. The course will cover..."
        """
        if dataset_id not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_id} not loaded. Call load_dataset() first.")
        
        data = self.loaded_datasets[dataset_id]
        
        # Handle DataFrame (CSV, Parquet)
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"Dataset {dataset_id} is not tabular (DataFrame)")
        
        # Filter columns if specified
        if config.columns_to_include:
            df = df[config.columns_to_include]
        
        processed_records = []
        
        # Check if this is USC Course Catalog format
        if all(col in df.columns for col in ['course_name', 'catalog_description', 'units', 'schedule_time']):
            # Use CourseRecord for structured conversion
            for _, row in df.iterrows():
                try:
                    record = CourseRecord(
                        course_name=str(row.get('course_name', '')),
                        units=int(row.get('units', 0)),
                        catalog_description=str(row.get('catalog_description', '')),
                        schedule_time=str(row.get('schedule_time', '')),
                        instructor=str(row.get('instructor', 'TBA')),
                        prerequisites=row.get('prerequisites', []) if isinstance(row.get('prerequisites'), list) else []
                    )
                    processed_records.append(record.to_embedding_string(config.include_labels))
                except Exception as e:
                    # Skip malformed rows
                    continue
        else:
            # Generic row-based processing
            for _, row in df.iterrows():
                if config.include_labels:
                    # Add column labels
                    parts = [f"{col}: {row[col]}" for col in df.columns]
                    processed_records.append(". ".join(parts))
                else:
                    # Concatenate values
                    parts = [str(row[col]) for col in df.columns]
                    processed_records.append(" ".join(parts))
        
        return ProcessedDataset(
            dataset_id=dataset_id,
            processed_records=processed_records,
            embeddings=None,
            metadata={
                "preprocessing_config": config.dict(),
                "num_records": len(processed_records),
                "original_columns": list(df.columns)
            }
        )
    
    def format_for_ragas(
        self,
        dataset_id: str,
        question_field: str = "question",
        context_field: str = "context",
        response_field: str = "response",
        ground_truth_field: str = "ground_truth"
    ) -> RagasDataset:
        """
        Format dataset for Ragas evaluation framework.
        
        Converts loaded data to Ragas-compatible format with user_input,
        retrieved_contexts, response, and ground_truth fields.
        
        Args:
            dataset_id: ID of loaded dataset
            question_field: Field name for questions
            context_field: Field name for contexts
            response_field: Field name for responses
            ground_truth_field: Field name for ground truths
            
        Returns:
            RagasDataset ready for evaluation
            
        Raises:
            ValueError: If dataset not loaded or missing required fields
            
        Example:
            >>> ragas_data = manager.format_for_ragas('amnesty_qa')
            >>> print(f"Formatted {len(ragas_data.user_inputs)} samples")
        """
        if dataset_id not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_id} not loaded. Call load_dataset() first.")
        
        data = self.loaded_datasets[dataset_id]
        
        # Handle different data formats
        if isinstance(data, pd.DataFrame):
            # DataFrame format
            if question_field not in data.columns:
                raise ValueError(f"Missing required field: {question_field}")
            
            user_inputs = data[question_field].tolist()
            
            # Handle contexts (may be list or string)
            if context_field in data.columns:
                contexts_raw = data[context_field].tolist()
                retrieved_contexts = []
                for ctx in contexts_raw:
                    if isinstance(ctx, list):
                        retrieved_contexts.append(ctx)
                    else:
                        retrieved_contexts.append([str(ctx)])
            else:
                retrieved_contexts = [[] for _ in user_inputs]
            
            # Handle responses
            if response_field in data.columns:
                responses = data[response_field].tolist()
            else:
                responses = ["" for _ in user_inputs]
            
            # Handle ground truths
            if ground_truth_field in data.columns:
                ground_truths = data[ground_truth_field].tolist()
            else:
                ground_truths = None
                
        elif isinstance(data, dict):
            # Dictionary format (JSON)
            user_inputs = data.get(question_field, data.get("user_input", []))
            
            contexts_raw = data.get(context_field, data.get("retrieved_contexts", []))
            retrieved_contexts = []
            for ctx in contexts_raw:
                if isinstance(ctx, list):
                    retrieved_contexts.append(ctx)
                else:
                    retrieved_contexts.append([str(ctx)])
            
            responses = data.get(response_field, data.get("response", []))
            ground_truths = data.get(ground_truth_field, data.get("reference", None))
            
        elif isinstance(data, list):
            # List of dicts format (JSONL)
            user_inputs = []
            retrieved_contexts = []
            responses = []
            ground_truths_list = []
            
            for item in data:
                user_inputs.append(item.get(question_field, item.get("user_input", "")))
                
                ctx = item.get(context_field, item.get("retrieved_contexts", []))
                if isinstance(ctx, list):
                    retrieved_contexts.append(ctx)
                else:
                    retrieved_contexts.append([str(ctx)])
                
                responses.append(item.get(response_field, item.get("response", "")))
                ground_truths_list.append(item.get(ground_truth_field, item.get("reference", "")))
            
            ground_truths = ground_truths_list if any(ground_truths_list) else None
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        return RagasDataset(
            dataset_id=dataset_id,
            user_inputs=user_inputs,
            retrieved_contexts=retrieved_contexts,
            responses=responses,
            ground_truths=ground_truths
        )
    
    def _infer_schema_from_json(self, data: Any) -> Dict[str, str]:
        """Infer schema from JSON data."""
        if isinstance(data, dict):
            return {key: type(value).__name__ for key, value in data.items()}
        elif isinstance(data, list) and data:
            return self._infer_schema_from_json(data[0])
        else:
            return {}
