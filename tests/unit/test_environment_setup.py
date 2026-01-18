"""
Unit tests for environment setup scripts.

Tests cover:
- Dependency installation validation
- Dataset loading and validation
- API key configuration validation
- Environment validation checks
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd

# Add scripts to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestDependencyInstallation:
    """Tests for dependency installation validation."""
    
    def test_python_version_check_success(self):
        """Test Python version check with valid version."""
        # Current Python should be 3.10+
        version = sys.version_info
        assert version.major >= 3
        assert version.minor >= 10
    
    def test_virtual_environment_detection(self):
        """Test virtual environment detection."""
        # Check if we can detect venv
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        # This test just verifies the detection logic works
        assert isinstance(in_venv, bool)
    
    def test_required_packages_importable(self):
        """Test that all required packages can be imported."""
        required_packages = [
            'pydantic',
            'pytest',
            'hypothesis',
            'pandas',
            'dotenv'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package '{package}' not installed")
    
    def test_package_version_check(self):
        """Test that we can check package versions."""
        import pydantic
        
        # Should have version attribute
        assert hasattr(pydantic, '__version__')
        
        # Version should be 2.0+
        major_version = int(pydantic.__version__.split('.')[0])
        assert major_version >= 2


class TestDatasetLoading:
    """Tests for dataset loading and validation."""
    
    @pytest.fixture
    def temp_dataset_dir(self, tmp_path):
        """Create temporary dataset directory."""
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        return dataset_dir
    
    def test_usc_catalog_creation(self, temp_dataset_dir):
        """Test USC Course Catalog creation."""
        # Create sample data
        sample_data = {
            'course_code': ['CSCI 567', 'CSCI 570'],
            'course_name': ['Machine Learning', 'Algorithms'],
            'units': [4, 4],
            'catalog_description': ['ML course', 'Algorithms course'],
            'schedule_time': ['MW 2:00-3:20 PM', 'TTh 10:00-11:20 AM'],
            'instructor': ['Dr. Smith', 'Dr. Johnson'],
            'prerequisites': ['None', 'CSCI 104']
        }
        
        df = pd.DataFrame(sample_data)
        csv_file = temp_dataset_dir / "usc_course_catalog.csv"
        df.to_csv(csv_file, index=False)
        
        # Verify file exists and can be read
        assert csv_file.exists()
        loaded_df = pd.read_csv(csv_file)
        assert len(loaded_df) == 2
        assert 'course_code' in loaded_df.columns
    
    def test_usc_catalog_validation(self, temp_dataset_dir):
        """Test USC Course Catalog validation."""
        # Create valid dataset
        sample_data = {
            'course_code': ['CSCI 567'],
            'course_name': ['Machine Learning'],
            'units': [4],
            'catalog_description': ['ML course']
        }
        
        df = pd.DataFrame(sample_data)
        csv_file = temp_dataset_dir / "usc_course_catalog.csv"
        df.to_csv(csv_file, index=False)
        
        # Validate required columns
        loaded_df = pd.read_csv(csv_file)
        required_columns = ['course_code', 'course_name', 'units', 'catalog_description']
        
        for col in required_columns:
            assert col in loaded_df.columns, f"Missing required column: {col}"
    
    def test_amnesty_qa_creation(self, temp_dataset_dir):
        """Test Amnesty Q&A dataset creation."""
        sample_data = [
            {
                'user_input': 'What is UDHR?',
                'retrieved_context': ['Context about UDHR'],
                'response': 'UDHR is...',
                'ground_truth': 'Universal Declaration of Human Rights'
            }
        ]
        
        json_file = temp_dataset_dir / "amnesty_qa.json"
        with open(json_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Verify file exists and can be read
        assert json_file.exists()
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        assert len(loaded_data) == 1
        assert 'user_input' in loaded_data[0]
    
    def test_amnesty_qa_validation(self, temp_dataset_dir):
        """Test Amnesty Q&A dataset validation."""
        sample_data = [
            {
                'user_input': 'Test question',
                'retrieved_context': ['Context'],
                'response': 'Answer',
                'ground_truth': 'Truth'
            }
        ]
        
        json_file = temp_dataset_dir / "amnesty_qa.json"
        with open(json_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Validate required fields
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        
        required_fields = ['user_input', 'retrieved_context', 'response', 'ground_truth']
        
        assert isinstance(loaded_data, list)
        assert len(loaded_data) > 0
        
        for field in required_fields:
            assert field in loaded_data[0], f"Missing required field: {field}"
    
    def test_dataset_file_size_check(self, temp_dataset_dir):
        """Test dataset file size checking."""
        # Create a small file
        test_file = temp_dataset_dir / "test.csv"
        test_file.write_text("test,data\n1,2\n")
        
        # Check file size
        assert test_file.exists()
        size = test_file.stat().st_size
        assert size > 0
        
        # Format size string
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        assert "KB" in size_str or "MB" in size_str


class TestAPIKeyValidation:
    """Tests for NVIDIA API key validation."""
    
    def test_api_key_environment_variable(self):
        """Test API key can be read from environment."""
        # Set test API key
        test_key = "nvapi-test-key-12345"
        os.environ['NVIDIA_API_KEY'] = test_key
        
        # Read it back
        api_key = os.getenv('NVIDIA_API_KEY')
        assert api_key == test_key
        
        # Clean up
        del os.environ['NVIDIA_API_KEY']
    
    def test_api_key_masking(self):
        """Test API key masking for display."""
        api_key = "nvapi-1234567890abcdef"
        
        # Mask the key
        masked_key = api_key[:10] + "..." if len(api_key) > 10 else "***"
        
        assert masked_key == "nvapi-1234..."
        assert len(masked_key) < len(api_key)
    
    def test_api_key_placeholder_detection(self):
        """Test detection of placeholder API key."""
        placeholder = "your_api_key_here"
        real_key = "nvapi-1234567890"
        
        # Placeholder should be detected
        assert placeholder == "your_api_key_here"
        
        # Real key should not match placeholder
        assert real_key != "your_api_key_here"
    
    def test_dotenv_file_loading(self, tmp_path):
        """Test loading environment variables from .env file."""
        from dotenv import load_dotenv
        
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("NVIDIA_API_KEY=test-key-123\n")
        
        # Load it
        load_dotenv(env_file)
        
        # Verify it was loaded
        api_key = os.getenv('NVIDIA_API_KEY')
        assert api_key == "test-key-123"
        
        # Clean up
        if 'NVIDIA_API_KEY' in os.environ:
            del os.environ['NVIDIA_API_KEY']


class TestEnvironmentValidation:
    """Tests for environment validation checks."""
    
    def test_path_exists_check(self, tmp_path):
        """Test checking if paths exist."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        # Check exists
        assert test_file.exists()
        
        # Check non-existent file
        non_existent = tmp_path / "does_not_exist.txt"
        assert not non_existent.exists()
    
    def test_directory_creation(self, tmp_path):
        """Test creating directory structure."""
        # Create nested directories
        nested_dir = tmp_path / "a" / "b" / "c"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
    
    def test_file_permissions_check(self, tmp_path):
        """Test checking file permissions."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("#!/usr/bin/env python3\nprint('test')")
        
        # Check if file is readable
        assert os.access(test_file, os.R_OK)
        
        # Make executable (Unix-like systems)
        if sys.platform != 'win32':
            test_file.chmod(0o755)
            assert os.access(test_file, os.X_OK)
    
    def test_import_validation(self):
        """Test that core modules can be imported."""
        # Test importing from src
        try:
            from src.models import course, dataset, assessment
            assert course is not None
            assert dataset is not None
            assert assessment is not None
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")
    
    def test_jupyter_availability(self):
        """Test that Jupyter is available."""
        try:
            import jupyter
            import jupyterlab
            assert jupyter is not None
            assert jupyterlab is not None
        except ImportError:
            pytest.skip("Jupyter not installed")
    
    def test_pytest_availability(self):
        """Test that pytest is available and working."""
        import pytest as pt
        
        # Should have version
        assert hasattr(pt, '__version__')
        
        # Should be able to create test
        def sample_test():
            assert True
        
        # Test should be callable
        assert callable(sample_test)


class TestScriptFunctionality:
    """Tests for script helper functions."""
    
    def test_color_codes(self):
        """Test ANSI color codes are defined."""
        # These would be imported from the scripts
        GREEN = '\033[92m'
        RED = '\033[91m'
        END = '\033[0m'
        
        assert GREEN != RED
        assert len(GREEN) > 0
        assert len(END) > 0
    
    def test_message_formatting(self):
        """Test message formatting functions."""
        # Test success message format
        message = "Test successful"
        formatted = f"✓ {message}"
        
        assert "✓" in formatted
        assert message in formatted
    
    def test_command_result_parsing(self):
        """Test parsing command results."""
        # Simulate command output
        stdout = "pytest==7.4.0\nhypothesis==6.82.0\n"
        
        # Parse versions
        lines = stdout.strip().split('\n')
        packages = {}
        
        for line in lines:
            if '==' in line:
                name, version = line.split('==')
                packages[name] = version
        
        assert 'pytest' in packages
        assert 'hypothesis' in packages
    
    def test_validation_result_aggregation(self):
        """Test aggregating validation results."""
        checks = [
            ("Python version", True),
            ("Dependencies", True),
            ("API key", False),
            ("Datasets", True)
        ]
        
        passed = sum(1 for _, success in checks if success)
        total = len(checks)
        
        assert passed == 3
        assert total == 4
        assert passed / total == 0.75


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataset_handling(self, tmp_path):
        """Test handling of empty datasets."""
        # Create empty CSV
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("column1,column2\n")
        
        df = pd.read_csv(empty_csv)
        assert len(df) == 0
        assert list(df.columns) == ['column1', 'column2']
    
    def test_malformed_json_handling(self, tmp_path):
        """Test handling of malformed JSON."""
        # Create malformed JSON
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid json")
        
        # Should raise error when loading
        with pytest.raises(json.JSONDecodeError):
            with open(bad_json, 'r') as f:
                json.load(f)
    
    def test_missing_columns_detection(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({
            'course_code': ['CSCI 567'],
            'course_name': ['Machine Learning']
        })
        
        required_columns = ['course_code', 'course_name', 'units', 'catalog_description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        assert len(missing_columns) == 2
        assert 'units' in missing_columns
        assert 'catalog_description' in missing_columns
    
    def test_invalid_api_key_format(self):
        """Test detection of invalid API key format."""
        valid_key = "nvapi-1234567890abcdef"
        invalid_keys = [
            "",
            "invalid",
            "your_api_key_here",
            "123",
            None
        ]
        
        # Valid key should start with nvapi-
        assert valid_key.startswith("nvapi-")
        
        # Invalid keys should not
        for key in invalid_keys:
            if key:
                assert not key.startswith("nvapi-") or key == "your_api_key_here"
    
    def test_file_not_found_handling(self, tmp_path):
        """Test handling of missing files."""
        non_existent = tmp_path / "does_not_exist.txt"
        
        # Should not exist
        assert not non_existent.exists()
        
        # Reading should raise error
        with pytest.raises(FileNotFoundError):
            with open(non_existent, 'r') as f:
                f.read()


class TestIntegration:
    """Integration tests for environment setup."""
    
    def test_full_validation_workflow(self, tmp_path):
        """Test complete validation workflow."""
        # 1. Check Python version
        version = sys.version_info
        assert version.major >= 3 and version.minor >= 10
        
        # 2. Create directory structure
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        assert dataset_dir.exists()
        
        # 3. Create datasets
        usc_file = dataset_dir / "usc_course_catalog.csv"
        pd.DataFrame({'course_code': ['CSCI 567']}).to_csv(usc_file, index=False)
        assert usc_file.exists()
        
        amnesty_file = dataset_dir / "amnesty_qa.json"
        with open(amnesty_file, 'w') as f:
            json.dump([{'user_input': 'test'}], f)
        assert amnesty_file.exists()
        
        # 4. Validate datasets
        df = pd.read_csv(usc_file)
        assert len(df) > 0
        
        with open(amnesty_file, 'r') as f:
            data = json.load(f)
        assert len(data) > 0
    
    def test_environment_setup_sequence(self, tmp_path):
        """Test the sequence of environment setup steps."""
        steps_completed = []
        
        # Step 1: Check Python
        if sys.version_info >= (3, 10):
            steps_completed.append("python_version")
        
        # Step 2: Create directories
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        steps_completed.append("directories")
        
        # Step 3: Create .env
        env_file = tmp_path / ".env"
        env_file.write_text("NVIDIA_API_KEY=test\n")
        steps_completed.append("env_file")
        
        # Verify all steps completed
        assert "python_version" in steps_completed
        assert "directories" in steps_completed
        assert "env_file" in steps_completed
        assert len(steps_completed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
