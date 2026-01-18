#!/usr/bin/env python3
"""
Dataset pre-loading script for RAG Evaluation Course.

This script downloads and prepares course datasets:
- USC Course Catalog
- Amnesty Q&A dataset

Usage:
    python scripts/preload_datasets.py [--force] [--validate-only]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pandas as pd
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required packages not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_step(message: str) -> None:
    """Print a step message in blue."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}▶ {message}{Colors.END}")


def print_success(message: str) -> None:
    """Print a success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def create_sample_usc_catalog() -> pd.DataFrame:
    """
    Create a sample USC Course Catalog dataset.
    
    In production, this would download from a real source.
    For now, we create sample data for testing.
    """
    sample_data = {
        'course_code': [
            'CSCI 567', 'CSCI 570', 'CSCI 571', 'CSCI 572', 'CSCI 573',
            'CSCI 574', 'CSCI 575', 'CSCI 576', 'CSCI 577', 'CSCI 578',
            'DSCI 510', 'DSCI 550', 'DSCI 551', 'DSCI 552', 'DSCI 553',
            'MATH 501', 'MATH 502', 'MATH 503', 'MATH 504', 'MATH 505'
        ],
        'course_name': [
            'Machine Learning', 'Analysis of Algorithms', 'Web Technologies',
            'Information Retrieval', 'Data Management', 'Computer Vision',
            'Natural Language Processing', 'Multimedia Systems', 'Software Engineering',
            'Database Systems', 'Data Science Fundamentals', 'Data Mining',
            'Foundations of Data Management', 'Machine Learning for Data Science',
            'Data Analytics', 'Linear Algebra', 'Probability Theory',
            'Statistical Methods', 'Optimization', 'Numerical Analysis'
        ],
        'units': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        'catalog_description': [
            'Fundamental concepts and techniques of machine learning including supervised and unsupervised learning, neural networks, and deep learning.',
            'Design and analysis of algorithms. Complexity analysis, sorting, searching, graph algorithms, dynamic programming.',
            'Web application development including HTML, CSS, JavaScript, server-side programming, and database integration.',
            'Techniques for indexing, searching, and ranking documents. Vector space models, probabilistic models, and evaluation metrics.',
            'Database design, SQL, NoSQL, transaction processing, and distributed databases.',
            'Image processing, feature extraction, object recognition, and deep learning for computer vision.',
            'Text processing, language models, sentiment analysis, machine translation, and transformer architectures.',
            'Audio and video processing, compression, streaming, and multimedia applications.',
            'Software development methodologies, design patterns, testing, and project management.',
            'Relational database theory, query optimization, indexing, and transaction management.',
            'Introduction to data science including data collection, cleaning, analysis, and visualization.',
            'Pattern discovery, clustering, classification, and association rule mining.',
            'Data modeling, database systems, and data warehousing.',
            'Advanced machine learning techniques for large-scale data analysis.',
            'Statistical analysis, hypothesis testing, and predictive modeling.',
            'Vector spaces, matrices, eigenvalues, and applications to data science.',
            'Probability distributions, random variables, and statistical inference.',
            'Regression analysis, ANOVA, and experimental design.',
            'Linear and nonlinear optimization, convex optimization, and applications.',
            'Numerical methods for solving equations, interpolation, and integration.'
        ],
        'schedule_time': [
            'MW 2:00-3:20 PM', 'TTh 10:00-11:20 AM', 'MW 4:00-5:20 PM',
            'TTh 2:00-3:20 PM', 'MW 10:00-11:20 AM', 'TTh 4:00-5:20 PM',
            'MW 6:00-7:20 PM', 'TTh 6:00-7:20 PM', 'MW 8:00-9:20 AM',
            'TTh 8:00-9:20 AM', 'MW 2:00-3:20 PM', 'TTh 10:00-11:20 AM',
            'MW 4:00-5:20 PM', 'TTh 2:00-3:20 PM', 'MW 10:00-11:20 AM',
            'TTh 4:00-5:20 PM', 'MW 6:00-7:20 PM', 'TTh 6:00-7:20 PM',
            'MW 8:00-9:20 AM', 'TTh 8:00-9:20 AM'
        ],
        'instructor': [
            'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Jones',
            'Dr. Garcia', 'Dr. Miller', 'Dr. Davis', 'Dr. Rodriguez', 'Dr. Martinez',
            'Dr. Anderson', 'Dr. Taylor', 'Dr. Thomas', 'Dr. Moore', 'Dr. Jackson',
            'Dr. White', 'Dr. Harris', 'Dr. Martin', 'Dr. Thompson', 'Dr. Lee'
        ],
        'prerequisites': [
            'CSCI 570 or equivalent', 'CSCI 104', 'CSCI 201',
            'CSCI 570', 'CSCI 201', 'CSCI 567, MATH 501',
            'CSCI 567', 'CSCI 201', 'CSCI 201', 'CSCI 201',
            'None', 'DSCI 510', 'DSCI 510', 'DSCI 510, MATH 501',
            'DSCI 510', 'None', 'MATH 501', 'MATH 501, MATH 502',
            'MATH 501', 'MATH 501'
        ]
    }
    
    return pd.DataFrame(sample_data)


def create_sample_amnesty_qa() -> list:
    """
    Create a sample Amnesty Q&A dataset.
    
    In production, this would download from a real source.
    For now, we create sample data for testing.
    """
    sample_data = [
        {
            'user_input': 'What is the Universal Declaration of Human Rights?',
            'retrieved_context': [
                'The Universal Declaration of Human Rights (UDHR) is a milestone document in the history of human rights. Drafted by representatives with different legal and cultural backgrounds from all regions of the world, the Declaration was proclaimed by the United Nations General Assembly in Paris on 10 December 1948 as a common standard of achievements for all peoples and all nations.',
                'The UDHR sets out, for the first time, fundamental human rights to be universally protected. It has been translated into over 500 languages and remains one of the most important documents in human rights history.'
            ],
            'response': 'The Universal Declaration of Human Rights (UDHR) is a historic document adopted by the United Nations General Assembly in 1948. It establishes fundamental human rights that should be universally protected for all people and nations.',
            'ground_truth': 'The UDHR is a foundational human rights document adopted by the UN in 1948 that sets out fundamental rights to be universally protected.'
        },
        {
            'user_input': 'What are the main categories of human rights?',
            'retrieved_context': [
                'Human rights can be classified into several categories: civil and political rights (such as the right to life, liberty, and freedom of expression), economic, social and cultural rights (such as the right to work, education, and health), and collective rights (such as the right to development and self-determination).',
                'The International Covenant on Civil and Political Rights (ICCPR) and the International Covenant on Economic, Social and Cultural Rights (ICESCR) are the two main international human rights treaties that elaborate on the rights set out in the UDHR.'
            ],
            'response': 'Human rights are generally categorized into three main groups: civil and political rights (like freedom of speech and right to vote), economic, social and cultural rights (like right to education and healthcare), and collective rights (like right to development).',
            'ground_truth': 'The main categories are civil and political rights, economic, social and cultural rights, and collective rights.'
        },
        {
            'user_input': 'What is the role of Amnesty International?',
            'retrieved_context': [
                'Amnesty International is a global movement of more than 10 million people who campaign for a world where human rights are enjoyed by all. We investigate and expose abuses, educate and mobilize the public, and help transform societies to create a safer, more just world.',
                'Founded in 1961, Amnesty International works to protect people wherever justice, freedom, truth and dignity are denied. We investigate human rights abuses, campaign for change, and help people claim their rights.'
            ],
            'response': 'Amnesty International is a global human rights organization with over 10 million members that investigates human rights abuses, campaigns for change, and helps people claim their rights. Founded in 1961, it works to protect justice, freedom, and dignity worldwide.',
            'ground_truth': 'Amnesty International is a global human rights organization that investigates abuses, campaigns for change, and helps people claim their rights.'
        },
        {
            'user_input': 'What is the right to freedom of expression?',
            'retrieved_context': [
                'Article 19 of the UDHR states: "Everyone has the right to freedom of opinion and expression; this right includes freedom to hold opinions without interference and to seek, receive and impart information and ideas through any media and regardless of frontiers."',
                'Freedom of expression is a fundamental human right that enables people to express their opinions, share information, and participate in public debate. However, it may be subject to certain restrictions necessary to protect the rights of others or public order.'
            ],
            'response': 'The right to freedom of expression, as stated in Article 19 of the UDHR, means everyone can hold and express opinions, and share information through any media without interference. While fundamental, this right may have some restrictions to protect others\' rights or public order.',
            'ground_truth': 'Freedom of expression is the right to hold and express opinions and share information without interference, as stated in Article 19 of the UDHR.'
        },
        {
            'user_input': 'What is the death penalty stance of Amnesty International?',
            'retrieved_context': [
                'Amnesty International opposes the death penalty in all cases without exception, regardless of the nature or circumstances of the crime, the guilt or innocence of the individual, or the method of execution. The death penalty is the ultimate cruel, inhuman and degrading punishment.',
                'As of 2023, more than two-thirds of countries have abolished the death penalty in law or practice. Amnesty International campaigns for the universal abolition of the death penalty and works to prevent executions and secure the release of prisoners on death row.'
            ],
            'response': 'Amnesty International completely opposes the death penalty in all cases, considering it the ultimate cruel, inhuman and degrading punishment. The organization campaigns for universal abolition and works to prevent executions worldwide.',
            'ground_truth': 'Amnesty International opposes the death penalty in all cases without exception and campaigns for its universal abolition.'
        }
    ]
    
    return sample_data


def load_or_create_usc_catalog(dataset_path: Path, force: bool = False) -> bool:
    """
    Load or create USC Course Catalog dataset.
    
    Args:
        dataset_path: Path to datasets directory
        force: Force recreation even if file exists
        
    Returns:
        True if successful, False otherwise
    """
    print_step("Processing USC Course Catalog...")
    
    usc_file = dataset_path / "usc_course_catalog.csv"
    
    if usc_file.exists() and not force:
        print_success(f"USC Course Catalog already exists: {usc_file}")
        
        # Validate existing file
        try:
            df = pd.read_csv(usc_file)
            print(f"  Records: {len(df)}")
            print(f"  Columns: {', '.join(df.columns)}")
            return True
        except Exception as e:
            print_error(f"Error reading existing file: {e}")
            print_warning("Will recreate file...")
    
    # Create sample dataset
    try:
        df = create_sample_usc_catalog()
        df.to_csv(usc_file, index=False)
        print_success(f"USC Course Catalog created: {usc_file}")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
        return True
    except Exception as e:
        print_error(f"Error creating USC Course Catalog: {e}")
        return False


def load_or_create_amnesty_qa(dataset_path: Path, force: bool = False) -> bool:
    """
    Load or create Amnesty Q&A dataset.
    
    Args:
        dataset_path: Path to datasets directory
        force: Force recreation even if file exists
        
    Returns:
        True if successful, False otherwise
    """
    print_step("Processing Amnesty Q&A dataset...")
    
    amnesty_file = dataset_path / "amnesty_qa.json"
    
    if amnesty_file.exists() and not force:
        print_success(f"Amnesty Q&A dataset already exists: {amnesty_file}")
        
        # Validate existing file
        try:
            with open(amnesty_file, 'r') as f:
                data = json.load(f)
            print(f"  Records: {len(data)}")
            if data:
                print(f"  Fields: {', '.join(data[0].keys())}")
            return True
        except Exception as e:
            print_error(f"Error reading existing file: {e}")
            print_warning("Will recreate file...")
    
    # Create sample dataset
    try:
        data = create_sample_amnesty_qa()
        with open(amnesty_file, 'w') as f:
            json.dump(data, f, indent=2)
        print_success(f"Amnesty Q&A dataset created: {amnesty_file}")
        print(f"  Records: {len(data)}")
        if data:
            print(f"  Fields: {', '.join(data[0].keys())}")
        return True
    except Exception as e:
        print_error(f"Error creating Amnesty Q&A dataset: {e}")
        return False


def validate_datasets(dataset_path: Path) -> bool:
    """
    Validate that datasets are properly formatted.
    
    Args:
        dataset_path: Path to datasets directory
        
    Returns:
        True if all datasets are valid, False otherwise
    """
    print_step("Validating datasets...")
    
    all_valid = True
    
    # Validate USC Course Catalog
    usc_file = dataset_path / "usc_course_catalog.csv"
    if usc_file.exists():
        try:
            df = pd.read_csv(usc_file)
            required_columns = ['course_code', 'course_name', 'units', 'catalog_description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print_error(f"USC Course Catalog missing columns: {', '.join(missing_columns)}")
                all_valid = False
            else:
                print_success("USC Course Catalog format valid")
        except Exception as e:
            print_error(f"Error validating USC Course Catalog: {e}")
            all_valid = False
    else:
        print_error("USC Course Catalog not found")
        all_valid = False
    
    # Validate Amnesty Q&A
    amnesty_file = dataset_path / "amnesty_qa.json"
    if amnesty_file.exists():
        try:
            with open(amnesty_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print_error("Amnesty Q&A should be a list of records")
                all_valid = False
            elif len(data) == 0:
                print_warning("Amnesty Q&A is empty")
            else:
                required_fields = ['user_input', 'retrieved_context', 'response', 'ground_truth']
                missing_fields = [field for field in required_fields if field not in data[0]]
                
                if missing_fields:
                    print_error(f"Amnesty Q&A missing fields: {', '.join(missing_fields)}")
                    all_valid = False
                else:
                    print_success("Amnesty Q&A format valid")
        except Exception as e:
            print_error(f"Error validating Amnesty Q&A: {e}")
            all_valid = False
    else:
        print_error("Amnesty Q&A not found")
        all_valid = False
    
    return all_valid


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Pre-load datasets for RAG Evaluation Course"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of datasets even if they exist"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing datasets without creating new ones"
    )
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}RAG Evaluation Course - Dataset Pre-loading{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    # Load environment variables
    load_dotenv()
    
    # Get dataset path from environment or use default
    dataset_path = Path(os.getenv("DATASET_PATH", "course_materials/datasets"))
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset directory: {dataset_path.absolute()}")
    
    if args.validate_only:
        # Only validate
        success = validate_datasets(dataset_path)
        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All datasets are valid!{Colors.END}\n")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}Dataset validation failed!{Colors.END}\n")
            sys.exit(1)
    else:
        # Load or create datasets
        usc_success = load_or_create_usc_catalog(dataset_path, args.force)
        amnesty_success = load_or_create_amnesty_qa(dataset_path, args.force)
        
        # Validate
        validation_success = validate_datasets(dataset_path)
        
        # Summary
        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Summary{Colors.END}")
        print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
        
        if usc_success:
            print_success("USC Course Catalog ready")
        else:
            print_error("USC Course Catalog failed")
        
        if amnesty_success:
            print_success("Amnesty Q&A dataset ready")
        else:
            print_error("Amnesty Q&A dataset failed")
        
        if validation_success:
            print_success("All datasets validated")
        else:
            print_warning("Some datasets have validation issues")
        
        if usc_success and amnesty_success and validation_success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}Dataset pre-loading completed successfully!{Colors.END}\n")
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}Dataset pre-loading completed with warnings{Colors.END}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
