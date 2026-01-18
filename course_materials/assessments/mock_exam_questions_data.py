"""
Question data for mock certification exam.

This module contains the question data organized by domain.
Questions are stored as dictionaries for easy maintenance and generation.
"""

# Question data organized by domain
EVALUATION_TUNING_QUESTIONS = [
    # Already created in main file: eval_1 through eval_8
]

AGENT_DEVELOPMENT_QUESTIONS = [
    {
        "id": "dev_2",
        "text": "In a RAG system, you notice that relevant documents are being retrieved but the generated answers are still incorrect. Which component should you debug first?",
        "options": [
            "Embedding model",
            "Vector store configuration",
            "Generation/LLM component",
            "Chunking strategy"
        ],
        "correct": 2,
        "explanation": "If retrieval is working (relevant documents found) but answers are wrong, the issue is in the generation stage. The LLM may be hallucinating, ignoring context, or misinterpreting it. Debug the generation component first, checking prompt engineering, context formatting, and LLM parameters.",
        "difficulty": "advanced"
    },
    {
        "id": "dev_3",
        "text": "What is the recommended approach for handling tabular data in RAG systems?",
        "options": [
            "Convert tables to images",
            "Ignore tabular data",
            "Convert rows to self-descriptive text strings with labels",
            "Use only column headers"
        ],
        "correct": 2,
        "explanation": "Tabular data should be converted to self-descriptive text strings where each row becomes a concatenated string with column labels. For example: 'Course: CSCI 567, Units: 4, Description: Machine Learning'. This makes the data understandable in embedding space without table structure.",
        "difficulty": "intermediate"
    },
    {
        "id": "dev_4",
        "text": "Which NVIDIA tool is specifically designed for synthetic test data generation?",
        "options": [
            "NVIDIA NIM",
            "NVIDIA Triton",
            "NVIDIA Nemotron-4-340B",
            "NVIDIA NeMo"
        ],
        "correct": 2,
        "explanation": "NVIDIA Nemotron-4-340B is specifically designed for synthetic data generation and can be used to create test datasets for RAG evaluation. NIM provides inference endpoints, Triton handles deployment, and NeMo is for agent development.",
        "difficulty": "intermediate"
    },
