"""
Setup configuration for RAG Evaluation Course system.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-evaluation-course",
    version="0.1.0",
    author="RAG Evaluation Course Team",
    description="A comprehensive course system for teaching RAG evaluation and semantic search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvidia/rag-evaluation-course",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "hypothesis>=6.82.0",
        "langchain>=0.1.0",
        "langchain-nvidia-ai-endpoints>=0.1.0",
        "ragas>=0.1.0",
        "llama-index>=0.9.0",
        "chromadb>=0.4.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "matplotlib>=3.7.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "full": [
            "milvus>=2.3.0",
            "pinecone-client>=2.2.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
            "transformers>=4.30.0",
            "datasets>=2.14.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-course=src.cli:main",
        ],
    },
)
