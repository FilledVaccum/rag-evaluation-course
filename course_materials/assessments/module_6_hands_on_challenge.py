"""
Module 6 Hands-On Challenge: Hybrid Search System Evaluation
Evaluating RAG and Semantic Search Systems Course

Challenge: Build and evaluate a hybrid search system that combines BM25 and
semantic search, then optimize it based on evaluation results.

Learning Objectives:
- Implement hybrid search with result fusion
- Evaluate multiple search paradigms comparatively
- Identify optimization opportunities from metrics
- Make data-driven decisions about search architecture

Requirements: 13.2
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


@dataclass
class ChallengeSpecification:
    """Specification for the hands-on challenge."""
    title: str
    description: str
    objectives: List[str]
    requirements: List[str]
    deliverables: List[str]
    evaluation_rubric: Dict[str, Dict[str, str]]
    bonus_tasks: List[str]
    estimated_time: str


CHALLENGE_SPEC = ChallengeSpecification(
    title="Hybrid Search System Evaluation Challenge",
    
    description="""
    You are a search engineer at a company with a legacy BM25-based search system.
    Users are complaining that the search doesn't understand their intent and misses
    relevant documents when they use different terminology.
    
    Your task is to:
    1. Evaluate the current BM25 system to quantify the problems
    2. Implement a semantic search alternative using embeddings
    3. Build a hybrid system combining both approaches
    4. Evaluate all three systems and recommend the best approach
    5. Optimize the chosen system based on evaluation results
    
    You'll work with a real dataset and use LLM-as-a-Judge for evaluation.
    """,
    
    objectives=[
        "Evaluate legacy BM25 system with modern metrics",
        "Implement semantic search using NVIDIA NIM embeddings",
        "Build hybrid search with Reciprocal Rank Fusion",
        "Compare all three approaches quantitatively",
        "Generate optimization recommendations based on data"
    ],
    
    requirements=[
        "Implement BM25 search (can use existing library like rank_bm25)",
        "Implement vector search using NVIDIA NV-Embed-v2",
        "Implement hybrid fusion using RRF or weighted combination",
        "Evaluate using Context Precision, Recall, and Relevance",
        "Create comparison report with visualizations",
        "Provide optimization suggestions with expected impact",
        "Document your methodology and findings"
    ],
    
    deliverables=[
        "Python notebook or script with all implementations",
        "Evaluation results for all three systems (BM25, Vector, Hybrid)",
        "Comparison report with metric tables and charts",
        "Optimization recommendations document",
        "README explaining your approach and findings"
    ],
    
    evaluation_rubric={
        "Implementation (40 points)": {
            "BM25 Search": "10 points - Correctly implements BM25 ranking",
            "Vector Search": "10 points - Uses embeddings and similarity search",
            "Hybrid Fusion": "10 points - Properly combines results from both systems",
            "Code Quality": "10 points - Clean, documented, well-structured code"
        },
        "Evaluation (30 points)": {
            "Metric Computation": "10 points - Correctly computes precision, recall, relevance",
            "LLM-as-a-Judge": "10 points - Uses LLM for semantic relevance assessment",
            "Comparative Analysis": "10 points - Fair comparison across all systems"
        },
        "Analysis (20 points)": {
            "Insights": "10 points - Identifies strengths/weaknesses of each approach",
            "Recommendations": "10 points - Data-driven optimization suggestions"
        },
        "Documentation (10 points)": {
            "Clarity": "5 points - Clear explanation of methodology",
            "Completeness": "5 points - All findings documented"
        }
    },
    
    bonus_tasks=[
        "+5 points: Implement query classification to route queries intelligently",
        "+5 points: Add re-ranking stage with cross-encoder",
        "+5 points: Implement A/B testing simulation",
        "+10 points: Deploy as API and create simple UI"
    ],
    
    estimated_time="4-6 hours"
)


# Sample Dataset for Challenge

CHALLENGE_DATASET = {
    "documents": [
        {
            "id": "doc_001",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables "
                      "computers to learn from data without explicit programming. It uses "
                      "statistical techniques to give computers the ability to learn patterns."
        },
        {
            "id": "doc_002",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a specialized branch of machine learning that uses "
                      "neural networks with multiple layers. These networks can learn "
                      "hierarchical representations of data."
        },
        {
            "id": "doc_003",
            "title": "Natural Language Processing",
            "content": "NLP is a field of AI focused on enabling computers to understand, "
                      "interpret, and generate human language. It combines linguistics with "
                      "machine learning techniques."
        },
        {
            "id": "doc_004",
            "title": "Computer Vision Applications",
            "content": "Computer vision enables machines to interpret visual information from "
                      "images and videos. Applications include object detection, facial "
                      "recognition, and autonomous vehicles."
        },
        {
            "id": "doc_005",
            "title": "Reinforcement Learning",
            "content": "Reinforcement learning trains agents to make decisions by rewarding "
                      "desired behaviors and punishing undesired ones. It's used in robotics, "
                      "game playing, and autonomous systems."
        },
        {
            "id": "doc_006",
            "title": "Supervised Learning Methods",
            "content": "Supervised learning uses labeled training data where each example has "
                      "a known output. Common algorithms include linear regression, decision "
                      "trees, and support vector machines."
        },
        {
            "id": "doc_007",
            "title": "Unsupervised Learning Techniques",
            "content": "Unsupervised learning discovers patterns in unlabeled data through "
                      "clustering, dimensionality reduction, and anomaly detection. K-means "
                      "and PCA are popular algorithms."
        },
        {
            "id": "doc_008",
            "title": "Transfer Learning",
            "content": "Transfer learning applies knowledge gained from one task to improve "
                      "performance on a related task. It's especially useful when training "
                      "data is limited."
        },
        {
            "id": "doc_009",
            "title": "Generative AI Models",
            "content": "Generative AI creates new content like text, images, and code. Large "
                      "language models like GPT and image generators like DALL-E are examples "
                      "of generative AI."
        },
        {
            "id": "doc_010",
            "title": "RAG Systems",
            "content": "Retrieval-Augmented Generation combines information retrieval with "
                      "language model generation. It grounds LLM responses in retrieved "
                      "documents to reduce hallucination."
        },
        {
            "id": "doc_011",
            "title": "Vector Databases",
            "content": "Vector databases store and retrieve high-dimensional embeddings "
                      "efficiently. They enable semantic search by finding similar vectors "
                      "using distance metrics."
        },
        {
            "id": "doc_012",
            "title": "BM25 Ranking Algorithm",
            "content": "BM25 is a probabilistic ranking function used in information retrieval. "
                      "It ranks documents based on query term frequency and document length "
                      "normalization."
        },
        {
            "id": "doc_013",
            "title": "Embeddings and Semantic Search",
            "content": "Embeddings represent text as dense vectors that capture semantic meaning. "
                      "Semantic search uses these embeddings to find conceptually similar "
                      "documents beyond keyword matching."
        },
        {
            "id": "doc_014",
            "title": "Hybrid Search Systems",
            "content": "Hybrid search combines keyword-based and semantic search approaches. "
                      "It leverages the precision of keyword matching with the semantic "
                      "understanding of vector search."
        },
        {
            "id": "doc_015",
            "title": "Evaluation Metrics for Search",
            "content": "Search systems are evaluated using metrics like precision, recall, "
                      "NDCG, and MRR. Modern approaches also use LLM-as-a-Judge for semantic "
                      "relevance assessment."
        }
    ],
    
    "test_queries": [
        {
            "query": "How do computers learn from data?",
            "ground_truth_doc_ids": ["doc_001", "doc_006"],
            "query_type": "conceptual"
        },
        {
            "query": "neural networks with multiple layers",
            "ground_truth_doc_ids": ["doc_002"],
            "query_type": "exact_match"
        },
        {
            "query": "AI that understands human language",
            "ground_truth_doc_ids": ["doc_003"],
            "query_type": "semantic"
        },
        {
            "query": "learning through rewards and penalties",
            "ground_truth_doc_ids": ["doc_005"],
            "query_type": "semantic"
        },
        {
            "query": "BM25 algorithm",
            "ground_truth_doc_ids": ["doc_012"],
            "query_type": "exact_match"
        },
        {
            "query": "combining retrieval with generation",
            "ground_truth_doc_ids": ["doc_010"],
            "query_type": "semantic"
        },
        {
            "query": "semantic search embeddings",
            "ground_truth_doc_ids": ["doc_013", "doc_011"],
            "query_type": "mixed"
        },
        {
            "query": "keyword and vector search together",
            "ground_truth_doc_ids": ["doc_014"],
            "query_type": "semantic"
        }
    ]
}


# Implementation Template

IMPLEMENTATION_TEMPLATE = """
# Hybrid Search System Evaluation Challenge
# Your Name
# Date

## Part 1: BM25 Implementation

```python
from rank_bm25 import BM25Okapi
import numpy as np

def implement_bm25_search(query: str, documents: List[Dict]) -> List[Dict]:
    '''
    Implement BM25 search.
    
    TODO:
    1. Tokenize documents
    2. Create BM25 index
    3. Search and rank results
    4. Return top 10 results with scores
    '''
    pass
```

## Part 2: Vector Search Implementation

```python
from sentence_transformers import SentenceTransformer

def implement_vector_search(query: str, documents: List[Dict]) -> List[Dict]:
    '''
    Implement semantic search using embeddings.
    
    TODO:
    1. Load NVIDIA NV-Embed-v2 or similar model
    2. Embed query and documents
    3. Compute cosine similarity
    4. Return top 10 results with scores
    '''
    pass
```

## Part 3: Hybrid Fusion

```python
def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    vector_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    '''
    Combine results using RRF.
    
    TODO:
    1. Implement RRF formula: score = Σ 1/(k + rank)
    2. Combine scores from both systems
    3. Re-rank by combined score
    4. Return top 10 results
    '''
    pass
```

## Part 4: Evaluation

```python
from src.evaluation.semantic_search import SemanticSearchEvaluator

def evaluate_all_systems():
    '''
    Evaluate BM25, Vector, and Hybrid systems.
    
    TODO:
    1. Run all systems on test queries
    2. Compute precision, recall, relevance
    3. Use LLM-as-a-Judge for semantic assessment
    4. Create comparison table
    '''
    pass
```

## Part 5: Analysis and Recommendations

'''
TODO:
1. Analyze which system performs best overall
2. Identify query types where each system excels
3. Provide optimization recommendations
4. Estimate expected impact of improvements
'''
"""


def display_challenge():
    """Display challenge specification."""
    print("=" * 80)
    print(f"HANDS-ON CHALLENGE: {CHALLENGE_SPEC.title}")
    print("=" * 80)
    
    print(f"\n{CHALLENGE_SPEC.description}")
    
    print(f"\nEstimated Time: {CHALLENGE_SPEC.estimated_time}")
    
    print("\nLearning Objectives:")
    for i, obj in enumerate(CHALLENGE_SPEC.objectives, 1):
        print(f"  {i}. {obj}")
    
    print("\nRequirements:")
    for i, req in enumerate(CHALLENGE_SPEC.requirements, 1):
        print(f"  {i}. {req}")
    
    print("\nDeliverables:")
    for i, deliv in enumerate(CHALLENGE_SPEC.deliverables, 1):
        print(f"  {i}. {deliv}")
    
    print("\nEvaluation Rubric:")
    for category, criteria in CHALLENGE_SPEC.evaluation_rubric.items():
        print(f"\n{category}:")
        for criterion, points in criteria.items():
            print(f"  - {criterion}: {points}")
    
    print("\nBonus Tasks:")
    for bonus in CHALLENGE_SPEC.bonus_tasks:
        print(f"  {bonus}")
    
    print("\n" + "=" * 80)
    print("Dataset Information:")
    print(f"  Documents: {len(CHALLENGE_DATASET['documents'])}")
    print(f"  Test Queries: {len(CHALLENGE_DATASET['test_queries'])}")
    print("=" * 80)


def get_challenge_dataset() -> Dict:
    """
    Get the challenge dataset.
    
    Returns:
        Dictionary with documents and test queries
    """
    return CHALLENGE_DATASET


def get_implementation_template() -> str:
    """
    Get the implementation template.
    
    Returns:
        String with code template
    """
    return IMPLEMENTATION_TEMPLATE


if __name__ == "__main__":
    display_challenge()
    
    print("\n\nTo get started:")
    print("1. Copy the implementation template")
    print("2. Load the challenge dataset")
    print("3. Implement each component step by step")
    print("4. Test thoroughly before final evaluation")
    print("5. Document your findings and recommendations")
