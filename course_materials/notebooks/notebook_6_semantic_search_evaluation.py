"""
Notebook 6: Semantic Search System Evaluation
Evaluating RAG and Semantic Search Systems Course

This notebook demonstrates:
1. Evaluating legacy BM25 search systems with LLM-as-a-Judge
2. Comparing RAG vs. traditional search performance
3. Implementing hybrid system evaluation
4. Debugging search quality issues

Learning Objectives:
- Apply modern evaluation techniques to legacy search systems
- Compare different search paradigms quantitatively
- Identify when to use BM25 vs. semantic search vs. RAG
- Optimize ranking algorithms based on evaluation results

Requirements: 8.6, 10.2
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

# Import evaluation components
from src.evaluation.semantic_search import (
    SemanticSearchEvaluator,
    SearchResult,
    SearchSystemResults,
    SearchSystemType
)
from src.evaluation.framework import TestSet, EvaluationFramework


# ============================================================================
# Part 1: Legacy BM25 System Evaluation
# ============================================================================

def simulate_bm25_search(query: str, corpus: List[str]) -> List[SearchResult]:
    """
    Simulate BM25 search results.
    
    In production, this would call your actual BM25 system (Elasticsearch, Solr, etc.).
    For this notebook, we simulate results based on keyword overlap.
    
    Args:
        query: Search query
        corpus: Document corpus
    
    Returns:
        List of SearchResult objects ranked by BM25 score
    """
    results = []
    query_terms = set(query.lower().split())
    
    for i, doc in enumerate(corpus):
        doc_terms = set(doc.lower().split())
        
        # Simple BM25 approximation: count matching terms
        overlap = len(query_terms & doc_terms)
        
        # Calculate score (simplified BM25)
        if overlap > 0:
            score = overlap / len(query_terms)
            results.append(
                SearchResult(
                    document_id=f"doc_{i}",
                    content=doc,
                    score=score,
                    rank=0  # Will be set after sorting
                )
            )
    
    # Sort by score and assign ranks
    results.sort(key=lambda x: x.score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    
    return results[:10]  # Return top 10


def simulate_vector_search(query: str, corpus: List[str]) -> List[SearchResult]:
    """
    Simulate vector search results.
    
    In production, this would use actual embeddings and vector similarity.
    For this notebook, we simulate semantic matching.
    
    Args:
        query: Search query
        corpus: Document corpus
    
    Returns:
        List of SearchResult objects ranked by semantic similarity
    """
    results = []
    
    # Simulate semantic understanding
    # In reality, this would use embeddings and cosine similarity
    for i, doc in enumerate(corpus):
        # Simulate semantic similarity (random for demo, would be real embeddings)
        # Add some logic to prefer semantically related content
        query_lower = query.lower()
        doc_lower = doc.lower()
        
        # Check for semantic relationships (simplified)
        semantic_score = 0.0
        
        # Exact match gets high score
        if query_lower in doc_lower:
            semantic_score = 0.9 + random.uniform(0, 0.1)
        # Partial match gets medium score
        elif any(term in doc_lower for term in query_lower.split()):
            semantic_score = 0.6 + random.uniform(0, 0.2)
        # Random baseline
        else:
            semantic_score = random.uniform(0.1, 0.4)
        
        results.append(
            SearchResult(
                document_id=f"doc_{i}",
                content=doc,
                score=semantic_score,
                rank=0
            )
        )
    
    # Sort by score and assign ranks
    results.sort(key=lambda x: x.score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    
    return results[:10]


# Sample corpus for evaluation
SAMPLE_CORPUS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing helps computers understand and generate human language.",
    "Computer vision allows machines to interpret and analyze visual information from images.",
    "Reinforcement learning trains agents through rewards and penalties in an environment.",
    "Supervised learning uses labeled data to train predictive models.",
    "Unsupervised learning discovers patterns in unlabeled data through clustering.",
    "Transfer learning applies knowledge from one task to improve performance on another.",
    "Generative AI creates new content like text, images, and code using large models.",
    "RAG combines retrieval with generation to provide accurate, grounded responses.",
    "Vector databases store embeddings for fast semantic search and retrieval.",
    "BM25 is a keyword-based ranking algorithm used in traditional search engines.",
    "Embeddings represent text as dense vectors in high-dimensional space.",
    "Semantic search understands query intent beyond exact keyword matching.",
    "Hybrid search combines keyword and semantic approaches for better results."
]


def exercise_1_evaluate_bm25():
    """
    Exercise 1: Evaluate Legacy BM25 System
    
    Task: Evaluate a BM25 search system using modern LLM-based metrics.
    
    Steps:
    1. Run BM25 search on sample queries
    2. Adapt results for Ragas evaluation
    3. Compute retrieval metrics
    4. Analyze performance
    """
    print("=" * 80)
    print("Exercise 1: Evaluating Legacy BM25 System")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = SemanticSearchEvaluator(
        llm_endpoint="nvidia/llama-3-70b",
        embedding_model="nvidia/nv-embed-v2"
    )
    
    # Sample queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain semantic search"
    ]
    
    # Run BM25 search for each query
    print("\n1. Running BM25 Search...")
    search_results_list = []
    
    for query in queries:
        results = simulate_bm25_search(query, SAMPLE_CORPUS)
        search_results_list.append(
            SearchSystemResults(
                query=query,
                results=results,
                system_type=SearchSystemType.BM25
            )
        )
        
        print(f"\nQuery: {query}")
        print(f"Top 3 Results:")
        for result in results[:3]:
            print(f"  Rank {result.rank}: {result.content[:60]}... (score: {result.score:.3f})")
    
    # Adapt for evaluation
    print("\n2. Adapting BM25 Results for Ragas Evaluation...")
    
    # INTENTIONAL BUG #1: Missing ground truths
    # This will cause recall calculation to fail
    # Students should add ground truths for proper evaluation
    test_set = evaluator.adapt_for_legacy(search_results_list)
    
    print(f"Adapted {len(test_set.questions)} queries for evaluation")
    print(f"Evaluation focus: {test_set.metadata['evaluation_focus']}")
    
    # Evaluate
    print("\n3. Computing Retrieval Metrics...")
    
    # INTENTIONAL BUG #2: Trying to compute recall without ground truths
    # This should be caught and handled gracefully
    try:
        results = evaluator.evaluate_search(
            queries=queries,
            search_results=[sr.results for sr in search_results_list],
            ground_truths=None,  # BUG: Should provide ground truths
            system_type=SearchSystemType.BM25
        )
        
        print("\nEvaluation Results:")
        print(f"Context Precision: {results.metrics.get('context_precision', 0.0):.3f}")
        print(f"Context Recall: {results.metrics.get('context_recall', 0.0):.3f}")
        print(f"Context Relevance: {results.metrics.get('context_relevance', 0.0):.3f}")
        
        print(f"\nSummary: {results.summary}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("Hint: Check if ground truths are provided for recall calculation")
    
    # Analysis
    print("\n4. Analysis:")
    print("BM25 Strengths:")
    print("  - Fast and efficient")
    print("  - Works well for exact keyword matches")
    print("  - No need for embeddings or LLMs")
    
    print("\nBM25 Limitations:")
    print("  - Misses semantic relationships")
    print("  - Vocabulary mismatch problems")
    print("  - Cannot understand synonyms or context")
    
    print("\n" + "=" * 80)


def exercise_2_compare_search_paradigms():
    """
    Exercise 2: Compare BM25 vs. Vector Search
    
    Task: Run the same queries through both systems and compare results.
    
    Steps:
    1. Run BM25 search
    2. Run vector search
    3. Evaluate both systems
    4. Compare performance
    """
    print("=" * 80)
    print("Exercise 2: Comparing BM25 vs. Vector Search")
    print("=" * 80)
    
    evaluator = SemanticSearchEvaluator(llm_endpoint="nvidia/llama-3-70b")
    
    # Test queries with different characteristics
    queries = [
        "machine learning algorithms",  # Exact keyword match
        "AI that learns from experience",  # Semantic/conceptual
        "neural network training"  # Mixed
    ]
    
    # Ground truths for proper evaluation
    ground_truths = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Reinforcement learning trains agents through rewards and penalties in an environment.",
        "Deep learning uses neural networks with multiple layers to process complex patterns."
    ]
    
    print("\n1. Running BM25 Search...")
    bm25_results = []
    for query in queries:
        results = simulate_bm25_search(query, SAMPLE_CORPUS)
        bm25_results.append(results)
        print(f"\nQuery: {query}")
        print(f"Top result: {results[0].content[:60]}...")
    
    print("\n2. Running Vector Search...")
    vector_results = []
    for query in queries:
        results = simulate_vector_search(query, SAMPLE_CORPUS)
        vector_results.append(results)
        print(f"\nQuery: {query}")
        print(f"Top result: {results[0].content[:60]}...")
    
    # Evaluate BM25
    print("\n3. Evaluating BM25...")
    bm25_eval = evaluator.evaluate_search(
        queries=queries,
        search_results=bm25_results,
        ground_truths=ground_truths,
        system_type=SearchSystemType.BM25
    )
    
    print(f"BM25 Metrics:")
    for metric, score in bm25_eval.metrics.items():
        print(f"  {metric}: {score:.3f}")
    
    # Evaluate Vector Search
    print("\n4. Evaluating Vector Search...")
    vector_eval = evaluator.evaluate_search(
        queries=queries,
        search_results=vector_results,
        ground_truths=ground_truths,
        system_type=SearchSystemType.VECTOR_SEARCH
    )
    
    print(f"Vector Search Metrics:")
    for metric, score in vector_eval.metrics.items():
        print(f"  {metric}: {score:.3f}")
    
    # Compare
    print("\n5. Comparison:")
    print(f"{'Metric':<20} {'BM25':<10} {'Vector':<10} {'Winner':<10}")
    print("-" * 50)
    
    for metric in bm25_eval.metrics.keys():
        bm25_score = bm25_eval.metrics[metric]
        vector_score = vector_eval.metrics[metric]
        winner = "BM25" if bm25_score > vector_score else "Vector"
        
        print(f"{metric:<20} {bm25_score:<10.3f} {vector_score:<10.3f} {winner:<10}")
    
    print("\n" + "=" * 80)


def exercise_3_rag_vs_search_comparison():
    """
    Exercise 3: Compare RAG vs. Traditional Search
    
    Task: Evaluate RAG system and compare with traditional search.
    
    Steps:
    1. Simulate RAG system (retrieval + generation)
    2. Evaluate RAG with full metrics
    3. Compare with search-only system
    4. Analyze when to use each approach
    """
    print("=" * 80)
    print("Exercise 3: RAG vs. Traditional Search Comparison")
    print("=" * 80)
    
    evaluator = SemanticSearchEvaluator(llm_endpoint="nvidia/llama-3-70b")
    
    queries = [
        "What is the difference between supervised and unsupervised learning?",
        "How does RAG improve LLM responses?"
    ]
    
    ground_truths = [
        "Supervised learning uses labeled data while unsupervised learning discovers patterns in unlabeled data.",
        "RAG combines retrieval with generation to provide accurate, grounded responses using retrieved context."
    ]
    
    # Simulate traditional search
    print("\n1. Traditional Search Results...")
    search_results = []
    for query in queries:
        results = simulate_vector_search(query, SAMPLE_CORPUS)
        search_results.append(results)
        print(f"\nQuery: {query}")
        print(f"Top result: {results[0].content}")
    
    # Simulate RAG system (retrieval + generation)
    print("\n2. RAG System Results...")
    rag_responses = [
        "Supervised learning uses labeled training data where each example has a known output, "
        "allowing the model to learn the mapping from inputs to outputs. Unsupervised learning "
        "works with unlabeled data and discovers hidden patterns through techniques like clustering.",
        
        "RAG (Retrieval-Augmented Generation) improves LLM responses by first retrieving relevant "
        "documents from a knowledge base, then using those documents as context for generation. "
        "This grounds the response in factual information and reduces hallucination."
    ]
    
    # INTENTIONAL BUG #3: Wrong metric selection for comparison
    # Using generation metrics for search-only system
    # Students should identify that search systems don't have generation metrics
    
    # Evaluate traditional search
    print("\n3. Evaluating Traditional Search...")
    search_eval = evaluator.evaluate_search(
        queries=queries,
        search_results=search_results,
        ground_truths=ground_truths,
        system_type=SearchSystemType.VECTOR_SEARCH
    )
    
    # Evaluate RAG system
    print("\n4. Evaluating RAG System...")
    
    # Create RAG test set
    rag_test_set = TestSet(
        questions=queries,
        contexts=[[r.content for r in results] for results in search_results],
        responses=rag_responses,
        ground_truths=ground_truths
    )
    
    eval_framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
    rag_eval = eval_framework.evaluate_rag(rag_test_set)
    
    print(f"\nRAG Metrics:")
    for metric, score in rag_eval.metrics.items():
        print(f"  {metric}: {score:.3f}")
    
    # Compare
    print("\n5. Comparison Analysis...")
    comparison = evaluator.compare_with_rag(search_eval, rag_eval)
    
    print(f"\nOverall Winner: {comparison.overall_winner.upper()}")
    
    print("\nKey Differences:")
    for diff in comparison.key_differences:
        print(f"  - {diff}")
    
    print("\nRecommendations:")
    for rec in comparison.recommendations:
        print(f"  - {rec}")
    
    print("\n" + "=" * 80)


def exercise_4_hybrid_system_evaluation():
    """
    Exercise 4: Hybrid System Evaluation
    
    Task: Evaluate a hybrid system combining BM25 and vector search.
    
    Steps:
    1. Implement result fusion (combine BM25 + vector search)
    2. Evaluate hybrid system
    3. Compare with individual systems
    4. Analyze when hybrid approach is beneficial
    """
    print("=" * 80)
    print("Exercise 4: Hybrid System Evaluation")
    print("=" * 80)
    
    evaluator = SemanticSearchEvaluator(llm_endpoint="nvidia/llama-3-70b")
    
    queries = [
        "deep learning neural networks",
        "AI learns from data"
    ]
    
    ground_truths = [
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    ]
    
    print("\n1. Running Individual Systems...")
    
    # BM25 results
    bm25_results = [simulate_bm25_search(q, SAMPLE_CORPUS) for q in queries]
    
    # Vector search results
    vector_results = [simulate_vector_search(q, SAMPLE_CORPUS) for q in queries]
    
    # Hybrid: Combine using Reciprocal Rank Fusion (RRF)
    print("\n2. Implementing Hybrid Fusion (RRF)...")
    
    def reciprocal_rank_fusion(
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score(doc) = Σ 1/(k + rank(doc))
        """
        scores = {}
        
        # Add BM25 scores
        for result in bm25_results:
            doc_id = result.document_id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + result.rank)
        
        # Add vector search scores
        for result in vector_results:
            doc_id = result.document_id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + result.rank)
        
        # Create combined results
        combined = []
        doc_content = {r.document_id: r.content for r in bm25_results + vector_results}
        
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            combined.append(
                SearchResult(
                    document_id=doc_id,
                    content=doc_content[doc_id],
                    score=score,
                    rank=len(combined) + 1
                )
            )
        
        return combined[:10]
    
    hybrid_results = []
    for i in range(len(queries)):
        fused = reciprocal_rank_fusion(bm25_results[i], vector_results[i])
        hybrid_results.append(fused)
        
        print(f"\nQuery: {queries[i]}")
        print(f"Hybrid top result: {fused[0].content[:60]}...")
    
    # Evaluate all three systems
    print("\n3. Evaluating All Systems...")
    
    bm25_eval = evaluator.evaluate_search(
        queries, bm25_results, ground_truths, SearchSystemType.BM25
    )
    
    vector_eval = evaluator.evaluate_search(
        queries, vector_results, ground_truths, SearchSystemType.VECTOR_SEARCH
    )
    
    hybrid_eval = evaluator.evaluate_search(
        queries, hybrid_results, ground_truths, SearchSystemType.HYBRID
    )
    
    # Compare
    print("\n4. Three-Way Comparison:")
    print(f"{'Metric':<20} {'BM25':<10} {'Vector':<10} {'Hybrid':<10}")
    print("-" * 60)
    
    for metric in bm25_eval.metrics.keys():
        bm25_score = bm25_eval.metrics[metric]
        vector_score = vector_eval.metrics[metric]
        hybrid_score = hybrid_eval.metrics[metric]
        
        print(f"{metric:<20} {bm25_score:<10.3f} {vector_score:<10.3f} {hybrid_score:<10.3f}")
    
    print("\n5. Analysis:")
    print("Hybrid systems typically excel when:")
    print("  - Queries have both exact keyword and semantic components")
    print("  - You want to leverage strengths of both approaches")
    print("  - Robustness is more important than peak performance")
    
    print("\n" + "=" * 80)


def exercise_5_ranking_optimization():
    """
    Exercise 5: Ranking Optimization
    
    Task: Analyze search results and get optimization suggestions.
    
    Steps:
    1. Evaluate current search system
    2. Identify performance bottlenecks
    3. Get prioritized optimization suggestions
    4. Implement improvements
    """
    print("=" * 80)
    print("Exercise 5: Ranking Optimization")
    print("=" * 80)
    
    evaluator = SemanticSearchEvaluator(llm_endpoint="nvidia/llama-3-70b")
    
    queries = [
        "machine learning",
        "neural networks",
        "semantic search"
    ]
    
    ground_truths = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Semantic search understands query intent beyond exact keyword matching."
    ]
    
    # Simulate a search system with suboptimal performance
    print("\n1. Evaluating Current System...")
    search_results = [simulate_bm25_search(q, SAMPLE_CORPUS) for q in queries]
    
    current_eval = evaluator.evaluate_search(
        queries, search_results, ground_truths, SearchSystemType.BM25
    )
    
    print(f"\nCurrent Performance:")
    for metric, score in current_eval.metrics.items():
        print(f"  {metric}: {score:.3f}")
    
    # Get optimization suggestions
    print("\n2. Generating Optimization Suggestions...")
    suggestions = evaluator.optimize_ranking(current_eval)
    
    print(f"\nIdentified Bottlenecks:")
    for bottleneck in suggestions.bottlenecks:
        print(f"  - {bottleneck}")
    
    print(f"\nPriority Actions:")
    for i, action in enumerate(suggestions.priority_order, 1):
        print(f"  {i}. {action}")
    
    print(f"\nDetailed Suggestions:")
    for improvement in suggestions.suggested_improvements:
        print(f"\nComponent: {improvement['component']}")
        print(f"Issue: {improvement['issue']}")
        print(f"Suggestion: {improvement['suggestion']}")
        print(f"Expected Impact: {improvement['expected_improvement']}")
    
    print("\n3. Implementation Guidance:")
    print("To implement these suggestions:")
    print("  1. Start with highest priority items")
    print("  2. Implement one change at a time")
    print("  3. Re-evaluate after each change")
    print("  4. Use A/B testing in production")
    print("  5. Monitor user engagement metrics")
    
    print("\n" + "=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main notebook execution.
    
    This notebook demonstrates semantic search evaluation techniques
    including legacy system evaluation, paradigm comparison, and
    ranking optimization.
    """
    print("\n" + "=" * 80)
    print("NOTEBOOK 6: SEMANTIC SEARCH SYSTEM EVALUATION")
    print("=" * 80)
    
    print("\nThis notebook covers:")
    print("1. Evaluating legacy BM25 systems with modern techniques")
    print("2. Comparing BM25 vs. vector search vs. RAG")
    print("3. Implementing hybrid system evaluation")
    print("4. Optimizing ranking algorithms")
    
    print("\nNote: This notebook contains intentional bugs for debugging practice:")
    print("  Bug #1: Missing ground truths in BM25 evaluation")
    print("  Bug #2: Attempting recall calculation without ground truths")
    print("  Bug #3: Using wrong metrics for search-only systems")
    
    print("\n" + "=" * 80)
    
    # Run exercises
    try:
        exercise_1_evaluate_bm25()
    except Exception as e:
        print(f"\nExercise 1 encountered an error: {e}")
        print("Debug this error as part of the learning exercise!")
    
    print("\n")
    exercise_2_compare_search_paradigms()
    
    print("\n")
    exercise_3_rag_vs_search_comparison()
    
    print("\n")
    exercise_4_hybrid_system_evaluation()
    
    print("\n")
    exercise_5_ranking_optimization()
    
    print("\n" + "=" * 80)
    print("NOTEBOOK COMPLETE")
    print("=" * 80)
    
    print("\nKey Takeaways:")
    print("1. Legacy systems can be evaluated with modern LLM-based techniques")
    print("2. Different search paradigms excel at different query types")
    print("3. Hybrid approaches often provide best overall performance")
    print("4. Systematic evaluation guides optimization decisions")
    print("5. Always validate improvements with real user metrics")


if __name__ == "__main__":
    main()
