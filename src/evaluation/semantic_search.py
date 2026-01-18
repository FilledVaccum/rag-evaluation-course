"""
Semantic Search System Evaluator

This module provides the SemanticSearchEvaluator class for evaluating legacy
semantic search systems (BM25, Elasticsearch, etc.) using modern LLM-based
evaluation techniques and Ragas framework adaptation.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.evaluation.framework import EvaluationFramework, TestSet
from src.models.evaluation import EvaluationResults, DetailedResult

logger = logging.getLogger(__name__)


class SearchSystemType(Enum):
    """Types of search systems."""
    BM25 = "bm25"
    VECTOR_SEARCH = "vector_search"
    HYBRID = "hybrid"
    RAG = "rag"
    ELASTICSEARCH = "elasticsearch"
    SOLR = "solr"


@dataclass
class SearchResult:
    """Single search result."""
    document_id: str
    content: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchSystemResults:
    """Results from a search system."""
    query: str
    results: List[SearchResult]
    system_type: SearchSystemType
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonReport:
    """Comparison report between search systems."""
    system_a_name: str
    system_b_name: str
    system_a_scores: Dict[str, float]
    system_b_scores: Dict[str, float]
    winner_by_metric: Dict[str, str]
    overall_winner: str
    key_differences: List[str]
    recommendations: List[str]


@dataclass
class OptimizationSuggestions:
    """Optimization suggestions for search system."""
    current_performance: Dict[str, float]
    bottlenecks: List[str]
    suggested_improvements: List[Dict[str, Any]]
    expected_impact: Dict[str, str]
    priority_order: List[str]


class SemanticSearchEvaluator:
    """
    Evaluator for semantic search systems using modern LLM-based techniques.
    
    This class enables:
    - Evaluation of legacy BM25 systems with Ragas-style metrics
    - Comparison between traditional search and RAG systems
    - Hybrid system evaluation
    - Ranking algorithm assessment and optimization
    
    Example:
        evaluator = SemanticSearchEvaluator(llm_endpoint="nvidia/llama-3-70b")
        
        # Adapt legacy BM25 system for evaluation
        adapted_system = evaluator.adapt_for_legacy(bm25_results)
        
        # Evaluate search quality
        results = evaluator.evaluate_search(search_queries, search_results)
        
        # Compare with RAG system
        comparison = evaluator.compare_with_rag(search_results, rag_results)
        
        # Get optimization suggestions
        suggestions = evaluator.optimize_ranking(current_results)
    """
    
    def __init__(
        self,
        llm_endpoint: str = "nvidia/llama-3-70b",
        embedding_model: str = "nvidia/nv-embed-v2",
        api_key: Optional[str] = None
    ):
        """
        Initialize semantic search evaluator.
        
        Args:
            llm_endpoint: LLM endpoint for LLM-as-a-Judge evaluation
            embedding_model: Embedding model for similarity metrics
            api_key: API key for LLM provider
        """
        self.llm_endpoint = llm_endpoint
        self.embedding_model = embedding_model
        self.api_key = api_key
        
        # Initialize evaluation framework for Ragas metrics
        self.eval_framework = EvaluationFramework(
            llm_endpoint=llm_endpoint,
            embedding_model=embedding_model,
            api_key=api_key
        )
        
        logger.info("Initialized SemanticSearchEvaluator")
    
    def adapt_for_legacy(
        self,
        search_results: List[SearchSystemResults],
        ground_truths: Optional[List[str]] = None
    ) -> TestSet:
        """
        Adapt legacy search system results for Ragas evaluation.
        
        This method converts traditional search results into the format expected
        by Ragas, allowing us to apply modern evaluation metrics to legacy systems.
        
        Key Adaptations:
        - Search results → Contexts (retrieved documents)
        - No generation stage (skip faithfulness, answer_relevancy)
        - Focus on retrieval metrics (precision, recall, relevance)
        
        Args:
            search_results: List of search system results
            ground_truths: Optional ground truth answers for recall calculation
        
        Returns:
            TestSet formatted for Ragas evaluation (retrieval metrics only)
        
        Example:
            # BM25 search results
            bm25_results = [
                SearchSystemResults(
                    query="What is machine learning?",
                    results=[
                        SearchResult(doc_id="1", content="ML is...", score=0.9, rank=1),
                        SearchResult(doc_id="2", content="AI includes...", score=0.7, rank=2)
                    ],
                    system_type=SearchSystemType.BM25
                )
            ]
            
            # Adapt for evaluation
            test_set = evaluator.adapt_for_legacy(bm25_results)
            
            # Evaluate using retrieval metrics
            results = evaluator.eval_framework.evaluate_rag(
                test_set,
                metrics=["context_precision", "context_recall", "context_relevance"]
            )
        """
        logger.info(f"Adapting {len(search_results)} search results for Ragas evaluation")
        
        questions = []
        contexts = []
        responses = []  # Empty for search-only systems
        
        for search_result in search_results:
            questions.append(search_result.query)
            
            # Convert search results to contexts
            result_contexts = [
                result.content for result in search_result.results
            ]
            contexts.append(result_contexts)
            
            # For search systems, we don't have generated responses
            # Use top result as "response" for compatibility
            if search_result.results:
                responses.append(search_result.results[0].content)
            else:
                responses.append("")
        
        test_set = TestSet(
            questions=questions,
            contexts=contexts,
            responses=responses,
            ground_truths=ground_truths,
            metadata={
                "adapted_from": "legacy_search",
                "system_type": search_results[0].system_type.value if search_results else "unknown",
                "evaluation_focus": "retrieval_only"
            }
        )
        
        logger.info("Successfully adapted search results to TestSet format")
        return test_set
    
    def evaluate_search(
        self,
        queries: List[str],
        search_results: List[List[SearchResult]],
        ground_truths: Optional[List[str]] = None,
        system_type: SearchSystemType = SearchSystemType.BM25
    ) -> EvaluationResults:
        """
        Evaluate semantic search system using LLM-as-a-Judge.
        
        This method evaluates search quality using:
        - Context Relevance: Are results relevant to the query?
        - Context Precision: Are relevant results ranked highly?
        - Context Recall: Are all relevant documents retrieved?
        
        Args:
            queries: List of search queries
            search_results: List of search results for each query
            ground_truths: Optional ground truth answers
            system_type: Type of search system being evaluated
        
        Returns:
            EvaluationResults with retrieval metrics
        
        Example:
            queries = ["What is RAG?", "How does BM25 work?"]
            results = [
                [SearchResult(doc_id="1", content="RAG combines...", score=0.9, rank=1)],
                [SearchResult(doc_id="2", content="BM25 is...", score=0.8, rank=1)]
            ]
            
            evaluation = evaluator.evaluate_search(queries, results)
            print(f"Context Relevance: {evaluation.metrics['context_relevance']}")
        """
        logger.info(f"Evaluating {len(queries)} search queries")
        
        # Convert to SearchSystemResults format
        search_system_results = []
        for i, query in enumerate(queries):
            search_system_results.append(
                SearchSystemResults(
                    query=query,
                    results=search_results[i],
                    system_type=system_type
                )
            )
        
        # Adapt for Ragas evaluation
        test_set = self.adapt_for_legacy(search_system_results, ground_truths)
        
        # Evaluate using retrieval metrics only
        retrieval_metrics = ["context_precision", "context_recall", "context_relevance"]
        
        results = self.eval_framework.evaluate_rag(
            test_set,
            metrics=retrieval_metrics
        )
        
        # Add search-specific metadata
        results.metadata["system_type"] = system_type.value
        results.metadata["evaluation_type"] = "semantic_search"
        
        # Enhance summary with search-specific insights
        results.summary = self._generate_search_summary(results.metrics, system_type)
        
        logger.info(f"Search evaluation complete: {results.metrics}")
        return results
    
    def compare_with_rag(
        self,
        search_results: EvaluationResults,
        rag_results: EvaluationResults
    ) -> ComparisonReport:
        """
        Compare traditional search system with RAG system.
        
        This method provides detailed comparison showing:
        - Metric-by-metric comparison
        - Strengths and weaknesses of each system
        - Recommendations for when to use each approach
        
        Args:
            search_results: Evaluation results from traditional search
            rag_results: Evaluation results from RAG system
        
        Returns:
            ComparisonReport with detailed analysis
        
        Example:
            # Evaluate both systems
            search_eval = evaluator.evaluate_search(queries, bm25_results)
            rag_eval = evaluator.eval_framework.evaluate_rag(rag_test_set)
            
            # Compare
            comparison = evaluator.compare_with_rag(search_eval, rag_eval)
            
            print(f"Overall Winner: {comparison.overall_winner}")
            print("Key Differences:")
            for diff in comparison.key_differences:
                print(f"  - {diff}")
        """
        logger.info("Comparing search system with RAG system")
        
        search_metrics = search_results.metrics
        rag_metrics = rag_results.metrics
        
        # Determine winner for each metric
        winner_by_metric = {}
        for metric in search_metrics.keys():
            if metric in rag_metrics:
                search_score = search_metrics[metric]
                rag_score = rag_metrics[metric]
                
                if search_score > rag_score:
                    winner_by_metric[metric] = "search"
                elif rag_score > search_score:
                    winner_by_metric[metric] = "rag"
                else:
                    winner_by_metric[metric] = "tie"
        
        # Calculate overall winner
        search_wins = sum(1 for w in winner_by_metric.values() if w == "search")
        rag_wins = sum(1 for w in winner_by_metric.values() if w == "rag")
        
        if search_wins > rag_wins:
            overall_winner = "search"
        elif rag_wins > search_wins:
            overall_winner = "rag"
        else:
            overall_winner = "tie"
        
        # Identify key differences
        key_differences = []
        
        # Retrieval quality comparison
        if "context_relevance" in search_metrics and "context_relevance" in rag_metrics:
            search_rel = search_metrics["context_relevance"]
            rag_rel = rag_metrics["context_relevance"]
            diff = abs(search_rel - rag_rel)
            
            if diff > 0.1:
                better_system = "RAG" if rag_rel > search_rel else "Search"
                key_differences.append(
                    f"{better_system} shows {diff:.1%} better context relevance"
                )
        
        # Precision comparison
        if "context_precision" in search_metrics and "context_precision" in rag_metrics:
            search_prec = search_metrics["context_precision"]
            rag_prec = rag_metrics["context_precision"]
            diff = abs(search_prec - rag_prec)
            
            if diff > 0.1:
                better_system = "RAG" if rag_prec > search_prec else "Search"
                key_differences.append(
                    f"{better_system} has {diff:.1%} better ranking quality (precision)"
                )
        
        # Generation quality (RAG only)
        if "faithfulness" in rag_metrics:
            key_differences.append(
                f"RAG provides generated responses with {rag_metrics['faithfulness']:.1%} faithfulness"
            )
            key_differences.append(
                "Search returns raw documents without synthesis"
            )
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(
            search_metrics,
            rag_metrics,
            overall_winner
        )
        
        report = ComparisonReport(
            system_a_name="Traditional Search",
            system_b_name="RAG System",
            system_a_scores=search_metrics,
            system_b_scores=rag_metrics,
            winner_by_metric=winner_by_metric,
            overall_winner=overall_winner,
            key_differences=key_differences,
            recommendations=recommendations
        )
        
        logger.info(f"Comparison complete. Overall winner: {overall_winner}")
        return report
    
    def optimize_ranking(
        self,
        current_results: EvaluationResults,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationSuggestions:
        """
        Generate optimization suggestions for improving ranking quality.
        
        This method analyzes current performance and suggests specific
        improvements to ranking algorithms, retrieval parameters, and
        system configuration.
        
        Args:
            current_results: Current evaluation results
            target_metrics: Optional target metric values to achieve
        
        Returns:
            OptimizationSuggestions with prioritized action items
        
        Example:
            # Evaluate current system
            results = evaluator.evaluate_search(queries, search_results)
            
            # Get optimization suggestions
            suggestions = evaluator.optimize_ranking(
                results,
                target_metrics={"context_precision": 0.8, "context_relevance": 0.85}
            )
            
            print("Priority Actions:")
            for action in suggestions.priority_order:
                print(f"  - {action}")
        """
        logger.info("Generating ranking optimization suggestions")
        
        metrics = current_results.metrics
        bottlenecks = []
        improvements = []
        expected_impact = {}
        
        # Analyze context precision (ranking quality)
        precision = metrics.get("context_precision", 0.0)
        if precision < 0.7:
            bottlenecks.append("Low ranking quality - relevant documents not appearing at top")
            improvements.append({
                "component": "ranking",
                "issue": "Poor ranking quality",
                "suggestion": "Implement re-ranking with cross-encoder model",
                "implementation": "Add cross-encoder re-ranking stage for top 20-50 results",
                "expected_improvement": "15-25% increase in precision"
            })
            expected_impact["context_precision"] = "+15-25%"
        
        # Analyze context recall (coverage)
        recall = metrics.get("context_recall", 0.0)
        if recall < 0.7:
            bottlenecks.append("Low recall - missing relevant documents")
            improvements.append({
                "component": "retrieval",
                "issue": "Insufficient coverage",
                "suggestion": "Increase k (number of retrieved documents) or lower similarity threshold",
                "implementation": "Increase k from current value to k+10, or reduce threshold by 0.1",
                "expected_improvement": "15-20% increase in recall"
            })
            expected_impact["context_recall"] = "+15-20%"
        
        # Analyze context relevance
        relevance = metrics.get("context_relevance", 0.0)
        if relevance < 0.7:
            bottlenecks.append("Low relevance - retrieved documents not matching query intent")
            improvements.append({
                "component": "embeddings",
                "issue": "Poor semantic matching",
                "suggestion": "Switch to domain-specific embedding model or fine-tune embeddings",
                "implementation": "Use NVIDIA NV-Embed-v2 or domain-specific model for your use case",
                "expected_improvement": "20-30% increase in relevance"
            })
            expected_impact["context_relevance"] = "+20-30%"
        
        # Check for precision-recall tradeoff issues
        if precision > 0.8 and recall < 0.6:
            bottlenecks.append("High precision but low recall - system too selective")
            improvements.append({
                "component": "retrieval_parameters",
                "issue": "Too selective retrieval",
                "suggestion": "Increase retrieval breadth while maintaining precision with re-ranking",
                "implementation": "Increase k by 50%, add re-ranking to maintain precision",
                "expected_improvement": "Balanced precision-recall tradeoff"
            })
            expected_impact["balance"] = "Better precision-recall tradeoff"
        
        if precision < 0.6 and recall > 0.8:
            bottlenecks.append("High recall but low precision - system too broad")
            improvements.append({
                "component": "filtering",
                "issue": "Too broad retrieval",
                "suggestion": "Add stricter filtering or improve ranking algorithm",
                "implementation": "Increase similarity threshold or add metadata filtering",
                "expected_improvement": "Improved precision without sacrificing recall"
            })
            expected_impact["balance"] = "Better precision-recall tradeoff"
        
        # General improvements if performance is moderate
        if 0.6 <= precision < 0.8 or 0.6 <= relevance < 0.8:
            improvements.append({
                "component": "hybrid_search",
                "issue": "Moderate performance",
                "suggestion": "Implement hybrid search combining BM25 and vector search",
                "implementation": "Use Reciprocal Rank Fusion (RRF) to combine BM25 and semantic search",
                "expected_improvement": "10-15% overall improvement"
            })
            expected_impact["overall"] = "+10-15%"
        
        # Prioritize improvements by impact
        priority_order = []
        
        # High priority: Low relevance (fundamental issue)
        if relevance < 0.7:
            priority_order.append("Improve embedding model for better semantic matching")
        
        # High priority: Low precision (user experience issue)
        if precision < 0.7:
            priority_order.append("Implement re-ranking to improve result quality")
        
        # Medium priority: Low recall (completeness issue)
        if recall < 0.7:
            priority_order.append("Increase retrieval breadth (k or threshold)")
        
        # Medium priority: Imbalanced precision-recall
        if (precision > 0.8 and recall < 0.6) or (precision < 0.6 and recall > 0.8):
            priority_order.append("Balance precision-recall tradeoff")
        
        # Low priority: General improvements
        if 0.6 <= precision < 0.8:
            priority_order.append("Consider hybrid search for incremental gains")
        
        # If no specific issues, suggest advanced optimizations
        if not priority_order:
            priority_order.append("Implement learning-to-rank with user feedback")
            priority_order.append("Add personalization based on user context")
            priority_order.append("Optimize for latency-quality tradeoff")
        
        suggestions = OptimizationSuggestions(
            current_performance=metrics,
            bottlenecks=bottlenecks,
            suggested_improvements=improvements,
            expected_impact=expected_impact,
            priority_order=priority_order
        )
        
        logger.info(f"Generated {len(improvements)} optimization suggestions")
        return suggestions
    
    # Private helper methods
    
    def _generate_search_summary(
        self,
        metrics: Dict[str, float],
        system_type: SearchSystemType
    ) -> str:
        """Generate search-specific summary."""
        summary_parts = []
        
        # System type
        summary_parts.append(f"{system_type.value.upper()} Search System Evaluation:")
        
        # Overall assessment
        avg_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        if avg_score > 0.8:
            summary_parts.append("Excellent retrieval performance.")
        elif avg_score > 0.6:
            summary_parts.append("Good retrieval performance with room for optimization.")
        else:
            summary_parts.append("Retrieval performance needs improvement.")
        
        # Specific insights
        precision = metrics.get("context_precision", 0.0)
        recall = metrics.get("context_recall", 0.0)
        relevance = metrics.get("context_relevance", 0.0)
        
        if precision > 0.8:
            summary_parts.append("Strong ranking quality.")
        elif precision < 0.6:
            summary_parts.append("Ranking quality needs improvement - consider re-ranking.")
        
        if recall > 0.8:
            summary_parts.append("Comprehensive coverage.")
        elif recall < 0.6:
            summary_parts.append("Coverage gaps - consider increasing k or lowering threshold.")
        
        if relevance > 0.8:
            summary_parts.append("High semantic relevance.")
        elif relevance < 0.6:
            summary_parts.append("Relevance issues - consider better embeddings or hybrid approach.")
        
        return " ".join(summary_parts)
    
    def _generate_comparison_recommendations(
        self,
        search_metrics: Dict[str, float],
        rag_metrics: Dict[str, float],
        overall_winner: str
    ) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []
        
        if overall_winner == "rag":
            recommendations.append(
                "RAG system shows better overall performance - consider migration"
            )
            recommendations.append(
                "RAG provides synthesized responses, better for question-answering use cases"
            )
            
            # Check if search has any advantages
            search_precision = search_metrics.get("context_precision", 0.0)
            rag_precision = rag_metrics.get("context_precision", 0.0)
            
            if search_precision > rag_precision:
                recommendations.append(
                    "Traditional search has better ranking - consider hybrid approach"
                )
        
        elif overall_winner == "search":
            recommendations.append(
                "Traditional search performs well - RAG may not be necessary"
            )
            recommendations.append(
                "Consider RAG only if you need synthesized responses"
            )
            
            # Check if RAG has any advantages
            rag_relevance = rag_metrics.get("context_relevance", 0.0)
            search_relevance = search_metrics.get("context_relevance", 0.0)
            
            if rag_relevance > search_relevance:
                recommendations.append(
                    "RAG shows better semantic understanding - consider for complex queries"
                )
        
        else:  # tie
            recommendations.append(
                "Both systems perform similarly - choose based on use case requirements"
            )
            recommendations.append(
                "Use traditional search for document retrieval, RAG for question-answering"
            )
            recommendations.append(
                "Consider hybrid approach combining strengths of both systems"
            )
        
        # General recommendations
        recommendations.append(
            "Monitor user satisfaction and engagement metrics in production"
        )
        recommendations.append(
            "Implement A/B testing to validate improvements with real users"
        )
        
        return recommendations
