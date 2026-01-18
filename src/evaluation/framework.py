"""
RAG Evaluation Framework using Ragas and custom metrics.

This module provides the EvaluationFramework class for comprehensive RAG
system evaluation, including metric customization and analysis.

Requirements: 7.2, 7.5, 7.6
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
import logging
from src.models.evaluation import (
    EvaluationResults,
    DetailedResult,
    AnalysisReport,
    CustomMetric,
    MetricDefinition
)

logger = logging.getLogger(__name__)


@dataclass
class TestSet:
    """Test set for RAG evaluation."""
    questions: List[str]
    contexts: List[List[str]]  # List of context lists (multiple docs per question)
    responses: List[str]
    ground_truths: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class EvaluationFramework:
    """
    Comprehensive RAG evaluation framework using Ragas.
    
    This class provides methods to:
    - Evaluate RAG systems using standard Ragas metrics
    - Customize existing metrics with prompt modifications
    - Create custom metrics from scratch
    - Analyze results and provide actionable insights
    
    Example:
        framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
        results = framework.evaluate_rag(test_set, rag_pipeline)
        analysis = framework.analyze_results(results)
    """

    
    def __init__(
        self,
        llm_endpoint: str = "nvidia/llama-3-70b",
        embedding_model: str = "nvidia/nv-embed-v2",
        framework: str = "ragas",
        api_key: Optional[str] = None
    ):
        """
        Initialize the evaluation framework.
        
        Args:
            llm_endpoint: LLM endpoint for LLM-as-a-Judge (NVIDIA NIM)
            embedding_model: Embedding model for similarity metrics
            framework: Evaluation framework to use (default: "ragas")
            api_key: API key for LLM provider
        """
        self.llm_endpoint = llm_endpoint
        self.embedding_model = embedding_model
        self.framework = framework
        self.api_key = api_key
        self.custom_metrics: Dict[str, CustomMetric] = {}
        
        logger.info(f"Initialized EvaluationFramework with {framework}")
    
    def evaluate_rag(
        self,
        test_set: TestSet,
        rag_pipeline: Optional[Any] = None,
        metrics: Optional[List[str]] = None
    ) -> EvaluationResults:
        """
        Evaluate a RAG system using Ragas metrics.
        
        This method runs comprehensive evaluation including:
        - Generation metrics: faithfulness, answer_relevancy, context_utilization
        - Retrieval metrics: context_precision, context_recall, context_relevance
        
        Args:
            test_set: Test dataset with questions, contexts, responses, ground_truths
            rag_pipeline: Optional RAG pipeline to evaluate (if not provided, uses test_set responses)
            metrics: List of metric names to compute (if None, uses all available)
        
        Returns:
            EvaluationResults with aggregated scores and detailed per-sample results
        
        Example:
            test_set = TestSet(
                questions=["What is RAG?"],
                contexts=[["RAG combines retrieval with generation..."]],
                responses=["RAG is a technique that..."],
                ground_truths=["RAG stands for..."]
            )
            results = framework.evaluate_rag(test_set)
        """
        if metrics is None:
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_utilization",
                "context_precision",
                "context_recall",
                "context_relevance"
            ]
        
        logger.info(f"Starting RAG evaluation with metrics: {metrics}")
        
        # If RAG pipeline provided, generate responses
        if rag_pipeline is not None:
            responses = []
            contexts = []
            for question in test_set.questions:
                result = rag_pipeline.process_query(question)
                responses.append(result.response)
                contexts.append(result.contexts)
            test_set.responses = responses
            test_set.contexts = contexts
        
        # Compute metrics for each sample
        detailed_results = []
        for i in range(len(test_set.questions)):
            sample_scores = {}
            
            # Generation metrics
            if "faithfulness" in metrics:
                sample_scores["faithfulness"] = self._compute_faithfulness(
                    test_set.responses[i],
                    test_set.contexts[i]
                )
            
            if "answer_relevancy" in metrics:
                sample_scores["answer_relevancy"] = self._compute_answer_relevancy(
                    test_set.questions[i],
                    test_set.responses[i]
                )
            
            if "context_utilization" in metrics:
                sample_scores["context_utilization"] = self._compute_context_utilization(
                    test_set.responses[i],
                    test_set.contexts[i]
                )
            
            # Retrieval metrics
            if "context_precision" in metrics and test_set.ground_truths:
                sample_scores["context_precision"] = self._compute_context_precision(
                    test_set.questions[i],
                    test_set.contexts[i],
                    test_set.ground_truths[i]
                )
            
            if "context_recall" in metrics and test_set.ground_truths:
                sample_scores["context_recall"] = self._compute_context_recall(
                    test_set.contexts[i],
                    test_set.ground_truths[i]
                )
            
            if "context_relevance" in metrics:
                sample_scores["context_relevance"] = self._compute_context_relevance(
                    test_set.questions[i],
                    test_set.contexts[i]
                )
            
            detailed_results.append(DetailedResult(
                sample_id=f"sample_{i:03d}",
                query=test_set.questions[i],
                response=test_set.responses[i],
                context=" ".join(test_set.contexts[i]),
                ground_truth=test_set.ground_truths[i] if test_set.ground_truths else None,
                metric_scores=sample_scores
            ))
        
        # Aggregate scores
        aggregated_metrics = {}
        for metric in metrics:
            scores = [r.metric_scores.get(metric, 0.0) for r in detailed_results if metric in r.metric_scores]
            if scores:
                aggregated_metrics[metric] = sum(scores) / len(scores)
        
        # Generate summary
        summary = self._generate_summary(aggregated_metrics)
        
        results = EvaluationResults(
            evaluation_id=f"eval_{len(test_set.questions)}samples",
            metrics=aggregated_metrics,
            detailed_results=detailed_results,
            summary=summary,
            recommendations=[],
            metadata={
                "framework": self.framework,
                "llm_endpoint": self.llm_endpoint,
                "num_samples": len(test_set.questions)
            }
        )
        
        logger.info(f"Evaluation complete. Average scores: {aggregated_metrics}")
        return results

    
    def customize_metric(
        self,
        metric_name: str,
        custom_prompt: str,
        scoring_rubric: Optional[Dict[str, Any]] = None
    ) -> CustomMetric:
        """
        Customize an existing metric by modifying its prompt template.
        
        This allows you to adapt standard metrics for domain-specific evaluation
        without creating entirely new metrics from scratch.
        
        Args:
            metric_name: Name of the metric to customize (e.g., "faithfulness")
            custom_prompt: Modified prompt template for evaluation
            scoring_rubric: Optional custom scoring rubric
        
        Returns:
            CustomMetric object with the modified configuration
        
        Example:
            # Customize faithfulness for medical domain
            custom_faithfulness = framework.customize_metric(
                metric_name="faithfulness",
                custom_prompt='''
                Evaluate if the medical response is supported by clinical context.
                Consider medical terminology accuracy and clinical guidelines.
                
                Response: {response}
                Context: {context}
                
                Score 0-1 where 1 means fully supported by medical evidence.
                ''',
                scoring_rubric={"0": "Not supported", "0.5": "Partially", "1": "Fully supported"}
            )
        """
        logger.info(f"Customizing metric: {metric_name}")
        
        if scoring_rubric is None:
            scoring_rubric = {"0": "Poor", "0.5": "Moderate", "1": "Excellent"}
        
        custom_metric = CustomMetric(
            metric_id=f"{metric_name}_custom",
            name=f"Custom {metric_name.title()}",
            prompt_template=custom_prompt,
            scoring_rubric=scoring_rubric,
            output_format="json"
        )
        
        self.custom_metrics[custom_metric.metric_id] = custom_metric
        logger.info(f"Created custom metric: {custom_metric.metric_id}")
        
        return custom_metric
    
    def create_custom_metric(
        self,
        definition: MetricDefinition
    ) -> CustomMetric:
        """
        Create a completely new custom metric from scratch.
        
        This allows you to define domain-specific evaluation criteria that
        aren't covered by standard Ragas metrics.
        
        Args:
            definition: MetricDefinition with name, criteria, prompt, scoring method
        
        Returns:
            CustomMetric object ready to use in evaluation
        
        Example:
            # Create custom metric for code quality
            code_quality_def = MetricDefinition(
                name="Code Quality",
                description="Evaluates code correctness and best practices",
                evaluation_criteria=[
                    "Syntax correctness",
                    "Follows best practices",
                    "Proper error handling",
                    "Clear variable names"
                ],
                prompt_template='''
                Evaluate the code quality of this response:
                
                Code: {response}
                
                Criteria:
                1. Syntax correctness (0-0.25)
                2. Best practices (0-0.25)
                3. Error handling (0-0.25)
                4. Code clarity (0-0.25)
                
                Return JSON: {{"score": 0.0-1.0, "reasoning": "..."}}
                ''',
                scoring_method="llm"
            )
            
            code_metric = framework.create_custom_metric(code_quality_def)
        """
        logger.info(f"Creating custom metric: {definition.name}")
        
        # Build scoring rubric from criteria
        scoring_rubric = {
            str(i / len(definition.evaluation_criteria)): criterion
            for i, criterion in enumerate(definition.evaluation_criteria)
        }
        
        custom_metric = CustomMetric(
            metric_id=definition.name.lower().replace(" ", "_"),
            name=definition.name,
            prompt_template=definition.prompt_template,
            scoring_rubric=scoring_rubric,
            output_format="json"
        )
        
        self.custom_metrics[custom_metric.metric_id] = custom_metric
        logger.info(f"Created custom metric: {custom_metric.metric_id}")
        
        return custom_metric
    
    def analyze_results(
        self,
        results: EvaluationResults
    ) -> AnalysisReport:
        """
        Analyze evaluation results and provide actionable insights.
        
        This method examines metric patterns to:
        - Identify system strengths and weaknesses
        - Diagnose common failure modes
        - Suggest specific optimization strategies
        - Prioritize actions based on impact
        
        Args:
            results: EvaluationResults from evaluate_rag()
        
        Returns:
            AnalysisReport with findings, suggestions, and priority actions
        
        Example:
            results = framework.evaluate_rag(test_set)
            analysis = framework.analyze_results(results)
            
            print("Key Findings:")
            for finding in analysis.key_findings:
                print(f"  - {finding}")
            
            print("\\nPriority Actions:")
            for action in analysis.priority_actions:
                print(f"  - {action}")
        """
        logger.info("Analyzing evaluation results")
        
        metrics = results.metrics
        key_findings = []
        strengths = []
        weaknesses = []
        optimization_suggestions = []
        priority_actions = []
        
        # Analyze generation metrics
        faithfulness = metrics.get("faithfulness", 0.0)
        relevancy = metrics.get("answer_relevancy", 0.0)
        utilization = metrics.get("context_utilization", 0.0)
        
        # Analyze retrieval metrics
        precision = metrics.get("context_precision", 0.0)
        recall = metrics.get("context_recall", 0.0)
        relevance = metrics.get("context_relevance", 0.0)
        
        # Pattern: High faithfulness + Low relevancy
        if faithfulness > 0.8 and relevancy < 0.6:
            key_findings.append(
                "High faithfulness but low relevancy suggests LLM is being too conservative"
            )
            weaknesses.append("Response relevancy needs improvement")
            optimization_suggestions.append({
                "component": "generation",
                "suggestion": "Adjust generation prompt to be more direct and comprehensive",
                "expected_impact": "Improve relevancy by 15-20%"
            })
            priority_actions.append("Review and strengthen generation prompts")
        
        # Pattern: Low faithfulness + High relevancy
        if faithfulness < 0.7 and relevancy > 0.8:
            key_findings.append(
                "Low faithfulness with high relevancy indicates hallucination issues"
            )
            weaknesses.append("LLM is hallucinating information")
            optimization_suggestions.append({
                "component": "generation",
                "suggestion": "Add stronger faithfulness constraints, lower temperature, or try different LLM",
                "expected_impact": "Improve faithfulness by 20-30%"
            })
            priority_actions.append("Strengthen faithfulness constraints in prompts")
            priority_actions.append("Consider using more reliable LLM model")
        
        # Pattern: High precision + Low recall
        if precision > 0.7 and recall < 0.6:
            key_findings.append(
                "High precision but low recall means retrieval is too selective"
            )
            weaknesses.append("Missing important information in retrieval")
            optimization_suggestions.append({
                "component": "retrieval",
                "suggestion": "Increase k (number of documents) or lower similarity threshold",
                "expected_impact": "Improve recall by 15-25%"
            })
            priority_actions.append("Tune retrieval parameters (k, threshold)")
        
        # Pattern: Low precision + High recall
        if precision < 0.6 and recall > 0.7:
            key_findings.append(
                "Low precision with high recall means retrieval is too broad"
            )
            weaknesses.append("Too much irrelevant content in retrieval")
            optimization_suggestions.append({
                "component": "retrieval",
                "suggestion": "Add re-ranking stage or improve embedding model",
                "expected_impact": "Improve precision by 20-30%"
            })
            priority_actions.append("Implement re-ranking or improve embeddings")
        
        # Pattern: Low context utilization
        if utilization < 0.6:
            key_findings.append(
                "Low context utilization suggests LLM is ignoring retrieved context"
            )
            weaknesses.append("LLM not using retrieved context effectively")
            optimization_suggestions.append({
                "component": "generation",
                "suggestion": "Strengthen prompt to emphasize context usage",
                "expected_impact": "Improve utilization by 20-30%"
            })
            priority_actions.append("Modify prompts to emphasize context usage")
        
        # Identify strengths
        if faithfulness > 0.85:
            strengths.append("Excellent faithfulness - minimal hallucination")
        if relevancy > 0.80:
            strengths.append("Strong answer relevancy - responses stay on topic")
        if precision > 0.75:
            strengths.append("Good retrieval precision - relevant documents ranked highly")
        if recall > 0.75:
            strengths.append("Good retrieval recall - comprehensive information coverage")
        
        # Overall assessment
        if not key_findings:
            key_findings.append("Overall good performance across all metrics")
        
        if not priority_actions:
            priority_actions.append("Continue monitoring and maintain current configuration")
        
        report = AnalysisReport(
            report_id=f"analysis_{results.evaluation_id}",
            evaluation_id=results.evaluation_id,
            key_findings=key_findings,
            strengths=strengths,
            weaknesses=weaknesses,
            optimization_suggestions=optimization_suggestions,
            priority_actions=priority_actions
        )
        
        logger.info(f"Analysis complete. Found {len(key_findings)} key findings")
        return report

    
    # Private helper methods for metric computation
    
    def _compute_faithfulness(self, response: str, contexts: List[str]) -> float:
        """
        Compute faithfulness score using multi-stage evaluation.
        
        Steps:
        1. Extract claims from response
        2. Verify each claim against context
        3. Calculate score as (verified_claims / total_claims)
        """
        # Simplified implementation - in production, use LLM-as-a-Judge
        # This is a placeholder that simulates the metric
        
        # For demonstration: check if key phrases from context appear in response
        context_text = " ".join(contexts).lower()
        response_lower = response.lower()
        
        # Simple heuristic: if response contains context phrases, higher faithfulness
        overlap_score = len(set(context_text.split()) & set(response_lower.split())) / max(len(response_lower.split()), 1)
        
        # Normalize to 0-1 range
        return min(overlap_score * 2, 1.0)
    
    def _compute_answer_relevancy(self, question: str, response: str) -> float:
        """
        Compute answer relevancy using embedding similarity.
        
        Formula: cosine_similarity(embed(question), embed(response))
        """
        # Simplified implementation - in production, use actual embeddings
        # This is a placeholder that simulates the metric
        
        question_lower = question.lower()
        response_lower = response.lower()
        
        # Simple heuristic: check if question keywords appear in response
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        
        overlap = len(question_words & response_words)
        relevancy = overlap / max(len(question_words), 1)
        
        return min(relevancy * 1.5, 1.0)
    
    def _compute_context_utilization(self, response: str, contexts: List[str]) -> float:
        """
        Compute context utilization score.
        
        Measures how much of the response comes from the context.
        """
        # Simplified implementation
        context_text = " ".join(contexts).lower()
        response_lower = response.lower()
        
        # Check overlap between response and context
        context_words = set(context_text.split())
        response_words = set(response_lower.split())
        
        overlap = len(context_words & response_words)
        utilization = overlap / max(len(response_words), 1)
        
        return min(utilization * 1.2, 1.0)
    
    def _compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        Compute context precision (ranking quality).
        
        Measures if relevant contexts appear higher in the list.
        """
        # Simplified implementation
        ground_truth_lower = ground_truth.lower()
        
        relevant_count = 0
        for i, context in enumerate(contexts):
            # Check if context is relevant to ground truth
            context_lower = context.lower()
            overlap = len(set(ground_truth_lower.split()) & set(context_lower.split()))
            
            if overlap > 3:  # Threshold for relevance
                relevant_count += 1
        
        precision = relevant_count / max(len(contexts), 1)
        return precision
    
    def _compute_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Compute context recall (information coverage).
        
        Measures if contexts contain all information from ground truth.
        """
        # Simplified implementation
        context_text = " ".join(contexts).lower()
        ground_truth_lower = ground_truth.lower()
        
        # Check how much of ground truth is covered by contexts
        ground_truth_words = set(ground_truth_lower.split())
        context_words = set(context_text.split())
        
        covered = len(ground_truth_words & context_words)
        recall = covered / max(len(ground_truth_words), 1)
        
        return recall
    
    def _compute_context_relevance(self, question: str, contexts: List[str]) -> float:
        """
        Compute context relevance.
        
        Measures percentage of contexts that are relevant to the question.
        """
        # Simplified implementation
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        relevant_count = 0
        for context in contexts:
            context_lower = context.lower()
            context_words = set(context_lower.split())
            
            # Check overlap with question
            overlap = len(question_words & context_words)
            if overlap > 2:  # Threshold for relevance
                relevant_count += 1
        
        relevance = relevant_count / max(len(contexts), 1)
        return relevance
    
    def _generate_summary(self, metrics: Dict[str, float]) -> str:
        """Generate a human-readable summary of evaluation results."""
        summary_parts = []
        
        # Overall assessment
        avg_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        if avg_score > 0.8:
            summary_parts.append("Overall excellent performance across metrics.")
        elif avg_score > 0.6:
            summary_parts.append("Overall good performance with room for improvement.")
        else:
            summary_parts.append("Performance needs significant improvement.")
        
        # Highlight best and worst metrics
        if metrics:
            best_metric = max(metrics.items(), key=lambda x: x[1])
            worst_metric = min(metrics.items(), key=lambda x: x[1])
            
            summary_parts.append(
                f"Strongest metric: {best_metric[0]} ({best_metric[1]:.2f})"
            )
            summary_parts.append(
                f"Weakest metric: {worst_metric[0]} ({worst_metric[1]:.2f})"
            )
        
        return " ".join(summary_parts)


class RagasEvaluator:
    """
    Ragas evaluator with error handling and graceful degradation.
    
    This class wraps Ragas evaluation with robust error handling,
    sample validation, and fallback mechanisms.
    
    Requirements: 7.2
    """
    
    def __init__(
        self,
        llm_endpoint: str = "nvidia/llama-3-70b",
        api_key: Optional[str] = None,
        enable_fallback: bool = True
    ):
        """
        Initialize Ragas evaluator.
        
        Args:
            llm_endpoint: LLM endpoint for evaluation
            api_key: API key for LLM provider
            enable_fallback: Whether to fallback to traditional metrics on errors
        """
        self.llm_endpoint = llm_endpoint
        self.api_key = api_key
        self.enable_fallback = enable_fallback
        self.failed_samples: List[str] = []
        
        logger.info("Initialized RagasEvaluator with error handling")
    
    def evaluate_with_error_handling(
        self,
        test_set: TestSet,
        metrics: List[str]
    ) -> EvaluationResults:
        """
        Evaluate with graceful error handling and sample skipping.
        
        Features:
        - Validates each sample before evaluation
        - Skips invalid samples with logging
        - Falls back to traditional metrics on LLM errors
        - Continues evaluation even if some samples fail
        
        Args:
            test_set: Test dataset
            metrics: List of metrics to compute
        
        Returns:
            EvaluationResults with successful evaluations
        """
        logger.info("Starting evaluation with error handling")
        
        valid_samples = []
        self.failed_samples = []
        
        # Validate samples
        for i in range(len(test_set.questions)):
            if self._validate_sample(
                test_set.questions[i],
                test_set.contexts[i],
                test_set.responses[i]
            ):
                valid_samples.append(i)
            else:
                self.failed_samples.append(f"sample_{i:03d}")
                logger.warning(f"Skipping invalid sample {i}")
        
        logger.info(f"Validated {len(valid_samples)}/{len(test_set.questions)} samples")
        
        # Create filtered test set
        filtered_test_set = TestSet(
            questions=[test_set.questions[i] for i in valid_samples],
            contexts=[test_set.contexts[i] for i in valid_samples],
            responses=[test_set.responses[i] for i in valid_samples],
            ground_truths=[test_set.ground_truths[i] for i in valid_samples] if test_set.ground_truths else None
        )
        
        # Evaluate with error handling
        try:
            framework = EvaluationFramework(
                llm_endpoint=self.llm_endpoint,
                api_key=self.api_key
            )
            results = framework.evaluate_rag(filtered_test_set, metrics=metrics)
            
            # Add metadata about failed samples
            results.metadata["failed_samples"] = self.failed_samples
            results.metadata["success_rate"] = len(valid_samples) / len(test_set.questions)
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            
            if self.enable_fallback:
                logger.info("Falling back to traditional metrics")
                return self._fallback_evaluation(filtered_test_set)
            else:
                raise
    
    def _validate_sample(
        self,
        question: str,
        contexts: List[str],
        response: str
    ) -> bool:
        """
        Validate a single sample.
        
        Checks:
        - Question is not empty
        - At least one context provided
        - Response is not empty
        - All strings are valid
        """
        if not question or not question.strip():
            return False
        
        if not contexts or len(contexts) == 0:
            return False
        
        if not response or not response.strip():
            return False
        
        # Check for valid strings (not just whitespace)
        if not any(ctx.strip() for ctx in contexts):
            return False
        
        return True
    
    def _fallback_evaluation(self, test_set: TestSet) -> EvaluationResults:
        """
        Fallback to traditional metrics when LLM evaluation fails.
        
        Uses simple heuristics instead of LLM-as-a-Judge:
        - Word overlap for faithfulness
        - Keyword matching for relevancy
        - Length ratios for utilization
        """
        logger.info("Using fallback traditional metrics")
        
        # Use simplified evaluation
        framework = EvaluationFramework(llm_endpoint=self.llm_endpoint)
        results = framework.evaluate_rag(test_set)
        
        results.metadata["evaluation_method"] = "fallback_traditional"
        results.summary += " (Note: Used traditional metrics due to LLM evaluation errors)"
        
        return results
