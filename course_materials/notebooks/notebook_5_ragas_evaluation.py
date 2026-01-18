"""
Notebook 5: Ragas Evaluation Implementation

This notebook demonstrates comprehensive RAG evaluation using the Ragas framework.
Students will learn to:
- Implement Ragas on the Amnesty Q&A dataset
- Compute faithfulness and context recall metrics
- Customize existing metrics with prompt modifications
- Create custom metrics from scratch
- Debug intentional bugs in metric prompts

Dataset: Amnesty Q&A (pre-formatted for Ragas)
Requirements: 7.7, 10.2

Learning Objectives:
1. Understand Ragas framework architecture and usage
2. Compute and interpret generation and retrieval metrics
3. Customize metrics for domain-specific evaluation
4. Create custom metrics from scratch
5. Debug common metric implementation issues
"""

# Standard library imports
import json
from typing import List, Dict, Any
from dataclasses import dataclass

# Course imports
from src.evaluation.framework import EvaluationFramework, RagasEvaluator, TestSet
from src.models.evaluation import MetricDefinition


# ============================================================================
# Part 1: Dataset Loading and Preparation
# ============================================================================

def load_amnesty_dataset() -> TestSet:
    """
    Load the Amnesty Q&A dataset pre-formatted for Ragas.
    
    The dataset contains:
    - user_input: Questions about human rights
    - retrieved_context: Relevant documents from Amnesty reports
    - response: Generated answers from RAG system
    - ground_truth: Reference answers for evaluation
    
    Returns:
        TestSet ready for Ragas evaluation
    """
    # Sample Amnesty Q&A data (in production, load from file)
    sample_data = [
        {
            "user_input": "What are the main human rights concerns in Country X?",
            "retrieved_context": [
                "Amnesty International documented widespread arbitrary detention in Country X.",
                "Freedom of expression is severely restricted with journalists facing harassment.",
                "The government has implemented surveillance programs affecting privacy rights."
            ],
            "response": "The main human rights concerns in Country X include arbitrary detention, "
                       "restrictions on freedom of expression with journalists being harassed, "
                       "and privacy violations through government surveillance programs.",
            "ground_truth": "Country X faces issues with arbitrary detention, freedom of expression "
                          "restrictions, and privacy rights violations."
        },
        {
            "user_input": "How does Amnesty International monitor human rights violations?",
            "retrieved_context": [
                "Amnesty International uses field research and interviews with victims.",
                "The organization analyzes government documents and legal frameworks.",
                "Satellite imagery is used to verify reports of mass atrocities."
            ],
            "response": "Amnesty International monitors violations through field research, "
                       "victim interviews, analysis of government documents, and satellite imagery "
                       "to verify reports.",
            "ground_truth": "Amnesty monitors through field research, document analysis, "
                          "interviews, and satellite verification."
        },
        {
            "user_input": "What is the Universal Declaration of Human Rights?",
            "retrieved_context": [
                "The Universal Declaration of Human Rights was adopted by the UN in 1948.",
                "It consists of 30 articles outlining fundamental human rights.",
                "The declaration is not legally binding but has moral authority."
            ],
            "response": "The Universal Declaration of Human Rights is a UN document from 1948 "
                       "with 30 articles on fundamental rights. While not legally binding, "
                       "it has significant moral authority globally.",
            "ground_truth": "A 1948 UN declaration with 30 articles on fundamental human rights, "
                          "morally authoritative but not legally binding."
        }
    ]
    
    # Convert to TestSet format
    test_set = TestSet(
        questions=[item["user_input"] for item in sample_data],
        contexts=[item["retrieved_context"] for item in sample_data],
        responses=[item["response"] for item in sample_data],
        ground_truths=[item["ground_truth"] for item in sample_data],
        metadata={"dataset": "amnesty_qa", "num_samples": len(sample_data)}
    )
    
    print(f"✓ Loaded Amnesty Q&A dataset with {len(sample_data)} samples")
    return test_set


# ============================================================================
# Part 2: Basic Ragas Evaluation
# ============================================================================

def run_basic_evaluation(test_set: TestSet) -> None:
    """
    Run basic Ragas evaluation with standard metrics.
    
    This demonstrates:
    - Setting up the evaluation framework
    - Running evaluation with default metrics
    - Interpreting results
    """
    print("\n" + "="*80)
    print("PART 2: Basic Ragas Evaluation")
    print("="*80)
    
    # Initialize evaluation framework
    framework = EvaluationFramework(
        llm_endpoint="nvidia/llama-3-70b",
        embedding_model="nvidia/nv-embed-v2",
        framework="ragas"
    )
    
    print("\n✓ Initialized EvaluationFramework with NVIDIA NIM endpoints")
    
    # Run evaluation with all standard metrics
    print("\nRunning evaluation with metrics:")
    print("  - Faithfulness (generation)")
    print("  - Answer Relevancy (generation)")
    print("  - Context Utilization (generation)")
    print("  - Context Precision (retrieval)")
    print("  - Context Recall (retrieval)")
    print("  - Context Relevance (retrieval)")
    
    results = framework.evaluate_rag(test_set)
    
    # Display results
    print("\n" + "-"*80)
    print("EVALUATION RESULTS")
    print("-"*80)
    
    print(f"\nAggregated Scores:")
    for metric_name, score in results.metrics.items():
        print(f"  {metric_name:25s}: {score:.3f}")
    
    print(f"\nSummary: {results.summary}")
    
    # Show detailed results for first sample
    print("\n" + "-"*80)
    print("DETAILED RESULTS (Sample 1)")
    print("-"*80)
    
    sample = results.detailed_results[0]
    print(f"\nQuestion: {sample.query}")
    print(f"\nResponse: {sample.response}")
    print(f"\nMetric Scores:")
    for metric, score in sample.metric_scores.items():
        print(f"  {metric:25s}: {score:.3f}")


# ============================================================================
# Part 3: Computing Specific Metrics (Faithfulness and Context Recall)
# ============================================================================

def compute_faithfulness_and_recall(test_set: TestSet) -> None:
    """
    Focus on faithfulness and context recall metrics.
    
    Faithfulness: Are response claims supported by context?
    Context Recall: Does context contain all necessary information?
    
    These are the two most critical metrics for RAG systems.
    """
    print("\n" + "="*80)
    print("PART 3: Faithfulness and Context Recall Deep Dive")
    print("="*80)
    
    framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
    
    # Evaluate with only faithfulness and recall
    results = framework.evaluate_rag(
        test_set,
        metrics=["faithfulness", "context_recall"]
    )
    
    print("\n" + "-"*80)
    print("FAITHFULNESS ANALYSIS")
    print("-"*80)
    print("\nFaithfulness measures if response claims are supported by context.")
    print("Score range: 0.0 (no support) to 1.0 (fully supported)")
    
    faithfulness_scores = [
        r.metric_scores.get("faithfulness", 0.0)
        for r in results.detailed_results
    ]
    
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    print(f"\nAverage Faithfulness: {avg_faithfulness:.3f}")
    
    if avg_faithfulness > 0.85:
        print("✓ Excellent! Minimal hallucination detected.")
    elif avg_faithfulness > 0.70:
        print("⚠ Good, but some claims may not be fully supported.")
    else:
        print("✗ Low faithfulness indicates hallucination issues.")
        print("  Action: Strengthen prompts or use more reliable LLM.")
    
    print("\n" + "-"*80)
    print("CONTEXT RECALL ANALYSIS")
    print("-"*80)
    print("\nContext Recall measures if retrieved context contains all necessary info.")
    print("Score range: 0.0 (missing info) to 1.0 (complete coverage)")
    
    recall_scores = [
        r.metric_scores.get("context_recall", 0.0)
        for r in results.detailed_results
    ]
    
    avg_recall = sum(recall_scores) / len(recall_scores)
    print(f"\nAverage Context Recall: {avg_recall:.3f}")
    
    if avg_recall > 0.80:
        print("✓ Excellent! Retrieval provides comprehensive information.")
    elif avg_recall > 0.65:
        print("⚠ Decent, but some information may be missing.")
    else:
        print("✗ Low recall indicates missing information.")
        print("  Action: Increase k (num docs) or improve retrieval.")


# ============================================================================
# Part 4: Metric Customization
# ============================================================================

def customize_metrics_for_domain(test_set: TestSet) -> None:
    """
    Customize existing metrics for human rights domain.
    
    This demonstrates how to adapt standard metrics with:
    - Domain-specific prompts
    - Custom scoring rubrics
    - Specialized evaluation criteria
    """
    print("\n" + "="*80)
    print("PART 4: Metric Customization for Human Rights Domain")
    print("="*80)
    
    framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
    
    # Customize faithfulness for human rights context
    print("\nCustomizing Faithfulness metric for human rights domain...")
    
    # INTENTIONAL BUG #1: Missing context placeholder in prompt
    # Students should fix this by adding {context} placeholder
    custom_faithfulness_prompt = """
    Evaluate if the response about human rights is supported by the provided context.
    
    Consider:
    - Accuracy of human rights terminology
    - Correct citation of international law and conventions
    - Proper attribution of violations to specific actors
    
    Response: {response}
    
    Score 0-1 where 1 means fully supported by evidence.
    Return JSON: {{"score": 0.0-1.0, "reasoning": "..."}}
    """
    
    custom_faithfulness = framework.customize_metric(
        metric_name="faithfulness",
        custom_prompt=custom_faithfulness_prompt,
        scoring_rubric={
            "0.0": "Not supported by context",
            "0.5": "Partially supported",
            "1.0": "Fully supported with accurate terminology"
        }
    )
    
    print(f"✓ Created custom metric: {custom_faithfulness.name}")
    print(f"  Metric ID: {custom_faithfulness.metric_id}")
    
    # Customize answer relevancy for human rights queries
    print("\nCustomizing Answer Relevancy for human rights queries...")
    
    custom_relevancy_prompt = """
    Evaluate if the response directly addresses the human rights question.
    
    Question: {question}
    Response: {response}
    
    Criteria:
    - Does it answer the specific human rights concern raised?
    - Is the response focused on human rights issues?
    - Does it avoid tangential political commentary?
    
    Score 0-1 where 1 means highly relevant.
    Return JSON: {{"score": 0.0-1.0, "reasoning": "..."}}
    """
    
    custom_relevancy = framework.customize_metric(
        metric_name="answer_relevancy",
        custom_prompt=custom_relevancy_prompt,
        scoring_rubric={
            "0.0": "Off-topic or irrelevant",
            "0.5": "Partially addresses question",
            "1.0": "Directly and comprehensively addresses question"
        }
    )
    
    print(f"✓ Created custom metric: {custom_relevancy.name}")
    
    print("\n💡 Exercise: Fix the bug in custom_faithfulness_prompt")
    print("   Hint: The prompt is missing a crucial placeholder for evaluation")


# ============================================================================
# Part 5: Creating Custom Metrics from Scratch
# ============================================================================

def create_custom_metrics(test_set: TestSet) -> None:
    """
    Create completely new custom metrics for human rights evaluation.
    
    This demonstrates:
    - Defining new evaluation criteria
    - Building custom prompts
    - Implementing domain-specific metrics
    """
    print("\n" + "="*80)
    print("PART 5: Creating Custom Metrics from Scratch")
    print("="*80)
    
    framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
    
    # Custom Metric 1: Legal Citation Accuracy
    print("\nCreating custom metric: Legal Citation Accuracy")
    
    legal_citation_def = MetricDefinition(
        name="Legal Citation Accuracy",
        description="Evaluates accuracy of legal and treaty citations",
        evaluation_criteria=[
            "Correct treaty names and dates",
            "Accurate article numbers",
            "Proper attribution to UN bodies",
            "Valid case law references"
        ],
        prompt_template="""
        Evaluate the accuracy of legal citations in this human rights response.
        
        Response: {response}
        Context: {context}
        
        Check for:
        1. Correct treaty names and dates (0-0.25)
        2. Accurate article numbers (0-0.25)
        3. Proper UN body attribution (0-0.25)
        4. Valid case law references (0-0.25)
        
        Return JSON: {{"score": 0.0-1.0, "reasoning": "...", "errors": []}}
        """,
        scoring_method="llm"
    )
    
    legal_citation_metric = framework.create_custom_metric(legal_citation_def)
    print(f"✓ Created: {legal_citation_metric.name}")
    print(f"  Criteria: {len(legal_citation_def.evaluation_criteria)} evaluation points")
    
    # Custom Metric 2: Victim Sensitivity
    print("\nCreating custom metric: Victim Sensitivity")
    
    # INTENTIONAL BUG #2: Scoring method typo
    # Students should fix "llm_judge" to "llm"
    sensitivity_def = MetricDefinition(
        name="Victim Sensitivity",
        description="Evaluates appropriate and respectful language about victims",
        evaluation_criteria=[
            "Avoids victim-blaming language",
            "Uses person-first language",
            "Respects dignity and privacy",
            "Appropriate tone for trauma"
        ],
        prompt_template="""
        Evaluate if the response uses appropriate and sensitive language about victims.
        
        Response: {response}
        
        Criteria:
        - No victim-blaming (0-0.25)
        - Person-first language (0-0.25)
        - Respects dignity (0-0.25)
        - Appropriate tone (0-0.25)
        
        Return JSON: {{"score": 0.0-1.0, "reasoning": "...", "issues": []}}
        """,
        scoring_method="llm_judge"  # BUG: Should be "llm"
    )
    
    sensitivity_metric = framework.create_custom_metric(sensitivity_def)
    print(f"✓ Created: {sensitivity_metric.name}")
    
    # Custom Metric 3: Actionability
    print("\nCreating custom metric: Response Actionability")
    
    actionability_def = MetricDefinition(
        name="Response Actionability",
        description="Evaluates if response provides actionable information",
        evaluation_criteria=[
            "Provides specific recommendations",
            "Includes relevant resources or contacts",
            "Explains next steps clearly",
            "Offers practical guidance"
        ],
        prompt_template="""
        Evaluate if the response provides actionable information for the user.
        
        Question: {question}
        Response: {response}
        
        Score based on:
        1. Specific recommendations (0-0.25)
        2. Relevant resources (0-0.25)
        3. Clear next steps (0-0.25)
        4. Practical guidance (0-0.25)
        
        Return JSON: {{"score": 0.0-1.0, "reasoning": "..."}}
        """,
        scoring_method="llm"
    )
    
    actionability_metric = framework.create_custom_metric(actionability_def)
    print(f"✓ Created: {actionability_metric.name}")
    
    print("\n💡 Exercise: Find and fix the bug in sensitivity_def")
    print("   Hint: Check the scoring_method field")
    
    print("\n" + "-"*80)
    print("CUSTOM METRICS SUMMARY")
    print("-"*80)
    print(f"\nTotal custom metrics created: {len(framework.custom_metrics)}")
    for metric_id, metric in framework.custom_metrics.items():
        print(f"  - {metric.name} ({metric_id})")


# ============================================================================
# Part 6: Results Analysis and Optimization Insights
# ============================================================================

def analyze_and_optimize(test_set: TestSet) -> None:
    """
    Analyze evaluation results and derive actionable optimization insights.
    
    This demonstrates:
    - Interpreting metric patterns
    - Identifying system weaknesses
    - Generating optimization recommendations
    - Prioritizing improvement actions
    """
    print("\n" + "="*80)
    print("PART 6: Results Analysis and Optimization")
    print("="*80)
    
    framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
    
    # Run full evaluation
    results = framework.evaluate_rag(test_set)
    
    # Analyze results
    print("\nAnalyzing evaluation results...")
    analysis = framework.analyze_results(results)
    
    print("\n" + "-"*80)
    print("ANALYSIS REPORT")
    print("-"*80)
    
    print("\n📊 Key Findings:")
    for i, finding in enumerate(analysis.key_findings, 1):
        print(f"  {i}. {finding}")
    
    if analysis.strengths:
        print("\n✓ Strengths:")
        for strength in analysis.strengths:
            print(f"  • {strength}")
    
    if analysis.weaknesses:
        print("\n⚠ Weaknesses:")
        for weakness in analysis.weaknesses:
            print(f"  • {weakness}")
    
    if analysis.optimization_suggestions:
        print("\n🔧 Optimization Suggestions:")
        for i, suggestion in enumerate(analysis.optimization_suggestions, 1):
            print(f"\n  {i}. Component: {suggestion['component']}")
            print(f"     Suggestion: {suggestion['suggestion']}")
            print(f"     Expected Impact: {suggestion['expected_impact']}")
    
    if analysis.priority_actions:
        print("\n🎯 Priority Actions:")
        for i, action in enumerate(analysis.priority_actions, 1):
            print(f"  {i}. {action}")
    
    # Show metric statistics
    print("\n" + "-"*80)
    print("METRIC STATISTICS")
    print("-"*80)
    
    for metric_name in results.metrics.keys():
        stats = results.get_metric_statistics(metric_name)
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Min:  {stats['min']:.3f}")
        print(f"  Max:  {stats['max']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")


# ============================================================================
# Part 7: Error Handling and Robustness
# ============================================================================

def demonstrate_error_handling() -> None:
    """
    Demonstrate robust evaluation with error handling.
    
    This shows:
    - Sample validation
    - Graceful degradation
    - Fallback to traditional metrics
    - Error logging and recovery
    """
    print("\n" + "="*80)
    print("PART 7: Error Handling and Robustness")
    print("="*80)
    
    # Create test set with some invalid samples
    test_set_with_errors = TestSet(
        questions=[
            "Valid question about human rights",
            "",  # Invalid: empty question
            "Another valid question",
            "Third valid question"
        ],
        contexts=[
            ["Valid context document"],
            [],  # Invalid: no contexts
            ["Valid context"],
            ["Valid context"]
        ],
        responses=[
            "Valid response",
            "Response to empty question",
            "",  # Invalid: empty response
            "Valid response"
        ],
        ground_truths=[
            "Valid ground truth",
            "Ground truth",
            "Ground truth",
            "Valid ground truth"
        ]
    )
    
    print("\nTest set contains 4 samples (2 valid, 2 invalid)")
    
    # Use RagasEvaluator with error handling
    evaluator = RagasEvaluator(
        llm_endpoint="nvidia/llama-3-70b",
        enable_fallback=True
    )
    
    print("\nRunning evaluation with error handling...")
    results = evaluator.evaluate_with_error_handling(
        test_set_with_errors,
        metrics=["faithfulness", "answer_relevancy"]
    )
    
    print("\n" + "-"*80)
    print("ERROR HANDLING RESULTS")
    print("-"*80)
    
    print(f"\nTotal samples: 4")
    print(f"Valid samples: {len(results.detailed_results)}")
    print(f"Failed samples: {len(evaluator.failed_samples)}")
    print(f"Success rate: {results.metadata.get('success_rate', 0):.1%}")
    
    if evaluator.failed_samples:
        print(f"\nFailed samples: {', '.join(evaluator.failed_samples)}")
    
    print("\n✓ Evaluation completed successfully despite invalid samples")
    print("  Invalid samples were skipped with proper logging")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution function for Notebook 5.
    
    Runs all parts of the Ragas evaluation demonstration.
    """
    print("="*80)
    print("NOTEBOOK 5: RAGAS EVALUATION IMPLEMENTATION")
    print("="*80)
    print("\nThis notebook demonstrates comprehensive RAG evaluation using Ragas.")
    print("You will learn to compute metrics, customize them, and create new ones.")
    
    # Load dataset
    test_set = load_amnesty_dataset()
    
    # Part 1: Basic evaluation
    run_basic_evaluation(test_set)
    
    # Part 2: Specific metrics
    compute_faithfulness_and_recall(test_set)
    
    # Part 3: Metric customization
    customize_metrics_for_domain(test_set)
    
    # Part 4: Custom metrics
    create_custom_metrics(test_set)
    
    # Part 5: Analysis
    analyze_and_optimize(test_set)
    
    # Part 6: Error handling
    demonstrate_error_handling()
    
    print("\n" + "="*80)
    print("NOTEBOOK COMPLETE")
    print("="*80)
    print("\n🎓 Key Takeaways:")
    print("  1. Ragas provides comprehensive RAG evaluation with 6 core metrics")
    print("  2. Faithfulness and Context Recall are the most critical metrics")
    print("  3. Metrics can be customized with domain-specific prompts")
    print("  4. Custom metrics can be created from scratch for specialized needs")
    print("  5. Analysis provides actionable optimization insights")
    print("  6. Robust error handling ensures evaluation continues despite failures")
    
    print("\n💡 Exercises:")
    print("  1. Fix the bug in custom_faithfulness_prompt (missing {context})")
    print("  2. Fix the bug in sensitivity_def (scoring_method typo)")
    print("  3. Create your own custom metric for a different domain")
    print("  4. Experiment with different metric combinations")
    print("  5. Analyze metric patterns to diagnose system issues")


if __name__ == "__main__":
    main()
