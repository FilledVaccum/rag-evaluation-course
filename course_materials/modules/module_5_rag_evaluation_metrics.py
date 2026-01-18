"""
Module 5: RAG Evaluation Metrics and Frameworks

This module provides comprehensive lecture materials on RAG evaluation metrics,
the Ragas framework, and LLM-as-a-Judge methodology. It covers both generation
and retrieval metrics with detailed explanations and practical guidance.

Learning Objectives:
- Understand LLM-as-a-Judge methodology and its limitations
- Master the Ragas framework architecture and capabilities
- Learn generation metrics: Faithfulness, Answer Relevancy, Context Utilization
- Learn retrieval metrics: Context Precision, Context Recall, Context Relevance
- Interpret metrics and derive actionable optimization insights

Requirements: 7.1, 7.2, 7.3, 7.4, 7.6
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class MetricCategory(Enum):
    """Categories of RAG evaluation metrics"""
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    END_TO_END = "end_to_end"


@dataclass
class LLMAsJudgeMethodology:
    """
    LLM-as-a-Judge: Using Language Models to Evaluate Other Language Models
    
    Core Concept:
    Instead of relying on traditional NLP metrics (BLEU, ROUGE, F1) which fail
    to capture semantic quality, we use powerful LLMs to evaluate RAG outputs
    based on human-like judgment criteria.
    
    Why Traditional Metrics Fail for RAG:
    - BLEU/ROUGE: Measure n-gram overlap, miss semantic equivalence
    - F1 Score: Requires exact token matching, ignores paraphrasing
    - Perplexity: Measures model confidence, not answer quality
    - These metrics can't assess faithfulness, relevance, or context usage
    """
    
    methodology_name: str = "LLM-as-a-Judge"
    description: str = "Using LLMs to evaluate RAG system outputs"
    
    @staticmethod
    def get_advantages() -> List[str]:
        """Advantages of LLM-as-a-Judge approach"""
        return [
            "Semantic Understanding: Captures meaning beyond exact word matching",
            "Scalability: Automated evaluation without human annotators",
            "Flexibility: Can evaluate diverse criteria (faithfulness, relevance, tone)",
            "Consistency: More consistent than human raters across large datasets",
            "Cost-Effective: Cheaper than hiring expert human evaluators",
            "Customizable: Prompts can be tailored to specific evaluation needs"
        ]
    
    @staticmethod
    def get_limitations() -> List[str]:
        """Critical limitations to understand"""
        return [
            "Bias Inheritance: LLM judges inherit biases from training data",
            "Hallucination Risk: Judge LLM may hallucinate facts during evaluation",
            "Prompt Sensitivity: Results highly dependent on prompt engineering quality",
            "Cost at Scale: API costs can accumulate with large evaluation sets",
            "Lack of Explainability: Hard to debug why judge made specific decisions",
            "Position Bias: May favor certain answer positions or formats",
            "Length Bias: May prefer longer or shorter responses regardless of quality",
            "Self-Preference: Some LLMs rate their own outputs higher",
            "Calibration Issues: Score distributions may not align with human judgment"
        ]
    
    @staticmethod
    def get_best_practices() -> Dict[str, str]:
        """Best practices for implementing LLM-as-a-Judge"""
        return {
            "Clear Rubrics": "Define explicit scoring criteria (0-1 scale, JSON output)",
            "Calibration Examples": "Provide examples for each score level",
            "Multi-Stage Evaluation": "Break complex metrics into simpler sub-evaluations",
            "Prompt Engineering": "Use specific, detailed prompts with examples",
            "Multiple Judges": "Use ensemble of different LLMs for robustness",
            "Human Validation": "Periodically validate against human judgments",
            "Error Handling": "Implement fallbacks for API failures or invalid outputs",
            "Logging": "Log all judge inputs/outputs for debugging and auditing"
        }


@dataclass
class RagasFramework:
    """
    Ragas: Retrieval-Augmented Generation Assessment Framework
    
    Ragas is an open-source framework specifically designed for evaluating RAG
    systems. It provides both component-level and end-to-end metrics using
    LLM-as-a-Judge methodology.
    
    Architecture:
    ```
    RAG System Output
           ↓
    ┌──────────────────────────────────┐
    │   Ragas Evaluation Framework     │
    │                                  │
    │  ┌────────────────────────────┐  │
    │  │  Generation Metrics        │  │
    │  │  - Faithfulness            │  │
    │  │  - Answer Relevancy        │  │
    │  │  - Context Utilization     │  │
    │  └────────────────────────────┘  │
    │                                  │
    │  ┌────────────────────────────┐  │
    │  │  Retrieval Metrics         │  │
    │  │  - Context Precision       │  │
    │  │  - Context Recall          │  │
    │  │  - Context Relevance       │  │
    │  └────────────────────────────┘  │
    │                                  │
    │  ┌────────────────────────────┐  │
    │  │  LLM-as-a-Judge Engine     │  │
    │  │  (NVIDIA NIM / OpenAI)     │  │
    │  └────────────────────────────┘  │
    └──────────────────────────────────┘
           ↓
    Evaluation Scores & Insights
    ```
    
    Key Features:
    - Component-level evaluation (retrieval vs generation)
    - Customizable metrics with prompt engineering
    - Support for custom metrics from scratch
    - Integration with popular LLM providers
    - Batch evaluation for efficiency
    """
    
    framework_name: str = "Ragas"
    version: str = "0.1.x"
    github_url: str = "https://github.com/explodinggradients/ragas"
    
    @staticmethod
    def get_architecture_components() -> Dict[str, str]:
        """Core architectural components of Ragas"""
        return {
            "Metrics Engine": "Computes individual metrics using LLM judges",
            "Prompt Templates": "Pre-defined prompts for each metric",
            "LLM Interface": "Abstraction layer for different LLM providers",
            "Dataset Handler": "Manages test sets with required fields",
            "Results Aggregator": "Combines individual scores into reports",
            "Customization Layer": "Allows metric modification and creation"
        }
    
    @staticmethod
    def get_required_fields() -> Dict[str, List[str]]:
        """Required fields for different evaluation types"""
        return {
            "Generation Metrics": [
                "question (user_input)",
                "answer (response)",
                "contexts (retrieved_context)",
                "ground_truth (optional for some metrics)"
            ],
            "Retrieval Metrics": [
                "question (user_input)",
                "contexts (retrieved_context)",
                "ground_truth (for recall/precision)"
            ],
            "End-to-End": [
                "question",
                "answer",
                "contexts",
                "ground_truth"
            ]
        }


@dataclass
class GenerationMetric:
    """Base class for generation metrics"""
    name: str
    description: str
    formula: str
    interpretation: str
    optimization_insights: List[str]


class GenerationMetrics:
    """
    Generation Metrics: Evaluate the quality of generated responses
    
    These metrics assess how well the LLM generates responses given the
    retrieved context and user question.
    """
    
    @staticmethod
    def faithfulness() -> GenerationMetric:
        """
        Faithfulness: Are claims in the response supported by retrieved context?
        
        This is the MOST CRITICAL metric for RAG systems. It measures whether
        the generated response makes claims that are actually supported by the
        retrieved documents.
        
        Why It Matters:
        - Prevents hallucination: Ensures LLM doesn't fabricate information
        - Builds trust: Users can verify claims against source documents
        - Legal/Compliance: Critical for regulated industries (healthcare, finance)
        
        How It Works (Multi-Stage):
        1. Extract Claims: Break response into individual factual claims
        2. Verify Each Claim: Check if each claim is supported by context
        3. Calculate Score: (verified_claims / total_claims)
        
        Example:
        Question: "What is the capital of France?"
        Context: "Paris is the capital and largest city of France."
        Response: "The capital of France is Paris, which has a population of 50 million."
        
        Claims:
        1. "Capital of France is Paris" ✓ (supported)
        2. "Paris has population of 50 million" ✗ (not in context - hallucination)
        
        Faithfulness Score: 1/2 = 0.5
        """
        return GenerationMetric(
            name="Faithfulness",
            description="Measures if generated claims are supported by retrieved context",
            formula="faithfulness = (verified_claims / total_claims)",
            interpretation="Range: 0.0 to 1.0. Higher is better. 1.0 = all claims supported.",
            optimization_insights=[
                "Low Score (<0.7): LLM is hallucinating. Try:",
                "  - Use stronger instruction prompts: 'Only use provided context'",
                "  - Add system message: 'If information not in context, say I don't know'",
                "  - Use more conservative temperature (0.1-0.3)",
                "  - Try different LLM with better instruction following",
                "  - Improve retrieval to provide more complete context",
                "",
                "High Score (>0.9): Good! But verify:",
                "  - Check if responses are too conservative (refusing to answer)",
                "  - Ensure responses are still helpful and complete"
            ]
        )
    
    @staticmethod
    def answer_relevancy() -> GenerationMetric:
        """
        Answer Relevancy: Is the response relevant to the user's question?
        
        Measures whether the generated answer actually addresses what the user
        asked, regardless of whether it's factually correct.
        
        How It Works:
        - Uses embedding similarity between question and response
        - Formula: cosine_similarity(embed(question), embed(response))
        - Can also use LLM-as-a-Judge with relevancy rubric
        
        Example:
        Question: "How do I reset my password?"
        Response A: "To reset your password, click 'Forgot Password' on login page."
        Response B: "Our company was founded in 1995 and has 500 employees."
        
        Response A: High relevancy (directly answers question)
        Response B: Low relevancy (completely off-topic)
        """
        return GenerationMetric(
            name="Answer Relevancy",
            description="Measures if response addresses the user's question",
            formula="relevancy = cosine_similarity(embed(question), embed(response))",
            interpretation="Range: 0.0 to 1.0. Higher is better. >0.8 is typically good.",
            optimization_insights=[
                "Low Score (<0.6): Response is off-topic. Try:",
                "  - Improve retrieval to get more relevant context",
                "  - Add question to generation prompt explicitly",
                "  - Use prompt: 'Answer the following question: {question}'",
                "  - Check if LLM is ignoring question and just summarizing context",
                "",
                "Medium Score (0.6-0.8): Partially relevant. Try:",
                "  - Make prompts more specific about answering the question",
                "  - Reduce context length if LLM is getting distracted",
                "",
                "High Score (>0.8): Good relevancy maintained"
            ]
        )
    
    @staticmethod
    def context_utilization() -> GenerationMetric:
        """
        Context Utilization: Does the response actually use the retrieved context?
        
        Measures whether the LLM is incorporating information from the retrieved
        documents into its response, or just using its parametric knowledge.
        
        Why It Matters:
        - Ensures RAG system is actually "retrieval-augmented"
        - Detects when LLM ignores provided context
        - Validates that retrieval is adding value
        
        How It Works:
        - Compare response with and without context
        - Measure information overlap between context and response
        - Use LLM judge to assess context usage
        """
        return GenerationMetric(
            name="Context Utilization",
            description="Measures if response incorporates retrieved context",
            formula="utilization = overlap(response, context) / length(response)",
            interpretation="Range: 0.0 to 1.0. Higher means more context usage.",
            optimization_insights=[
                "Low Score (<0.5): LLM ignoring context. Try:",
                "  - Strengthen prompt: 'Use ONLY the provided context'",
                "  - Place context prominently in prompt (before question)",
                "  - Use few-shot examples showing context usage",
                "  - Check if context is actually relevant (may need better retrieval)",
                "",
                "High Score (>0.8): Good context usage. Verify:",
                "  - Response isn't just copying context verbatim",
                "  - Response still answers the question (check relevancy)"
            ]
        )


@dataclass
class RetrievalMetric:
    """Base class for retrieval metrics"""
    name: str
    description: str
    formula: str
    interpretation: str
    optimization_insights: List[str]


class RetrievalMetrics:
    """
    Retrieval Metrics: Evaluate the quality of document retrieval
    
    These metrics assess how well the retrieval system finds relevant documents
    before generation happens.
    """
    
    @staticmethod
    def context_precision() -> RetrievalMetric:
        """
        Context Precision: Are the top-ranked retrieved documents relevant?
        
        Measures the ranking quality - whether relevant documents appear higher
        in the retrieved list. This is critical because LLMs pay more attention
        to context that appears earlier.
        
        How It Works:
        - For each position k, check if document is relevant
        - Weight by position (earlier positions matter more)
        - Formula: precision@k = (relevant_in_top_k / k)
        
        Example:
        Retrieved docs (in order): [Relevant, Irrelevant, Relevant, Relevant, Irrelevant]
        
        Precision@1: 1/1 = 1.0 (first doc is relevant)
        Precision@3: 2/3 = 0.67 (2 out of first 3 are relevant)
        Precision@5: 3/5 = 0.6 (3 out of 5 are relevant)
        """
        return RetrievalMetric(
            name="Context Precision",
            description="Measures ranking quality of retrieved documents",
            formula="precision@k = (relevant_docs_in_top_k / k)",
            interpretation="Range: 0.0 to 1.0. Higher is better. Measures ranking quality.",
            optimization_insights=[
                "Low Score (<0.5): Poor ranking. Try:",
                "  - Tune vector store similarity threshold",
                "  - Experiment with different embedding models",
                "  - Add re-ranking stage (cross-encoder)",
                "  - Use hybrid search (BM25 + vector)",
                "  - Adjust number of retrieved documents (k)",
                "",
                "Medium Score (0.5-0.7): Decent but improvable. Try:",
                "  - Fine-tune embedding model on domain data",
                "  - Optimize chunk size and overlap",
                "  - Add metadata filtering",
                "",
                "High Score (>0.7): Good ranking quality"
            ]
        )
    
    @staticmethod
    def context_recall() -> RetrievalMetric:
        """
        Context Recall: Did we retrieve all necessary information?
        
        Measures coverage - whether the retrieved documents contain all the
        information needed to answer the question (compared to ground truth).
        
        Why It Matters:
        - Low recall = missing information = incomplete answers
        - Critical for questions requiring multiple facts
        - Helps diagnose retrieval failures
        
        How It Works:
        - Compare retrieved context with ground truth answer
        - Check if all facts in ground truth are present in context
        - Formula: recall = (ground_truth_facts_in_context / total_ground_truth_facts)
        
        Example:
        Question: "What are the symptoms of flu?"
        Ground Truth: "Fever, cough, fatigue, body aches"
        Retrieved Context: "Flu symptoms include fever and cough."
        
        Recall: 2/4 = 0.5 (only fever and cough retrieved, missing fatigue and body aches)
        """
        return RetrievalMetric(
            name="Context Recall",
            description="Measures if retrieved context contains all necessary information",
            formula="recall = (ground_truth_covered / total_ground_truth)",
            interpretation="Range: 0.0 to 1.0. Higher is better. 1.0 = complete coverage.",
            optimization_insights=[
                "Low Score (<0.6): Missing information. Try:",
                "  - Increase number of retrieved documents (k)",
                "  - Reduce similarity threshold (retrieve more broadly)",
                "  - Improve chunking strategy (larger chunks or more overlap)",
                "  - Check if information exists in knowledge base",
                "  - Use query expansion or reformulation",
                "",
                "Medium Score (0.6-0.8): Partial coverage. Try:",
                "  - Analyze which facts are consistently missed",
                "  - Add those topics to knowledge base",
                "  - Tune retrieval parameters",
                "",
                "High Score (>0.8): Good coverage. But check:",
                "  - Precision might be low (retrieving too much irrelevant content)",
                "  - Balance recall with precision"
            ]
        )
    
    @staticmethod
    def context_relevance() -> RetrievalMetric:
        """
        Context Relevance: Are retrieved documents relevant to the question?
        
        Measures what percentage of retrieved documents are actually relevant
        to answering the user's question. Unlike precision, this doesn't
        consider ranking - just binary relevant/not relevant.
        
        How It Works:
        - For each retrieved document, classify as relevant or not
        - Use LLM-as-a-Judge or embedding similarity
        - Formula: relevance = (relevant_chunks / total_chunks)
        
        Example:
        Question: "How do I install Python?"
        Retrieved docs:
        1. "Python installation guide for Windows" ✓ Relevant
        2. "Python history and philosophy" ✗ Not relevant
        3. "Step-by-step Python setup tutorial" ✓ Relevant
        4. "Java installation guide" ✗ Not relevant
        
        Relevance: 2/4 = 0.5
        """
        return RetrievalMetric(
            name="Context Relevance",
            description="Measures percentage of retrieved documents that are relevant",
            formula="relevance = (relevant_chunks / total_chunks)",
            interpretation="Range: 0.0 to 1.0. Higher is better. >0.7 is typically good.",
            optimization_insights=[
                "Low Score (<0.5): Too much irrelevant content. Try:",
                "  - Increase similarity threshold (be more selective)",
                "  - Reduce number of retrieved documents (k)",
                "  - Improve embedding model quality",
                "  - Add metadata filtering (date, category, etc.)",
                "  - Check query preprocessing (stopword removal, etc.)",
                "",
                "Medium Score (0.5-0.7): Some noise. Try:",
                "  - Fine-tune retrieval parameters",
                "  - Add re-ranking to filter irrelevant docs",
                "  - Improve chunking to create more focused chunks",
                "",
                "High Score (>0.7): Good relevance. But check:",
                "  - Recall might be low (being too selective)",
                "  - Balance relevance with coverage"
            ]
        )


@dataclass
class MetricInterpretationGuide:
    """
    Guide for interpreting RAG evaluation metrics and deriving actionable insights
    """
    
    @staticmethod
    def get_common_patterns() -> Dict[str, Dict[str, str]]:
        """Common metric patterns and their diagnoses"""
        return {
            "High Faithfulness + Low Relevancy": {
                "diagnosis": "LLM is being too conservative or retrieval is poor",
                "action": "Check if retrieval is getting relevant context. If yes, adjust generation prompt to be more direct."
            },
            "Low Faithfulness + High Relevancy": {
                "diagnosis": "LLM is hallucinating but staying on topic",
                "action": "Strengthen faithfulness constraints in prompt. Use lower temperature. Consider different LLM."
            },
            "High Precision + Low Recall": {
                "diagnosis": "Retrieval is too selective, missing information",
                "action": "Increase k (number of docs). Lower similarity threshold. Check chunking strategy."
            },
            "Low Precision + High Recall": {
                "diagnosis": "Retrieval is too broad, getting irrelevant content",
                "action": "Decrease k. Raise similarity threshold. Add re-ranking. Improve embedding model."
            },
            "Low Context Utilization + High Faithfulness": {
                "diagnosis": "LLM using parametric knowledge instead of context",
                "action": "Strengthen prompt to use context. May indicate context isn't adding value."
            },
            "All Metrics Low": {
                "diagnosis": "Fundamental system issues",
                "action": "Start with retrieval evaluation. Check embedding model, chunking, and knowledge base quality."
            }
        }
    
    @staticmethod
    def get_optimization_workflow() -> List[str]:
        """Step-by-step workflow for optimizing RAG based on metrics"""
        return [
            "1. BASELINE: Run evaluation on test set, record all metrics",
            "",
            "2. DIAGNOSE RETRIEVAL FIRST:",
            "   - Check Context Precision, Recall, Relevance",
            "   - If retrieval metrics are low, fix retrieval before generation",
            "   - Common fixes: embedding model, chunking, k value, similarity threshold",
            "",
            "3. DIAGNOSE GENERATION SECOND:",
            "   - Check Faithfulness, Answer Relevancy, Context Utilization",
            "   - Only after retrieval is good (precision >0.7, recall >0.7)",
            "   - Common fixes: prompt engineering, temperature, LLM selection",
            "",
            "4. ITERATE:",
            "   - Make ONE change at a time",
            "   - Re-run evaluation after each change",
            "   - Track metrics in spreadsheet or dashboard",
            "   - Look for trade-offs (e.g., improving recall may hurt precision)",
            "",
            "5. VALIDATE:",
            "   - Test on held-out dataset",
            "   - Get human feedback on sample outputs",
            "   - Monitor metrics in production",
            "",
            "6. CONTINUOUS IMPROVEMENT:",
            "   - Collect user feedback",
            "   - Add failed cases to test set",
            "   - Re-evaluate periodically",
            "   - Update knowledge base as information changes"
        ]
    
    @staticmethod
    def get_metric_targets() -> Dict[str, Dict[str, float]]:
        """Recommended target scores for different use cases"""
        return {
            "High-Stakes (Healthcare, Legal, Finance)": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.85,
                "context_precision": 0.80,
                "context_recall": 0.85,
                "context_relevance": 0.80
            },
            "General Enterprise (Customer Support, Internal KB)": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.80,
                "context_precision": 0.70,
                "context_recall": 0.75,
                "context_relevance": 0.70
            },
            "Exploratory (Research, Discovery)": {
                "faithfulness": 0.75,
                "answer_relevancy": 0.75,
                "context_precision": 0.60,
                "context_recall": 0.80,  # Prioritize coverage
                "context_relevance": 0.65
            }
        }


# Lecture content structure
LECTURE_OUTLINE = """
Module 5: RAG Evaluation Metrics and Frameworks
Duration: 120-150 minutes (2-2.5 hours)

Part 1: Introduction and Motivation (20 min)
- Why traditional NLP metrics fail for RAG
- The need for semantic evaluation
- Overview of LLM-as-a-Judge methodology

Part 2: LLM-as-a-Judge Deep Dive (30 min)
- How it works: Using LLMs to evaluate LLMs
- Advantages: Scalability, semantic understanding, flexibility
- Limitations: Bias, hallucination, prompt sensitivity, cost
- Best practices: Clear rubrics, calibration, multi-stage evaluation

Part 3: Ragas Framework (30 min)
- Architecture and components
- Integration with LLM providers (NVIDIA NIM, OpenAI)
- Required data format
- Installation and setup

Part 4: Generation Metrics (40 min)
- Faithfulness: Preventing hallucination
  * Multi-stage evaluation process
  * Examples and interpretation
  * Optimization strategies
- Answer Relevancy: Staying on topic
  * Embedding-based measurement
  * Common failure modes
- Context Utilization: Using retrieved information
  * Detecting when LLM ignores context

Part 5: Retrieval Metrics (40 min)
- Context Precision: Ranking quality
  * Why ranking matters for LLMs
  * Precision@k calculation
- Context Recall: Information coverage
  * Detecting missing information
  * Balancing with precision
- Context Relevance: Document relevance
  * Binary classification approach
  * Filtering irrelevant content

Part 6: Metric Interpretation and Optimization (30 min)
- Common metric patterns and diagnoses
- Optimization workflow
- Target scores for different use cases
- Trade-offs and balancing metrics

Part 7: Hands-On Preview (10 min)
- Overview of Notebook 3
- Amnesty Q&A dataset introduction
- Metric customization preview
"""


def get_module_summary() -> str:
    """Get a concise summary of Module 5"""
    return """
    Module 5: RAG Evaluation Metrics and Frameworks
    
    Key Takeaways:
    1. Traditional NLP metrics (BLEU, ROUGE, F1) fail for RAG - use LLM-as-a-Judge
    2. Ragas provides comprehensive RAG evaluation with generation + retrieval metrics
    3. Generation Metrics: Faithfulness (no hallucination), Answer Relevancy (on-topic), Context Utilization
    4. Retrieval Metrics: Context Precision (ranking), Context Recall (coverage), Context Relevance (filtering)
    5. Diagnose retrieval BEFORE generation - fix retrieval issues first
    6. Interpret metrics in combination - look for patterns and trade-offs
    7. Set targets based on use case (high-stakes vs exploratory)
    8. Iterate: baseline → diagnose → fix → re-evaluate → validate
    
    Critical Skills:
    - Implement Ragas evaluation pipelines
    - Customize metrics with prompt engineering
    - Create custom metrics from scratch
    - Interpret results and derive actionable insights
    - Balance competing metrics (precision vs recall, faithfulness vs relevancy)
    """


if __name__ == "__main__":
    # Example usage for instructors
    print("=== Module 5: RAG Evaluation Metrics and Frameworks ===\n")
    
    print("LLM-as-a-Judge Methodology:")
    methodology = LLMAsJudgeMethodology()
    print(f"Advantages: {len(methodology.get_advantages())}")
    print(f"Limitations: {len(methodology.get_limitations())}")
    print(f"Best Practices: {len(methodology.get_best_practices())}\n")
    
    print("Generation Metrics:")
    faithfulness = GenerationMetrics.faithfulness()
    print(f"- {faithfulness.name}: {faithfulness.description}")
    relevancy = GenerationMetrics.answer_relevancy()
    print(f"- {relevancy.name}: {relevancy.description}")
    utilization = GenerationMetrics.context_utilization()
    print(f"- {utilization.name}: {utilization.description}\n")
    
    print("Retrieval Metrics:")
    precision = RetrievalMetrics.context_precision()
    print(f"- {precision.name}: {precision.description}")
    recall = RetrievalMetrics.context_recall()
    print(f"- {recall.name}: {recall.description}")
    relevance = RetrievalMetrics.context_relevance()
    print(f"- {relevance.name}: {relevance.description}\n")
    
    print("Metric Targets (General Enterprise):")
    targets = MetricInterpretationGuide.get_metric_targets()
    for metric, score in targets["General Enterprise (Customer Support, Internal KB)"].items():
        print(f"  {metric}: {score}")
