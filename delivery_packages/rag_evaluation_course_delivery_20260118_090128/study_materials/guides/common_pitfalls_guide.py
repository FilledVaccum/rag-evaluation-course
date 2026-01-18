"""
Common Pitfalls and Best Practices Guide for RAG Evaluation

This module provides comprehensive documentation of common pitfalls encountered
when building and evaluating RAG systems, along with best practices to avoid them.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class PitfallCategory(Enum):
    """Categories of common pitfalls in RAG systems"""
    SYNTHETIC_DATA = "synthetic_data"
    COMPONENT_DIAGNOSIS = "component_diagnosis"
    EMBEDDING_SELECTION = "embedding_selection"
    CHUNKING_STRATEGY = "chunking_strategy"
    EVALUATION_METRICS = "evaluation_metrics"
    PROMPT_ENGINEERING = "prompt_engineering"
    PRODUCTION_MONITORING = "production_monitoring"


@dataclass
class Pitfall:
    """Represents a common pitfall with solutions"""
    title: str
    category: PitfallCategory
    description: str
    symptoms: List[str]
    root_cause: str
    solution: str
    before_example: Optional[str] = None
    after_example: Optional[str] = None
    prevention_tips: List[str] = None


# Pitfall 1: Over-Generic Synthetic Data
PITFALL_OVER_GENERIC_SYNTHETIC_DATA = Pitfall(
    title="Over-Generic Synthetic Data Generation",
    category=PitfallCategory.SYNTHETIC_DATA,
    description="""
    Generated synthetic test questions are too philosophical, broad, or generic,
    failing to reflect realistic user queries for the specific domain.
    """,
    symptoms=[
        "Generated questions are overly abstract or philosophical",
        "Questions don't match actual user query patterns",
        "Test data doesn't reflect domain-specific terminology",
        "Evaluation results don't correlate with real-world performance",
        "Questions are too long or complex compared to actual user queries"
    ],
    root_cause="""
    The LLM synthesizer lacks sufficient domain context and constraints. Without
    specific examples and explicit instructions, LLMs tend to generate academic
    or philosophical questions rather than practical, user-focused queries.
    """,
    solution="""
    Apply targeted prompt engineering with the 3-5 example pattern:
    1. Provide 3-5 concrete examples of desired question style
    2. Specify user persona explicitly (e.g., "undergraduate student", not "user")
    3. Add explicit negative examples (what NOT to generate)
    4. Constrain question length and complexity
    5. Include domain-specific terminology requirements
    6. Iterate and validate with domain experts
    """,
    before_example="""
# BEFORE: Over-generic prompt
prompt = '''
Generate questions about the course catalog.
'''

# Generated questions (too philosophical):
- "What is the epistemological foundation of computer science education?"
- "How does the curriculum reflect contemporary pedagogical theories?"
- "What philosophical principles guide course design?"
    """,
    after_example="""
# AFTER: Specific, constrained prompt with examples
prompt = '''
Generate questions that an undergraduate student would ask when searching
for courses to take next semester. Questions should be practical and specific.

Examples of GOOD questions:
1. "What are the prerequisites for CSCI 567?"
2. "Which courses offer 4 units and meet on Tuesdays?"
3. "Is CSCI 401 offered in Fall 2024?"

Examples of BAD questions (do NOT generate):
- Philosophical questions about education theory
- Questions longer than 20 words
- Abstract questions about curriculum design

Generate 5 questions following the GOOD examples pattern.
'''

# Generated questions (practical and specific):
- "What time does CSCI 570 meet?"
- "How many units is CSCI 585?"
- "Who teaches CSCI 544 this semester?"
    """,
    prevention_tips=[
        "Always include 3-5 concrete examples in your prompt",
        "Specify user persona with demographic details",
        "Add explicit negative examples",
        "Test prompts on small batches before full generation",
        "Validate generated data with domain experts",
        "Use domain-specific synthesizers when available"
    ]
)


# Pitfall 2: Retrieval vs Generation Failure Misdiagnosis
PITFALL_COMPONENT_MISDIAGNOSIS = Pitfall(
    title="Retrieval vs Generation Failure Misdiagnosis",
    category=PitfallCategory.COMPONENT_DIAGNOSIS,
    description="""
    Incorrectly attributing RAG system failures to the wrong component, leading
    to wasted optimization effort on the generation stage when the retrieval
    stage is actually the problem (or vice versa).
    """,
    symptoms=[
        "Optimization efforts don't improve system performance",
        "Changing LLM models has no effect on accuracy",
        "Prompt engineering doesn't fix incorrect responses",
        "System performs well on some queries but fails on similar ones",
        "Generated responses are fluent but factually incorrect"
    ],
    root_cause="""
    Lack of component-level evaluation. Teams often evaluate RAG systems
    end-to-end without isolating retrieval and generation stages. This makes
    it impossible to determine which component is causing failures.
    """,
    solution="""
    Implement component-level debugging workflow:
    
    1. EVALUATE RETRIEVAL INDEPENDENTLY:
       - Check if relevant documents are retrieved (Context Recall)
       - Verify retrieved documents are actually relevant (Context Relevance)
       - Measure ranking quality (Context Precision)
       - Use ground truth to validate retrieval
    
    2. EVALUATE GENERATION INDEPENDENTLY:
       - Given CORRECT context, does LLM generate correct response?
       - Check faithfulness (are claims supported by context?)
       - Verify answer relevancy (is response relevant to question?)
       - Test with manually curated "perfect" context
    
    3. DECISION MATRIX:
       - Retrieval Good + Generation Bad → Optimize LLM/prompts
       - Retrieval Bad + Generation Good → Optimize embeddings/chunking
       - Both Bad → Fix retrieval first (garbage in, garbage out)
       - Both Good → Check orchestration/augmentation logic
    """,
    before_example="""
# BEFORE: End-to-end evaluation only
def evaluate_rag(query: str, rag_system: RAGPipeline) -> float:
    response = rag_system.process_query(query)
    score = compare_with_ground_truth(response, ground_truth)
    return score

# Problem: Can't tell if low score is due to retrieval or generation!
    """,
    after_example="""
# AFTER: Component-level evaluation
def evaluate_rag_components(query: str, rag_system: RAGPipeline) -> Dict:
    # Step 1: Evaluate retrieval independently
    retrieved_docs = rag_system.retrieve(query)
    retrieval_metrics = {
        'context_recall': compute_context_recall(retrieved_docs, ground_truth),
        'context_relevance': compute_context_relevance(retrieved_docs, query),
        'context_precision': compute_context_precision(retrieved_docs, ground_truth)
    }
    
    # Step 2: Evaluate generation independently
    # Use PERFECT context (ground truth) to isolate generation
    perfect_context = get_ground_truth_context(query)
    response_with_perfect_context = rag_system.generate(query, perfect_context)
    generation_metrics = {
        'faithfulness': compute_faithfulness(response_with_perfect_context, perfect_context),
        'answer_relevancy': compute_answer_relevancy(response_with_perfect_context, query)
    }
    
    # Step 3: Diagnose failure point
    diagnosis = diagnose_failure(retrieval_metrics, generation_metrics)
    
    return {
        'retrieval': retrieval_metrics,
        'generation': generation_metrics,
        'diagnosis': diagnosis,
        'recommendation': get_optimization_recommendation(diagnosis)
    }
    """,
    prevention_tips=[
        "Always evaluate retrieval and generation independently",
        "Use ground truth context to test generation in isolation",
        "Implement component-level metrics from the start",
        "Create debugging dashboards showing component-level performance",
        "Fix retrieval issues before optimizing generation",
        "Maintain separate test sets for retrieval and generation"
    ]
)


# Pitfall 3: Wrong Embedding Model Selection
PITFALL_EMBEDDING_SELECTION = Pitfall(
    title="Inappropriate Embedding Model for Domain",
    category=PitfallCategory.EMBEDDING_SELECTION,
    description="""
    Using general-purpose embedding models for specialized domains, resulting
    in poor semantic understanding and retrieval quality.
    """,
    symptoms=[
        "Semantically similar queries return irrelevant results",
        "Domain-specific terminology not captured correctly",
        "Poor performance on technical or specialized content",
        "Multilingual content not handled properly",
        "Code snippets or structured data poorly embedded"
    ],
    root_cause="""
    General-purpose embedding models (e.g., trained on web text) lack the
    specialized vocabulary and semantic understanding required for specific
    domains like medical, legal, financial, or multilingual content.
    """,
    solution="""
    Use the embedding model selection decision matrix:
    
    DOMAIN → RECOMMENDED MODEL
    ├── General English text → NV-Embed-v2, text-embedding-ada-002
    ├── Code/Programming → CodeBERT, GraphCodeBERT, StarEncoder
    ├── Medical/Healthcare → BioBERT, PubMedBERT, ClinicalBERT
    ├── Legal → Legal-BERT, LegalBERT-base
    ├── Financial → FinBERT, FinancialBERT
    ├── Multilingual → mBERT, XLM-RoBERTa, LaBSE
    ├── Low-resource languages → XLM-RoBERTa, mT5
    ├── Scientific papers → SciBERT, SPECTER
    └── Custom domain → Fine-tune on domain data
    
    EVALUATION PROCESS:
    1. Benchmark multiple models on your domain
    2. Use retrieval metrics (recall@k, MRR, NDCG)
    3. Test with domain-specific queries
    4. Consider fine-tuning if off-the-shelf models insufficient
    """,
    before_example="""
# BEFORE: Using general model for specialized domain
embedding_model = "text-embedding-ada-002"  # General purpose

# Medical query example
query = "What are the contraindications for metformin in CKD patients?"
# Problem: Model doesn't understand medical terminology well
# Retrieved: Generic diabetes articles instead of CKD-specific guidance
    """,
    after_example="""
# AFTER: Using domain-specific model
embedding_model = "PubMedBERT"  # Medical domain specialist

# Same medical query
query = "What are the contraindications for metformin in CKD patients?"
# Better: Model understands medical context
# Retrieved: Specific guidelines for metformin use in chronic kidney disease

# Decision matrix implementation
def select_embedding_model(domain: str, language: str = "en") -> str:
    if domain == "medical":
        return "PubMedBERT"
    elif domain == "legal":
        return "Legal-BERT"
    elif domain == "code":
        return "CodeBERT"
    elif domain == "finance":
        return "FinBERT"
    elif language != "en":
        return "XLM-RoBERTa"
    else:
        return "NV-Embed-v2"
    """,
    prevention_tips=[
        "Benchmark embedding models on domain-specific test sets",
        "Don't assume general models work for specialized domains",
        "Consider fine-tuning for highly specialized domains",
        "Test multilingual models for non-English content",
        "Evaluate retrieval quality before full deployment",
        "Keep model selection flexible for A/B testing"
    ]
)


# Pitfall 4: Inappropriate Chunk Size
PITFALL_CHUNK_SIZE = Pitfall(
    title="Inappropriate Chunk Size Configuration",
    category=PitfallCategory.CHUNKING_STRATEGY,
    description="""
    Using chunk sizes that are too small (missing context) or too large
    (inefficient retrieval, exceeding context windows), leading to poor
    retrieval quality and system performance.
    """,
    symptoms=[
        "Retrieved chunks missing critical context",
        "Responses lack necessary details",
        "High retrieval latency",
        "Context window exceeded errors",
        "Irrelevant information in retrieved chunks",
        "Poor ranking of relevant documents"
    ],
    root_cause="""
    Chunk size is often set arbitrarily without experimentation. Optimal
    chunk size depends on document structure, query types, and use case.
    Too small chunks fragment context; too large chunks dilute relevance.
    """,
    solution="""
    Implement systematic chunk size experimentation framework:
    
    1. BASELINE RECOMMENDATIONS:
       - Short Q&A: 200-400 tokens
       - Technical docs: 400-800 tokens
       - Long-form content: 800-1200 tokens
       - Code: Function/class level (variable)
    
    2. EXPERIMENTATION FRAMEWORK:
       ```python
       chunk_sizes = [200, 400, 600, 800, 1000]
       overlap_ratios = [0, 0.1, 0.2]
       
       for size in chunk_sizes:
           for overlap in overlap_ratios:
               chunks = create_chunks(documents, size, overlap)
               metrics = evaluate_retrieval(chunks, test_queries)
               results.append((size, overlap, metrics))
       
       best_config = max(results, key=lambda x: x[2]['recall@5'])
       ```
    
    3. METRICS TO TRACK:
       - Context Recall: Are relevant chunks retrieved?
       - Context Precision: Are retrieved chunks relevant?
       - Latency: Retrieval speed
       - Coverage: Is full answer contained in chunks?
    
    4. OVERLAP STRATEGY:
       - Use 10-20% overlap to preserve context across boundaries
       - Higher overlap for narrative content
       - Lower overlap for structured content
    """,
    before_example="""
# BEFORE: Arbitrary chunk size
chunk_size = 500  # Why 500? No experimentation!
chunks = split_documents(documents, chunk_size=chunk_size)

# Problems:
# - May miss context that spans chunk boundaries
# - May be too small for complex technical content
# - No validation of chunk quality
    """,
    after_example="""
# AFTER: Systematic experimentation
def find_optimal_chunk_size(documents, test_queries, ground_truth):
    results = []
    
    # Test multiple configurations
    for chunk_size in [200, 400, 600, 800, 1000]:
        for overlap_pct in [0, 10, 20]:
            # Create chunks with overlap
            chunks = split_documents(
                documents,
                chunk_size=chunk_size,
                overlap_tokens=int(chunk_size * overlap_pct / 100)
            )
            
            # Evaluate retrieval quality
            metrics = evaluate_retrieval_quality(
                chunks=chunks,
                queries=test_queries,
                ground_truth=ground_truth
            )
            
            results.append({
                'chunk_size': chunk_size,
                'overlap_pct': overlap_pct,
                'context_recall': metrics['recall'],
                'context_precision': metrics['precision'],
                'latency_ms': metrics['latency']
            })
    
    # Select best configuration balancing recall and latency
    best = max(results, key=lambda x: x['context_recall'] - 0.1 * x['latency_ms'] / 100)
    
    return best['chunk_size'], best['overlap_pct']

# Use optimal configuration
optimal_size, optimal_overlap = find_optimal_chunk_size(docs, queries, ground_truth)
chunks = split_documents(docs, chunk_size=optimal_size, overlap_tokens=optimal_overlap)
    """,
    prevention_tips=[
        "Never use arbitrary chunk sizes without testing",
        "Experiment with multiple chunk sizes on your data",
        "Use overlap (10-20%) to preserve context",
        "Consider document structure (paragraphs, sections)",
        "Monitor chunk quality metrics in production",
        "Re-evaluate chunk size when content types change"
    ]
)


# Pitfall 5: Using Traditional NLP Metrics for RAG
PITFALL_TRADITIONAL_METRICS = Pitfall(
    title="Relying on Traditional NLP Metrics for RAG Evaluation",
    category=PitfallCategory.EVALUATION_METRICS,
    description="""
    Using traditional NLP metrics (BLEU, ROUGE, F1) to evaluate RAG systems,
    which fail to capture semantic correctness, faithfulness, and context
    utilization that are critical for RAG performance.
    """,
    symptoms=[
        "High metric scores but poor actual performance",
        "Metrics don't correlate with user satisfaction",
        "Paraphrased correct answers score poorly",
        "Factually incorrect responses score well",
        "Can't detect hallucinations or unfaithful responses"
    ],
    root_cause="""
    Traditional NLP metrics (BLEU, ROUGE, F1) measure surface-level text
    overlap, not semantic correctness or faithfulness. RAG systems can
    generate semantically correct responses with different wording, or
    generate fluent but factually incorrect responses.
    """,
    solution="""
    Use RAG-specific evaluation metrics:
    
    1. GENERATION METRICS:
       ✓ Faithfulness: Are claims supported by retrieved context?
       ✓ Answer Relevancy: Is response relevant to question?
       ✓ Context Utilization: Does response use retrieved context?
       ✗ BLEU/ROUGE: Surface-level text overlap (not suitable)
    
    2. RETRIEVAL METRICS:
       ✓ Context Recall: Is ground truth information retrieved?
       ✓ Context Precision: Are retrieved chunks relevant?
       ✓ Context Relevance: Is context relevant to query?
       ✗ F1/Precision/Recall: Document-level (not suitable for RAG)
    
    3. IMPLEMENTATION:
       - Use Ragas framework for RAG-specific metrics
       - Implement LLM-as-a-Judge for semantic evaluation
       - Create custom metrics for domain-specific requirements
       - Combine multiple metrics for comprehensive evaluation
    
    4. WHEN TO USE TRADITIONAL METRICS:
       - Extractive QA (exact answer extraction)
       - Summarization (with reference summaries)
       - Translation (with reference translations)
       - NOT for open-ended RAG generation
    """,
    before_example="""
# BEFORE: Using BLEU for RAG evaluation
from nltk.translate.bleu_score import sentence_bleu

query = "What are the prerequisites for CSCI 570?"
ground_truth = "CSCI 570 requires CSCI 270 and CSCI 350 as prerequisites"
response = "You need to complete CSCI 270 and CSCI 350 before taking CSCI 570"

# BLEU score is low despite correct answer!
bleu = sentence_bleu([ground_truth.split()], response.split())
# bleu ≈ 0.15 (very low, but answer is correct!)

# Problem: Different wording = low score, even if semantically correct
    """,
    after_example="""
# AFTER: Using RAG-specific metrics
from ragas.metrics import faithfulness, answer_relevancy, context_recall

query = "What are the prerequisites for CSCI 570?"
context = "CSCI 570 - Analysis of Algorithms. Prerequisites: CSCI 270, CSCI 350"
response = "You need to complete CSCI 270 and CSCI 350 before taking CSCI 570"
ground_truth = "CSCI 570 requires CSCI 270 and CSCI 350 as prerequisites"

# Evaluate with RAG-specific metrics
metrics = {
    'faithfulness': faithfulness.score(
        question=query,
        answer=response,
        contexts=[context]
    ),  # High: Claims supported by context
    
    'answer_relevancy': answer_relevancy.score(
        question=query,
        answer=response
    ),  # High: Response answers the question
    
    'context_recall': context_recall.score(
        question=query,
        answer=response,
        contexts=[context],
        ground_truth=ground_truth
    )  # High: Retrieved context contains answer
}

# All metrics are high, correctly identifying good response!
# faithfulness ≈ 1.0, answer_relevancy ≈ 0.95, context_recall ≈ 1.0
    """,
    prevention_tips=[
        "Never use BLEU/ROUGE for RAG evaluation",
        "Implement faithfulness as primary generation metric",
        "Use context recall/precision for retrieval evaluation",
        "Combine multiple RAG-specific metrics",
        "Validate metrics correlate with user satisfaction",
        "Use LLM-as-a-Judge for semantic evaluation"
    ]
)


# Pitfall 6: Prompt Engineering Mistakes
PITFALL_PROMPT_ENGINEERING = Pitfall(
    title="Common Prompt Engineering Mistakes",
    category=PitfallCategory.PROMPT_ENGINEERING,
    description="""
    Ineffective prompts for synthetic data generation and LLM-as-a-Judge
    evaluation due to lack of specificity, insufficient examples, or
    missing constraints.
    """,
    symptoms=[
        "Inconsistent synthetic data quality",
        "LLM-as-a-Judge scores are unreliable",
        "Generated data doesn't match requirements",
        "Evaluation results vary significantly across runs",
        "Prompts require extensive trial-and-error"
    ],
    root_cause="""
    Prompts lack the specificity, examples, and constraints needed to
    reliably steer LLM behavior. The 3-5 example pattern is optimal but
    often ignored in favor of vague instructions.
    """,
    solution="""
    Follow the 3-5 example rule and extreme specificity principle:
    
    1. EXTREME SPECIFICITY:
       - Write as if explaining to a child
       - Define every term explicitly
       - Specify format, length, style
       - Include what NOT to do
    
    2. 3-5 EXAMPLE PATTERN:
       - Fewer than 3: Insufficient steering
       - More than 5: Overfitting, diminishing returns
       - Exactly 3-5: Optimal balance
    
    3. EXPLICIT NEGATIVES:
       - Show examples of what NOT to generate
       - Prevents common failure modes
       - Clarifies boundaries
    
    4. USER PERSONA:
       - Specify demographics, role, context
       - "undergraduate student" not "user"
       - Include realistic scenarios
    
    5. FOR LLM-AS-A-JUDGE:
       - Define explicit scoring rubrics (0-1 scale)
       - Provide calibration examples for each score level
       - Request structured output (JSON)
       - Break complex evaluations into stages
    """,
    before_example="""
# BEFORE: Vague prompt
prompt = '''
Generate some questions about courses.
Make them realistic.
'''

# Problems:
# - No examples
# - "Realistic" is undefined
# - No constraints on length, style, or content
# - No user persona specified
    """,
    after_example="""
# AFTER: Specific prompt with 3-5 examples
prompt = '''
You are generating questions that an undergraduate computer science student
would ask when searching for courses to register for next semester.

REQUIREMENTS:
- Questions must be 5-15 words long
- Questions must be practical and specific (not philosophical)
- Questions should focus on prerequisites, schedule, or course content
- Use casual student language, not formal academic language

GOOD EXAMPLES (generate questions like these):
1. "What are the prereqs for CSCI 570?"
2. "Does CSCI 544 meet on Fridays?"
3. "How many units is CSCI 585?"
4. "Who's teaching CSCI 401 this fall?"
5. "Is CSCI 567 offered in spring?"

BAD EXAMPLES (do NOT generate questions like these):
- "What is the epistemological foundation of computer science?" (too philosophical)
- "How does the curriculum reflect contemporary pedagogical theories?" (too academic)
- "What are all the courses in the computer science department?" (too broad)

Generate 10 questions following the GOOD examples pattern.
Output format: One question per line, numbered.
'''
    """,
    prevention_tips=[
        "Always include 3-5 concrete examples",
        "Write prompts with extreme specificity",
        "Add explicit negative examples",
        "Define user persona with details",
        "Test prompts on small batches first",
        "Use structured output formats (JSON)",
        "For LLM-as-a-Judge, provide scoring rubrics with calibration examples"
    ]
)


# Pitfall 7: Production Monitoring Gaps
PITFALL_PRODUCTION_MONITORING = Pitfall(
    title="Inadequate Production Monitoring and Observability",
    category=PitfallCategory.PRODUCTION_MONITORING,
    description="""
    Deploying RAG systems to production without comprehensive monitoring,
    leading to undetected performance degradation, failures, and inability
    to diagnose issues quickly.
    """,
    symptoms=[
        "Performance issues discovered by users, not monitoring",
        "Unable to diagnose production failures quickly",
        "No visibility into component-level performance",
        "Can't correlate failures with specific changes",
        "No early warning of degrading quality"
    ],
    root_cause="""
    Teams focus on development and initial deployment but neglect ongoing
    monitoring and observability. RAG systems are complex with many failure
    modes that require continuous monitoring to detect and diagnose.
    """,
    solution="""
    Implement comprehensive monitoring strategy:
    
    1. METRICS TO TRACK:
       System Metrics:
       - Latency (p50, p95, p99)
       - Throughput (queries/second)
       - Error rates
       - Resource utilization (CPU, memory, GPU)
       
       Component Metrics:
       - Retrieval latency and recall
       - Generation latency and quality
       - Cache hit rates
       - API call success rates
       
       Quality Metrics:
       - Faithfulness scores (sampled)
       - Answer relevancy (sampled)
       - User feedback (thumbs up/down)
       - Session abandonment rates
    
    2. ALERTING STRATEGY:
       Critical Alerts (immediate):
       - Error rate > 5%
       - Latency p95 > 5 seconds
       - API failures > 10%
       
       Warning Alerts (investigate):
       - Faithfulness score drops > 10%
       - User feedback negative > 30%
       - Cache hit rate drops > 20%
    
    3. CONTINUOUS EVALUATION:
       - Run evaluation pipeline on production traffic (sampled)
       - Maintain golden test set for regression testing
       - A/B test changes before full rollout
       - Collect user feedback systematically
    
    4. OBSERVABILITY:
       - Log all queries, contexts, responses
       - Trace requests through pipeline
       - Visualize component-level performance
       - Create debugging dashboards
    """,
    before_example="""
# BEFORE: No monitoring
def rag_endpoint(query: str) -> str:
    response = rag_pipeline.process_query(query)
    return response

# Problems:
# - No latency tracking
# - No error handling
# - No quality monitoring
# - No logging for debugging
    """,
    after_example="""
# AFTER: Comprehensive monitoring
import time
import logging
from prometheus_client import Counter, Histogram

# Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
latency_histogram = Histogram('rag_latency_seconds', 'RAG query latency')
error_counter = Counter('rag_errors_total', 'Total RAG errors', ['error_type'])
faithfulness_gauge = Gauge('rag_faithfulness_score', 'Faithfulness score')

def rag_endpoint(query: str) -> Dict:
    query_counter.inc()
    start_time = time.time()
    
    try:
        # Process query with component-level tracking
        retrieval_start = time.time()
        contexts = rag_pipeline.retrieve(query)
        retrieval_latency = time.time() - retrieval_start
        
        generation_start = time.time()
        response = rag_pipeline.generate(query, contexts)
        generation_latency = time.time() - generation_start
        
        # Sample quality evaluation (10% of traffic)
        if random.random() < 0.1:
            faithfulness = evaluate_faithfulness(query, response, contexts)
            faithfulness_gauge.set(faithfulness)
        
        # Log for debugging
        logging.info({
            'query': query,
            'retrieval_latency': retrieval_latency,
            'generation_latency': generation_latency,
            'num_contexts': len(contexts),
            'response_length': len(response)
        })
        
        # Record total latency
        total_latency = time.time() - start_time
        latency_histogram.observe(total_latency)
        
        return {
            'response': response,
            'metadata': {
                'latency_ms': total_latency * 1000,
                'num_contexts': len(contexts)
            }
        }
        
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        logging.error(f"RAG error: {e}", exc_info=True)
        raise
    """,
    prevention_tips=[
        "Implement monitoring before production deployment",
        "Track both system and quality metrics",
        "Set up alerting for critical issues",
        "Run continuous evaluation on production traffic",
        "Log all requests for debugging",
        "Create dashboards for component-level visibility",
        "Collect user feedback systematically",
        "Maintain golden test sets for regression testing"
    ]
)


# Compile all pitfalls
ALL_PITFALLS = [
    PITFALL_OVER_GENERIC_SYNTHETIC_DATA,
    PITFALL_COMPONENT_MISDIAGNOSIS,
    PITFALL_EMBEDDING_SELECTION,
    PITFALL_CHUNK_SIZE,
    PITFALL_TRADITIONAL_METRICS,
    PITFALL_PROMPT_ENGINEERING,
    PITFALL_PRODUCTION_MONITORING
]


def get_pitfalls_by_category(category: PitfallCategory) -> List[Pitfall]:
    """Get all pitfalls for a specific category"""
    return [p for p in ALL_PITFALLS if p.category == category]


def get_pitfall_summary() -> str:
    """Generate a summary of all pitfalls"""
    summary = "# Common Pitfalls in RAG Systems\n\n"
    
    for category in PitfallCategory:
        pitfalls = get_pitfalls_by_category(category)
        if pitfalls:
            summary += f"## {category.value.replace('_', ' ').title()}\n\n"
            for pitfall in pitfalls:
                summary += f"### {pitfall.title}\n"
                summary += f"{pitfall.description}\n\n"
                summary += f"**Key Symptoms:**\n"
                for symptom in pitfall.symptoms:
                    summary += f"- {symptom}\n"
                summary += "\n"
    
    return summary


if __name__ == "__main__":
    # Print summary of all pitfalls
    print(get_pitfall_summary())
    
    # Example: Get synthetic data pitfalls
    synthetic_pitfalls = get_pitfalls_by_category(PitfallCategory.SYNTHETIC_DATA)
    print(f"\nFound {len(synthetic_pitfalls)} pitfalls in synthetic data category")
