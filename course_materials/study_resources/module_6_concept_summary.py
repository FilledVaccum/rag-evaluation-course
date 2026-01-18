"""
Module 6 Concept Summary: Semantic Search System Evaluation
Evaluating RAG and Semantic Search Systems Course

One-page summary of key concepts, formulas, and decision frameworks
for quick review and exam preparation.

Requirements: 17.2
"""

CONCEPT_SUMMARY = """
================================================================================
MODULE 6: SEMANTIC SEARCH SYSTEM EVALUATION - CONCEPT SUMMARY
================================================================================

1. LEGACY SYSTEM EVALUATION
────────────────────────────────────────────────────────────────────────────

BM25 Characteristics:
  ✓ Strengths: Exact keyword matching, fast, no embeddings needed
  ✗ Weaknesses: No semantic understanding, vocabulary mismatch, synonym blindness

When BM25 Excels:
  • Technical terms, product codes, IDs
  • Legal documents (exact phrase matching)
  • Code search (function/variable names)
  • Regulatory compliance

Modern Evaluation Approach:
  1. LLM-as-a-Judge for semantic relevance
  2. Comparative analysis (BM25 vs semantic)
  3. User intent classification
  4. Failure mode analysis

────────────────────────────────────────────────────────────────────────────

2. ADAPTING RAGAS FOR NON-RAG SYSTEMS
────────────────────────────────────────────────────────────────────────────

Metric Adaptation:
  ✓ USE: Context Precision, Context Recall, Context Relevance
  ✗ SKIP: Faithfulness, Answer Relevancy (generation metrics)

Adaptation Process:
  1. Convert search results → contexts
  2. Focus on retrieval metrics only
  3. Use LLM-as-a-Judge for relevance
  4. Require ground truth for recall

Key Insight: Search systems = Retrieval only, no generation

────────────────────────────────────────────────────────────────────────────

3. HYBRID SEARCH SYSTEMS
────────────────────────────────────────────────────────────────────────────

Architecture:
  Query → [BM25 Search] ──┐
                          ├→ Fusion → Re-ranking → Results
  Query → [Vector Search] ┘

Reciprocal Rank Fusion (RRF):
  Formula: score(doc) = Σ 1/(k + rank(doc))
  Default k: 60
  Combines rankings from multiple systems

Evaluation Strategy:
  1. Component-level: Test each retriever independently
  2. Ablation studies: BM25 only, Vector only, Hybrid
  3. Query type analysis: Performance per query type
  4. Cost-benefit: Latency vs relevance tradeoff

When to Use Hybrid:
  • Queries have both keyword and semantic components
  • Want robustness over peak performance
  • Need to support diverse query types

────────────────────────────────────────────────────────────────────────────

4. RANKING METRICS
────────────────────────────────────────────────────────────────────────────

Context Precision (Ranking Quality):
  • Measures if relevant docs appear at top
  • Formula: P@k = (relevant in top k) / k
  • High precision = good ranking

Context Recall (Coverage):
  • Measures if all relevant docs retrieved
  • Formula: Recall = (retrieved relevant) / (total relevant)
  • Requires ground truth

Context Relevance (Semantic Match):
  • Percentage of retrieved docs that are relevant
  • Formula: Relevance = (relevant docs) / (total retrieved)
  • Use LLM-as-a-Judge for assessment

Mean Reciprocal Rank (MRR):
  • Position of first relevant result
  • Formula: MRR = 1 / rank_of_first_relevant
  • Use case: Single correct answer queries

NDCG (Normalized Discounted Cumulative Gain):
  • Ranking quality with graded relevance
  • Formula: DCG = Σ (relevance / log₂(position + 1))
  • Penalizes relevant docs appearing lower

────────────────────────────────────────────────────────────────────────────

5. PERFORMANCE PATTERNS & OPTIMIZATION
────────────────────────────────────────────────────────────────────────────

Pattern: High Precision + Low Recall
  Problem: Too selective, missing relevant docs
  Solution: Increase k or lower threshold
  Impact: +15-20% recall

Pattern: Low Precision + High Recall
  Problem: Too broad, irrelevant docs included
  Solution: Add re-ranking or stricter filtering
  Impact: +20-30% precision

Pattern: Low Relevance
  Problem: Poor semantic matching
  Solution: Better embeddings or domain-specific model
  Impact: +20-30% relevance

Optimization Strategies:
  1. Re-ranking with cross-encoders (top 20-50 results)
  2. Learning to Rank (LTR) with user feedback
  3. Query expansion for better recall
  4. Personalization based on user context
  5. A/B testing for validation

────────────────────────────────────────────────────────────────────────────

6. SYSTEM COMPARISON FRAMEWORK
────────────────────────────────────────────────────────────────────────────

BM25 vs Vector Search vs RAG:

                    BM25        Vector      RAG
  Exact Match       ★★★★★       ★★          ★★★
  Semantic          ★           ★★★★★       ★★★★★
  Speed             ★★★★★       ★★★         ★★
  Cost              ★★★★★       ★★★         ★
  Synthesis         ✗           ✗           ★★★★★

Decision Framework:
  • Document retrieval → BM25 or Vector
  • Question answering → RAG
  • Mixed queries → Hybrid (BM25 + Vector)
  • Exact terms critical → BM25
  • Semantic understanding critical → Vector or RAG

────────────────────────────────────────────────────────────────────────────

7. ENTERPRISE INTEGRATION
────────────────────────────────────────────────────────────────────────────

Integration Patterns:
  • Sidecar: Evaluate alongside production
  • Shadow: Compare old vs new systems
  • Gradual: Incremental traffic migration (5% → 100%)
  • Fallback: New primary, old fallback

Practical Considerations:
  ✓ Data privacy and compliance (GDPR, HIPAA)
  ✓ Performance impact on production
  ✓ Cost management (LLM-as-a-Judge at scale)
  ✓ Stakeholder buy-in with pilot projects
  ✓ Continuous monitoring for drift

────────────────────────────────────────────────────────────────────────────

8. KEY FORMULAS (QUICK REFERENCE)
────────────────────────────────────────────────────────────────────────────

Precision@K:        P@k = (relevant in top k) / k

Recall:             R = (retrieved relevant) / (total relevant)

F1 Score:           F1 = 2 × (P × R) / (P + R)

MRR:                MRR = 1 / rank_of_first_relevant

RRF:                score(d) = Σ 1/(k + rank(d))

Cosine Similarity:  cos(θ) = (A · B) / (||A|| × ||B||)

────────────────────────────────────────────────────────────────────────────

9. EXAM TIPS
────────────────────────────────────────────────────────────────────────────

Key Concepts to Remember:
  ✓ Ragas adapts to search-only systems (skip generation metrics)
  ✓ RRF combines rankings from multiple systems
  ✓ High precision + low recall = too selective
  ✓ Low precision + high recall = too broad
  ✓ LLM-as-a-Judge provides semantic assessment
  ✓ Hybrid systems leverage strengths of both approaches

Common Mistakes to Avoid:
  ✗ Using generation metrics for search-only systems
  ✗ Comparing systems without ground truth
  ✗ Ignoring query type differences
  ✗ Replacing legacy systems without evaluation
  ✗ Optimizing single metric at expense of others

────────────────────────────────────────────────────────────────────────────

10. NVIDIA PLATFORM INTEGRATION
────────────────────────────────────────────────────────────────────────────

NVIDIA NIM for Embeddings:
  • NV-Embed-v2 for semantic search
  • Domain-specific models available
  • Fast inference with optimized serving

NVIDIA NIMs for LLM-as-a-Judge:
  • Llama-3.1-70B-Instruct for relevance scoring
  • Mistral-Large for complex evaluations
  • Cost-effective at scale

NVIDIA Triton:
  • Deploy evaluation pipelines in production
  • Batch processing for efficiency
  • Real-time monitoring

Reference: NVIDIA Agent Intelligence Toolkit

================================================================================
CERTIFICATION ALIGNMENT: Evaluation & Tuning (13%), Knowledge Integration (10%)
================================================================================
"""


def get_concept_summary() -> str:
    """
    Get the one-page concept summary.
    
    Returns:
        Formatted concept summary string
    """
    return CONCEPT_SUMMARY


def print_summary():
    """Print the concept summary."""
    print(CONCEPT_SUMMARY)


# Quick reference cards for specific topics

QUICK_REFERENCE_CARDS = {
    "metric_selection": """
    METRIC SELECTION GUIDE
    ══════════════════════════════════════════════════════════════
    
    Use Case                          Recommended Metrics
    ──────────────────────────────────────────────────────────────
    Search-only system                Context Precision, Recall, Relevance
    RAG system                        All Ragas metrics
    Ranking quality                   Context Precision, NDCG
    Coverage assessment               Context Recall
    Semantic relevance                Context Relevance (LLM-as-a-Judge)
    Single answer queries             MRR
    Multiple relevant docs            NDCG, MAP
    ══════════════════════════════════════════════════════════════
    """,
    
    "optimization_decision_tree": """
    OPTIMIZATION DECISION TREE
    ══════════════════════════════════════════════════════════════
    
    Low Precision?
      ├─ Yes → Add re-ranking or stricter filtering
      └─ No → Check recall
    
    Low Recall?
      ├─ Yes → Increase k or lower threshold
      └─ No → Check relevance
    
    Low Relevance?
      ├─ Yes → Improve embeddings or use hybrid
      └─ No → System performing well
    
    High Precision + Low Recall?
      └─ Too selective → Increase breadth + re-rank
    
    Low Precision + High Recall?
      └─ Too broad → Add filtering or better ranking
    ══════════════════════════════════════════════════════════════
    """,
    
    "system_selection": """
    SYSTEM SELECTION GUIDE
    ══════════════════════════════════════════════════════════════
    
    Query Type              Best System         Why
    ──────────────────────────────────────────────────────────────
    Exact keywords          BM25                Fast, precise matching
    Conceptual              Vector Search       Semantic understanding
    Mixed                   Hybrid              Combines both strengths
    Question answering      RAG                 Synthesizes responses
    Document retrieval      BM25 or Vector      No synthesis needed
    Technical terms         BM25                Exact matching critical
    Natural language        Vector or RAG       Semantic understanding
    ══════════════════════════════════════════════════════════════
    """
}


def get_quick_reference(topic: str) -> str:
    """
    Get a quick reference card for a specific topic.
    
    Args:
        topic: One of "metric_selection", "optimization_decision_tree", "system_selection"
    
    Returns:
        Quick reference card string
    """
    return QUICK_REFERENCE_CARDS.get(topic, "Topic not found")


if __name__ == "__main__":
    print_summary()
    
    print("\n\nQUICK REFERENCE CARDS:")
    print("=" * 80)
    for topic, card in QUICK_REFERENCE_CARDS.items():
        print(f"\n{topic.upper().replace('_', ' ')}")
        print(card)
