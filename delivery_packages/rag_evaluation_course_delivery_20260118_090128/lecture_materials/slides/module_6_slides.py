"""
Module 6: Semantic Search System Evaluation - Slide Deck
Focus on legacy systems and hybrid evaluation strategies
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Slide:
    """Represents a single slide in the presentation"""
    slide_number: int
    title: str
    content: List[str]
    mermaid_diagram: Optional[str] = None
    visual_notes: Optional[str] = None
    speaker_notes: Optional[str] = None


# Module 6 Slide Deck
MODULE_6_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 6: Semantic Search System Evaluation",
        content=[
            "Evaluating Legacy and Hybrid Systems",
            "",
            "Learning Objectives:",
            "• Evaluate legacy BM25 systems with modern techniques",
            "• Apply Ragas to non-RAG search systems",
            "• Implement hybrid evaluation strategies",
            "• Assess ranking algorithm performance",
            "• Support enterprise migration scenarios"
        ],
        visual_notes="Title slide with legacy-to-modern transition visual",
        speaker_notes="90-120 minutes. Enterprise-focused. Addresses real-world migration challenges."
    ),
    
    Slide(
        slide_number=2,
        title="The Enterprise Reality: Legacy Systems",
        content=[
            "Why Legacy Search Matters",
            "",
            "Enterprise Challenges:",
            "• Existing BM25 systems in production",
            "• Can't replace overnight (risk, cost, complexity)",
            "• Need to evaluate current performance",
            "• Gradual migration to semantic search",
            "• Hybrid approaches during transition",
            "",
            "Key Insight:",
            "Modern evaluation techniques can assess legacy systems",
            "Don't need to rebuild to measure quality"
        ],
        visual_notes="Enterprise architecture showing legacy and modern components",
        speaker_notes="This resonates with students working in enterprises. Most have legacy systems."
    ),
    
    Slide(
        slide_number=3,
        title="Applying Ragas to Non-RAG Systems",
        content=[
            "Adapting RAG Metrics for Search",
            "",
            "Key Adaptation:",
            "• RAG = Retrieval + Generation",
            "• Search = Retrieval only",
            "• Solution: Use retrieval metrics, skip generation metrics",
            "",
            "Applicable Metrics:",
            "✓ Context Precision (ranking quality)",
            "✓ Context Recall (coverage)",
            "✓ Context Relevance (retrieval quality)",
            "",
            "Not Applicable:",
            "✗ Faithfulness (no generation)",
            "✗ Answer Relevancy (no generation)"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[Query] --> B[BM25 Search]
    B --> C[Retrieved Results]
    C --> D[Ragas Retrieval Metrics]
    D --> E[Evaluation Report]
    
    style B fill:#ffe1e1
    style D fill:#e1f5ff
```
""",
        visual_notes="Adaptation diagram showing metric mapping",
        speaker_notes="This is the key insight. Ragas is flexible enough for non-RAG systems."
    ),
    
    Slide(
        slide_number=4,
        title="Evaluating BM25 with LLM-as-a-Judge",
        content=[
            "Modern Evaluation for Traditional Search",
            "",
            "Process:",
            "1. Run BM25 search on test queries",
            "2. Collect top-k results",
            "3. Use LLM to judge relevance of each result",
            "4. Calculate precision, recall, NDCG",
            "5. Compare with ground truth or human judgments",
            "",
            "Benefits:",
            "• Semantic relevance assessment (not just keyword match)",
            "• Scalable evaluation",
            "• Consistent scoring",
            "• Identifies specific failure cases"
        ],
        visual_notes="BM25 evaluation pipeline diagram",
        speaker_notes="This gives enterprises a way to measure current system quality objectively."
    ),
    
    Slide(
        slide_number=5,
        title="Hybrid Evaluation Strategies",
        content=[
            "Evaluating Combined Systems",
            "",
            "Hybrid System Architecture:",
            "• BM25 retrieval (keyword matching)",
            "• Vector search (semantic matching)",
            "• Re-ranking (LLM-based scoring)",
            "• Final merged results",
            "",
            "Evaluation Approach:",
            "1. Evaluate each component independently",
            "2. Measure component contribution",
            "3. Assess final merged results",
            "4. Identify bottlenecks",
            "5. Optimize weakest component"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Query] --> B[BM25]
    A --> C[Vector Search]
    B --> D[Merge]
    C --> D
    D --> E[Re-ranker]
    E --> F[Final Results]
    
    B -.->|Eval| G[BM25 Metrics]
    C -.->|Eval| H[Vector Metrics]
    E -.->|Eval| I[Re-ranking Metrics]
    F -.->|Eval| J[End-to-End Metrics]
```
""",
        visual_notes="Multi-stage evaluation architecture",
        speaker_notes="Component-level evaluation is critical. Can't optimize what you don't measure."
    ),
    
    Slide(
        slide_number=6,
        title="Ranking Algorithm Assessment",
        content=[
            "Measuring Ranking Quality",
            "",
            "Key Metrics:",
            "• Precision@k: Relevance in top-k results",
            "• Recall@k: Coverage in top-k results",
            "• NDCG: Normalized Discounted Cumulative Gain",
            "• MRR: Mean Reciprocal Rank",
            "",
            "NDCG Explained:",
            "• Considers position of relevant results",
            "• Higher weight for top positions",
            "• Score: 0 (worst) to 1 (perfect)",
            "• Industry standard for ranking evaluation",
            "",
            "Example: NDCG@10 = 0.85 (good ranking)"
        ],
        visual_notes="NDCG calculation example with visualization",
        speaker_notes="NDCG is the gold standard for ranking. Students should understand the formula."
    ),
    
    Slide(
        slide_number=7,
        title="Comparing RAG vs Traditional Search",
        content=[
            "Head-to-Head Evaluation",
            "",
            "Comparison Dimensions:",
            "• Retrieval quality (precision, recall)",
            "• Response quality (for RAG)",
            "• Latency and throughput",
            "• Cost per query",
            "• User satisfaction",
            "",
            "Typical Findings:",
            "• RAG: Better semantic understanding, higher cost",
            "• BM25: Faster, cheaper, good for exact matches",
            "• Hybrid: Best of both, more complex",
            "",
            "Decision Framework:",
            "• Use case requirements",
            "• Budget constraints",
            "• Latency requirements",
            "• Accuracy needs"
        ],
        visual_notes="Comparison matrix with scores",
        speaker_notes="No universal winner. Choice depends on requirements and constraints."
    ),
    
    Slide(
        slide_number=8,
        title="Integration with Enterprise Systems",
        content=[
            "Real-World Deployment Considerations",
            "",
            "Integration Challenges:",
            "• Existing search infrastructure",
            "• Data pipelines and ETL",
            "• Authentication and authorization",
            "• Monitoring and logging",
            "• SLA requirements",
            "",
            "Evaluation in Context:",
            "• Test with production data",
            "• Measure end-to-end latency",
            "• Assess scalability",
            "• Validate compliance requirements",
            "• User acceptance testing"
        ],
        visual_notes="Enterprise integration architecture diagram",
        speaker_notes="Evaluation doesn't stop at accuracy. Must consider operational requirements."
    ),
    
    Slide(
        slide_number=9,
        title="Migration Strategies",
        content=[
            "From Legacy to Modern Search",
            "",
            "Gradual Migration Approach:",
            "1. Baseline: Evaluate current BM25 system",
            "2. Pilot: Deploy semantic search for subset",
            "3. A/B Test: Compare BM25 vs semantic",
            "4. Hybrid: Combine both approaches",
            "5. Optimize: Tune based on metrics",
            "6. Scale: Gradual rollout",
            "",
            "Risk Mitigation:",
            "• Maintain BM25 as fallback",
            "• Monitor metrics continuously",
            "• Rollback capability",
            "• Incremental user migration"
        ],
        visual_notes="Migration timeline with milestones",
        speaker_notes="This is how real enterprises migrate. Not big-bang replacement."
    ),
    
    Slide(
        slide_number=10,
        title="Hands-On: Notebook 6",
        content=[
            "Semantic Search Evaluation Exercise",
            "",
            "You will:",
            "• Evaluate semantic search with LLM-as-a-Judge",
            "• Apply Ragas metrics to BM25 system",
            "• Compare RAG vs traditional search performance",
            "• Implement hybrid evaluation workflow",
            "• Analyze component contributions",
            "• Debug intentional bugs",
            "",
            "Key Skills:",
            "• Legacy system evaluation",
            "• Comparative analysis",
            "• Hybrid system assessment"
        ],
        visual_notes="Notebook screenshot with comparison dashboard",
        speaker_notes="Give 25-30 minutes. Focus on comparative analysis skills."
    ),
    
    Slide(
        slide_number=11,
        title="Module 6 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ Legacy BM25 systems can be evaluated with modern techniques",
            "✓ Ragas metrics apply to non-RAG search systems",
            "✓ Hybrid evaluation requires component-level assessment",
            "✓ NDCG is the standard for ranking quality",
            "✓ Migration requires gradual, measured approach",
            "",
            "Next Module: Production Deployment and Advanced Topics",
            "• Continuous evaluation in production",
            "• Monitoring and observability",
            "• A/B testing frameworks"
        ],
        visual_notes="Summary with migration roadmap",
        speaker_notes="This module bridges theory and enterprise reality. Take questions. 5-minute break."
    )
]


def get_module_6_slides() -> List[Slide]:
    """Return all slides for Module 6"""
    return MODULE_6_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format"""
    markdown = "# Module 6: Semantic Search System Evaluation\n\n"
    
    for slide in slides:
        markdown += f"## Slide {slide.slide_number}: {slide.title}\n\n"
        
        for line in slide.content:
            markdown += f"{line}\n"
        markdown += "\n"
        
        if slide.mermaid_diagram:
            markdown += f"{slide.mermaid_diagram}\n\n"
        
        if slide.visual_notes:
            markdown += f"**Visual Notes:** {slide.visual_notes}\n\n"
        
        if slide.speaker_notes:
            markdown += f"**Speaker Notes:** {slide.speaker_notes}\n\n"
        
        markdown += "---\n\n"
    
    return markdown


if __name__ == "__main__":
    slides = get_module_6_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
