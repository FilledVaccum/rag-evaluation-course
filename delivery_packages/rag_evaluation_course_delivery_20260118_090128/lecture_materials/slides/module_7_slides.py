"""
Module 7: Production Deployment and Advanced Topics - Slide Deck
Focus on production considerations, monitoring, and advanced techniques
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


# Module 7 Slide Deck
MODULE_7_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 7: Production Deployment and Advanced Topics",
        content=[
            "Taking RAG to Production Scale",
            "",
            "Learning Objectives:",
            "• Handle temporal data and time-weighted retrieval",
            "• Ensure regulatory compliance (GDPR, HIPAA)",
            "• Implement continuous evaluation",
            "• Optimize performance and cost trade-offs",
            "• Deploy monitoring and observability",
            "• Design A/B testing frameworks"
        ],
        visual_notes="Title slide with production architecture preview",
        speaker_notes="60-90 minutes. Final module. Focus on production readiness and advanced topics."
    ),
    
    Slide(
        slide_number=2,
        title="Temporal Data Handling",
        content=[
            "Managing Time-Sensitive Information",
            "",
            "Challenge:",
            "• Information changes over time",
            "• Old data may be outdated or incorrect",
            "• Need to prioritize recent information",
            "",
            "Solutions:",
            "• Time-weighted retrieval (boost recent docs)",
            "• Temporal metadata in embeddings",
            "• Periodic re-indexing",
            "• Document versioning",
            "• Expiration policies",
            "",
            "Example: News articles, product catalogs, regulations"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[Query] --> B[Retrieval]
    C[Document Age] --> D[Time Weight]
    D --> B
    B --> E[Weighted Results]
    E --> F[Recent Docs Prioritized]
```
""",
        visual_notes="Timeline showing document recency weighting",
        speaker_notes="Temporal handling is often overlooked but critical for many use cases."
    ),
    
    Slide(
        slide_number=3,
        title="Regulatory Compliance",
        content=[
            "GDPR, HIPAA, and Data Privacy",
            "",
            "Key Requirements:",
            "• Data minimization (only collect what's needed)",
            "• Right to deletion (remove user data)",
            "• Data encryption (at rest and in transit)",
            "• Access controls and audit logs",
            "• Consent management",
            "",
            "RAG-Specific Considerations:",
            "• Embedding storage (contains sensitive data)",
            "• LLM training data (privacy concerns)",
            "• Retrieved context (may expose PII)",
            "• Response logging (compliance requirements)",
            "",
            "Solution: Privacy-preserving RAG architectures"
        ],
        visual_notes="Compliance checklist with regulatory logos",
        speaker_notes="Compliance is non-negotiable for healthcare, finance, legal domains."
    ),
    
    Slide(
        slide_number=4,
        title="Continuous Evaluation in Production",
        content=[
            "Monitoring RAG Quality Over Time",
            "",
            "Why Continuous Evaluation?",
            "• Data drift (documents change)",
            "• Model drift (LLM behavior changes)",
            "• User behavior changes",
            "• New edge cases emerge",
            "",
            "Implementation:",
            "1. Automated test set generation",
            "2. Scheduled evaluation runs (daily/weekly)",
            "3. Metric tracking and alerting",
            "4. Anomaly detection",
            "5. Automated rollback on quality degradation"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Production RAG] --> B[Synthetic Test Gen]
    B --> C[Evaluation Pipeline]
    C --> D[Metrics Dashboard]
    D --> E{Quality OK?}
    E -->|Yes| F[Continue]
    E -->|No| G[Alert + Rollback]
    F --> B
```
""",
        visual_notes="Continuous evaluation loop diagram",
        speaker_notes="This is how mature ML teams operate. Evaluation doesn't stop at deployment."
    ),
    
    Slide(
        slide_number=5,
        title="Performance Profiling",
        content=[
            "Identifying Bottlenecks",
            "",
            "Key Metrics:",
            "• Latency: End-to-end response time",
            "• Throughput: Queries per second",
            "• Resource utilization: CPU, GPU, memory",
            "• Cost per query",
            "",
            "Common Bottlenecks:",
            "• Embedding generation (GPU-bound)",
            "• Vector search (memory-bound)",
            "• LLM inference (GPU-bound, expensive)",
            "• Network latency (API calls)",
            "",
            "Optimization Strategies:",
            "• Caching (embeddings, responses)",
            "• Batch processing",
            "• Model quantization",
            "• Async processing"
        ],
        visual_notes="Performance profiling dashboard with bottleneck highlights",
        speaker_notes="Performance optimization is iterative. Measure, identify bottleneck, optimize, repeat."
    ),
    
    Slide(
        slide_number=6,
        title="Cost-Efficiency vs Accuracy Trade-offs",
        content=[
            "Balancing Quality and Budget",
            "",
            "Cost Drivers:",
            "• Embedding API calls ($)",
            "• LLM inference ($$)",
            "• Vector store operations ($)",
            "• Compute resources ($$)",
            "",
            "Optimization Strategies:",
            "• Use smaller models for simple queries",
            "• Cache frequent queries",
            "• Batch processing for efficiency",
            "• Tiered service levels (fast/accurate/cheap)",
            "• Query routing (simple → cheap, complex → expensive)",
            "",
            "Example: 80% of queries can use smaller model, 20% need GPT-4"
        ],
        visual_notes="Cost vs accuracy curve with optimal operating point",
        speaker_notes="Cost optimization is critical for production. Can't just use GPT-4 for everything."
    ),
    
    Slide(
        slide_number=7,
        title="A/B Testing for RAG Systems",
        content=[
            "Comparing System Variants",
            "",
            "What to A/B Test:",
            "• Embedding models (NV-Embed vs OpenAI)",
            "• Chunking strategies (512 vs 1024 tokens)",
            "• Retrieval methods (BM25 vs vector vs hybrid)",
            "• LLM models (GPT-4 vs Llama vs Mistral)",
            "• Prompt templates",
            "",
            "Implementation:",
            "1. Define success metrics",
            "2. Split traffic (50/50 or 90/10)",
            "3. Run for statistical significance",
            "4. Analyze results",
            "5. Roll out winner",
            "",
            "Key: Need sufficient sample size for valid conclusions"
        ],
        visual_notes="A/B test dashboard with statistical significance indicators",
        speaker_notes="A/B testing is the gold standard for production optimization. Requires discipline."
    ),
    
    Slide(
        slide_number=8,
        title="Monitoring and Observability",
        content=[
            "Visibility into Production Systems",
            "",
            "What to Monitor:",
            "• System metrics: Latency, throughput, errors",
            "• Quality metrics: Faithfulness, relevancy, precision",
            "• Business metrics: User satisfaction, task completion",
            "• Cost metrics: API spend, compute costs",
            "",
            "Observability Tools:",
            "• Logging: Structured logs for debugging",
            "• Metrics: Time-series data (Prometheus, Grafana)",
            "• Tracing: Request flow visualization (Jaeger)",
            "• Alerting: Automated notifications (PagerDuty)",
            "",
            "Best Practice: Comprehensive dashboards for all stakeholders"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[RAG System] --> B[Logs]
    A --> C[Metrics]
    A --> D[Traces]
    B --> E[Dashboard]
    C --> E
    D --> E
    E --> F[Alerts]
    F --> G[On-Call Engineer]
```
""",
        visual_notes="Monitoring architecture with dashboard examples",
        speaker_notes="Observability is essential. Can't fix what you can't see."
    ),
    
    Slide(
        slide_number=9,
        title="Feedback Loops and Iterative Improvement",
        content=[
            "Learning from Production Data",
            "",
            "Feedback Sources:",
            "• Explicit: User ratings, thumbs up/down",
            "• Implicit: Click-through rates, dwell time",
            "• Support tickets: Common failure patterns",
            "• Manual review: Expert evaluation",
            "",
            "Improvement Cycle:",
            "1. Collect feedback",
            "2. Identify failure patterns",
            "3. Generate new test cases",
            "4. Improve system components",
            "5. Evaluate improvements",
            "6. Deploy and monitor",
            "",
            "Key: Close the loop from production to development"
        ],
        visual_notes="Feedback loop diagram with improvement cycle",
        speaker_notes="Best RAG systems continuously improve based on real usage. This is the virtuous cycle."
    ),
    
    Slide(
        slide_number=10,
        title="Multi-Language and Low-Resource Languages",
        content=[
            "Global Deployment Challenges",
            "",
            "Challenges:",
            "• Embedding models trained on English",
            "• Limited training data for low-resource languages",
            "• Cross-lingual retrieval",
            "• Translation quality issues",
            "",
            "Solutions:",
            "• Multilingual embedding models (mBERT, XLM-R)",
            "• Language-specific models where available",
            "• Translation-based approaches",
            "• Cross-lingual transfer learning",
            "",
            "Example Languages:",
            "• High-resource: English, Chinese, Spanish",
            "• Low-resource: Swahili, Icelandic, Maori"
        ],
        visual_notes="World map showing language coverage",
        speaker_notes="Multi-language support is complex. Often requires language-specific strategies."
    ),
    
    Slide(
        slide_number=11,
        title="Advanced Topics Preview",
        content=[
            "Beyond This Course",
            "",
            "Advanced Techniques:",
            "• Multi-modal RAG (text + images + tables)",
            "• Graph-based retrieval (knowledge graphs)",
            "• Agentic RAG (autonomous agents)",
            "• Federated RAG (distributed data sources)",
            "• Streaming RAG (real-time updates)",
            "",
            "Emerging Research:",
            "• Self-correcting RAG",
            "• Uncertainty quantification",
            "• Explainable retrieval",
            "• Privacy-preserving embeddings",
            "",
            "Resources: NVIDIA courses, research papers, community forums"
        ],
        visual_notes="Technology roadmap with emerging trends",
        speaker_notes="RAG is rapidly evolving. This course provides foundation for continued learning."
    ),
    
    Slide(
        slide_number=12,
        title="Hands-On: Notebook 7",
        content=[
            "Production Monitoring Exercise",
            "",
            "You will:",
            "• Implement production monitoring pipeline",
            "• Design A/B test for RAG variants",
            "• Set up continuous evaluation",
            "• Create alerting rules",
            "• Analyze performance metrics",
            "• Optimize cost vs accuracy trade-offs",
            "",
            "Key Skills:",
            "• Production deployment",
            "• Monitoring and observability",
            "• A/B testing methodology"
        ],
        visual_notes="Notebook screenshot with monitoring dashboard",
        speaker_notes="Give 20-25 minutes. Focus on production readiness mindset."
    ),
    
    Slide(
        slide_number=13,
        title="Module 7 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ Temporal data requires time-weighted retrieval",
            "✓ Compliance is critical for regulated industries",
            "✓ Continuous evaluation prevents quality degradation",
            "✓ Performance optimization balances cost and accuracy",
            "✓ A/B testing validates improvements",
            "✓ Monitoring and observability are essential",
            "✓ Feedback loops enable continuous improvement",
            "",
            "Course Complete!",
            "• Review certification study guide",
            "• Practice with mock exam",
            "• Complete capstone project"
        ],
        visual_notes="Course completion celebration with next steps",
        speaker_notes="Congratulations! Review key concepts. Discuss certification preparation. Q&A session."
    )
]


def get_module_7_slides() -> List[Slide]:
    """Return all slides for Module 7"""
    return MODULE_7_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format"""
    markdown = "# Module 7: Production Deployment and Advanced Topics\n\n"
    
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
    slides = get_module_7_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
