"""
Module 5: RAG Evaluation Metrics and Frameworks - Slide Deck
Deep dive into evaluation methodologies and Ragas framework
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


# Module 5 Slide Deck
MODULE_5_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 5: RAG Evaluation Metrics and Frameworks",
        content=[
            "Mastering RAG Evaluation",
            "",
            "Learning Objectives:",
            "• Understand LLM-as-a-Judge methodology",
            "• Master Ragas framework and metrics",
            "• Implement generation and retrieval metrics",
            "• Customize existing metrics",
            "• Create custom metrics from scratch",
            "• Interpret results for optimization"
        ],
        visual_notes="Title slide with metrics dashboard preview",
        speaker_notes="120-150 minutes. Core certification content. Primary focus of this course."
    ),
    
    Slide(
        slide_number=2,
        title="Why Traditional NLP Metrics Fail for RAG",
        content=[
            "The Limitations of BLEU, ROUGE, F1",
            "",
            "Problems:",
            "• BLEU/ROUGE: Measure n-gram overlap, not meaning",
            "• F1: Requires exact token matches",
            "• All: Miss semantic equivalence",
            "",
            "Example:",
            "Reference: 'The cat sat on the mat'",
            "Response: 'A feline rested on the rug'",
            "BLEU score: ~0.0 (no word overlap)",
            "Semantic similarity: High (same meaning)",
            "",
            "Conclusion: Need semantic evaluation methods"
        ],
        visual_notes="Side-by-side comparison showing metric failures",
        speaker_notes="This is a critical insight. Traditional metrics are misleading for RAG."
    ),
    
    Slide(
        slide_number=3,
        title="LLM-as-a-Judge: The Paradigm Shift",
        content=[
            "Using LLMs to Evaluate LLM Outputs",
            "",
            "Core Idea:",
            "• Use powerful LLM (GPT-4, Claude) as evaluator",
            "• Provide evaluation criteria and rubrics",
            "• LLM scores other LLM outputs",
            "",
            "Advantages:",
            "• Semantic understanding",
            "• Flexible criteria",
            "• Scalable automation",
            "• Nuanced scoring",
            "",
            "Limitations:",
            "• Cost (API calls)",
            "• Latency",
            "• Potential biases",
            "• Requires careful prompt engineering"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[RAG Output] --> B[Judge LLM]
    C[Evaluation Criteria] --> B
    D[Scoring Rubric] --> B
    B --> E[Score + Explanation]
```
""",
        visual_notes="LLM-as-a-Judge architecture diagram",
        speaker_notes="This is the foundation of modern RAG evaluation. Emphasize prompt engineering importance."
    ),
    
    Slide(
        slide_number=4,
        title="Ragas Framework Overview",
        content=[
            "Retrieval-Augmented Generation Assessment",
            "",
            "What is Ragas?",
            "• Open-source evaluation framework",
            "• Specialized for RAG systems",
            "• Built-in metrics for retrieval and generation",
            "• LLM-as-a-Judge implementation",
            "• Extensible and customizable",
            "",
            "Key Features:",
            "• Pre-built metrics (faithfulness, relevancy, etc.)",
            "• Metric customization",
            "• Batch evaluation",
            "• Integration with popular LLM frameworks"
        ],
        visual_notes="Ragas logo and architecture overview",
        speaker_notes="Ragas is the industry standard. Students must be proficient for certification."
    ),
    
    Slide(
        slide_number=5,
        title="Generation Metrics: Faithfulness",
        content=[
            "Are Claims Supported by Context?",
            "",
            "Definition:",
            "Measures if generated response is grounded in retrieved context",
            "",
            "Multi-Stage Process:",
            "1. Extract claims from response",
            "2. For each claim, check if supported by context",
            "3. Calculate: verified_claims / total_claims",
            "",
            "Example:",
            "Context: 'CSCI 567 is a 4-unit course'",
            "Response: 'CSCI 567 is worth 4 units and meets on Mondays'",
            "Claims: [4 units ✓, meets Mondays ✗]",
            "Faithfulness: 0.5"
        ],
        visual_notes="Claim extraction and verification flowchart",
        speaker_notes="Faithfulness is critical for preventing hallucinations. Walk through example carefully."
    ),
    
    Slide(
        slide_number=6,
        title="Generation Metrics: Answer Relevancy",
        content=[
            "Is Response Relevant to Question?",
            "",
            "Definition:",
            "Measures if generated answer addresses the original question",
            "",
            "Method:",
            "• Embed question and response",
            "• Calculate cosine similarity",
            "• Score: similarity(embed(question), embed(response))",
            "",
            "Example:",
            "Question: 'What are the prerequisites for CSCI 567?'",
            "Response A: 'Prerequisites are CSCI 270 and MATH 225' → High relevancy",
            "Response B: 'CSCI 567 is a great course' → Low relevancy"
        ],
        visual_notes="Embedding similarity visualization",
        speaker_notes="Relevancy is different from faithfulness. Can be faithful but not relevant."
    ),
    
    Slide(
        slide_number=7,
        title="Generation Metrics: Context Utilization",
        content=[
            "Does Response Use Retrieved Context?",
            "",
            "Definition:",
            "Measures if response actually uses the provided context",
            "",
            "Why It Matters:",
            "• LLM might ignore context and use parametric knowledge",
            "• Want to ensure RAG is actually working",
            "• Detect when retrieval is being bypassed",
            "",
            "Evaluation:",
            "• Check overlap between response and context",
            "• Verify citations and references",
            "• Compare with no-context baseline"
        ],
        visual_notes="Context utilization measurement diagram",
        speaker_notes="This catches cases where LLM ignores retrieved context. Important for debugging."
    ),
    
    Slide(
        slide_number=8,
        title="Retrieval Metrics: Context Precision",
        content=[
            "Ranking Quality of Retrieved Contexts",
            "",
            "Definition:",
            "Measures if relevant contexts appear higher in ranking",
            "",
            "Formula:",
            "precision@k = (relevant_in_top_k) / k",
            "",
            "Example:",
            "Retrieved docs: [Relevant, Irrelevant, Relevant, Irrelevant, Relevant]",
            "Precision@1: 1/1 = 1.0",
            "Precision@3: 2/3 = 0.67",
            "Precision@5: 3/5 = 0.60",
            "",
            "Goal: High precision at low k (relevant docs ranked first)"
        ],
        visual_notes="Ranking visualization with precision scores",
        speaker_notes="Precision measures ranking quality. Important for systems that use top-k retrieval."
    ),
    
    Slide(
        slide_number=9,
        title="Retrieval Metrics: Context Recall",
        content=[
            "Coverage of Ground Truth",
            "",
            "Definition:",
            "Measures if all necessary information was retrieved",
            "",
            "Formula:",
            "recall = (ground_truth_covered) / (total_ground_truth)",
            "",
            "Example:",
            "Ground truth: 'Prerequisites: CSCI 270, MATH 225, CSCI 360'",
            "Retrieved: 'Prerequisites: CSCI 270, MATH 225'",
            "Recall: 2/3 = 0.67 (missing CSCI 360)",
            "",
            "Trade-off: Recall vs Precision (retrieve more vs retrieve better)"
        ],
        visual_notes="Venn diagram showing coverage",
        speaker_notes="Recall measures completeness. Can have high precision but low recall."
    ),
    
    Slide(
        slide_number=10,
        title="Retrieval Metrics: Context Relevance",
        content=[
            "Is Retrieved Context Relevant?",
            "",
            "Definition:",
            "Measures if retrieved chunks are relevant to query",
            "",
            "Method:",
            "• Binary classification for each chunk",
            "• LLM judges: relevant or not relevant",
            "• Score: relevant_chunks / total_chunks",
            "",
            "Example:",
            "Query: 'CSCI 567 prerequisites'",
            "Chunk 1: 'Prerequisites are CSCI 270...' → Relevant",
            "Chunk 2: 'The course covers ML topics...' → Not relevant",
            "Chunk 3: 'Required: MATH 225...' → Relevant",
            "Relevance: 2/3 = 0.67"
        ],
        visual_notes="Relevance scoring interface",
        speaker_notes="This is the most direct measure of retrieval quality. Use for debugging."
    ),
    
    Slide(
        slide_number=11,
        title="Metric Interpretation and Insights",
        content=[
            "Turning Scores into Actions",
            "",
            "Low Faithfulness:",
            "→ LLM is hallucinating, improve prompt or use better model",
            "",
            "Low Answer Relevancy:",
            "→ LLM not addressing question, improve prompt specificity",
            "",
            "Low Context Precision:",
            "→ Ranking is poor, improve re-ranking or embedding model",
            "",
            "Low Context Recall:",
            "→ Missing information, retrieve more docs or improve chunking",
            "",
            "Low Context Relevance:",
            "→ Retrieval is broken, fix embeddings or query processing"
        ],
        visual_notes="Decision tree for metric interpretation",
        speaker_notes="This is the payoff. Metrics guide optimization. Walk through each scenario."
    ),
    
    Slide(
        slide_number=12,
        title="Customizing Ragas Metrics",
        content=[
            "Adapting Metrics to Your Needs",
            "",
            "Why Customize?",
            "• Domain-specific evaluation criteria",
            "• Different scoring rubrics",
            "• Custom prompt templates",
            "• Specialized use cases",
            "",
            "How to Customize:",
            "1. Modify existing metric prompts",
            "2. Adjust scoring thresholds",
            "3. Add domain-specific examples",
            "4. Change evaluation criteria",
            "",
            "Example: Customize faithfulness for medical domain",
            "• Add medical terminology",
            "• Stricter verification requirements",
            "• Citation format requirements"
        ],
        visual_notes="Before/after customization examples",
        speaker_notes="Customization is powerful. Show concrete example of prompt modification."
    ),
    
    Slide(
        slide_number=13,
        title="Creating Custom Metrics from Scratch",
        content=[
            "Building New Evaluation Dimensions",
            "",
            "When to Create Custom Metrics:",
            "• Unique evaluation requirements",
            "• Domain-specific quality criteria",
            "• Regulatory compliance needs",
            "• Novel RAG architectures",
            "",
            "Steps:",
            "1. Define evaluation criteria clearly",
            "2. Create scoring rubric (0-1 scale)",
            "3. Write LLM-as-a-Judge prompt",
            "4. Add calibration examples",
            "5. Test and validate",
            "",
            "Example: Citation accuracy metric",
            "• Check if citations are properly formatted",
            "• Verify citations match retrieved sources",
            "• Score based on accuracy percentage"
        ],
        visual_notes="Custom metric development workflow",
        speaker_notes="Advanced topic. Most students will customize, not create from scratch."
    ),
    
    Slide(
        slide_number=14,
        title="Hands-On: Notebook 5",
        content=[
            "Ragas Evaluation Implementation",
            "",
            "You will:",
            "• Implement Ragas on Amnesty Q&A dataset",
            "• Compute faithfulness and context recall",
            "• Analyze metric outputs and patterns",
            "• Customize existing metrics with new prompts",
            "• Create custom metric from scratch",
            "• Compare metric performance",
            "• Debug intentional bugs in metric prompts",
            "",
            "Key Skills:",
            "• Metric selection and interpretation",
            "• Prompt engineering for evaluation",
            "• Custom metric development"
        ],
        visual_notes="Notebook screenshot with metrics dashboard",
        speaker_notes="This is the most important hands-on. Give 35-40 minutes. This is core certification content."
    ),
    
    Slide(
        slide_number=15,
        title="Module 5 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ Traditional NLP metrics fail for RAG evaluation",
            "✓ LLM-as-a-Judge enables semantic evaluation",
            "✓ Ragas provides comprehensive RAG metrics",
            "✓ Generation metrics: Faithfulness, Relevancy, Utilization",
            "✓ Retrieval metrics: Precision, Recall, Relevance",
            "✓ Metrics guide optimization decisions",
            "✓ Customization enables domain-specific evaluation",
            "",
            "Next Module: Semantic Search System Evaluation",
            "• Evaluating legacy BM25 systems",
            "• Hybrid evaluation strategies"
        ],
        visual_notes="Comprehensive metrics summary table",
        speaker_notes="This is the heart of the course. Ensure solid understanding. Take questions. 10-minute break."
    )
]


def get_module_5_slides() -> List[Slide]:
    """Return all slides for Module 5"""
    return MODULE_5_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format"""
    markdown = "# Module 5: RAG Evaluation Metrics and Frameworks\n\n"
    
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
    slides = get_module_5_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
