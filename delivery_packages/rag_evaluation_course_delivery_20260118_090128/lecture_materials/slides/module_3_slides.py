"""
Module 3: RAG Architecture and Component Analysis - Slide Deck
Comprehensive slide deck focusing on pipeline architecture and debugging
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


# Module 3 Slide Deck: RAG Architecture and Component Analysis
MODULE_3_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 3: RAG Architecture and Component Analysis",
        content=[
            "Mastering the RAG Pipeline",
            "",
            "Learning Objectives:",
            "• Understand three-stage RAG architecture",
            "• Diagnose component-level failures",
            "• Evaluate retrieval vs generation independently",
            "• Implement debugging workflows",
            "• Build end-to-end RAG systems"
        ],
        visual_notes="Title slide with RAG pipeline visualization",
        speaker_notes="60-90 minute module. Core content for certification. Focus on debugging skills."
    ),
    
    Slide(
        slide_number=2,
        title="RAG Pipeline: Three-Stage Architecture",
        content=[
            "The Complete RAG Flow",
            "",
            "Stage 1: Retrieval",
            "• Query understanding and expansion",
            "• Similarity search in vector store",
            "• Candidate document selection",
            "",
            "Stage 2: Augmentation",
            "• Context assembly and formatting",
            "• Prompt engineering",
            "• Token budget management",
            "",
            "Stage 3: Generation",
            "• LLM inference",
            "• Response synthesis",
            "• Citation and source attribution"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[User Query] --> B[Stage 1: Retrieval]
    B --> C[Retrieved Documents]
    C --> D[Stage 2: Augmentation]
    A --> D
    D --> E[Augmented Prompt]
    E --> F[Stage 3: Generation]
    F --> G[Final Response]
    
    style B fill:#e1f5ff
    style D fill:#fff4e1
    style F fill:#e8f5e9
```
""",
        visual_notes="Color-coded stages with data flow arrows",
        speaker_notes="Most failures happen in Stage 1 (retrieval), but developers often blame Stage 3 (generation)."
    ),
    
    Slide(
        slide_number=3,
        title="Stage 1: Retrieval Deep Dive",
        content=[
            "Finding Relevant Information",
            "",
            "Query Processing:",
            "• Query expansion (synonyms, related terms)",
            "• Query rewriting for clarity",
            "• Multi-query generation",
            "",
            "Retrieval Methods:",
            "• Dense retrieval (vector similarity)",
            "• Sparse retrieval (BM25)",
            "• Hybrid retrieval (combination)",
            "",
            "Ranking:",
            "• Initial scoring",
            "• Re-ranking with cross-encoders",
            "• Diversity filtering"
        ],
        visual_notes="Flowchart showing query processing steps",
        speaker_notes="Retrieval quality determines RAG quality. Garbage in, garbage out."
    ),
    
    Slide(
        slide_number=4,
        title="Stage 2: Augmentation Deep Dive",
        content=[
            "Preparing Context for the LLM",
            "",
            "Context Assembly:",
            "• Select top-k documents (typically 3-5)",
            "• Order by relevance or chronology",
            "• Format with clear delimiters",
            "",
            "Prompt Engineering:",
            "• System instructions",
            "• Context injection",
            "• Task specification",
            "• Output format requirements",
            "",
            "Token Management:",
            "• Context window limits (4K, 8K, 32K, 128K)",
            "• Truncation strategies",
            "• Compression techniques"
        ],
        visual_notes="Example prompt template with highlighted sections",
        speaker_notes="Prompt engineering is critical. Small changes can dramatically affect output quality."
    ),
    
    Slide(
        slide_number=5,
        title="Stage 3: Generation Deep Dive",
        content=[
            "Synthesizing the Final Response",
            "",
            "LLM Configuration:",
            "• Model selection (GPT-4, Llama, Mistral)",
            "• Temperature and sampling parameters",
            "• Max tokens and stop sequences",
            "",
            "Response Quality:",
            "• Faithfulness to retrieved context",
            "• Relevance to original query",
            "• Coherence and fluency",
            "• Citation accuracy",
            "",
            "Post-Processing:",
            "• Format validation",
            "• Safety filtering",
            "• Source attribution"
        ],
        visual_notes="LLM parameter tuning guide with examples",
        speaker_notes="Generation is usually not the problem. If output is bad, check retrieval first."
    ),
    
    Slide(
        slide_number=6,
        title="Component-Level Failure Diagnosis",
        content=[
            "The Critical Debugging Skill",
            "",
            "Common Mistake:",
            "❌ 'The LLM is hallucinating!' (blaming generation)",
            "",
            "Systematic Approach:",
            "✓ Step 1: Check retrieval quality",
            "✓ Step 2: Verify context relevance",
            "✓ Step 3: Inspect prompt construction",
            "✓ Step 4: Evaluate generation",
            "",
            "Key Insight:",
            "80% of RAG failures are retrieval problems,",
            "not generation problems"
        ],
        visual_notes="Decision tree for failure diagnosis",
        speaker_notes="This is the most important slide. Emphasize: always check retrieval first!"
    ),
    
    Slide(
        slide_number=7,
        title="Debugging Retrieval Failures",
        content=[
            "Identifying Retrieval Problems",
            "",
            "Symptoms:",
            "• Irrelevant documents retrieved",
            "• Missing key information",
            "• Low similarity scores",
            "",
            "Root Causes:",
            "• Wrong embedding model for domain",
            "• Poor chunking strategy",
            "• Query-document mismatch",
            "• Insufficient index coverage",
            "",
            "Solutions:",
            "• Switch to domain-specific embeddings",
            "• Adjust chunk size and overlap",
            "• Implement query expansion",
            "• Add more documents to index"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Poor RAG Output] --> B{Check Retrieved Docs}
    B -->|Irrelevant| C[Retrieval Failure]
    B -->|Relevant| D[Generation Failure]
    C --> E[Fix: Embeddings]
    C --> F[Fix: Chunking]
    C --> G[Fix: Query Processing]
```
""",
        visual_notes="Troubleshooting flowchart with color-coded paths",
        speaker_notes="Walk through real example. Show how to inspect retrieved documents."
    ),
    
    Slide(
        slide_number=8,
        title="Debugging Generation Failures",
        content=[
            "When Generation is Actually the Problem",
            "",
            "Symptoms:",
            "• Retrieved context is relevant",
            "• But response is still wrong/irrelevant",
            "",
            "Root Causes:",
            "• Poor prompt engineering",
            "• Insufficient context in prompt",
            "• Wrong LLM parameters",
            "• Model limitations",
            "",
            "Solutions:",
            "• Improve prompt clarity and specificity",
            "• Add examples (few-shot learning)",
            "• Adjust temperature and top-p",
            "• Try different/larger model"
        ],
        visual_notes="Side-by-side comparison of bad vs good prompts",
        speaker_notes="Show concrete examples of prompt improvements and their impact."
    ),
    
    Slide(
        slide_number=9,
        title="Context Relevance Assessment",
        content=[
            "Evaluating Retrieved Context Quality",
            "",
            "Manual Assessment:",
            "• Read retrieved documents",
            "• Check if they contain answer",
            "• Verify relevance to query",
            "",
            "Automated Metrics:",
            "• Similarity scores (cosine, dot product)",
            "• Overlap with ground truth",
            "• LLM-as-a-Judge relevance scoring",
            "",
            "Best Practice:",
            "• Always inspect top-3 retrieved documents",
            "• If they don't contain the answer, retrieval failed",
            "• Don't expect LLM to fix bad retrieval"
        ],
        visual_notes="Example showing query, retrieved docs, and relevance scores",
        speaker_notes="Emphasize: human inspection is still valuable. Don't rely only on metrics."
    ),
    
    Slide(
        slide_number=10,
        title="Response Accuracy and Faithfulness",
        content=[
            "Ensuring Grounded Responses",
            "",
            "Faithfulness:",
            "• Are claims supported by retrieved context?",
            "• No hallucinated information",
            "• Proper attribution to sources",
            "",
            "Accuracy:",
            "• Factually correct information",
            "• Relevant to the query",
            "• Complete answer (not partial)",
            "",
            "Evaluation Methods:",
            "• Claim extraction and verification",
            "• Entailment checking",
            "• Human evaluation",
            "• LLM-as-a-Judge scoring"
        ],
        visual_notes="Diagram showing claim verification process",
        speaker_notes="Faithfulness vs accuracy: faithfulness is about context, accuracy is about ground truth."
    ),
    
    Slide(
        slide_number=11,
        title="Orchestration and Multi-Step Reasoning",
        content=[
            "Beyond Simple RAG",
            "",
            "Advanced Patterns:",
            "• Multi-hop reasoning: Multiple retrieval rounds",
            "• Iterative refinement: Generate → Critique → Refine",
            "• Decomposition: Break complex queries into sub-queries",
            "• Verification: Fact-check generated responses",
            "",
            "Orchestration Frameworks:",
            "• LangChain: Python framework for LLM apps",
            "• LlamaIndex: Data framework for LLMs",
            "• Haystack: End-to-end NLP framework",
            "• Custom: Build your own orchestration logic"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Complex Query] --> B[Decompose]
    B --> C[Sub-Query 1]
    B --> D[Sub-Query 2]
    C --> E[Retrieve 1]
    D --> F[Retrieve 2]
    E --> G[Synthesize]
    F --> G
    G --> H[Final Answer]
```
""",
        visual_notes="Multi-step reasoning flow diagram",
        speaker_notes="Advanced topic. Most students will use simple RAG, but good to know what's possible."
    ),
    
    Slide(
        slide_number=12,
        title="Hands-On: Notebook 2",
        content=[
            "RAG Debugging Exercise",
            "",
            "You will:",
            "• Build complete RAG pipeline from scratch",
            "• Identify failure points in broken pipeline",
            "• Debug retrieval vs generation issues",
            "• Implement component-level evaluation",
            "• Fix intentional bugs in the code",
            "",
            "Key Skills:",
            "• Systematic debugging approach",
            "• Component isolation",
            "• Metric-driven optimization"
        ],
        visual_notes="Screenshot of notebook with debugging sections",
        speaker_notes="This is the most important hands-on. Give 30-35 minutes. Circulate actively."
    ),
    
    Slide(
        slide_number=13,
        title="Module 3 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ RAG has three stages: Retrieval, Augmentation, Generation",
            "✓ Most failures are retrieval problems, not generation",
            "✓ Always debug systematically: check retrieval first",
            "✓ Context relevance determines RAG quality",
            "✓ Component-level evaluation is essential",
            "",
            "Next Module: Synthetic Test Data Generation",
            "• LLM-based test data creation",
            "• Prompt engineering for data steering",
            "• Quality validation"
        ],
        visual_notes="Summary with debugging workflow diagram",
        speaker_notes="Reinforce the key message: check retrieval first! Take questions. 5-minute break."
    )
]


def get_module_3_slides() -> List[Slide]:
    """Return all slides for Module 3"""
    return MODULE_3_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format"""
    markdown = "# Module 3: RAG Architecture and Component Analysis\n\n"
    
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
    slides = get_module_3_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
