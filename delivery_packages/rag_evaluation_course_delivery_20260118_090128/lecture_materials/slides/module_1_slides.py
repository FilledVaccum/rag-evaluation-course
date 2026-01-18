"""
Module 1: Evolution of Search to RAG Systems - Slide Deck
Comprehensive slide deck with visual diagrams and Mermaid architecture diagrams
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


# Module 1 Slide Deck: Evolution of Search to RAG Systems
MODULE_1_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 1: Evolution of Search to RAG Systems",
        content=[
            "Understanding the Journey from Traditional Search to Modern RAG",
            "",
            "Learning Objectives:",
            "• Understand classic search architecture",
            "• Compare BM25 vs semantic search paradigms",
            "• Explore enterprise hybrid systems",
            "• Learn when to use each approach"
        ],
        visual_notes="Title slide with course branding and module icon",
        speaker_notes="Welcome students. This module covers 30-45 minutes. Set expectations for hands-on notebook at the end."
    ),
    
    Slide(
        slide_number=2,
        title="Classic Search Architecture",
        content=[
            "The Traditional Search Pipeline:",
            "",
            "1. Crawling: Discover and fetch web pages",
            "2. Analysis: Extract text and metadata",
            "3. Indexing: Build inverted index structures",
            "4. Ranking: Score and order results"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[Web Pages] --> B[Crawler]
    B --> C[Analyzer]
    C --> D[Indexer]
    D --> E[Index Storage]
    F[User Query] --> G[Ranking Engine]
    E --> G
    G --> H[Ranked Results]
```
""",
        visual_notes="Show data flow from web pages through to ranked results",
        speaker_notes="Emphasize that this architecture dominated for decades. Google, Bing, etc. all use variations of this."
    ),
    
    Slide(
        slide_number=3,
        title="BM25: The Keyword Ranking Algorithm",
        content=[
            "Best Match 25 (BM25) - Industry Standard",
            "",
            "Key Characteristics:",
            "• Term frequency (TF): How often does term appear?",
            "• Inverse document frequency (IDF): How rare is the term?",
            "• Document length normalization",
            "• Tunable parameters (k1, b)",
            "",
            "Strengths: Fast, interpretable, works well for exact matches",
            "Limitations: No semantic understanding, synonym problems"
        ],
        visual_notes="Show BM25 formula with color-coded components",
        speaker_notes="BM25 is still widely used in production. It's not obsolete - it's complementary to semantic search."
    ),
    
    Slide(
        slide_number=4,
        title="The Semantic Search Revolution",
        content=[
            "From Keywords to Meaning",
            "",
            "Semantic Search Capabilities:",
            "• Understand query intent, not just keywords",
            "• Handle synonyms and paraphrases",
            "• Capture contextual meaning",
            "• Multi-dimensional similarity matching",
            "",
            "Powered by: Embedding models + Vector stores"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Query: 'car repair'] --> B[Embedding Model]
    B --> C[Vector: [0.2, 0.8, ...]]
    D[Documents] --> E[Embedding Model]
    E --> F[Vector Store]
    C --> G[Similarity Search]
    F --> G
    G --> H[Results: 'automobile maintenance', 'vehicle service']
```
""",
        visual_notes="Contrast with BM25 - show how semantic search finds 'automobile maintenance' for 'car repair'",
        speaker_notes="This is the key innovation. Embeddings capture meaning in high-dimensional space."
    ),
    
    Slide(
        slide_number=5,
        title="BM25 vs Semantic Search: When to Use Each",
        content=[
            "Decision Framework:",
            "",
            "Use BM25 When:",
            "• Exact keyword matching is critical (legal, compliance)",
            "• Query contains specific identifiers (product codes, IDs)",
            "• Low latency is paramount",
            "• Interpretability is required",
            "",
            "Use Semantic Search When:",
            "• Natural language queries",
            "• Synonym and paraphrase handling needed",
            "• Conceptual similarity matters",
            "• Multi-language support required"
        ],
        visual_notes="Side-by-side comparison table with checkmarks",
        speaker_notes="Most production systems use BOTH. This isn't either/or - it's about combining strengths."
    ),
    
    Slide(
        slide_number=6,
        title="Enterprise Hybrid Systems",
        content=[
            "The Best of Both Worlds",
            "",
            "Hybrid Architecture Components:",
            "1. BM25 Retrieval: Fast keyword matching",
            "2. Vector Search: Semantic understanding",
            "3. Re-ranking: LLM-based relevance scoring",
            "",
            "Benefits:",
            "• Precision of keyword search",
            "• Recall of semantic search",
            "• Flexibility to tune for specific use cases"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[User Query] --> B[BM25 Search]
    A --> C[Vector Search]
    B --> D[Candidate Set 1]
    C --> E[Candidate Set 2]
    D --> F[Merge & Deduplicate]
    E --> F
    F --> G[Re-ranking Model]
    G --> H[Final Ranked Results]
```
""",
        visual_notes="Show parallel paths merging into re-ranker",
        speaker_notes="This is what you'll see in production. Companies like Google, Microsoft use hybrid approaches."
    ),
    
    Slide(
        slide_number=7,
        title="From Search to RAG: The Next Evolution",
        content=[
            "Retrieval-Augmented Generation (RAG)",
            "",
            "Key Innovation: Combine retrieval with generation",
            "",
            "RAG Pipeline:",
            "1. Retrieve: Find relevant documents (BM25/Vector/Hybrid)",
            "2. Augment: Add retrieved context to prompt",
            "3. Generate: LLM produces informed response",
            "",
            "Why RAG?",
            "• Reduces hallucinations",
            "• Grounds responses in facts",
            "• Enables knowledge updates without retraining"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[User Question] --> B[Retrieval System]
    B --> C[Relevant Documents]
    C --> D[Augmented Prompt]
    A --> D
    D --> E[LLM Generator]
    E --> F[Grounded Response]
```
""",
        visual_notes="Highlight the augmentation step - this is what makes RAG different from pure search",
        speaker_notes="RAG is the foundation of modern AI assistants. ChatGPT with web search, Perplexity, etc."
    ),
    
    Slide(
        slide_number=8,
        title="RAG Architecture Deep Dive",
        content=[
            "Three-Stage Pipeline:",
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
        visual_notes="Three-column layout showing each stage with icons",
        speaker_notes="Understanding these stages is critical for debugging. Most failures happen in retrieval, not generation."
    ),
    
    Slide(
        slide_number=9,
        title="Enterprise Considerations",
        content=[
            "Real-World Challenges:",
            "",
            "• Legacy Systems: Can't replace existing BM25 overnight",
            "• Hybrid Requirements: Need both keyword and semantic",
            "• Scale: Millions of documents, thousands of queries/sec",
            "• Latency: Sub-second response times",
            "• Cost: Embedding and LLM inference costs",
            "• Compliance: Data privacy, regulatory requirements",
            "",
            "Solution: Gradual migration with hybrid approaches"
        ],
        visual_notes="Show enterprise architecture diagram with legacy and modern components",
        speaker_notes="This is why we focus on hybrid systems. Pure semantic search isn't always the answer."
    ),
    
    Slide(
        slide_number=10,
        title="Hands-On: Notebook 0",
        content=[
            "Search Paradigm Comparison Exercise",
            "",
            "You will:",
            "• Implement BM25 search on sample dataset",
            "• Implement vector search with NVIDIA NIM embeddings",
            "• Build hybrid search with re-ranking",
            "• Compare results on identical queries",
            "",
            "Key Learning:",
            "• See strengths and weaknesses of each approach",
            "• Understand when to use which method",
            "• Debug intentional bugs in the implementation"
        ],
        visual_notes="Screenshot of notebook interface",
        speaker_notes="Give students 20-25 minutes for this exercise. Circulate to help with bugs."
    ),
    
    Slide(
        slide_number=11,
        title="Module 1 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ Classic search uses BM25 for keyword matching",
            "✓ Semantic search uses embeddings for meaning",
            "✓ Hybrid systems combine both approaches",
            "✓ RAG adds generation to retrieval",
            "✓ Enterprise systems need gradual migration strategies",
            "",
            "Next Module: Embeddings and Vector Stores",
            "• Deep dive into embedding models",
            "• Vector store configuration",
            "• Chunking strategies"
        ],
        visual_notes="Summary slide with checkmarks and forward arrow",
        speaker_notes="Take questions. Preview next module. 5-minute break before Module 2."
    )
]


def get_module_1_slides() -> List[Slide]:
    """Return all slides for Module 1"""
    return MODULE_1_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format for easy viewing"""
    markdown = "# Module 1: Evolution of Search to RAG Systems\n\n"
    
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
    # Export slides to markdown for review
    slides = get_module_1_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
