"""
Module 2: Embeddings and Vector Stores - Slide Deck
Comprehensive slide deck with visual diagrams and technical architecture
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


# Module 2 Slide Deck: Embeddings and Vector Stores
MODULE_2_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 2: Embeddings and Vector Stores",
        content=[
            "Deep Dive into Semantic Representation",
            "",
            "Learning Objectives:",
            "• Understand embedding fundamentals",
            "• Explore domain-specific embedding models",
            "• Configure vector stores for optimal retrieval",
            "• Master chunking strategies",
            "• Handle tabular data in embedding space"
        ],
        visual_notes="Title slide with embedding visualization preview",
        speaker_notes="45-60 minute module. Heavy on technical concepts but with practical exercises."
    ),
    
    Slide(
        slide_number=2,
        title="What Are Embeddings?",
        content=[
            "Embeddings: Numerical Representations of Meaning",
            "",
            "Key Concepts:",
            "• Convert text to high-dimensional vectors (768, 1024, 1536 dims)",
            "• Similar meanings → Similar vectors",
            "• Distance metrics: Cosine similarity, Euclidean distance",
            "",
            "Example:",
            "'cat' → [0.2, 0.8, 0.1, ...]",
            "'kitten' → [0.3, 0.7, 0.2, ...]  (close to 'cat')",
            "'car' → [0.9, 0.1, 0.8, ...]  (far from 'cat')"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Text: 'cat'] --> B[Embedding Model]
    B --> C[Vector: 768 dimensions]
    D[Text: 'kitten'] --> B
    B --> E[Vector: 768 dimensions]
    C --> F[Cosine Similarity: 0.92]
    E --> F
```
""",
        visual_notes="3D visualization of word embeddings with 'cat', 'kitten', 'car' plotted",
        speaker_notes="Use analogy: embeddings are like GPS coordinates for meaning. Close in space = similar meaning."
    ),
    
    Slide(
        slide_number=3,
        title="Multi-Dimensional Similarity",
        content=[
            "Why High Dimensions Matter",
            "",
            "2D Space: Limited expressiveness",
            "• Can only capture 2 aspects of meaning",
            "",
            "768D Space: Rich semantic representation",
            "• Captures: topic, sentiment, style, domain, etc.",
            "• Enables nuanced similarity matching",
            "",
            "Visualization Challenge:",
            "• We can't visualize 768 dimensions",
            "• Use dimensionality reduction (t-SNE, UMAP) for 2D plots",
            "• But remember: full dimensions contain the real information"
        ],
        visual_notes="Show t-SNE plot of embeddings colored by category",
        speaker_notes="Students often struggle with high dimensions. Emphasize that it's okay not to 'see' it."
    ),
    
    Slide(
        slide_number=4,
        title="Domain-Specific Embedding Models",
        content=[
            "One Size Does NOT Fit All",
            "",
            "General Purpose:",
            "• OpenAI text-embedding-ada-002",
            "• NVIDIA NV-Embed-v2",
            "• Sentence-BERT",
            "",
            "Domain-Specific:",
            "• Code: CodeBERT, GraphCodeBERT",
            "• Finance: FinBERT, BloombergGPT embeddings",
            "• Healthcare: BioBERT, PubMedBERT",
            "• Legal: LegalBERT",
            "• Multilingual: mBERT, XLM-RoBERTa"
        ],
        visual_notes="Table comparing model performance across domains",
        speaker_notes="Common mistake: using general embeddings for specialized domains. Performance can drop 20-30%."
    ),
    
    Slide(
        slide_number=5,
        title="NVIDIA NIM Embedding Models",
        content=[
            "NVIDIA Inference Microservices (NIM)",
            "",
            "Key Models:",
            "• NV-Embed-v2: State-of-the-art general embeddings",
            "• Optimized for retrieval tasks",
            "• Fast inference with GPU acceleration",
            "",
            "Integration:",
            "```python",
            "from nvidia_nim import EmbeddingClient",
            "client = EmbeddingClient(api_key=API_KEY)",
            "embeddings = client.embed(['query text'])",
            "```",
            "",
            "Benefits: Managed service, automatic scaling, low latency"
        ],
        visual_notes="Architecture diagram showing NIM API integration",
        speaker_notes="NIM is key for NCP-AAI certification. Students should be comfortable with NVIDIA ecosystem."
    ),
    
    Slide(
        slide_number=6,
        title="Vector Stores: Storage and Retrieval",
        content=[
            "Specialized Databases for Embeddings",
            "",
            "Popular Vector Stores:",
            "• Pinecone: Managed, cloud-native",
            "• Milvus: Open-source, scalable",
            "• Chroma: Lightweight, developer-friendly",
            "• Weaviate: GraphQL interface",
            "• FAISS: Facebook's library (not a database)",
            "",
            "Key Features:",
            "• Approximate Nearest Neighbor (ANN) search",
            "• Horizontal scaling",
            "• Metadata filtering",
            "• Hybrid search capabilities"
        ],
        mermaid_diagram="""
```mermaid
graph TB
    A[Application] --> B[Vector Store API]
    B --> C[Index Layer]
    C --> D[Storage Layer]
    E[Query Vector] --> B
    B --> F[ANN Search]
    C --> F
    F --> G[Top-K Results]
```
""",
        visual_notes="Comparison table of vector store features and pricing",
        speaker_notes="Choice depends on scale, budget, and existing infrastructure. No single best option."
    ),
    
    Slide(
        slide_number=7,
        title="Vector Store Configuration",
        content=[
            "Optimization Strategies",
            "",
            "Index Types:",
            "• HNSW: Hierarchical Navigable Small World (fast, memory-intensive)",
            "• IVF: Inverted File Index (balanced)",
            "• LSH: Locality-Sensitive Hashing (approximate)",
            "",
            "Configuration Parameters:",
            "• Distance metric: Cosine vs Euclidean vs Dot Product",
            "• Index size: Trade-off between speed and accuracy",
            "• Sharding strategy: Horizontal scaling",
            "",
            "Performance Tuning:",
            "• Benchmark with your data",
            "• Monitor query latency",
            "• Adjust based on recall requirements"
        ],
        visual_notes="Performance graphs showing latency vs accuracy trade-offs",
        speaker_notes="This is where theory meets practice. Configuration matters for production systems."
    ),
    
    Slide(
        slide_number=8,
        title="Chunking Strategies: The Critical Decision",
        content=[
            "How to Split Documents for Embedding",
            "",
            "Why Chunking Matters:",
            "• Embeddings have token limits (512, 1024, 2048)",
            "• Chunk size affects retrieval quality",
            "• Too small: Loss of context",
            "• Too large: Diluted relevance",
            "",
            "Common Strategies:",
            "• Fixed size: 256, 512, 1024 tokens",
            "• Sentence-based: Natural boundaries",
            "• Paragraph-based: Semantic units",
            "• Sliding window: Overlapping chunks",
            "• Semantic chunking: AI-driven boundaries"
        ],
        visual_notes="Visual comparison of different chunking approaches on sample text",
        speaker_notes="No universal answer. Must experiment with your specific data and use case."
    ),
    
    Slide(
        slide_number=9,
        title="Chunking for Different Data Types",
        content=[
            "Data Type-Specific Strategies",
            "",
            "Text Documents:",
            "• Paragraph or section-based chunking",
            "• Preserve semantic coherence",
            "",
            "Code:",
            "• Function or class-level chunks",
            "• Include docstrings and comments",
            "",
            "Tabular Data:",
            "• Row-based: Each row becomes a chunk",
            "• Column selection: Only relevant fields",
            "• Label addition: Make self-descriptive",
            "",
            "Structured Documents (JSON, XML):",
            "• Respect hierarchical structure",
            "• Chunk at logical boundaries"
        ],
        visual_notes="Side-by-side examples of chunking different data types",
        speaker_notes="Tabular data is tricky. We'll practice this in the hands-on exercise."
    ),
    
    Slide(
        slide_number=10,
        title="Handling Tabular Data: USC Course Catalog Example",
        content=[
            "Challenge: Convert Table Rows to Embeddings",
            "",
            "Approach 1: Row-based with labels",
            "```",
            "Class name: CSCI 567. The course will cover:",
            "Machine Learning fundamentals. Units: 4.",
            "Schedule: MW 2:00-3:20 PM.",
            "```",
            "",
            "Approach 2: Column concatenation",
            "```",
            "CSCI 567 Machine Learning fundamentals 4 MW 2:00-3:20 PM",
            "```",
            "",
            "Best Practice: Add descriptive labels for context"
        ],
        visual_notes="Before/after comparison showing table row transformation",
        speaker_notes="This is a common real-world problem. Many enterprises have tabular data to search."
    ),
    
    Slide(
        slide_number=11,
        title="Embedding Pipeline Architecture",
        content=[
            "End-to-End Workflow",
            "",
            "Steps:",
            "1. Data ingestion and preprocessing",
            "2. Chunking with chosen strategy",
            "3. Embedding generation (batch processing)",
            "4. Vector store indexing",
            "5. Metadata attachment",
            "6. Query-time retrieval",
            "",
            "Considerations:",
            "• Batch size for embedding API",
            "• Rate limiting and retry logic",
            "• Cost optimization (caching, deduplication)",
            "• Incremental updates vs full reindex"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[Raw Documents] --> B[Chunking]
    B --> C[Embedding Model]
    C --> D[Vector Store]
    E[Metadata] --> D
    F[Query] --> C
    C --> G[Query Vector]
    G --> D
    D --> H[Retrieved Chunks]
```
""",
        visual_notes="Show data flow with emphasis on batch processing",
        speaker_notes="Production pipelines need error handling, monitoring, and cost controls."
    ),
    
    Slide(
        slide_number=12,
        title="Hands-On: Notebook 1",
        content=[
            "Embeddings and Chunking Exercise",
            "",
            "You will:",
            "• Implement embeddings using NVIDIA NIM",
            "• Experiment with different embedding models",
            "• Transform USC course catalog (tabular data)",
            "• Test various chunking strategies",
            "• Measure retrieval quality with different approaches",
            "",
            "Debug Challenge:",
            "• Find and fix intentional bugs in the pipeline",
            "• Optimize for your specific use case"
        ],
        visual_notes="Screenshot of notebook with key sections highlighted",
        speaker_notes="Give 25-30 minutes. This is hands-on heavy. Encourage experimentation."
    ),
    
    Slide(
        slide_number=13,
        title="Module 2 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ Embeddings convert text to numerical vectors",
            "✓ Domain-specific models outperform general models",
            "✓ Vector stores enable fast similarity search",
            "✓ Chunking strategy significantly impacts retrieval",
            "✓ Tabular data requires special handling",
            "",
            "Next Module: RAG Architecture and Component Analysis",
            "• Three-stage RAG pipeline",
            "• Component-level debugging",
            "• Failure diagnosis"
        ],
        visual_notes="Summary with visual icons for each takeaway",
        speaker_notes="Check understanding with quick questions. 5-minute break before Module 3."
    )
]


def get_module_2_slides() -> List[Slide]:
    """Return all slides for Module 2"""
    return MODULE_2_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format"""
    markdown = "# Module 2: Embeddings and Vector Stores\n\n"
    
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
    slides = get_module_2_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
