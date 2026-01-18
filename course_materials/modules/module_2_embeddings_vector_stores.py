"""
Module 2: Embeddings and Vector Stores
Evaluating RAG and Semantic Search Systems Course

This module covers embedding fundamentals, domain-specific models, vector store configuration,
and chunking strategies for different data types.

Learning Objectives:
- Understand multi-dimensional similarity and embedding fundamentals
- Select appropriate domain-specific embedding models
- Configure and optimize vector stores
- Apply effective chunking strategies for different data types

Certification Alignment:
- Knowledge Integration and Data Handling (10% - CORE)
- NVIDIA Platform Implementation (7%)

Requirements: 4.1, 4.2, 4.4, 4.5
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# SECTION 1: EMBEDDING FUNDAMENTALS
# ============================================================================

EMBEDDING_FUNDAMENTALS = """
# Embedding Fundamentals and Multi-Dimensional Similarity

## What Are Embeddings?

Embeddings are numerical vector representations of text that capture semantic meaning in 
high-dimensional space. Unlike keyword-based approaches (BM25), embeddings enable semantic 
similarity matching.

### Key Concepts:

1. **Vector Representation**: Text → Dense numerical vector (e.g., 768 or 1024 dimensions)
2. **Semantic Similarity**: Similar meanings → Similar vectors (measured by cosine similarity)
3. **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
4. **Dimensionality**: Higher dimensions capture more nuanced semantic relationships

## Multi-Dimensional Similarity Visualization

```
Text: "The cat sat on the mat"
Embedding: [0.23, -0.45, 0.67, ..., 0.12]  # 768 dimensions

Text: "A feline rested on the rug"
Embedding: [0.21, -0.43, 0.69, ..., 0.15]  # Similar vector!

Cosine Similarity: 0.92 (very similar semantically)
```

### Visualization Concept:

```mermaid
graph TB
    subgraph "High-Dimensional Embedding Space"
        A["'cat on mat'<br/>[0.23, -0.45, ...]"]
        B["'feline on rug'<br/>[0.21, -0.43, ...]"]
        C["'dog in park'<br/>[0.15, 0.32, ...]"]
        D["'stock market crash'<br/>[-0.67, 0.89, ...]"]
    end
    
    A -.->|High Similarity| B
    A -.->|Medium Similarity| C
    A -.->|Low Similarity| D
    
    style A fill:#90EE90
    style B fill:#90EE90
    style C fill:#FFD700
    style D fill:#FF6B6B
```

## How Embeddings Enable Semantic Search

**Traditional Keyword Search (BM25)**:
- Query: "feline on rug" → No match for "cat on mat" (different words)
- Relies on exact word matching and term frequency

**Semantic Search (Embeddings)**:
- Query: "feline on rug" → High match for "cat on mat" (similar meaning)
- Captures semantic relationships and synonyms

## Embedding Model Architecture

```mermaid
graph LR
    A[Input Text] --> B[Tokenization]
    B --> C[Transformer Encoder]
    C --> D[Pooling Layer]
    D --> E[Embedding Vector]
    
    subgraph "Transformer Layers"
        C
    end
```

### Common Embedding Models:
- **BERT-based**: General-purpose, 768 dimensions
- **Sentence-BERT**: Optimized for sentence similarity
- **NVIDIA NV-Embed-v2**: State-of-the-art, optimized for retrieval
- **Domain-specific**: Specialized for code, finance, healthcare, etc.
"""


# ============================================================================
# SECTION 2: DOMAIN-SPECIFIC EMBEDDING MODELS
# ============================================================================

@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model."""
    name: str
    provider: str
    dimensions: int
    domain: str
    use_cases: List[str]
    strengths: List[str]
    limitations: List[str]
    nvidia_nim_available: bool


DOMAIN_SPECIFIC_MODELS = {
    "general": [
        EmbeddingModelInfo(
            name="NV-Embed-v2",
            provider="NVIDIA",
            dimensions=4096,
            domain="General Purpose",
            use_cases=[
                "General text retrieval",
                "Question answering",
                "Semantic search across domains"
            ],
            strengths=[
                "State-of-the-art retrieval performance",
                "Optimized for RAG applications",
                "Available via NVIDIA NIM"
            ],
            limitations=[
                "Higher computational cost",
                "May be overkill for simple tasks"
            ],
            nvidia_nim_available=True
        ),
        EmbeddingModelInfo(
            name="sentence-transformers/all-MiniLM-L6-v2",
            provider="Hugging Face",
            dimensions=384,
            domain="General Purpose",
            use_cases=[
                "Fast semantic search",
                "Resource-constrained environments",
                "Prototyping"
            ],
            strengths=[
                "Fast inference",
                "Small model size",
                "Good general performance"
            ],
            limitations=[
                "Lower accuracy than larger models",
                "Less effective for specialized domains"
            ],
            nvidia_nim_available=False
        )
    ],
    "code": [
        EmbeddingModelInfo(
            name="CodeBERT",
            provider="Microsoft",
            dimensions=768,
            domain="Code",
            use_cases=[
                "Code search",
                "Code documentation retrieval",
                "API documentation search"
            ],
            strengths=[
                "Understands programming syntax",
                "Trained on code-text pairs",
                "Multi-language support"
            ],
            limitations=[
                "May struggle with natural language queries",
                "Requires code-specific preprocessing"
            ],
            nvidia_nim_available=False
        )
    ],
    "finance": [
        EmbeddingModelInfo(
            name="FinBERT",
            provider="ProsusAI",
            dimensions=768,
            domain="Finance",
            use_cases=[
                "Financial document retrieval",
                "Regulatory compliance search",
                "Financial news analysis"
            ],
            strengths=[
                "Trained on financial corpus",
                "Understands financial terminology",
                "Good for sentiment analysis"
            ],
            limitations=[
                "Limited to financial domain",
                "May need fine-tuning for specific use cases"
            ],
            nvidia_nim_available=False
        )
    ],
    "healthcare": [
        EmbeddingModelInfo(
            name="BioBERT",
            provider="DMIS Lab",
            dimensions=768,
            domain="Healthcare/Biomedical",
            use_cases=[
                "Medical literature search",
                "Clinical documentation retrieval",
                "Drug information lookup"
            ],
            strengths=[
                "Trained on PubMed and PMC",
                "Understands medical terminology",
                "Good for biomedical NER"
            ],
            limitations=[
                "Requires medical domain knowledge",
                "May need updates for new medical terms"
            ],
            nvidia_nim_available=False
        )
    ],
    "multilingual": [
        EmbeddingModelInfo(
            name="multilingual-e5-large",
            provider="Microsoft",
            dimensions=1024,
            domain="Multilingual",
            use_cases=[
                "Cross-lingual search",
                "Multilingual document retrieval",
                "Translation-free semantic search"
            ],
            strengths=[
                "Supports 100+ languages",
                "Cross-lingual similarity",
                "No translation required"
            ],
            limitations=[
                "Lower performance on low-resource languages",
                "Larger model size"
            ],
            nvidia_nim_available=False
        )
    ]
}


DOMAIN_MODEL_SELECTION_GUIDE = """
# Domain-Specific Embedding Model Selection Guide

## Decision Matrix

| Domain | Recommended Model | When to Use | NVIDIA NIM Available |
|--------|------------------|-------------|---------------------|
| **General Purpose** | NV-Embed-v2 | High-accuracy RAG, production systems | ✅ Yes |
| **Code** | CodeBERT | Code search, API documentation | ❌ No |
| **Finance** | FinBERT | Financial documents, compliance | ❌ No |
| **Healthcare** | BioBERT | Medical literature, clinical docs | ❌ No |
| **Multilingual** | multilingual-e5 | Cross-lingual search, global apps | ❌ No |
| **Legal** | Legal-BERT | Legal documents, case law | ❌ No |

## Selection Criteria

### 1. Domain Specificity
- **High domain specificity** (finance, healthcare, legal) → Use domain-specific model
- **General content** → Use general-purpose model (NV-Embed-v2)

### 2. Language Requirements
- **Single language (English)** → General or domain-specific model
- **Multiple languages** → Multilingual model
- **Low-resource languages** → May need custom training

### 3. Performance vs. Cost
- **Production, high-accuracy** → NV-Embed-v2 (4096 dim)
- **Prototyping, cost-sensitive** → MiniLM (384 dim)
- **Specialized domain** → Domain-specific model

### 4. Integration Requirements
- **NVIDIA ecosystem** → NV-Embed-v2 via NIM
- **Custom deployment** → Hugging Face models
- **On-premise** → Self-hosted models

## Example Use Cases

### Use Case 1: Financial RAG System
**Scenario**: Search financial reports and regulatory filings

**Recommendation**: FinBERT
- Understands financial terminology (EBITDA, P/E ratio, etc.)
- Trained on financial corpus
- Good for compliance-related queries

### Use Case 2: Medical Q&A System
**Scenario**: Answer questions about medical conditions

**Recommendation**: BioBERT
- Trained on PubMed medical literature
- Understands medical terminology
- Good for clinical documentation

### Use Case 3: Code Documentation Search
**Scenario**: Search API documentation and code examples

**Recommendation**: CodeBERT
- Understands programming syntax
- Trained on code-text pairs
- Supports multiple programming languages

### Use Case 4: General Enterprise RAG
**Scenario**: Search across diverse enterprise documents

**Recommendation**: NV-Embed-v2
- State-of-the-art general performance
- Available via NVIDIA NIM
- Optimized for RAG applications
"""


# ============================================================================
# SECTION 3: VECTOR STORE CONFIGURATION
# ============================================================================

VECTOR_STORE_FUNDAMENTALS = """
# Vector Store Configuration and Optimization

## What is a Vector Store?

A vector store (vector database) is a specialized database optimized for storing and 
retrieving high-dimensional embeddings efficiently.

### Key Features:
1. **Efficient Similarity Search**: Find nearest neighbors quickly
2. **Scalability**: Handle millions/billions of vectors
3. **Indexing**: Optimize retrieval speed vs. accuracy trade-offs
4. **Metadata Filtering**: Combine vector search with traditional filters

## Popular Vector Stores

```mermaid
graph TB
    subgraph "Vector Store Ecosystem"
        A[Pinecone]
        B[Milvus]
        C[Chroma]
        D[Weaviate]
        E[Qdrant]
        F[FAISS]
    end
    
    A -->|Cloud-Native| G[Managed Service]
    B -->|Open Source| H[Self-Hosted]
    C -->|Lightweight| I[Embedded]
    D -->|GraphQL API| G
    E -->|Rust-Based| H
    F -->|Facebook AI| I
```

### Comparison:

| Vector Store | Type | Best For | Scalability | Ease of Use |
|-------------|------|----------|-------------|-------------|
| **Pinecone** | Managed | Production, cloud-native | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Milvus** | Open Source | Large-scale, on-premise | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Chroma** | Embedded | Prototyping, small-scale | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Weaviate** | Open Source | GraphQL integration | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Qdrant** | Open Source | High performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **FAISS** | Library | Research, prototyping | ⭐⭐⭐ | ⭐⭐ |

## Vector Store Configuration

### 1. Index Type Selection

**HNSW (Hierarchical Navigable Small World)**:
- **Best for**: High recall, moderate dataset size
- **Trade-off**: Higher memory usage, fast queries
- **Use when**: Accuracy is critical

**IVF (Inverted File Index)**:
- **Best for**: Large datasets, memory-constrained
- **Trade-off**: Lower recall, faster indexing
- **Use when**: Speed and scale are priorities

**Flat Index**:
- **Best for**: Small datasets, exact search
- **Trade-off**: Slow for large datasets
- **Use when**: Dataset < 10K vectors

### 2. Distance Metrics

**Cosine Similarity**:
```python
similarity = dot(A, B) / (norm(A) * norm(B))
# Range: [-1, 1], higher is more similar
```
- **Best for**: Text embeddings, normalized vectors
- **Use when**: Direction matters more than magnitude

**Euclidean Distance**:
```python
distance = sqrt(sum((A - B)^2))
# Range: [0, ∞], lower is more similar
```
- **Best for**: Image embeddings, spatial data
- **Use when**: Absolute distance matters

**Dot Product**:
```python
similarity = dot(A, B)
# Range: [-∞, ∞], higher is more similar
```
- **Best for**: Pre-normalized embeddings
- **Use when**: Magnitude encodes importance

### 3. Optimization Parameters

**For HNSW Index**:
```python
config = {
    "M": 16,  # Number of connections per layer (higher = better recall, more memory)
    "efConstruction": 200,  # Build-time search depth (higher = better quality, slower build)
    "efSearch": 100,  # Query-time search depth (higher = better recall, slower queries)
}
```

**For IVF Index**:
```python
config = {
    "nlist": 100,  # Number of clusters (sqrt(N) is a good starting point)
    "nprobe": 10,  # Number of clusters to search (higher = better recall, slower)
}
```

## Configuration Examples

### Example 1: High-Accuracy Production System
```python
# Use HNSW with high parameters
vector_store_config = {
    "index_type": "HNSW",
    "metric": "cosine",
    "params": {
        "M": 32,
        "efConstruction": 400,
        "efSearch": 200
    }
}
# Trade-off: Higher memory, slower indexing, best recall
```

### Example 2: Large-Scale Cost-Optimized System
```python
# Use IVF with moderate parameters
vector_store_config = {
    "index_type": "IVF_FLAT",
    "metric": "cosine",
    "params": {
        "nlist": 1000,
        "nprobe": 20
    }
}
# Trade-off: Lower memory, faster indexing, good recall
```

### Example 3: Prototyping/Development
```python
# Use Flat index for simplicity
vector_store_config = {
    "index_type": "FLAT",
    "metric": "cosine",
    "params": {}
}
# Trade-off: Exact search, slow for large datasets
```
"""


# ============================================================================
# SECTION 4: CHUNKING STRATEGIES
# ============================================================================

CHUNKING_STRATEGIES = """
# Chunking Strategies for Different Data Types

## Why Chunking Matters

Chunking is the process of dividing documents into smaller segments for embedding and retrieval.
Proper chunking is critical for RAG performance:

- **Too small**: Loss of context, incomplete information
- **Too large**: Irrelevant information, poor retrieval precision
- **Just right**: Balanced context and precision

## Chunking Strategy Decision Tree

```mermaid
graph TD
    A[Document Type?] --> B[Long-form Text]
    A --> C[Structured Data]
    A --> D[Code]
    A --> E[Conversational]
    
    B --> F[Fixed-size chunks<br/>500-1000 tokens<br/>with overlap]
    C --> G[Row-based or<br/>semantic units]
    D --> H[Function/class<br/>level chunks]
    E --> I[Turn-based<br/>chunks]
    
    F --> J[Overlap: 10-20%]
    G --> K[Add labels]
    H --> L[Include context]
    I --> M[Include history]
```

## Strategy 1: Fixed-Size Chunking (Text Documents)

**Best for**: Articles, documentation, long-form content

**Parameters**:
- **Chunk size**: 500-1000 tokens (roughly 2-4 paragraphs)
- **Overlap**: 50-200 tokens (10-20% of chunk size)
- **Separator**: Sentence boundaries, paragraphs

**Example**:
```python
def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity
    return chunks
```

**Pros**:
- Simple to implement
- Predictable chunk sizes
- Good for general text

**Cons**:
- May split mid-sentence or mid-concept
- Doesn't respect document structure

## Strategy 2: Semantic Chunking (Intelligent Splitting)

**Best for**: Documents with clear structure (sections, headings)

**Approach**:
- Split on semantic boundaries (headings, sections, paragraphs)
- Maintain logical units of information
- Variable chunk sizes based on content

**Example**:
```python
def semantic_chunking(text: str, max_chunk_size: int = 1000):
    # Split on headings and paragraphs
    sections = split_on_headings(text)
    chunks = []
    
    for section in sections:
        if len(section) <= max_chunk_size:
            chunks.append(section)
        else:
            # Further split large sections
            paragraphs = split_on_paragraphs(section)
            chunks.extend(paragraphs)
    
    return chunks
```

**Pros**:
- Respects document structure
- Maintains semantic coherence
- Better context preservation

**Cons**:
- More complex implementation
- Variable chunk sizes
- Requires document structure

## Strategy 3: Tabular Data Chunking (Structured Data)

**Best for**: CSV files, database tables, spreadsheets

**Approach**: Row-based chunking with label addition

### Challenge: Tabular Data in Embeddings

**Problem**: Embeddings work best with natural language, not structured data

**Solution**: Transform rows into self-descriptive strings

**Example - USC Course Catalog**:

**Original Table**:
| Course Name | Units | Description | Schedule |
|------------|-------|-------------|----------|
| CSCI 567 | 4 | Machine Learning | MW 2:00-3:20 PM |

**Transformed String (WITH labels)**:
```
Class name: CSCI 567. The course will cover the following topics: Machine Learning 
fundamentals. Units: 4. Schedule: MW 2:00-3:20 PM.
```

**Transformed String (WITHOUT labels)**:
```
CSCI 567 Machine Learning fundamentals 4 MW 2:00-3:20 PM
```

**Recommendation**: Use labels for better semantic understanding

### Implementation:
```python
def tabular_chunking(df, include_labels: bool = True):
    chunks = []
    for _, row in df.iterrows():
        if include_labels:
            chunk = f"Class name: {row['course_name']}. "
            chunk += f"The course will cover the following topics: {row['description']}. "
            chunk += f"Units: {row['units']}. "
            chunk += f"Schedule: {row['schedule']}."
        else:
            chunk = f"{row['course_name']} {row['description']} {row['units']} {row['schedule']}"
        
        chunks.append(chunk)
    
    return chunks
```

**Key Decisions**:
1. **Column Selection**: Only include relevant columns
2. **Label Addition**: Add descriptive labels ("Class name:", "Units:")
3. **Row vs. Column**: Usually row-based (each row = one chunk)
4. **Aggregation**: Sometimes combine related rows

## Strategy 4: Code Chunking

**Best for**: Source code, API documentation

**Approach**: Function/class-level chunking with context

**Example**:
```python
def code_chunking(code: str):
    # Parse code into functions/classes
    functions = extract_functions(code)
    chunks = []
    
    for func in functions:
        # Include function signature, docstring, and body
        chunk = f"Function: {func.name}\n"
        chunk += f"Signature: {func.signature}\n"
        chunk += f"Docstring: {func.docstring}\n"
        chunk += f"Code: {func.body}"
        chunks.append(chunk)
    
    return chunks
```

**Pros**:
- Maintains code structure
- Includes context (signatures, docstrings)
- Good for code search

**Cons**:
- Requires code parsing
- May miss cross-function dependencies

## Strategy 5: Conversational Chunking

**Best for**: Chat logs, customer support transcripts

**Approach**: Turn-based chunking with conversation history

**Example**:
```python
def conversational_chunking(messages: List[Dict], context_turns: int = 3):
    chunks = []
    
    for i, message in enumerate(messages):
        # Include previous N turns for context
        start = max(0, i - context_turns)
        context = messages[start:i+1]
        
        chunk = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        chunks.append(chunk)
    
    return chunks
```

## Chunking Best Practices

### 1. Experiment with Chunk Sizes
- Start with 500-1000 tokens
- Measure retrieval quality
- Adjust based on domain and use case

### 2. Use Overlap for Context
- 10-20% overlap prevents information loss
- Especially important for fixed-size chunking

### 3. Preserve Semantic Units
- Don't split mid-sentence
- Respect document structure
- Maintain logical coherence

### 4. Add Metadata
- Include document title, section, page number
- Helps with filtering and ranking
- Provides context for retrieved chunks

### 5. Test and Iterate
- Evaluate retrieval quality with test queries
- Measure precision and recall
- Adjust strategy based on results

## Chunking Evaluation Framework

```python
def evaluate_chunking_strategy(chunks, test_queries):
    metrics = {
        "avg_chunk_size": calculate_avg_size(chunks),
        "chunk_size_variance": calculate_variance(chunks),
        "retrieval_precision": measure_precision(chunks, test_queries),
        "retrieval_recall": measure_recall(chunks, test_queries),
        "context_completeness": measure_completeness(chunks, test_queries)
    }
    return metrics
```

**Key Metrics**:
- **Precision**: Are retrieved chunks relevant?
- **Recall**: Are all relevant chunks retrieved?
- **Context Completeness**: Do chunks contain enough context?
- **Chunk Size Distribution**: Are chunks reasonably sized?
"""


# ============================================================================
# LECTURE OUTLINE
# ============================================================================

LECTURE_OUTLINE = """
# Module 2 Lecture Outline: Embeddings and Vector Stores

## Part 1: Embedding Fundamentals (15 minutes)
1. What are embeddings? (5 min)
   - Vector representations of text
   - Multi-dimensional similarity
   - Visualization and intuition

2. How embeddings enable semantic search (5 min)
   - Comparison with BM25
   - Cosine similarity
   - Semantic relationships

3. Embedding model architecture (5 min)
   - Transformer-based models
   - Pooling strategies
   - Output dimensions

## Part 2: Domain-Specific Models (15 minutes)
1. General-purpose models (5 min)
   - NV-Embed-v2 (NVIDIA NIM)
   - Sentence-BERT
   - When to use general models

2. Domain-specific models (7 min)
   - CodeBERT for code search
   - FinBERT for finance
   - BioBERT for healthcare
   - Multilingual models

3. Model selection decision matrix (3 min)
   - Domain specificity
   - Performance vs. cost
   - Integration requirements

## Part 3: Vector Store Configuration (10 minutes)
1. Vector store fundamentals (3 min)
   - What is a vector store?
   - Popular options (Pinecone, Milvus, Chroma)

2. Index types and trade-offs (4 min)
   - HNSW vs. IVF vs. Flat
   - Distance metrics (cosine, Euclidean, dot product)

3. Optimization parameters (3 min)
   - Configuration examples
   - Performance tuning

## Part 4: Chunking Strategies (15 minutes)
1. Why chunking matters (3 min)
   - Context vs. precision trade-off
   - Impact on retrieval quality

2. Chunking strategies by data type (8 min)
   - Fixed-size chunking (text)
   - Semantic chunking (structured docs)
   - Tabular data chunking (USC catalog example)
   - Code chunking
   - Conversational chunking

3. Best practices and evaluation (4 min)
   - Experimentation framework
   - Metrics for chunking quality
   - Iterative improvement

## Part 5: Hands-On Preview (5 minutes)
- Overview of upcoming exercises
- NVIDIA NIM integration
- USC course catalog transformation
- Chunking experiments

**Total Duration**: 60 minutes (lecture/demo)
**Hands-On Duration**: 75 minutes (separate exercises)
**Discussion/Q&A**: 15 minutes
"""


# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

INSTRUCTOR_NOTES = """
# Instructor Notes: Module 2

## Key Teaching Points

### 1. Embeddings Are Not Magic
- Emphasize that embeddings are learned representations
- Quality depends on training data
- Domain-specific models perform better in specialized domains

### 2. Vector Store Selection Matters
- Different use cases require different vector stores
- Trade-offs between accuracy, speed, and cost
- Start simple (Chroma), scale up (Pinecone/Milvus)

### 3. Chunking Is Critical
- Spend extra time on chunking strategies
- Show before/after examples
- Emphasize experimentation

### 4. Tabular Data Is Tricky
- Students often struggle with tabular data in RAG
- Demonstrate label addition clearly
- Show multiple approaches

## Common Student Questions

**Q: "Why not just use the largest embedding model?"**
A: Larger models have higher computational cost, latency, and may overfit to training data. 
   Choose based on use case requirements.

**Q: "How do I know which chunk size to use?"**
A: Start with 500-1000 tokens, then experiment. Measure retrieval quality with test queries.
   Domain and document structure matter.

**Q: "Can I use multiple embedding models in one system?"**
A: Yes! Hybrid approaches can combine general and domain-specific models. However, vectors
   must be in the same space for comparison.

**Q: "What if my domain isn't covered by existing models?"**
A: Options: (1) Fine-tune a general model, (2) Use NV-Embed-v2 (strong general performance),
   (3) Train custom model (expensive).

## Live Demo Tips

### Demo 1: Embedding Similarity
- Show cosine similarity between similar/dissimilar sentences
- Visualize in 2D using dimensionality reduction (t-SNE/UMAP)
- Demonstrate semantic relationships

### Demo 2: Chunking Comparison
- Take same document, chunk with different strategies
- Show retrieval results for same query
- Highlight differences in context and relevance

### Demo 3: Tabular Data Transformation
- Live transform USC course catalog row
- Show with/without labels
- Query and compare results

## Timing Guidance

- **Embedding Fundamentals**: Don't rush, this is foundation
- **Domain Models**: Can be faster, students will explore in exercises
- **Vector Stores**: Focus on concepts, not implementation details
- **Chunking**: Allocate extra time, very important for RAG success

## Troubleshooting

### Issue: Students confused about dimensions
**Solution**: Use 2D/3D analogies, show visualization

### Issue: Overwhelmed by vector store options
**Solution**: Recommend Chroma for learning, defer production decisions

### Issue: Chunking seems arbitrary
**Solution**: Show metrics, demonstrate impact on retrieval quality

## Connection to Certification Exam

- **Knowledge Integration (10%)**: Embedding selection, data handling
- **NVIDIA Platform (7%)**: NV-Embed-v2, NVIDIA NIM integration
- **Evaluation (13%)**: Chunking impacts evaluation metrics (covered in Module 5)

Emphasize that proper embedding and chunking choices directly impact RAG evaluation results.
"""


if __name__ == "__main__":
    print("Module 2: Embeddings and Vector Stores - Lecture Materials")
    print("=" * 70)
    print("\nThis module covers:")
    print("- Embedding fundamentals and multi-dimensional similarity")
    print("- Domain-specific embedding models (code, finance, healthcare, multilingual)")
    print("- Vector store configuration and optimization")
    print("- Chunking strategies for different data types")
    print("\nCertification Alignment:")
    print("- Knowledge Integration and Data Handling (10% - CORE)")
    print("- NVIDIA Platform Implementation (7%)")
