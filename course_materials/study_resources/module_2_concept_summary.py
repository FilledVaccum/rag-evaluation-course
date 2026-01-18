"""
Module 2: Embeddings and Vector Stores - One-Page Concept Summary
Evaluating RAG and Semantic Search Systems Course

This one-page summary provides key concepts, formulas, and decision frameworks
for quick review and exam preparation.

Requirements: 17.2
"""

MODULE_2_CONCEPT_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODULE 2: EMBEDDINGS AND VECTOR STORES                    ║
║                           ONE-PAGE CONCEPT SUMMARY                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. EMBEDDING FUNDAMENTALS                                                    │
└──────────────────────────────────────────────────────────────────────────────┘

• Definition: Numerical vector representations of text that capture semantic meaning
• Dimension: Typically 384-4096 dimensions (higher = more nuanced relationships)
• Key Property: Similar meanings → Similar vectors

Cosine Similarity Formula:
    cos(θ) = (A · B) / (||A|| × ||B||)
    Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

BM25 vs. Embeddings:
    BM25: Exact word matching, fast, no semantic understanding
    Embeddings: Semantic matching, slower, captures meaning

┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. DOMAIN-SPECIFIC EMBEDDING MODELS                                          │
└──────────────────────────────────────────────────────────────────────────────┘

Model Selection Decision Matrix:

Domain              | Recommended Model        | Dimensions | NVIDIA NIM
--------------------|--------------------------|------------|------------
General Purpose     | NV-Embed-v2             | 4096       | ✅ Yes
Code                | CodeBERT                | 768        | ❌ No
Finance             | FinBERT                 | 768        | ❌ No
Healthcare/Medical  | BioBERT                 | 768        | ❌ No
Multilingual        | multilingual-e5-large   | 1024       | ❌ No
Fast/Lightweight    | all-MiniLM-L6-v2        | 384        | ❌ No

Key Principle: Domain-specific models outperform general models in specialized domains

┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. VECTOR STORE CONFIGURATION                                                │
└──────────────────────────────────────────────────────────────────────────────┘

Index Types:

Type        | Best For              | Trade-off                    | Use When
------------|-----------------------|------------------------------|------------------
FLAT        | Small datasets        | Exact search, O(n) slow      | < 10K vectors
HNSW        | High accuracy         | High memory, fast queries    | Accuracy critical
IVF_FLAT    | Large datasets        | Lower recall, memory-efficient| Scale > speed

Distance Metrics:

Metric              | Formula                  | Best For
--------------------|--------------------------|---------------------------
Cosine Similarity   | dot(A,B)/(||A||×||B||)  | Text embeddings (default)
Euclidean Distance  | sqrt(Σ(A-B)²)           | Image embeddings
Dot Product         | dot(A,B)                | Pre-normalized vectors

HNSW Parameters:
    M: Connections per layer (16-32, higher = better recall, more memory)
    efConstruction: Build quality (200-400, higher = better index, slower build)
    efSearch: Query thoroughness (100-200, higher = better recall, slower queries)

IVF Parameters:
    nlist: Number of clusters (sqrt(N) is good starting point)
    nprobe: Clusters to search (10-20, higher = better recall, slower queries)

┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. CHUNKING STRATEGIES                                                       │
└──────────────────────────────────────────────────────────────────────────────┘

Core Trade-off:
    Larger chunks: More context, may include irrelevant info
    Smaller chunks: More precise, may lack context

Strategy Comparison:

Strategy        | Chunk Size    | Overlap | Best For
----------------|---------------|---------|--------------------------------
Fixed-Size      | 500-1000 tok  | 10-20%  | General text, simple implementation
Semantic        | Variable      | N/A     | Structured docs (headings, sections)
Tabular         | Row-based     | N/A     | CSV, database tables
Code            | Function-level| N/A     | Source code, API docs
Conversational  | Turn-based    | 3 turns | Chat logs, support transcripts

Overlap Purpose: Prevent information loss at chunk boundaries

Tabular Data Best Practice:
    ❌ Bad:  "CSCI 567 4 Machine Learning MW 2-3:20"
    ✅ Good: "Class name: CSCI 567. Topics: Machine Learning. Units: 4. Schedule: MW 2-3:20"
    
    Key: Add labels for self-descriptive strings

┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. OPTIMIZATION GUIDELINES                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Problem                          | Solution
---------------------------------|------------------------------------------
Low recall (missing relevant)    | Increase efSearch/nprobe, check chunk size
Poor ranking (relevant at bottom)| Increase efSearch/nprobe, tune similarity
Slow queries                     | Use IVF instead of HNSW, reduce efSearch
High memory usage                | Use IVF, reduce M parameter, smaller model
Wrong language support           | Use multilingual model
Domain-specific poor results     | Switch to domain-specific model

Experimentation Framework:
    1. Start with defaults (chunk_size=1000, overlap=200, HNSW with M=16)
    2. Measure retrieval quality (precision, recall, relevance)
    3. Adjust one parameter at a time
    4. Re-measure and compare
    5. Iterate until satisfactory

┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. NVIDIA PLATFORM INTEGRATION                                               │
└──────────────────────────────────────────────────────────────────────────────┘

NVIDIA NIM for Embeddings:
    • Model: NV-Embed-v2 (4096 dimensions)
    • Advantages: State-of-the-art retrieval, optimized for RAG
    • Access: Via NVIDIA NIM API with API key
    • Use Case: Production RAG systems requiring high accuracy

Integration Pattern:
    1. Set up NVIDIA API key
    2. Initialize NIM client
    3. Call embedding endpoint with text
    4. Store embeddings in vector store
    5. Query vector store for retrieval

┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. EXAM PREPARATION CHECKLIST                                                │
└──────────────────────────────────────────────────────────────────────────────┘

✓ Understand cosine similarity and why it's used for text embeddings
✓ Know when to use domain-specific vs. general-purpose models
✓ Understand HNSW vs. IVF trade-offs
✓ Know how to configure vector store parameters for different scenarios
✓ Understand chunking trade-offs (size, overlap, strategy)
✓ Know how to transform tabular data for embeddings
✓ Understand multilingual embedding considerations
✓ Know NVIDIA NIM embedding capabilities

Exam Domain Coverage:
    • Knowledge Integration and Data Handling (10% - CORE) ⭐⭐⭐
    • NVIDIA Platform Implementation (7% - INTEGRATED) ⭐⭐

┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. KEY FORMULAS AND METRICS                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Cosine Similarity:
    similarity = dot(A, B) / (||A|| × ||B||)

Euclidean Distance:
    distance = sqrt(Σ(Aᵢ - Bᵢ)²)

Dot Product:
    similarity = Σ(Aᵢ × Bᵢ)

Optimal Chunk Overlap:
    overlap = chunk_size × 0.10 to 0.20  (10-20% of chunk size)

IVF nlist (number of clusters):
    nlist ≈ sqrt(N)  where N = total number of vectors

┌──────────────────────────────────────────────────────────────────────────────┐
│ 9. COMMON PITFALLS TO AVOID                                                  │
└──────────────────────────────────────────────────────────────────────────────┘

❌ Using general model for specialized domain (e.g., MiniLM for medical docs)
❌ Forgetting to normalize vectors before computing cosine similarity
❌ Using too small chunks (< 200 tokens) - loses context
❌ Using too large chunks (> 2000 tokens) - includes irrelevant info
❌ No overlap in fixed-size chunking - loses boundary information
❌ Concatenating tabular data without labels - ambiguous for embeddings
❌ Using FLAT index for large datasets (> 100K vectors) - too slow
❌ Setting efSearch/nprobe too low - poor recall
❌ Ignoring language requirements - using English-only model for multilingual data

┌──────────────────────────────────────────────────────────────────────────────┐
│ 10. QUICK REFERENCE: WHEN TO USE WHAT                                        │
└──────────────────────────────────────────────────────────────────────────────┘

Scenario                                    | Recommendation
--------------------------------------------|----------------------------------
Production RAG, high accuracy               | NV-Embed-v2 + HNSW (M=32)
Cost-sensitive, large scale                 | MiniLM + IVF_FLAT
Financial documents                         | FinBERT + HNSW
Medical literature                          | BioBERT + HNSW
Code search                                 | CodeBERT + HNSW
Multilingual (English + Spanish + Arabic)  | multilingual-e5 + HNSW
Prototyping, small dataset (< 10K)         | Any model + FLAT
Long documents with clear structure         | Semantic chunking
Unstructured text                          | Fixed-size chunking (1000 tok)
Tabular data (CSV, database)               | Row-based with labels
Source code                                | Function/class-level chunking

═══════════════════════════════════════════════════════════════════════════════
                              END OF SUMMARY
═══════════════════════════════════════════════════════════════════════════════
"""


# ============================================================================
# VISUAL DIAGRAMS
# ============================================================================

EMBEDDING_SPACE_DIAGRAM = """
# Embedding Space Visualization (Conceptual 2D Projection)

    Semantic Similarity in Vector Space
    
    │
    │     • "cat on mat"
    │     • "feline on rug"
    │         (High similarity: 0.92)
    │
    │              • "dog in park"
    │                (Medium similarity to cat: 0.65)
    │
    │
    │
    │                                    • "stock market crash"
    │                                      (Low similarity to cat: 0.15)
    │
    └────────────────────────────────────────────────────────────────────────
    
    Key Insight: Distance in embedding space reflects semantic similarity
"""


CHUNKING_STRATEGY_FLOWCHART = """
# Chunking Strategy Decision Flowchart

    Document Type?
         │
         ├─── Long-form text ──────────> Fixed-size (500-1000 tok, 10-20% overlap)
         │
         ├─── Structured docs ─────────> Semantic (respect headings/sections)
         │
         ├─── Tabular data ────────────> Row-based with labels
         │
         ├─── Source code ─────────────> Function/class-level
         │
         └─── Conversational ──────────> Turn-based (3-5 turn context)
"""


VECTOR_STORE_SELECTION_FLOWCHART = """
# Vector Store Index Selection Flowchart

    Dataset Size?
         │
         ├─── < 10K vectors ───────────> FLAT (exact search)
         │
         ├─── 10K - 1M vectors ────────> HNSW (high accuracy)
         │                                  │
         │                                  ├─ Accuracy critical? ──> M=32, efSearch=200
         │                                  └─ Balanced? ──────────> M=16, efSearch=100
         │
         └─── > 1M vectors ────────────> IVF_FLAT (scalable)
                                            │
                                            ├─ High recall? ───────> nprobe=20
                                            └─ Fast queries? ──────> nprobe=5
"""


# ============================================================================
# EXPORT FOR USE IN COURSE
# ============================================================================

if __name__ == "__main__":
    print(MODULE_2_CONCEPT_SUMMARY)
    print("\n" + "=" * 80)
    print("VISUAL DIAGRAMS")
    print("=" * 80)
    print(EMBEDDING_SPACE_DIAGRAM)
    print(CHUNKING_STRATEGY_FLOWCHART)
    print(VECTOR_STORE_SELECTION_FLOWCHART)
