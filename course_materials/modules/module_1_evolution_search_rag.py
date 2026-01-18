"""
Module 1: Evolution of Search to RAG Systems

This module provides lecture materials covering the evolution from traditional
search systems to modern RAG architectures.

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from src.models.course import LectureMaterial, Slide, MermaidDiagram, CaseStudy


def create_module_1_lecture_materials() -> LectureMaterial:
    """
    Create comprehensive lecture materials for Module 1.
    
    Returns:
        LectureMaterial object with slides, diagrams, and case studies
    """
    
    # Define slides
    slides = [
        Slide(
            slide_number=1,
            title="Module 1: Evolution of Search to RAG Systems",
            content="""
# Evolution of Search to RAG Systems

## Learning Objectives
- Understand the progression from traditional search to RAG
- Compare keyword-based (BM25) vs semantic search paradigms
- Analyze enterprise hybrid systems
- Develop decision frameworks for search approach selection

## Duration: 30-45 minutes
## Certification Alignment: Knowledge Integration (10%), Agent Architecture (15%)
            """,
            speaker_notes="Start with motivation: Why do we need to understand search evolution? "
                         "Emphasize that RAG builds on decades of search technology."
        ),
        
        Slide(
            slide_number=2,
            title="Classic Search Architecture",
            content="""
# Classic Search Architecture

## Four Core Stages:

1. **Crawling**: Discovering and fetching web pages
   - Web crawlers traverse links
   - Respect robots.txt and rate limits
   
2. **Analysis**: Extracting content and metadata
   - Parse HTML, extract text
   - Identify language, structure, entities
   
3. **Indexing**: Building searchable data structures
   - Inverted index: term → document mappings
   - Store term frequencies, positions
   
4. **Ranking**: Ordering results by relevance
   - BM25, PageRank, and other algorithms
   - Combine multiple signals
            """,
            speaker_notes="Draw parallels to RAG: retrieval is like search, but with semantic understanding. "
                         "Emphasize that indexing is still crucial in RAG (vector stores)."
        ),
        
        Slide(
            slide_number=3,
            title="BM25: Best Match 25 Algorithm",
            content="""
# BM25: Keyword-Based Ranking

## Core Concept
BM25 ranks documents based on term frequency and document length normalization.

## Formula Components:
- **TF (Term Frequency)**: How often does the query term appear?
- **IDF (Inverse Document Frequency)**: How rare is the term across all documents?
- **Document Length Normalization**: Penalize very long documents

## Strengths:
✓ Fast and efficient
✓ Interpretable results
✓ Works well for exact keyword matches
✓ No training required

## Limitations:
✗ No semantic understanding (synonyms, context)
✗ Vocabulary mismatch problems
✗ Cannot handle paraphrasing
✗ Struggles with ambiguous queries
            """,
            speaker_notes="Give example: searching 'car' won't find 'automobile'. "
                         "This is the key limitation that semantic search addresses."
        ),
        
        Slide(
            slide_number=4,
            title="Semantic Search: Understanding Meaning",
            content="""
# Semantic Search with Embeddings

## Core Concept
Convert text to high-dimensional vectors that capture semantic meaning.

## How It Works:
1. **Embedding Model**: Neural network encodes text → vector
2. **Vector Store**: Efficient storage and retrieval of embeddings
3. **Similarity Search**: Find vectors close in embedding space
4. **Ranking**: Order by cosine similarity or distance metric

## Strengths:
✓ Understands synonyms and paraphrasing
✓ Captures semantic relationships
✓ Handles vocabulary mismatch
✓ Works across languages (multilingual models)

## Limitations:
✗ Requires embedding model (computational cost)
✗ Less interpretable than keyword search
✗ May miss exact keyword matches
✗ Sensitive to embedding model quality
            """,
            speaker_notes="Emphasize: semantic search complements, not replaces, keyword search. "
                         "This is why hybrid systems are common in enterprise."
        ),
        
        Slide(
            slide_number=5,
            title="BM25 vs Semantic Search Comparison",
            content="""
# When to Use Each Approach

| Aspect | BM25 | Semantic Search |
|--------|------|-----------------|
| **Best For** | Exact matches, technical terms | Conceptual queries, paraphrasing |
| **Speed** | Very fast | Slower (embedding computation) |
| **Accuracy** | High for keyword matches | High for semantic matches |
| **Setup** | Simple, no training | Requires embedding model |
| **Interpretability** | High (term matching) | Lower (vector similarity) |
| **Cost** | Low | Higher (GPU for embeddings) |

## Decision Framework:
- **Use BM25** when: Exact terminology matters, speed critical, simple setup
- **Use Semantic** when: Understanding context matters, handling synonyms, multilingual
- **Use Hybrid** when: Enterprise production, need both precision and recall
            """,
            speaker_notes="Most production systems use hybrid approaches. "
                         "This is a key insight for the certification exam."
        ),
        
        Slide(
            slide_number=6,
            title="Enterprise Hybrid Systems",
            content="""
# Hybrid Search Architecture

## Three-Stage Pipeline:

### Stage 1: Parallel Retrieval
- **BM25 Retrieval**: Fast keyword matching
- **Vector Search**: Semantic similarity
- Retrieve top-k from each (e.g., k=100)

### Stage 2: Fusion
- Combine results from both retrievers
- Reciprocal Rank Fusion (RRF) or weighted combination
- Deduplicate results

### Stage 3: Re-ranking
- Apply more sophisticated model to top results
- Cross-encoder or LLM-based re-ranker
- Final top-k selection (e.g., k=10)

## Benefits:
✓ Best of both worlds: precision + recall
✓ Robust to different query types
✓ Tunable trade-offs (speed vs accuracy)
            """,
            speaker_notes="This is the architecture most enterprises use. "
                         "Re-ranking is expensive but only applied to top candidates."
        ),
        
        Slide(
            slide_number=7,
            title="From Search to RAG",
            content="""
# Retrieval-Augmented Generation (RAG)

## Key Difference from Search:
Search returns **documents**. RAG returns **generated answers**.

## RAG Pipeline:
1. **Retrieval**: Find relevant documents (like search)
2. **Augmentation**: Combine query + retrieved context
3. **Generation**: LLM generates answer from context

## Why RAG?
- **Grounded Responses**: Answers based on retrieved facts
- **Up-to-date Information**: No need to retrain LLM
- **Source Attribution**: Can cite retrieved documents
- **Domain Adaptation**: Works with specialized knowledge bases

## RAG vs Fine-tuning:
- RAG: Dynamic knowledge, no retraining needed
- Fine-tuning: Baked-in knowledge, requires retraining for updates
            """,
            speaker_notes="Emphasize: RAG is search + generation. "
                         "The retrieval component uses everything we just learned about search."
        ),
        
        Slide(
            slide_number=8,
            title="Decision Framework: Choosing Your Approach",
            content="""
# Search Approach Selection Framework

## Questions to Ask:

### 1. What type of queries do users ask?
- **Exact terms** → BM25
- **Conceptual/paraphrased** → Semantic
- **Mixed** → Hybrid

### 2. What's your performance budget?
- **Low latency critical** → BM25 or cached embeddings
- **Accuracy critical** → Hybrid with re-ranking
- **Balanced** → Semantic search

### 3. Do you need generated answers?
- **No, documents are fine** → Search (BM25/Semantic/Hybrid)
- **Yes, need synthesized answers** → RAG

### 4. How specialized is your domain?
- **General knowledge** → General embedding models
- **Specialized (medical, legal, finance)** → Domain-specific embeddings or hybrid

### 5. What's your infrastructure?
- **Simple setup** → BM25
- **GPU available** → Semantic or RAG
- **Enterprise scale** → Hybrid with caching
            """,
            speaker_notes="Walk through a few examples with the class. "
                         "This framework will be tested in the module quiz."
        ),
        
        Slide(
            slide_number=9,
            title="Real-World Example: Enterprise Search Evolution",
            content="""
# Case Study: Financial Services Company

## Initial System (2015-2020):
- Pure BM25 search over regulatory documents
- Fast but missed semantic matches
- Users frustrated with vocabulary mismatch

## Hybrid System (2020-2023):
- Added semantic search with financial domain embeddings
- Hybrid fusion of BM25 + vector search
- 40% improvement in user satisfaction

## RAG System (2023-Present):
- Added LLM generation layer
- Generates summaries with source citations
- Compliance team can verify sources
- 60% reduction in time-to-answer

## Key Lessons:
- Evolution, not revolution
- Hybrid approaches reduce risk
- Domain-specific embeddings matter
- Source attribution critical for compliance
            """,
            speaker_notes="This progression is typical in enterprise. "
                         "Emphasize the importance of source attribution for regulated industries."
        ),
        
        Slide(
            slide_number=10,
            title="Hands-On: Notebook 0",
            content="""
# Notebook 0: Search Paradigm Comparison

## What You'll Build:
1. Implement BM25 search on sample dataset
2. Implement semantic search with NVIDIA NIM embeddings
3. Implement hybrid search with fusion
4. Compare all three on identical queries

## Learning Goals:
- Understand practical differences between approaches
- See when each approach excels
- Debug intentional bugs in the implementation
- Develop intuition for search system design

## Dataset: USC Course Catalog
- Structured tabular data
- Good for comparing keyword vs semantic matching
- Realistic enterprise scenario

## Time: 20-25 minutes hands-on
            """,
            speaker_notes="Emphasize that there are intentional bugs to find. "
                         "This is a debugging exercise, not just implementation."
        ),
        
        Slide(
            slide_number=11,
            title="Key Takeaways",
            content="""
# Module 1 Summary

## Core Concepts:
1. **Classic Search**: Crawl → Analyze → Index → Rank
2. **BM25**: Fast keyword matching, no semantic understanding
3. **Semantic Search**: Embedding-based, captures meaning
4. **Hybrid Systems**: Combine BM25 + semantic + re-ranking
5. **RAG**: Search + generation for synthesized answers

## Decision Framework:
- Consider query types, performance budget, domain specificity
- Hybrid approaches common in enterprise
- RAG when you need generated answers with grounding

## Next Module:
Module 2 will dive deep into embeddings and vector stores—
the foundation of semantic search and RAG.

## Certification Relevance:
- Agent Architecture (15%): Understanding RAG pipeline stages
- Knowledge Integration (10%): Retrieval strategies and data handling
            """,
            speaker_notes="Recap the decision framework. "
                         "Preview Module 2 to maintain continuity."
        )
    ]
    
    # Define Mermaid diagrams
    diagrams = [
        MermaidDiagram(
            diagram_id="classic_search_flow",
            title="Classic Search Architecture Flow",
            mermaid_code="""
graph TB
    A[Web Pages] --> B[Crawler]
    B --> C[Content Analysis]
    C --> D[Indexing]
    D --> E[Inverted Index]
    F[User Query] --> G[Query Processing]
    G --> H[Ranking Algorithm]
    E --> H
    H --> I[Ranked Results]
    
    style A fill:#e1f5ff
    style E fill:#fff4e1
    style I fill:#e8f5e9
            """,
            description="Traditional search engine architecture showing the four main stages: "
                       "crawling, analysis, indexing, and ranking."
        ),
        
        MermaidDiagram(
            diagram_id="bm25_vs_semantic",
            title="BM25 vs Semantic Search Comparison",
            mermaid_code="""
graph LR
    subgraph "BM25 Search"
        A1[Query: car] --> B1[Term Matching]
        B1 --> C1[Documents with 'car']
        C1 --> D1[BM25 Scoring]
        D1 --> E1[Ranked Results]
    end
    
    subgraph "Semantic Search"
        A2[Query: car] --> B2[Embedding Model]
        B2 --> C2[Query Vector]
        C2 --> D2[Vector Similarity]
        E2[Document Vectors] --> D2
        D2 --> F2[Ranked Results]
    end
    
    style E1 fill:#ffebee
    style F2 fill:#e8f5e9
            """,
            description="Side-by-side comparison of BM25 keyword matching vs semantic search "
                       "with embeddings. BM25 matches exact terms, semantic search finds similar meanings."
        ),
        
        MermaidDiagram(
            diagram_id="hybrid_search_architecture",
            title="Enterprise Hybrid Search Architecture",
            mermaid_code="""
graph TB
    A[User Query] --> B[Query Processing]
    
    B --> C1[BM25 Retriever]
    B --> C2[Vector Retriever]
    
    C1 --> D1[Top 100 BM25 Results]
    C2 --> D2[Top 100 Vector Results]
    
    D1 --> E[Fusion Layer]
    D2 --> E
    
    E --> F[Combined Top 100]
    F --> G[Re-ranker Model]
    G --> H[Final Top 10 Results]
    
    I[Document Index] --> C1
    J[Vector Store] --> C2
    
    style A fill:#e3f2fd
    style E fill:#fff9c4
    style H fill:#c8e6c9
            """,
            description="Three-stage hybrid search: parallel BM25 and vector retrieval, "
                       "fusion of results, and re-ranking for final selection."
        ),
        
        MermaidDiagram(
            diagram_id="search_to_rag_evolution",
            title="Evolution from Search to RAG",
            mermaid_code="""
graph LR
    A[Traditional Search] --> B[Semantic Search]
    B --> C[Hybrid Search]
    C --> D[RAG System]
    
    A1[Returns: Documents] -.-> A
    B1[Returns: Documents] -.-> B
    C1[Returns: Documents] -.-> C
    D1[Returns: Generated Answers] -.-> D
    
    style A fill:#ffebee
    style B fill:#fff3e0
    style C fill:#e1f5fe
    style D fill:#c8e6c9
            """,
            description="The progression from traditional keyword search to RAG systems. "
                       "Key transition: from returning documents to generating answers."
        ),
        
        MermaidDiagram(
            diagram_id="rag_pipeline_overview",
            title="RAG Pipeline: Three Stages",
            mermaid_code="""
graph LR
    A[User Query] --> B[Stage 1: Retrieval]
    B --> C[Retrieved Documents]
    C --> D[Stage 2: Augmentation]
    D --> E[Query + Context]
    E --> F[Stage 3: Generation]
    F --> G[Generated Answer]
    
    H[Vector Store] --> B
    I[LLM] --> F
    
    style B fill:#e3f2fd
    style D fill:#fff9c4
    style F fill:#c8e6c9
            """,
            description="RAG pipeline showing the three core stages: retrieval of relevant documents, "
                       "augmentation of query with context, and generation of final answer."
        )
    ]
    
    # Define case studies
    case_studies = [
        CaseStudy(
            title="Financial Services: Regulatory Document Search",
            industry="finance",
            problem="Financial analysts needed to search thousands of regulatory documents. "
                   "Pure keyword search missed semantic matches. Queries like 'capital requirements' "
                   "wouldn't find documents discussing 'reserve ratios' even though they're related concepts.",
            solution="Implemented hybrid search system:\n"
                    "1. BM25 for exact regulatory term matching\n"
                    "2. Domain-specific financial embeddings for semantic search\n"
                    "3. Cross-encoder re-ranker for final ranking\n"
                    "4. Added RAG layer for generating summaries with citations",
            outcomes=[
                "40% improvement in search relevance (measured by user satisfaction)",
                "60% reduction in time-to-answer for analysts",
                "100% source attribution for compliance requirements",
                "Handles both exact regulatory terms and conceptual queries"
            ]
        ),
        
        CaseStudy(
            title="Healthcare: Medical Literature Search",
            industry="healthcare",
            problem="Doctors needed to search medical literature for diagnosis support. "
                   "General-purpose embeddings didn't understand medical terminology. "
                   "HIPAA compliance required on-premise deployment.",
            solution="Deployed specialized medical RAG system:\n"
                    "1. BioBERT embeddings for medical domain\n"
                    "2. On-premise vector store for HIPAA compliance\n"
                    "3. RAG with medical LLM for generating evidence-based summaries\n"
                    "4. Citation tracking for medical literature references",
            outcomes=[
                "95% accuracy on medical query benchmarks",
                "HIPAA compliant on-premise deployment",
                "2-second average response time",
                "Trusted by 500+ physicians in pilot program"
            ]
        ),
        
        CaseStudy(
            title="E-commerce: Product Search Evolution",
            industry="e-commerce",
            problem="Customers searched using natural language ('comfortable running shoes for flat feet') "
                   "but product catalog used technical specifications. "
                   "Pure keyword search returned irrelevant results.",
            solution="Built semantic product search:\n"
                    "1. Fine-tuned embeddings on product descriptions and user queries\n"
                    "2. Hybrid search: keywords for exact model numbers, semantic for descriptions\n"
                    "3. Personalization layer using user history\n"
                    "4. A/B tested against baseline keyword search",
            outcomes=[
                "23% increase in click-through rate",
                "18% increase in conversion rate",
                "Handles natural language queries effectively",
                "Reduced 'no results' pages by 45%"
            ]
        ),
        
        CaseStudy(
            title="Legal: Case Law Research",
            industry="legal",
            problem="Lawyers needed to find relevant case precedents. "
                   "Legal terminology is highly specific, but concepts span multiple phrasings. "
                   "Both exact citation matching and conceptual search required.",
            solution="Hybrid legal search system:\n"
                    "1. BM25 for exact case citations and statute numbers\n"
                    "2. Legal-domain embeddings (trained on case law corpus)\n"
                    "3. Temporal weighting (recent cases ranked higher)\n"
                    "4. RAG for generating case summaries with precedent analysis",
            outcomes=[
                "Handles both exact citations and conceptual queries",
                "Temporal relevance improves research quality",
                "Reduces legal research time by 35%",
                "Generates summaries with proper legal citations"
            ]
        )
    ]
    
    # Speaker notes for each slide
    speaker_notes = {
        1: "Start with motivation: Why do we need to understand search evolution? "
           "Emphasize that RAG builds on decades of search technology. "
           "Set expectations: this is a foundation module.",
        
        2: "Draw parallels to RAG: retrieval is like search, but with semantic understanding. "
           "Emphasize that indexing is still crucial in RAG (vector stores). "
           "Ask: Who has worked with search systems before?",
        
        3: "Give example: searching 'car' won't find 'automobile'. "
           "This is the key limitation that semantic search addresses. "
           "BM25 is still widely used because it's fast and interpretable.",
        
        4: "Emphasize: semantic search complements, not replaces, keyword search. "
           "This is why hybrid systems are common in enterprise. "
           "Show embedding visualization if time permits.",
        
        5: "Most production systems use hybrid approaches. "
           "This is a key insight for the certification exam. "
           "Decision framework will be tested in quiz.",
        
        6: "This is the architecture most enterprises use. "
           "Re-ranking is expensive but only applied to top candidates. "
           "Reciprocal Rank Fusion is a common fusion technique.",
        
        7: "Emphasize: RAG is search + generation. "
           "The retrieval component uses everything we just learned about search. "
           "RAG vs fine-tuning is a common exam question.",
        
        8: "Walk through a few examples with the class. "
           "This framework will be tested in the module quiz. "
           "Encourage students to think about their own use cases.",
        
        9: "This progression is typical in enterprise. "
           "Emphasize the importance of source attribution for regulated industries. "
           "Evolution, not revolution, is the key message.",
        
        10: "Emphasize that there are intentional bugs to find. "
            "This is a debugging exercise, not just implementation. "
            "Encourage experimentation with different approaches.",
        
        11: "Recap the decision framework. "
            "Preview Module 2 to maintain continuity. "
            "Open floor for questions before moving to hands-on."
    }
    
    # Create and return lecture material
    return LectureMaterial(
        module_number=1,
        title="Evolution of Search to RAG Systems",
        slides=slides,
        speaker_notes=speaker_notes,
        diagrams=diagrams,
        case_studies=case_studies
    )


# Create the lecture materials
module_1_lecture = create_module_1_lecture_materials()
