"""
Module 1 Concept Summary: Evolution of Search to RAG Systems

One-page summary of key concepts, decision frameworks, and visual aids for Module 1.

Requirements: 17.2
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ConceptSummary:
    """Represents a key concept with definition and examples."""
    concept_name: str
    definition: str
    key_points: List[str]
    examples: List[str]


class Module1ConceptSummary:
    """
    One-page concept summary for Module 1: Evolution of Search to RAG Systems.
    
    This summary provides concise coverage of:
    - Classic search architecture
    - BM25 vs semantic search
    - RAG system fundamentals
    - Hybrid system design
    - Decision frameworks for search approach selection
    """
    
    def __init__(self):
        """Initialize the concept summary with all key concepts."""
        self.module_number = 1
        self.module_title = "Evolution of Search to RAG Systems"
        self.concepts = self._define_concepts()
        self.decision_framework = self._create_decision_framework()
        self.visual_diagrams = self._define_visual_diagrams()
    
    def _define_concepts(self) -> List[ConceptSummary]:
        """Define all key concepts for Module 1."""
        return [
            ConceptSummary(
                concept_name="Classic Search Architecture",
                definition="Traditional search systems that crawl, index, and rank documents based on keyword matching and relevance signals.",
                key_points=[
                    "Four stages: Crawling → Analysis → Indexing → Ranking",
                    "Relies on keyword matching and statistical relevance",
                    "Uses inverted indexes for fast lookup",
                    "Ranking algorithms often proprietary (closed-source)"
                ],
                examples=[
                    "Google Search (web-scale)",
                    "Enterprise document search",
                    "E-commerce product search"
                ]
            ),
            ConceptSummary(
                concept_name="BM25 (Best Match 25)",
                definition="A keyword-based ranking algorithm that scores documents based on term frequency, inverse document frequency, and document length normalization.",
                key_points=[
                    "Probabilistic ranking function",
                    "Considers term frequency (TF) and inverse document frequency (IDF)",
                    "Normalizes for document length",
                    "No understanding of semantic meaning",
                    "Fast and efficient for exact keyword matches"
                ],
                examples=[
                    "Elasticsearch default ranking",
                    "Academic paper search",
                    "Legal document retrieval"
                ]
            ),
            ConceptSummary(
                concept_name="Semantic Search",
                definition="Search systems that understand the meaning and context of queries and documents using embeddings and vector similarity.",
                key_points=[
                    "Uses neural embeddings to represent text as vectors",
                    "Captures semantic meaning, not just keywords",
                    "Finds conceptually similar content",
                    "Handles synonyms and paraphrasing naturally",
                    "Requires vector stores for efficient retrieval"
                ],
                examples=[
                    "\"affordable car\" matches \"inexpensive vehicle\"",
                    "\"machine learning\" matches \"artificial intelligence\"",
                    "Cross-lingual search (query in English, find Spanish docs)"
                ]
            ),
            ConceptSummary(
                concept_name="RAG (Retrieval-Augmented Generation)",
                definition="A system that combines information retrieval with language model generation to produce accurate, grounded responses.",
                key_points=[
                    "Three stages: Retrieval → Augmentation → Generation",
                    "Retrieves relevant context from knowledge base",
                    "Augments LLM prompt with retrieved context",
                    "Generates response grounded in retrieved information",
                    "Reduces hallucinations by providing factual context"
                ],
                examples=[
                    "Customer support chatbot with company knowledge base",
                    "Medical Q&A system with clinical guidelines",
                    "Code assistant with documentation retrieval"
                ]
            ),
            ConceptSummary(
                concept_name="Hybrid Search Systems",
                definition="Enterprise systems that combine multiple search approaches (BM25 + Vector + Re-ranking) to leverage strengths of each method.",
                key_points=[
                    "Combines keyword and semantic search",
                    "BM25 for exact matches and rare terms",
                    "Vector search for semantic similarity",
                    "Re-ranking stage for final optimization",
                    "Balances precision and recall"
                ],
                examples=[
                    "Enterprise knowledge management systems",
                    "E-commerce with filters + semantic search",
                    "Legal research platforms"
                ]
            ),
            ConceptSummary(
                concept_name="Vector Stores",
                definition="Specialized databases optimized for storing and retrieving high-dimensional embedding vectors using approximate nearest neighbor search.",
                key_points=[
                    "Store embeddings (dense vectors)",
                    "Fast similarity search using ANN algorithms",
                    "Support filtering and metadata",
                    "Horizontal scaling for large datasets",
                    "Examples: Milvus, Pinecone, Chroma, Weaviate"
                ],
                examples=[
                    "Milvus for billion-scale vector search",
                    "Pinecone for managed vector database",
                    "Chroma for lightweight local development"
                ]
            )
        ]
    
    def _create_decision_framework(self) -> Dict[str, str]:
        """
        Create decision framework for selecting appropriate search approach.
        
        Returns a dictionary mapping use cases to recommended approaches.
        """
        return {
            "Exact keyword matching required": "BM25 (keyword search)",
            "Rare or technical terms": "BM25 (keyword search)",
            "Semantic understanding needed": "Vector search (semantic)",
            "Synonyms and paraphrasing": "Vector search (semantic)",
            "Cross-lingual search": "Vector search with multilingual embeddings",
            "Enterprise with legacy systems": "Hybrid (BM25 + Vector + Re-ranking)",
            "Need both precision and recall": "Hybrid (BM25 + Vector + Re-ranking)",
            "Question answering with context": "RAG (Retrieval-Augmented Generation)",
            "Reduce LLM hallucinations": "RAG (Retrieval-Augmented Generation)",
            "Grounded, factual responses": "RAG (Retrieval-Augmented Generation)"
        }
    
    def _define_visual_diagrams(self) -> Dict[str, str]:
        """Define Mermaid diagrams for visual aids."""
        return {
            "classic_search_flow": """
```mermaid
graph LR
    A[Web Crawling] --> B[Document Analysis]
    B --> C[Indexing]
    C --> D[Ranking]
    D --> E[Search Results]
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#fff4e1
    style E fill:#e8f5e9
```
            """,
            "rag_pipeline": """
```mermaid
graph LR
    A[User Query] --> B[Retrieval Stage]
    B --> C[Augmentation Stage]
    C --> D[Generation Stage]
    D --> E[Response]
    
    B -.->|Vector Search| F[Knowledge Base]
    C -.->|Context Injection| G[LLM Prompt]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#fff4e1
    style E fill:#e8f5e9
```
            """,
            "hybrid_system": """
```mermaid
graph TB
    A[User Query] --> B[BM25 Search]
    A --> C[Vector Search]
    
    B --> D[Candidate Results]
    C --> D
    
    D --> E[Re-ranking Stage]
    E --> F[Final Results]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#ffe1f5
    style E fill:#ffe1f5
    style F fill:#e8f5e9
```
            """
        }
    
    def get_concept(self, concept_name: str) -> ConceptSummary:
        """Get a specific concept by name."""
        for concept in self.concepts:
            if concept.concept_name == concept_name:
                return concept
        raise ValueError(f"Concept '{concept_name}' not found")
    
    def generate_one_page_summary(self) -> str:
        """
        Generate a one-page summary suitable for quick review and study.
        
        Returns formatted text summary with all key concepts, decision framework,
        and references to visual diagrams.
        """
        summary = []
        summary.append("=" * 80)
        summary.append(f"MODULE {self.module_number}: {self.module_title}")
        summary.append("ONE-PAGE CONCEPT SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Key concepts
        summary.append("KEY CONCEPTS")
        summary.append("-" * 80)
        for concept in self.concepts:
            summary.append(f"\n{concept.concept_name}")
            summary.append(f"  Definition: {concept.definition}")
            summary.append("  Key Points:")
            for point in concept.key_points:
                summary.append(f"    • {point}")
            if concept.examples:
                summary.append("  Examples:")
                for example in concept.examples:
                    summary.append(f"    - {example}")
        
        summary.append("")
        summary.append("DECISION FRAMEWORK: When to Use Each Approach")
        summary.append("-" * 80)
        for use_case, approach in self.decision_framework.items():
            summary.append(f"  {use_case}")
            summary.append(f"    → {approach}")
        
        summary.append("")
        summary.append("COMPARISON TABLE")
        summary.append("-" * 80)
        summary.append("| Aspect          | BM25 (Keyword)      | Semantic Search     | RAG                 |")
        summary.append("|-----------------|---------------------|---------------------|---------------------|")
        summary.append("| Matching        | Exact keywords      | Semantic meaning    | Context + Generation|")
        summary.append("| Synonyms        | No                  | Yes                 | Yes                 |")
        summary.append("| Speed           | Very fast           | Fast (with ANN)     | Moderate            |")
        summary.append("| Rare terms      | Excellent           | Poor                | Good                |")
        summary.append("| Understanding   | None                | High                | Very high           |")
        summary.append("| Hallucinations  | N/A                 | N/A                 | Reduced             |")
        summary.append("| Use case        | Exact match         | Similarity search   | Q&A, generation     |")
        
        summary.append("")
        summary.append("KEY FORMULAS")
        summary.append("-" * 80)
        summary.append("BM25 Score:")
        summary.append("  score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))")
        summary.append("  where:")
        summary.append("    - f(qi,D) = term frequency of qi in document D")
        summary.append("    - |D| = document length")
        summary.append("    - avgdl = average document length")
        summary.append("    - k1, b = tuning parameters (typically k1=1.5, b=0.75)")
        summary.append("")
        summary.append("Vector Similarity (Cosine):")
        summary.append("  similarity(A,B) = (A · B) / (||A|| × ||B||)")
        summary.append("  Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite")
        
        summary.append("")
        summary.append("VISUAL DIAGRAMS")
        summary.append("-" * 80)
        summary.append("See course materials for:")
        summary.append("  1. Classic Search Flow (Crawling → Analysis → Indexing → Ranking)")
        summary.append("  2. RAG Pipeline (Retrieval → Augmentation → Generation)")
        summary.append("  3. Hybrid System Architecture (BM25 + Vector + Re-ranking)")
        
        summary.append("")
        summary.append("EXAM RELEVANCE")
        summary.append("-" * 80)
        summary.append("This module covers:")
        summary.append("  • Agent Architecture and Design (15% of NCP-AAI exam)")
        summary.append("  • Knowledge Integration and Data Handling (10% of NCP-AAI exam)")
        summary.append("")
        summary.append("Key exam topics:")
        summary.append("  - Foundational agent structuring and design")
        summary.append("  - Search system evolution and architecture")
        summary.append("  - RAG architecture patterns")
        summary.append("  - Hybrid system design considerations")
        
        summary.append("")
        summary.append("STUDY TIPS")
        summary.append("-" * 80)
        summary.append("  • Understand when to use each search approach (decision framework)")
        summary.append("  • Know the trade-offs: BM25 (fast, exact) vs Vector (semantic, flexible)")
        summary.append("  • Remember RAG reduces hallucinations by grounding in retrieved context")
        summary.append("  • Practice: Run Notebook 0 to compare search paradigms hands-on")
        summary.append("  • Memorize: RAG = Retrieval → Augmentation → Generation")
        
        summary.append("")
        return "\n".join(summary)
    
    def export_to_markdown(self, filepath: str) -> None:
        """Export the one-page summary to a markdown file."""
        content = self.generate_one_page_summary()
        with open(filepath, 'w') as f:
            f.write(content)


# Example usage
if __name__ == "__main__":
    summary = Module1ConceptSummary()
    
    # Print one-page summary
    print(summary.generate_one_page_summary())
    
    # Example: Get specific concept
    rag_concept = summary.get_concept("RAG (Retrieval-Augmented Generation)")
    print(f"\nRAG Definition: {rag_concept.definition}")
