"""
Notebook 0: Search Paradigm Comparison

This notebook implements and compares three search paradigms:
1. BM25 keyword search
2. Semantic search with NVIDIA NIM embeddings
3. Hybrid search with fusion and re-ranking

Requirements: 3.5, 10.2

NOTE: This notebook contains intentional bugs for students to debug as part of the learning exercise.
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Document:
    """Represents a document in the search corpus."""
    doc_id: str
    title: str
    content: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: str
    title: str
    score: float
    content: str
    method: str  # "bm25", "semantic", "hybrid"


class BM25Searcher:
    """
    BM25 keyword-based search implementation.
    
    BM25 (Best Match 25) is a ranking function used by search engines to estimate
    the relevance of documents to a given search query based on term frequency.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 searcher.
        
        Args:
            k1: Term frequency saturation parameter (typical: 1.2-2.0)
            b: Length normalization parameter (typical: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.term_doc_freq: Dict[str, Dict[str, int]] = {}
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split on whitespace.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_doc_freq = {}
        
        # Calculate document lengths and term frequencies
        for doc in documents:
            tokens = self.tokenize(doc.content)
            self.doc_lengths.append(len(tokens))
            
            # Track term frequencies in this document
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Update document frequency for each unique term
            for term in term_freq:
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                    self.term_doc_freq[term] = {}
                self.doc_freqs[term] += 1
                self.term_doc_freq[term][doc.doc_id] = term_freq[term]
        
        # Calculate average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF for each term
        N = len(documents)
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects ranked by BM25 score
        """
        query_tokens = self.tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.documents):
            score = 0.0
            doc_length = self.doc_lengths[idx]
            
            for term in query_tokens:
                if term not in self.term_doc_freq:
                    continue
                
                if doc.doc_id not in self.term_doc_freq[term]:
                    continue
                
                # Get term frequency in document
                tf = self.term_doc_freq[term][doc.doc_id]
                
                # Get IDF
                idf = self.idf.get(term, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                score += idf * (numerator / denominator)
            
            scores.append((doc, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for doc, score in scores[:top_k]:
            results.append(SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                score=score,
                content=doc.content,
                method="bm25"
            ))
        
        return results


class SemanticSearcher:
    """
    Semantic search using embeddings and vector similarity.
    
    This implementation uses NVIDIA NIM embeddings for encoding text
    into high-dimensional vectors and cosine similarity for ranking.
    """
    
    def __init__(self, embedding_model: str = "nvidia/nv-embed-v2"):
        """
        Initialize semantic searcher.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.doc_embeddings: Optional[np.ndarray] = None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using NVIDIA NIM.
        
        INTENTIONAL BUG: This is a placeholder that returns random embeddings.
        Students should replace this with actual NVIDIA NIM API calls.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector (768-dimensional)
        """
        # PLACEHOLDER: Replace with actual NVIDIA NIM API call
        # Example:
        # from nvidia_nim import EmbeddingClient
        # client = EmbeddingClient(api_key=os.getenv("NVIDIA_API_KEY"))
        # response = client.embed(text, model=self.embedding_model)
        # return np.array(response.embedding)
        
        # For now, return random embedding (THIS IS THE BUG!)
        np.random.seed(hash(text) % (2**32))  # Deterministic for same text
        return np.random.randn(768)
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents by computing their embeddings.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        embeddings = []
        
        for doc in documents:
            # Embed document content
            embedding = self.get_embedding(doc.content)
            embeddings.append(embedding)
        
        self.doc_embeddings = np.array(embeddings)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search documents using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects ranked by cosine similarity
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities
        scores = []
        for idx, doc in enumerate(self.documents):
            doc_embedding = self.doc_embeddings[idx]
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            scores.append((doc, similarity))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for doc, score in scores[:top_k]:
            results.append(SearchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                score=score,
                content=doc.content,
                method="semantic"
            ))
        
        return results


class HybridSearcher:
    """
    Hybrid search combining BM25 and semantic search with fusion and re-ranking.
    
    This implements a three-stage pipeline:
    1. Parallel retrieval from BM25 and semantic search
    2. Reciprocal Rank Fusion (RRF) to combine results
    3. Re-ranking using a cross-encoder (simplified here)
    """
    
    def __init__(self, bm25_searcher: BM25Searcher, semantic_searcher: SemanticSearcher):
        """
        Initialize hybrid searcher.
        
        Args:
            bm25_searcher: BM25 searcher instance
            semantic_searcher: Semantic searcher instance
        """
        self.bm25_searcher = bm25_searcher
        self.semantic_searcher = semantic_searcher
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = sum(1 / (k + rank(d)))
        where rank(d) is the rank of document d in each result list.
        
        Args:
            bm25_results: Results from BM25 search
            semantic_results: Results from semantic search
            k: RRF parameter (typical: 60)
            
        Returns:
            List of (doc_id, fused_score) tuples
        """
        scores = {}
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.doc_id
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
        # Add semantic scores
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result.doc_id
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
        # Sort by fused score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores
    
    def rerank(self, doc_ids: List[str], query: str, top_k: int = 10) -> List[str]:
        """
        Re-rank documents using a more sophisticated model.
        
        INTENTIONAL BUG: This is a placeholder that just returns the input order.
        Students should implement actual re-ranking logic (e.g., cross-encoder).
        
        Args:
            doc_ids: List of document IDs to re-rank
            query: Original query
            top_k: Number of results to return
            
        Returns:
            Re-ranked list of document IDs
        """
        # PLACEHOLDER: Replace with actual re-ranking model
        # Example:
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # pairs = [[query, self.get_doc_content(doc_id)] for doc_id in doc_ids]
        # scores = model.predict(pairs)
        # ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        # return [doc_id for doc_id, _ in ranked[:top_k]]
        
        # For now, just return input order (THIS IS THE BUG!)
        return doc_ids[:top_k]
    
    def search(self, query: str, top_k: int = 10, retrieve_k: int = 100) -> List[SearchResult]:
        """
        Hybrid search with fusion and re-ranking.
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            retrieve_k: Number of results to retrieve from each searcher
            
        Returns:
            List of SearchResult objects after fusion and re-ranking
        """
        # Stage 1: Parallel retrieval
        bm25_results = self.bm25_searcher.search(query, top_k=retrieve_k)
        semantic_results = self.semantic_searcher.search(query, top_k=retrieve_k)
        
        # Stage 2: Fusion
        fused_scores = self.reciprocal_rank_fusion(bm25_results, semantic_results)
        
        # Get top candidates for re-ranking
        candidate_ids = [doc_id for doc_id, _ in fused_scores[:retrieve_k]]
        
        # Stage 3: Re-ranking
        reranked_ids = self.rerank(candidate_ids, query, top_k=top_k)
        
        # Build final results
        # Create doc_id to document mapping
        doc_map = {doc.doc_id: doc for doc in self.bm25_searcher.documents}
        score_map = dict(fused_scores)
        
        results = []
        for doc_id in reranked_ids:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                results.append(SearchResult(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    score=score_map[doc_id],
                    content=doc.content,
                    method="hybrid"
                ))
        
        return results


def load_sample_dataset() -> List[Document]:
    """
    Load sample USC Course Catalog dataset.
    
    Returns:
        List of Document objects
    """
    # Sample course data (in practice, load from CSV/JSON)
    courses = [
        {
            "doc_id": "CSCI-567",
            "title": "CSCI 567: Machine Learning",
            "content": "Machine Learning fundamentals including supervised learning, "
                      "unsupervised learning, neural networks, and deep learning. "
                      "Topics: regression, classification, clustering, dimensionality reduction. "
                      "Prerequisites: linear algebra, probability, programming."
        },
        {
            "doc_id": "CSCI-561",
            "title": "CSCI 561: Foundations of Artificial Intelligence",
            "content": "Artificial Intelligence foundations covering search algorithms, "
                      "knowledge representation, planning, reasoning under uncertainty. "
                      "Topics: A* search, constraint satisfaction, Bayesian networks, "
                      "decision theory. Prerequisites: data structures, algorithms."
        },
        {
            "doc_id": "CSCI-544",
            "title": "CSCI 544: Applied Natural Language Processing",
            "content": "Natural Language Processing techniques for text analysis. "
                      "Topics: tokenization, part-of-speech tagging, parsing, "
                      "sentiment analysis, machine translation, language models. "
                      "Prerequisites: machine learning, programming."
        },
        {
            "doc_id": "CSCI-585",
            "title": "CSCI 585: Database Systems",
            "content": "Database management systems covering relational databases, "
                      "SQL, transaction processing, query optimization, indexing. "
                      "Topics: ER modeling, normalization, ACID properties, "
                      "NoSQL databases. Prerequisites: data structures."
        },
        {
            "doc_id": "CSCI-570",
            "title": "CSCI 570: Analysis of Algorithms",
            "content": "Algorithm design and analysis techniques. "
                      "Topics: divide and conquer, dynamic programming, greedy algorithms, "
                      "graph algorithms, NP-completeness, approximation algorithms. "
                      "Prerequisites: data structures, discrete mathematics."
        },
        {
            "doc_id": "CSCI-599",
            "title": "CSCI 599: Deep Learning",
            "content": "Deep learning architectures and applications. "
                      "Topics: convolutional neural networks, recurrent neural networks, "
                      "transformers, attention mechanisms, generative models, "
                      "reinforcement learning. Prerequisites: machine learning, linear algebra."
        },
        {
            "doc_id": "CSCI-572",
            "title": "CSCI 572: Information Retrieval and Web Search Engines",
            "content": "Information retrieval systems and web search technology. "
                      "Topics: indexing, ranking algorithms, BM25, PageRank, "
                      "semantic search, query processing, evaluation metrics. "
                      "Prerequisites: data structures, algorithms."
        },
        {
            "doc_id": "CSCI-566",
            "title": "CSCI 566: Deep Learning and Its Applications",
            "content": "Advanced deep learning methods and real-world applications. "
                      "Topics: computer vision, natural language understanding, "
                      "speech recognition, generative adversarial networks, "
                      "transfer learning. Prerequisites: machine learning, neural networks."
        }
    ]
    
    return [Document(**course) for course in courses]


def compare_search_methods(query: str, documents: List[Document]) -> Dict[str, List[SearchResult]]:
    """
    Compare all three search methods on the same query.
    
    Args:
        query: Search query
        documents: Document corpus
        
    Returns:
        Dictionary mapping method name to results
    """
    # Initialize searchers
    bm25 = BM25Searcher()
    semantic = SemanticSearcher()
    
    # Index documents
    bm25.index_documents(documents)
    semantic.index_documents(documents)
    
    # Create hybrid searcher
    hybrid = HybridSearcher(bm25, semantic)
    
    # Run searches
    results = {
        "bm25": bm25.search(query, top_k=5),
        "semantic": semantic.search(query, top_k=5),
        "hybrid": hybrid.search(query, top_k=5)
    }
    
    return results


def print_results(results: Dict[str, List[SearchResult]]) -> None:
    """
    Print search results in a readable format.
    
    Args:
        results: Dictionary mapping method name to results
    """
    for method, result_list in results.items():
        print(f"\n{'='*80}")
        print(f"{method.upper()} SEARCH RESULTS")
        print(f"{'='*80}")
        
        for i, result in enumerate(result_list, 1):
            print(f"\n{i}. {result.title} (Score: {result.score:.4f})")
            print(f"   {result.content[:150]}...")


def main():
    """
    Main notebook execution: compare search paradigms.
    """
    print("Notebook 0: Search Paradigm Comparison")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading USC Course Catalog dataset...")
    documents = load_sample_dataset()
    print(f"Loaded {len(documents)} courses")
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "natural language processing",
        "database management",
        "AI and intelligent systems"
    ]
    
    # Compare methods for each query
    for query in test_queries:
        print(f"\n\n{'#'*80}")
        print(f"QUERY: {query}")
        print(f"{'#'*80}")
        
        results = compare_search_methods(query, documents)
        print_results(results)
    
    print("\n\n" + "="*80)
    print("DEBUGGING EXERCISE")
    print("="*80)
    print("""
This notebook contains TWO intentional bugs for you to find and fix:

1. BUG #1: The semantic search is using random embeddings instead of real NVIDIA NIM embeddings.
   - Location: SemanticSearcher.get_embedding() method
   - Fix: Replace the placeholder with actual NVIDIA NIM API calls
   - Hint: You'll need to use the NVIDIA NIM client library

2. BUG #2: The hybrid search re-ranker is not actually re-ranking.
   - Location: HybridSearcher.rerank() method
   - Fix: Implement actual re-ranking logic using a cross-encoder or similar model
   - Hint: Consider using sentence-transformers CrossEncoder

TASK: Find and fix both bugs, then re-run the comparison to see improved results!
    """)


if __name__ == "__main__":
    main()
