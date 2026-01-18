"""
Notebook 1: Embeddings and Chunking Strategies
Evaluating RAG and Semantic Search Systems Course

This notebook provides hands-on exercises for:
1. Implementing embeddings with NVIDIA NIM
2. Experimenting with chunking strategies
3. Transforming tabular data (USC course catalog)
4. Debugging intentional bugs in embedding pipelines

Learning Objectives:
- Generate embeddings using NVIDIA NIM
- Compare different chunking strategies
- Handle tabular data in RAG systems
- Debug common embedding pipeline issues

Requirements: 4.6, 10.2
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


# ============================================================================
# EXERCISE 1: IMPLEMENTING EMBEDDINGS WITH NVIDIA NIM
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Implementing Embeddings with NVIDIA NIM")
print("=" * 70)

EXERCISE_1_INSTRUCTIONS = """
# Exercise 1: Implementing Embeddings with NVIDIA NIM

## Objective
Learn to generate embeddings using NVIDIA NIM and understand semantic similarity.

## Tasks
1. Set up NVIDIA NIM client
2. Generate embeddings for sample texts
3. Calculate cosine similarity
4. Visualize semantic relationships

## Expected Outcomes
- Understand how embeddings capture semantic meaning
- See how similar texts have similar embeddings
- Learn to use NVIDIA NIM API

## Intentional Bug Alert! 🐛
There is an intentional bug in the embedding normalization function below.
Can you find and fix it?
"""

print(EXERCISE_1_INSTRUCTIONS)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    text: str
    embedding: List[float]
    model: str


class NVIDIANIMEmbedder:
    """
    Client for NVIDIA NIM embedding service.
    
    This is a simplified implementation for educational purposes.
    In production, use the official NVIDIA NIM SDK.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "nvidia/nv-embed-v2"):
        """
        Initialize NVIDIA NIM embedder.
        
        Args:
            api_key: NVIDIA API key (reads from NVIDIA_API_KEY env var if not provided)
            model: Embedding model name
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model
        self.dimension = 4096  # NV-Embed-v2 dimension
        
        if not self.api_key:
            print("⚠️  Warning: NVIDIA_API_KEY not set. Using mock embeddings for demonstration.")
            self.use_mock = True
        else:
            self.use_mock = False
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            EmbeddingResult with embedding vector
        """
        if self.use_mock:
            # Generate mock embedding for demonstration
            # In real implementation, call NVIDIA NIM API
            embedding = self._generate_mock_embedding(text)
        else:
            # Real implementation would call NVIDIA NIM API here
            # embedding = self._call_nvidia_nim_api(text)
            embedding = self._generate_mock_embedding(text)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model
        )
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of EmbeddingResults
        """
        return [self.embed_text(text) for text in texts]
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate mock embedding based on text content.
        
        This creates deterministic embeddings for demonstration.
        Similar texts will have similar embeddings.
        """
        # Use text hash for deterministic generation
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        
        # Generate base embedding
        embedding = np.random.randn(self.dimension)
        
        # Add semantic signals based on keywords
        if "cat" in text.lower() or "feline" in text.lower():
            embedding[0:10] += 2.0  # Boost animal-related dimensions
        if "dog" in text.lower() or "canine" in text.lower():
            embedding[0:10] += 1.8  # Similar but slightly different
        if "finance" in text.lower() or "stock" in text.lower():
            embedding[10:20] += 2.0  # Boost finance-related dimensions
        if "medical" in text.lower() or "health" in text.lower():
            embedding[20:30] += 2.0  # Boost medical-related dimensions
        
        # INTENTIONAL BUG: Incorrect normalization
        # Bug: Should normalize to unit length, but this divides by sum instead of L2 norm
        # This will cause incorrect cosine similarity calculations
        embedding = embedding / np.sum(np.abs(embedding))  # 🐛 BUG HERE!
        
        return embedding.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# Example usage
print("\n--- Example: Generating Embeddings ---\n")

embedder = NVIDIANIMEmbedder()

# Sample texts with semantic relationships
texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The dog played in the park",
    "Stock market crashed today",
    "Medical research shows promising results"
]

print("Generating embeddings for sample texts...")
results = embedder.embed_batch(texts)

print(f"\nGenerated {len(results)} embeddings")
print(f"Embedding dimension: {len(results[0].embedding)}")

# Calculate similarity matrix
print("\n--- Similarity Matrix ---\n")
print("Comparing semantic similarity between texts:\n")

for i, result1 in enumerate(results):
    for j, result2 in enumerate(results):
        if i < j:  # Only upper triangle
            similarity = cosine_similarity(result1.embedding, result2.embedding)
            print(f"Text {i+1} vs Text {j+1}: {similarity:.4f}")
            print(f"  '{result1.text[:40]}...'")
            print(f"  '{result2.text[:40]}...'")
            print()

print("\n💡 Expected Behavior:")
print("- Texts 1 and 2 (cat/feline) should have HIGH similarity")
print("- Texts 1 and 3 (cat/dog) should have MEDIUM similarity")
print("- Texts 1 and 4 (cat/stock) should have LOW similarity")
print("\n🐛 Debug Challenge: If similarities look wrong, find the bug!")


# ============================================================================
# EXERCISE 2: EXPERIMENTING WITH CHUNKING STRATEGIES
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Experimenting with Chunking Strategies")
print("=" * 70)

EXERCISE_2_INSTRUCTIONS = """
# Exercise 2: Experimenting with Chunking Strategies

## Objective
Compare different chunking strategies and understand their impact on retrieval.

## Tasks
1. Implement fixed-size chunking
2. Implement semantic chunking
3. Compare retrieval quality
4. Analyze trade-offs

## Expected Outcomes
- Understand chunk size vs. context trade-offs
- Learn when to use different strategies
- Measure impact on retrieval quality

## Intentional Bug Alert! 🐛
The overlap calculation in fixed-size chunking has a bug.
Can you spot it?
"""

print(EXERCISE_2_INSTRUCTIONS)


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text into segments."""
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize fixed-size chunking.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into fixed-size segments with overlap.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # INTENTIONAL BUG: Overlap calculation is wrong
            # Bug: Should be (end - overlap), but this adds instead of subtracts
            start = end + self.overlap  # 🐛 BUG HERE!
        
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking based on paragraph boundaries."""
    
    def __init__(self, max_chunk_size: int = 1000):
        """
        Initialize semantic chunking.
        
        Args:
            max_chunk_size: Maximum size of each chunk
        """
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text based on paragraph boundaries.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Split on double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Sample long text for chunking
SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a powerful technique that combines information retrieval with language model generation. RAG systems retrieve relevant documents from a knowledge base and use them to augment the context provided to a language model.

The RAG pipeline consists of three main stages: retrieval, augmentation, and generation. In the retrieval stage, the system searches for relevant documents based on the input query. This typically involves embedding the query and finding similar documents in a vector store.

In the augmentation stage, the retrieved documents are processed and formatted to provide context to the language model. This may involve ranking, filtering, and combining multiple documents.

Finally, in the generation stage, the language model generates a response based on both the original query and the retrieved context. This allows the model to provide more accurate and grounded responses.

Evaluation of RAG systems is critical for ensuring quality. Common metrics include faithfulness (whether the response is supported by the context), answer relevancy (whether the response addresses the query), and context precision (whether the retrieved documents are relevant).

Different chunking strategies can significantly impact RAG performance. Fixed-size chunking is simple but may split concepts. Semantic chunking respects document structure but produces variable-sized chunks. The choice depends on the specific use case and document type.
"""

print("\n--- Comparing Chunking Strategies ---\n")

# Fixed-size chunking
fixed_chunker = FixedSizeChunking(chunk_size=200, overlap=50)
fixed_chunks = fixed_chunker.chunk(SAMPLE_TEXT)

print(f"Fixed-size chunking (size=200, overlap=50):")
print(f"  Number of chunks: {len(fixed_chunks)}")
print(f"  Avg chunk size: {np.mean([len(c) for c in fixed_chunks]):.1f} chars")
print(f"  First chunk: '{fixed_chunks[0][:80]}...'")

# Semantic chunking
semantic_chunker = SemanticChunking(max_chunk_size=300)
semantic_chunks = semantic_chunker.chunk(SAMPLE_TEXT)

print(f"\nSemantic chunking (max_size=300):")
print(f"  Number of chunks: {len(semantic_chunks)}")
print(f"  Avg chunk size: {np.mean([len(c) for c in semantic_chunks]):.1f} chars")
print(f"  First chunk: '{semantic_chunks[0][:80]}...'")

print("\n💡 Analysis:")
print("- Fixed-size: Predictable sizes, may split concepts")
print("- Semantic: Respects structure, variable sizes")
print("\n🐛 Debug Challenge: Check if overlap is working correctly!")


# ============================================================================
# EXERCISE 3: TABULAR DATA TRANSFORMATION (USC COURSE CATALOG)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Tabular Data Transformation")
print("=" * 70)

EXERCISE_3_INSTRUCTIONS = """
# Exercise 3: Tabular Data Transformation

## Objective
Learn to transform tabular data into embedding-friendly text format.

## Tasks
1. Load USC course catalog data
2. Transform rows with and without labels
3. Compare embedding quality
4. Understand best practices

## Expected Outcomes
- Handle structured data in RAG systems
- Understand label addition benefits
- Learn column selection strategies

## Intentional Bug Alert! 🐛
The label formatting has an inconsistency that affects embedding quality.
Can you find it?
"""

print(EXERCISE_3_INSTRUCTIONS)


# Sample USC course catalog data
usc_courses_data = {
    'course_name': ['CSCI 567', 'CSCI 570', 'CSCI 585', 'BUAD 310', 'BUAD 425'],
    'units': [4, 4, 4, 4, 3],
    'description': [
        'Machine Learning fundamentals including supervised and unsupervised learning',
        'Analysis of Algorithms covering complexity theory and algorithm design',
        'Database Systems including SQL, NoSQL, and distributed databases',
        'Applied Business Statistics with focus on data analysis',
        'Strategic Management and competitive analysis'
    ],
    'schedule': ['MW 2:00-3:20 PM', 'TTh 3:30-4:50 PM', 'MW 4:00-5:20 PM', 'TTh 2:00-3:20 PM', 'MW 10:00-11:20 AM'],
    'instructor': ['Prof. Smith', 'Prof. Johnson', 'Prof. Williams', 'Prof. Brown', 'Prof. Davis']
}

usc_courses_df = pd.DataFrame(usc_courses_data)


def transform_row_with_labels(row: pd.Series) -> str:
    """
    Transform table row to self-descriptive string WITH labels.
    
    Args:
        row: DataFrame row
        
    Returns:
        Formatted string with labels
    """
    # INTENTIONAL BUG: Inconsistent label formatting
    # Bug: Some labels end with ":", others don't. This affects embedding quality.
    text = f"Class name: {row['course_name']}. "
    text += f"The course will cover the following topics {row['description']}. "  # 🐛 Missing colon!
    text += f"Units: {row['units']}. "
    text += f"Schedule: {row['schedule']}. "
    text += f"Instructor {row['instructor']}"  # 🐛 Missing colon!
    
    return text


def transform_row_without_labels(row: pd.Series) -> str:
    """
    Transform table row to string WITHOUT labels.
    
    Args:
        row: DataFrame row
        
    Returns:
        Formatted string without labels
    """
    return f"{row['course_name']} {row['description']} {row['units']} {row['schedule']} {row['instructor']}"


print("\n--- Tabular Data Transformation Examples ---\n")

# Transform first course with labels
print("WITH LABELS:")
print(transform_row_with_labels(usc_courses_df.iloc[0]))

print("\nWITHOUT LABELS:")
print(transform_row_without_labels(usc_courses_df.iloc[0]))

# Transform all courses
print("\n--- Transforming All Courses ---\n")

with_labels = [transform_row_with_labels(row) for _, row in usc_courses_df.iterrows()]
without_labels = [transform_row_without_labels(row) for _, row in usc_courses_df.iterrows()]

print(f"Transformed {len(with_labels)} courses")
print(f"Avg length with labels: {np.mean([len(t) for t in with_labels]):.1f} chars")
print(f"Avg length without labels: {np.mean([len(t) for t in without_labels]):.1f} chars")

# Generate embeddings for comparison
print("\n--- Embedding Quality Comparison ---\n")

embeddings_with_labels = embedder.embed_batch(with_labels[:3])
embeddings_without_labels = embedder.embed_batch(without_labels[:3])

print("Comparing embedding quality for similar courses:")
print("(CSCI 567 - Machine Learning vs CSCI 570 - Algorithms)\n")

sim_with_labels = cosine_similarity(
    embeddings_with_labels[0].embedding,
    embeddings_with_labels[1].embedding
)
sim_without_labels = cosine_similarity(
    embeddings_without_labels[0].embedding,
    embeddings_without_labels[1].embedding
)

print(f"Similarity WITH labels: {sim_with_labels:.4f}")
print(f"Similarity WITHOUT labels: {sim_without_labels:.4f}")

print("\n💡 Recommendation: Use labels for better semantic understanding")
print("🐛 Debug Challenge: Check label formatting consistency!")


# ============================================================================
# EXERCISE 4: DEBUGGING CHALLENGE
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Debugging Challenge")
print("=" * 70)

DEBUGGING_INSTRUCTIONS = """
# Exercise 4: Debugging Challenge

## Bugs to Find

This notebook contains THREE intentional bugs:

1. **Bug in NVIDIANIMEmbedder._generate_mock_embedding()**
   - Location: Embedding normalization
   - Symptom: Incorrect cosine similarity values
   - Hint: Check how vectors are normalized

2. **Bug in FixedSizeChunking.chunk()**
   - Location: Overlap calculation
   - Symptom: Chunks don't overlap correctly
   - Hint: Should overlap go forward or backward?

3. **Bug in transform_row_with_labels()**
   - Location: Label formatting
   - Symptom: Inconsistent label format
   - Hint: Check all labels end with ":"

## Your Task

1. Find all three bugs
2. Understand why they cause problems
3. Fix them
4. Verify fixes work correctly

## Hints

- Use print statements to inspect intermediate values
- Compare expected vs. actual behavior
- Test edge cases

Good luck! 🐛🔍
"""

print(DEBUGGING_INSTRUCTIONS)


# ============================================================================
# SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY AND NEXT STEPS")
print("=" * 70)

SUMMARY = """
# What You Learned

1. **Embedding Generation**
   - How to use NVIDIA NIM for embeddings
   - Understanding semantic similarity
   - Calculating cosine similarity

2. **Chunking Strategies**
   - Fixed-size vs. semantic chunking
   - Trade-offs between strategies
   - Impact on retrieval quality

3. **Tabular Data Handling**
   - Transforming structured data for embeddings
   - Benefits of label addition
   - Column selection strategies

4. **Debugging Skills**
   - Finding bugs in embedding pipelines
   - Understanding error symptoms
   - Testing and verification

# Next Steps

1. **Fix the bugs** in this notebook
2. **Experiment** with different chunk sizes and overlap values
3. **Try** different embedding models (if you have API access)
4. **Apply** these techniques to your own datasets
5. **Move on** to Module 3: RAG Architecture and Component Analysis

# Additional Resources

- NVIDIA NIM Documentation: https://docs.nvidia.com/nim/
- NV-Embed-v2 Model Card: https://huggingface.co/nvidia/nv-embed-v2
- Chunking Best Practices: See Module 2 lecture materials
- USC Course Catalog Dataset: course_materials/datasets/

# Questions?

- Review Module 2 lecture materials
- Check the course discussion forum
- Consult with instructors during office hours
"""

print(SUMMARY)

print("\n" + "=" * 70)
print("End of Notebook 1: Embeddings and Chunking Strategies")
print("=" * 70)
