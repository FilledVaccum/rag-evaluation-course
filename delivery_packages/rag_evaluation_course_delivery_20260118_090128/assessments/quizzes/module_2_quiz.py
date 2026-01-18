"""
Module 2 Quiz: Embeddings and Vector Stores
Evaluating RAG and Semantic Search Systems Course

This quiz tests understanding of embedding fundamentals, domain-specific models,
vector store configuration, and chunking strategies.

Requirements: 13.1, 17.2, 2.1
"""

from src.models.assessment import (
    Assessment,
    AssessmentType,
    Question,
    QuestionType,
    Difficulty,
    EvaluationRubric
)


# ============================================================================
# MODULE 2 QUIZ QUESTIONS
# ============================================================================

QUIZ_QUESTIONS = [
    Question(
        question_id="m2_q1",
        question_text=(
            "What is the primary advantage of semantic search using embeddings "
            "over traditional keyword-based search (BM25)?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Semantic search is always faster than keyword search",
            "B) Semantic search can match queries to documents with similar meanings even when exact words differ",
            "C) Semantic search requires less computational resources",
            "D) Semantic search works without any training data"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Semantic search using embeddings captures the meaning of text in high-dimensional "
            "vector space, allowing it to match queries to documents based on semantic similarity "
            "rather than exact word matching. For example, a query about 'feline' can match "
            "documents about 'cat' because they have similar embeddings.\n\n"
            "Why other options are incorrect:\n"
            "A) Semantic search is typically slower than BM25 due to embedding generation and "
            "vector similarity computation.\n"
            "C) Semantic search requires more computational resources (embedding models, vector stores).\n"
            "D) Semantic search requires pre-trained embedding models trained on large text corpora."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q2",
        question_text=(
            "You are building a RAG system for a financial services company to search "
            "regulatory documents. Which embedding model would be most appropriate?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) sentence-transformers/all-MiniLM-L6-v2 (general-purpose, 384 dimensions)",
            "B) FinBERT (finance-specific, 768 dimensions)",
            "C) CodeBERT (code-specific, 768 dimensions)",
            "D) BioBERT (medical-specific, 768 dimensions)"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "FinBERT is specifically trained on financial corpus and understands financial "
            "terminology (EBITDA, P/E ratio, regulatory terms, etc.). For a financial services "
            "RAG system searching regulatory documents, domain-specific models significantly "
            "outperform general-purpose models.\n\n"
            "Why other options are incorrect:\n"
            "A) While MiniLM is fast and general-purpose, it lacks understanding of specialized "
            "financial terminology and would miss important semantic relationships in regulatory text.\n"
            "C) CodeBERT is optimized for programming code, not financial documents.\n"
            "D) BioBERT is optimized for medical/biomedical text, not financial documents."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q3",
        question_text=(
            "What is cosine similarity and why is it commonly used for comparing embeddings?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) It measures the absolute distance between vectors and is fast to compute",
            "B) It measures the angle between vectors, making it independent of vector magnitude",
            "C) It measures the sum of element-wise differences between vectors",
            "D) It measures the maximum difference between any two dimensions"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Cosine similarity measures the cosine of the angle between two vectors, ranging from "
            "-1 to 1. It's calculated as: cos(θ) = (A·B) / (||A|| × ||B||). This makes it "
            "independent of vector magnitude, focusing on direction/orientation. For text embeddings, "
            "this is ideal because we care about semantic similarity (direction) rather than "
            "absolute magnitude.\n\n"
            "Why other options are incorrect:\n"
            "A) This describes Euclidean distance, which is affected by vector magnitude.\n"
            "C) This describes Manhattan distance (L1 norm).\n"
            "D) This describes Chebyshev distance (L∞ norm)."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q4",
        question_text=(
            "When configuring a vector store for a production RAG system with 10 million documents "
            "where accuracy is critical, which index type and configuration would be most appropriate?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) FLAT index with exact search",
            "B) IVF_FLAT index with nlist=100, nprobe=5",
            "C) HNSW index with M=32, efConstruction=400, efSearch=200",
            "D) IVF_FLAT index with nlist=10, nprobe=1"
        ],
        correct_answer="C",
        explanation=(
            "Correct Answer: C\n\n"
            "For a production system with 10M documents where accuracy is critical, HNSW "
            "(Hierarchical Navigable Small World) with high parameters provides the best balance. "
            "M=32 creates more connections per layer (better recall), efConstruction=400 ensures "
            "high-quality index building, and efSearch=200 provides thorough search at query time. "
            "This configuration prioritizes accuracy over speed and memory.\n\n"
            "Why other options are incorrect:\n"
            "A) FLAT index provides exact search but is too slow for 10M documents (O(n) complexity).\n"
            "B) IVF with low nprobe=5 will have poor recall, missing relevant documents.\n"
            "D) IVF with nlist=10 and nprobe=1 is severely under-configured, resulting in very poor recall."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.ADVANCED
    ),
    
    Question(
        question_id="m2_q5",
        question_text=(
            "You are implementing a RAG system for a university course catalog. The data is in "
            "tabular format (CSV) with columns: course_name, units, description, schedule, instructor. "
            "What is the recommended approach for transforming this data for embedding?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Embed each column separately and store 5 embeddings per course",
            "B) Concatenate all columns without labels: 'CSCI 567 4 Machine Learning MW 2-3:20 Prof. Smith'",
            "C) Create self-descriptive strings with labels: 'Class name: CSCI 567. Topics: Machine Learning. Units: 4...'",
            "D) Only embed the description column and discard other columns"
        ],
        correct_answer="C",
        explanation=(
            "Correct Answer: C\n\n"
            "For tabular data in RAG systems, the recommended approach is to transform each row "
            "into a self-descriptive string with labels. This makes the text understandable without "
            "the table structure and helps the embedding model capture semantic relationships. "
            "Labels like 'Class name:', 'Topics:', 'Units:' provide context that improves embedding quality.\n\n"
            "Why other options are incorrect:\n"
            "A) Embedding columns separately loses relationships between fields and requires complex "
            "retrieval logic to combine results.\n"
            "B) Concatenation without labels creates ambiguous text that's hard for embedding models "
            "to interpret correctly.\n"
            "D) Discarding columns loses valuable information (course name, schedule, instructor) "
            "that may be relevant for queries."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q6",
        question_text=(
            "What is the primary trade-off when choosing chunk size for document splitting in RAG systems?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Larger chunks are always better because they provide more context",
            "B) Smaller chunks are always better because they are more precise",
            "C) Larger chunks provide more context but may include irrelevant information; smaller chunks are more precise but may lack context",
            "D) Chunk size doesn't significantly impact RAG performance"
        ],
        correct_answer="C",
        explanation=(
            "Correct Answer: C\n\n"
            "Chunk size involves a fundamental trade-off: larger chunks (e.g., 1500-2000 tokens) "
            "provide more context and reduce the risk of splitting related information, but may "
            "include irrelevant content that dilutes retrieval precision. Smaller chunks (e.g., "
            "300-500 tokens) are more precise and focused, but may lack sufficient context for "
            "the LLM to generate accurate responses. The optimal size depends on document structure "
            "and use case.\n\n"
            "Why other options are incorrect:\n"
            "A) Larger chunks can include too much irrelevant information, reducing precision.\n"
            "B) Smaller chunks may lack context needed for accurate generation.\n"
            "D) Chunk size significantly impacts both retrieval quality and generation accuracy."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q7",
        question_text=(
            "When should you use semantic chunking instead of fixed-size chunking?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) When documents have clear structure (headings, sections) and you want to preserve logical units",
            "B) When you need predictable chunk sizes for consistent processing",
            "C) When documents are unstructured text without clear boundaries",
            "D) When you want the fastest chunking implementation"
        ],
        correct_answer="A",
        explanation=(
            "Correct Answer: A\n\n"
            "Semantic chunking is ideal when documents have clear structure (headings, sections, "
            "paragraphs) because it preserves logical units of information. This maintains semantic "
            "coherence and prevents splitting concepts mid-thought. For example, keeping an entire "
            "section about 'RAG Architecture' together rather than splitting it arbitrarily.\n\n"
            "Why other options are incorrect:\n"
            "B) Semantic chunking produces variable-sized chunks based on content structure, not "
            "predictable sizes. Use fixed-size for predictable sizes.\n"
            "C) For unstructured text without clear boundaries, fixed-size chunking is more appropriate.\n"
            "D) Fixed-size chunking is typically faster and simpler to implement than semantic chunking."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q8",
        question_text=(
            "You notice that your RAG system retrieves relevant documents but they are ranked poorly "
            "(relevant docs appear at positions 8-10 instead of 1-3). Which vector store parameter "
            "should you adjust first?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Increase the embedding dimension",
            "B) Change from cosine similarity to Euclidean distance",
            "C) Increase efSearch (for HNSW) or nprobe (for IVF) to search more thoroughly",
            "D) Decrease the chunk size"
        ],
        correct_answer="C",
        explanation=(
            "Correct Answer: C\n\n"
            "Poor ranking of relevant documents (low precision at top-k) indicates the search is "
            "not thorough enough. For HNSW indexes, increasing efSearch makes the search explore "
            "more candidates at query time. For IVF indexes, increasing nprobe searches more clusters. "
            "Both improve ranking quality at the cost of slightly slower queries.\n\n"
            "Why other options are incorrect:\n"
            "A) Embedding dimension is determined by the model and cannot be changed without "
            "retraining or switching models.\n"
            "B) Changing distance metrics requires re-indexing and may not improve ranking. Cosine "
            "similarity is generally preferred for text embeddings.\n"
            "D) Chunk size affects what content is retrieved, not how well it's ranked. This is a "
            "retrieval configuration issue, not a chunking issue."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.ADVANCED
    ),
    
    Question(
        question_id="m2_q9",
        question_text=(
            "What is the purpose of overlap in fixed-size chunking?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) To increase the total number of chunks for better coverage",
            "B) To prevent information loss at chunk boundaries by including context from adjacent chunks",
            "C) To make chunks larger and provide more context",
            "D) To improve embedding quality by repeating important information"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Overlap (typically 10-20% of chunk size) ensures that information at chunk boundaries "
            "is not lost. For example, if a sentence is split across two chunks, the overlap ensures "
            "both chunks contain the complete sentence. This prevents context loss and improves "
            "retrieval quality. Without overlap, important information at boundaries might be "
            "incomplete in both chunks.\n\n"
            "Why other options are incorrect:\n"
            "A) While overlap does increase chunk count, that's a side effect, not the purpose.\n"
            "C) Overlap doesn't make individual chunks larger; it creates redundancy between chunks.\n"
            "D) Overlap is about preserving boundary context, not improving embedding quality through repetition."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m2_q10",
        question_text=(
            "Your company needs to build a RAG system that searches across documents in English, "
            "Spanish, French, and Arabic. What is the best approach for embedding model selection?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Use separate embedding models for each language and maintain separate vector stores",
            "B) Use a multilingual embedding model like multilingual-e5 that supports all languages in a shared vector space",
            "C) Translate all documents to English and use an English-only embedding model",
            "D) Use NVIDIA NV-Embed-v2 which automatically handles all languages"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Multilingual embedding models like multilingual-e5 are specifically designed to handle "
            "multiple languages in a shared vector space. This means a query in English can retrieve "
            "relevant documents in Spanish, French, or Arabic without translation. The model "
            "understands cross-lingual semantic similarity, making it ideal for multilingual RAG systems.\n\n"
            "Why other options are incorrect:\n"
            "A) Separate models and vector stores add complexity and prevent cross-lingual search "
            "(English query can't find Spanish documents).\n"
            "C) Translation adds latency, cost, and potential errors. It also loses language-specific "
            "nuances and requires translation at both indexing and query time.\n"
            "D) While NV-Embed-v2 is excellent for general-purpose use, it's primarily optimized for "
            "English. For true multilingual support including Arabic, a specialized multilingual model is better."
        ),
        points=1,
        exam_domain="Knowledge Integration and Data Handling (10%)",
        difficulty=Difficulty.ADVANCED
    )
]


# ============================================================================
# QUIZ CONFIGURATION
# ============================================================================

# Create evaluation rubric
QUIZ_RUBRIC = EvaluationRubric(
    rubric_id="module_2_quiz_rubric",
    criteria={
        "embedding_fundamentals": {
            "points": 3,
            "description": "Understanding of embedding concepts and semantic similarity"
        },
        "model_selection": {
            "points": 3,
            "description": "Ability to select appropriate embedding models for different domains"
        },
        "vector_store_configuration": {
            "points": 2,
            "description": "Knowledge of vector store configuration and optimization"
        },
        "chunking_strategies": {
            "points": 2,
            "description": "Understanding of chunking strategies and trade-offs"
        }
    },
    total_points=10,
    passing_score=7
)


# Create the quiz assessment
MODULE_2_QUIZ = Assessment(
    assessment_id="module_2_quiz",
    assessment_type=AssessmentType.QUIZ,
    module_number=2,
    title="Module 2 Quiz: Embeddings and Vector Stores",
    description=(
        "Test your understanding of embedding fundamentals, domain-specific models, "
        "vector store configuration, and chunking strategies. This quiz covers key concepts "
        "from Module 2 that are essential for building effective RAG systems."
    ),
    questions=QUIZ_QUESTIONS,
    rubric=QUIZ_RUBRIC,
    time_limit_minutes=30
)


# ============================================================================
# EXAM DOMAIN MAPPING
# ============================================================================

EXAM_DOMAIN_MAPPING = {
    "Knowledge Integration and Data Handling": {
        "weight": 10.0,
        "coverage": "⭐⭐⭐ CORE",
        "topics_covered": [
            "Embedding fundamentals and semantic similarity",
            "Domain-specific embedding model selection",
            "Vector store configuration and optimization",
            "Chunking strategies for different data types",
            "Tabular data transformation for RAG",
            "Multilingual embedding considerations"
        ]
    },
    "NVIDIA Platform Implementation": {
        "weight": 7.0,
        "coverage": "⭐⭐ INTEGRATED",
        "topics_covered": [
            "NVIDIA NIM for embeddings (NV-Embed-v2)",
            "NVIDIA embedding model integration",
            "NVIDIA platform best practices"
        ]
    }
}


# ============================================================================
# QUIZ USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 2 QUIZ: Embeddings and Vector Stores")
    print("=" * 70)
    print(f"\nQuiz ID: {MODULE_2_QUIZ.assessment_id}")
    print(f"Module: {MODULE_2_QUIZ.module_number}")
    print(f"Number of Questions: {len(MODULE_2_QUIZ.questions)}")
    print(f"Time Limit: {MODULE_2_QUIZ.time_limit_minutes} minutes")
    print(f"Passing Score: {MODULE_2_QUIZ.rubric.passing_score}/{MODULE_2_QUIZ.rubric.total_points}")
    
    print("\n" + "=" * 70)
    print("EXAM DOMAIN MAPPING")
    print("=" * 70)
    for domain, info in EXAM_DOMAIN_MAPPING.items():
        print(f"\n{domain} ({info['weight']}% of exam)")
        print(f"Coverage: {info['coverage']}")
        print("Topics:")
        for topic in info['topics_covered']:
            print(f"  - {topic}")
    
    print("\n" + "=" * 70)
    print("SAMPLE QUESTIONS")
    print("=" * 70)
    for i, question in enumerate(MODULE_2_QUIZ.questions[:3], 1):
        print(f"\nQuestion {i}:")
        print(question.question_text)
        print("\nOptions:")
        for option in question.options:
            print(f"  {option}")
        print(f"\nDifficulty: {question.difficulty.value}")
        print(f"Exam Domain: {question.exam_domain}")
