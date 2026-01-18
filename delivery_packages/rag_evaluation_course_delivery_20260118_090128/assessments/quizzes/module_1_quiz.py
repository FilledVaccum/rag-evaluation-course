"""
Module 1 Quiz: Evolution of Search to RAG Systems

This quiz tests understanding of search evolution, BM25 vs semantic search,
hybrid systems, and RAG architecture.

Requirements: 13.1, 17.3
"""

from src.models.assessment import (
    Assessment,
    AssessmentType,
    Question,
    QuestionType,
    Difficulty,
    EvaluationRubric
)


def create_module_1_quiz() -> Assessment:
    """
    Create Module 1 quiz with 8 questions covering search evolution concepts.
    
    Returns:
        Assessment object with quiz questions and rubric
    """
    
    questions = [
        Question(
            question_id="m1_q1",
            question_text=(
                "What is the primary limitation of BM25 keyword-based search that "
                "semantic search addresses?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "BM25 is too slow for large-scale applications",
                "BM25 cannot handle synonyms and paraphrasing",
                "BM25 requires expensive GPU hardware",
                "BM25 cannot rank documents by relevance"
            ],
            correct_answer="BM25 cannot handle synonyms and paraphrasing",
            explanation=(
                "The primary limitation of BM25 is its inability to understand semantic meaning. "
                "BM25 performs exact term matching, so searching for 'car' will not find documents "
                "containing 'automobile' even though they are synonyms. Semantic search using "
                "embeddings captures meaning in vector space, allowing it to match semantically "
                "similar terms even if the exact words differ. This is called the vocabulary "
                "mismatch problem.\n\n"
                "Why other options are incorrect:\n"
                "- BM25 is actually very fast (faster than semantic search)\n"
                "- BM25 does not require GPU hardware\n"
                "- BM25 does rank documents by relevance using term frequency and IDF"
            ),
            exam_domain="Agent Architecture",
            difficulty=Difficulty.INTERMEDIATE,
            points=1
        ),
        
        Question(
            question_id="m1_q2",
            question_text=(
                "In a hybrid search system, what is the purpose of the re-ranking stage?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "To combine results from BM25 and vector search",
                "To apply a more sophisticated model to refine the top candidates",
                "To remove duplicate documents from results",
                "To convert documents into embeddings"
            ],
            correct_answer="To apply a more sophisticated model to refine the top candidates",
            explanation=(
                "Re-ranking is the third stage in a hybrid search pipeline. After parallel "
                "retrieval (BM25 + vector search) and fusion, re-ranking applies a more "
                "sophisticated model (like a cross-encoder) to the top candidates. This is "
                "computationally expensive, so it's only applied to a small set of top results "
                "(e.g., top 100 → top 10).\n\n"
                "The three stages are:\n"
                "1. Parallel Retrieval: BM25 and vector search retrieve top-k each\n"
                "2. Fusion: Combine results using techniques like Reciprocal Rank Fusion\n"
                "3. Re-ranking: Apply expensive model to refine final ranking\n\n"
                "Why other options are incorrect:\n"
                "- Combining results happens in the fusion stage, not re-ranking\n"
                "- Deduplication typically happens during fusion\n"
                "- Embedding conversion happens before retrieval, not during re-ranking"
            ),
            exam_domain="Agent Architecture",
            difficulty=Difficulty.INTERMEDIATE,
            points=1
        ),
        
        Question(
            question_id="m1_q3",
            question_text=(
                "What is the key difference between RAG and traditional search systems?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "RAG uses embeddings while search uses keywords",
                "RAG returns generated answers while search returns documents",
                "RAG is faster than traditional search",
                "RAG does not require an index"
            ],
            correct_answer="RAG returns generated answers while search returns documents",
            explanation=(
                "The fundamental difference is in the output: traditional search systems return "
                "documents or passages, while RAG (Retrieval-Augmented Generation) returns "
                "generated answers synthesized by an LLM.\n\n"
                "RAG pipeline:\n"
                "1. Retrieval: Find relevant documents (like search)\n"
                "2. Augmentation: Combine query + retrieved context\n"
                "3. Generation: LLM generates answer from context\n\n"
                "Both RAG and modern search can use embeddings. The retrieval component of RAG "
                "is essentially a search system. The key innovation is adding the generation "
                "layer to synthesize answers rather than just returning documents.\n\n"
                "Why other options are incorrect:\n"
                "- Both can use embeddings or keywords\n"
                "- RAG is typically slower due to LLM generation\n"
                "- RAG still requires an index for retrieval"
            ),
            exam_domain="Agent Architecture",
            difficulty=Difficulty.BEGINNER,
            points=1
        ),
        
        Question(
            question_id="m1_q4",
            question_text=(
                "When would you choose BM25 over semantic search for a production system?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "When users search using natural language questions",
                "When exact terminology matching is critical and speed is important",
                "When the domain requires understanding synonyms",
                "When working with multilingual content"
            ],
            correct_answer="When exact terminology matching is critical and speed is important",
            explanation=(
                "BM25 is the best choice when:\n"
                "1. Exact terminology matters (technical terms, product codes, legal citations)\n"
                "2. Speed is critical (BM25 is much faster than embedding-based search)\n"
                "3. Simple setup is required (no embedding model needed)\n"
                "4. Interpretability is important (can see which terms matched)\n\n"
                "Example use cases:\n"
                "- Legal case citation search (exact case numbers)\n"
                "- Product catalog search by SKU\n"
                "- Technical documentation with specific terminology\n"
                "- Regulatory compliance search (exact regulatory terms)\n\n"
                "Why other options are incorrect:\n"
                "- Natural language questions → semantic search is better\n"
                "- Understanding synonyms → semantic search is better\n"
                "- Multilingual content → semantic search with multilingual embeddings is better"
            ),
            exam_domain="Knowledge Integration",
            difficulty=Difficulty.INTERMEDIATE,
            points=1
        ),
        
        Question(
            question_id="m1_q5",
            question_text=(
                "What is Reciprocal Rank Fusion (RRF) used for in hybrid search?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "To embed documents into vector space",
                "To combine rankings from multiple retrievers into a single ranking",
                "To re-rank documents using a cross-encoder",
                "To filter out irrelevant documents"
            ],
            correct_answer="To combine rankings from multiple retrievers into a single ranking",
            explanation=(
                "Reciprocal Rank Fusion (RRF) is a fusion technique that combines rankings from "
                "multiple retrievers (e.g., BM25 and vector search) into a single unified ranking.\n\n"
                "RRF formula: score(d) = Σ 1/(k + rank(d))\n"
                "where rank(d) is the rank of document d in each retriever's results, "
                "and k is a constant (typically 60).\n\n"
                "How it works:\n"
                "1. BM25 retrieves top 100 documents with rankings\n"
                "2. Vector search retrieves top 100 documents with rankings\n"
                "3. RRF assigns scores based on ranks in both lists\n"
                "4. Documents appearing high in both lists get higher scores\n"
                "5. Final ranking is by combined RRF score\n\n"
                "Benefits:\n"
                "- Simple and effective\n"
                "- No training required\n"
                "- Robust to different score scales from different retrievers\n\n"
                "Why other options are incorrect:\n"
                "- Embedding happens before retrieval\n"
                "- Re-ranking happens after fusion\n"
                "- Filtering is a separate step"
            ),
            exam_domain="Agent Architecture",
            difficulty=Difficulty.ADVANCED,
            points=1
        ),
        
        Question(
            question_id="m1_q6",
            question_text=(
                "In the classic search architecture, what is the purpose of the inverted index?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "To store documents in reverse chronological order",
                "To map terms to the documents containing them for fast lookup",
                "To rank documents by relevance",
                "To convert documents into embeddings"
            ],
            correct_answer="To map terms to the documents containing them for fast lookup",
            explanation=(
                "An inverted index is a data structure that maps terms to the documents containing "
                "them. It's called 'inverted' because it inverts the document-to-terms relationship "
                "into a term-to-documents relationship.\n\n"
                "Structure:\n"
                "- Term → List of (document_id, term_frequency, positions)\n"
                "- Example: 'machine' → [(doc1, 3, [5,12,45]), (doc5, 1, [23]), ...]\n\n"
                "Benefits:\n"
                "- Fast lookup: Given a query term, instantly find all documents containing it\n"
                "- Efficient: Only need to check documents containing query terms\n"
                "- Supports ranking: Stores term frequencies for BM25 scoring\n\n"
                "The four stages of classic search:\n"
                "1. Crawling: Discover and fetch documents\n"
                "2. Analysis: Extract text and metadata\n"
                "3. Indexing: Build inverted index\n"
                "4. Ranking: Use index to find and rank relevant documents\n\n"
                "Why other options are incorrect:\n"
                "- Not about chronological order\n"
                "- Ranking uses the index but is a separate step\n"
                "- Embeddings are for semantic search, not inverted indexes"
            ),
            exam_domain="Knowledge Integration",
            difficulty=Difficulty.BEGINNER,
            points=1
        ),
        
        Question(
            question_id="m1_q7",
            question_text=(
                "Why is source attribution important in RAG systems, especially for "
                "regulated industries like finance and healthcare?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "To improve the accuracy of generated answers",
                "To allow verification of claims and ensure compliance with regulations",
                "To reduce the computational cost of generation",
                "To enable faster retrieval of documents"
            ],
            correct_answer="To allow verification of claims and ensure compliance with regulations",
            explanation=(
                "Source attribution is critical in RAG systems for regulated industries because:\n\n"
                "1. Verification: Users can verify that generated claims are supported by retrieved sources\n"
                "2. Compliance: Regulations (GDPR, HIPAA, financial regulations) often require "
                "   traceability of information\n"
                "3. Trust: Users can assess the credibility of sources\n"
                "4. Accountability: Organizations can demonstrate due diligence\n"
                "5. Error correction: If a source is wrong, it can be identified and corrected\n\n"
                "Example use cases:\n"
                "- Healthcare: Doctors need to verify medical claims against literature\n"
                "- Finance: Analysts need to cite regulatory documents\n"
                "- Legal: Lawyers need to cite case precedents\n\n"
                "Implementation:\n"
                "- Track which retrieved documents contributed to the answer\n"
                "- Include citations in generated responses\n"
                "- Provide links to original sources\n\n"
                "Why other options are incorrect:\n"
                "- Source attribution doesn't directly improve accuracy (though it enables verification)\n"
                "- It doesn't reduce computational cost\n"
                "- It doesn't affect retrieval speed"
            ),
            exam_domain="Knowledge Integration",
            difficulty=Difficulty.INTERMEDIATE,
            points=1
        ),
        
        Question(
            question_id="m1_q8",
            question_text=(
                "What is the main advantage of RAG over fine-tuning an LLM for "
                "domain-specific knowledge?"
            ),
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=[
                "RAG is always more accurate than fine-tuning",
                "RAG allows dynamic knowledge updates without retraining the model",
                "RAG requires less computational resources than fine-tuning",
                "RAG works better for all types of tasks"
            ],
            correct_answer="RAG allows dynamic knowledge updates without retraining the model",
            explanation=(
                "The key advantage of RAG is dynamic knowledge updates. With RAG, you can update "
                "the knowledge base by simply adding/modifying documents in the retrieval corpus. "
                "No model retraining is required.\n\n"
                "RAG vs Fine-tuning comparison:\n\n"
                "RAG:\n"
                "+ Dynamic knowledge: Update corpus anytime\n"
                "+ Source attribution: Can cite sources\n"
                "+ Lower training cost: No model retraining\n"
                "+ Handles rare/new information well\n"
                "- Retrieval overhead: Slower inference\n"
                "- Depends on retrieval quality\n\n"
                "Fine-tuning:\n"
                "+ Faster inference: No retrieval needed\n"
                "+ Better for style/format adaptation\n"
                "- Static knowledge: Requires retraining for updates\n"
                "- No source attribution\n"
                "- Expensive retraining for knowledge updates\n\n"
                "Best practice: Use both!\n"
                "- Fine-tune for task-specific behavior and style\n"
                "- Use RAG for dynamic knowledge and factual grounding\n\n"
                "Why other options are incorrect:\n"
                "- RAG is not always more accurate (depends on retrieval quality)\n"
                "- RAG can require more resources at inference time (retrieval + generation)\n"
                "- Neither is universally better for all tasks"
            ),
            exam_domain="Agent Architecture",
            difficulty=Difficulty.INTERMEDIATE,
            points=1
        )
    ]
    
    # Create evaluation rubric
    rubric = EvaluationRubric(
        rubric_id="module_1_quiz_rubric",
        criteria={
            "understanding": {
                "points": 8,
                "description": "Demonstrates understanding of search evolution and RAG concepts"
            }
        },
        total_points=8,
        passing_score=6  # 75% passing score
    )
    
    # Create assessment
    return Assessment(
        assessment_id="quiz_module_1",
        assessment_type=AssessmentType.QUIZ,
        module_number=1,
        title="Module 1 Quiz: Evolution of Search to RAG Systems",
        description=(
            "This quiz tests your understanding of the evolution from traditional search "
            "to RAG systems, including BM25 keyword search, semantic search with embeddings, "
            "hybrid search architectures, and the key differences between search and RAG. "
            "\n\n"
            "Topics covered:\n"
            "- Classic search architecture (crawling, indexing, ranking)\n"
            "- BM25 vs semantic search comparison\n"
            "- Hybrid search systems with fusion and re-ranking\n"
            "- RAG architecture and use cases\n"
            "- Decision frameworks for choosing search approaches\n"
            "\n\n"
            "Time limit: 30 minutes\n"
            "Passing score: 6/8 (75%)\n"
            "Question types: Multiple choice with detailed explanations"
        ),
        questions=questions,
        rubric=rubric,
        time_limit_minutes=30
    )


# Create the quiz
module_1_quiz = create_module_1_quiz()
