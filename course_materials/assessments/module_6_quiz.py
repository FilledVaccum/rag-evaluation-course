"""
Module 6 Quiz: Semantic Search System Evaluation
Evaluating RAG and Semantic Search Systems Course

This quiz tests understanding of:
- Legacy BM25 system evaluation with modern techniques
- Applying Ragas to non-RAG search systems
- Hybrid evaluation strategies
- Ranking algorithm assessment

Requirements: 13.1
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class QuestionType(Enum):
    """Types of quiz questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SCENARIO = "scenario"


@dataclass
class QuizQuestion:
    """Quiz question with answer and explanation."""
    question_id: str
    question_text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    exam_domain: str


# Module 6 Quiz Questions

QUIZ_QUESTIONS = [
    QuizQuestion(
        question_id="m6_q1",
        question_text=(
            "What is the primary advantage of BM25 over semantic search for certain use cases?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Better semantic understanding",
            "B) Exact keyword matching for technical terms",
            "C) Lower computational cost",
            "D) Both B and C"
        ],
        correct_answer="D",
        explanation=(
            "BM25 excels at exact keyword matching, which is crucial for technical terms, "
            "product codes, and legal documents. It's also computationally cheaper than "
            "semantic search since it doesn't require embeddings or neural networks. "
            "However, BM25 lacks semantic understanding (option A is incorrect)."
        ),
        difficulty="beginner",
        exam_domain="Evaluation and Tuning (13%)"
    ),
    
    QuizQuestion(
        question_id="m6_q2",
        question_text=(
            "When adapting Ragas for legacy BM25 systems, which metrics should you focus on?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Faithfulness and Answer Relevancy",
            "B) Context Precision, Context Recall, and Context Relevance",
            "C) All Ragas metrics equally",
            "D) Only Context Precision"
        ],
        correct_answer="B",
        explanation=(
            "When evaluating search-only systems (no generation), focus on retrieval metrics: "
            "Context Precision (ranking quality), Context Recall (coverage), and Context Relevance "
            "(semantic match). Generation metrics like Faithfulness and Answer Relevancy (option A) "
            "only apply when an LLM generates responses. Option D is too narrow - you need multiple "
            "metrics for comprehensive evaluation."
        ),
        difficulty="intermediate",
        exam_domain="Evaluation and Tuning (13%)"
    ),
    
    QuizQuestion(
        question_id="m6_q3",
        question_text=(
            "What does Reciprocal Rank Fusion (RRF) do in hybrid search systems?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Replaces BM25 with vector search",
            "B) Combines rankings from multiple search systems",
            "C) Improves embedding quality",
            "D) Generates synthetic queries"
        ],
        correct_answer="B",
        explanation=(
            "Reciprocal Rank Fusion (RRF) is a method for combining rankings from multiple "
            "search systems (e.g., BM25 + vector search). It uses the formula: "
            "score(doc) = Σ 1/(k + rank(doc)) to merge results. This allows hybrid systems "
            "to leverage strengths of both keyword and semantic search. Options A, C, and D "
            "describe different techniques not related to result fusion."
        ),
        difficulty="intermediate",
        exam_domain="Knowledge Integration and Data Handling (10%)"
    ),
    
    QuizQuestion(
        question_id="m6_q4",
        question_text=(
            "A search system has high precision (0.85) but low recall (0.55). What does this indicate?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) The system retrieves too many irrelevant documents",
            "B) The system is too selective and misses relevant documents",
            "C) The system has perfect performance",
            "D) The ranking algorithm needs improvement"
        ],
        correct_answer="B",
        explanation=(
            "High precision with low recall means the system is too selective - the documents "
            "it retrieves are highly relevant (high precision), but it's missing many other "
            "relevant documents (low recall). This typically happens when similarity thresholds "
            "are too strict or k (number of results) is too small. Option A describes low precision. "
            "Option D might help but doesn't directly address the recall issue."
        ),
        difficulty="intermediate",
        exam_domain="Evaluation and Tuning (13%)"
    ),
    
    QuizQuestion(
        question_id="m6_q5",
        question_text=(
            "When should you choose traditional search over RAG?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) When you need synthesized, natural language responses",
            "B) When you need to return raw documents without modification",
            "C) When semantic understanding is critical",
            "D) Never - RAG is always better"
        ],
        correct_answer="B",
        explanation=(
            "Traditional search is appropriate when you need to return raw documents without "
            "synthesis or modification. Use cases include document libraries, legal databases, "
            "and research repositories where users want to see original sources. RAG (option A) "
            "is better for question-answering where synthesis is needed. Option D is incorrect - "
            "each approach has appropriate use cases. Option C favors semantic search but doesn't "
            "necessarily require RAG."
        ),
        difficulty="beginner",
        exam_domain="Agent Architecture and Design (15%)"
    ),
    
    QuizQuestion(
        question_id="m6_q6",
        question_text=(
            "What is the main limitation of using LLM-as-a-Judge for search evaluation at scale?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) It's less accurate than traditional metrics",
            "B) It can be expensive and slow for large datasets",
            "C) It cannot evaluate semantic relevance",
            "D) It only works for RAG systems"
        ],
        correct_answer="B",
        explanation=(
            "LLM-as-a-Judge provides excellent semantic evaluation but can be expensive and slow "
            "when evaluating thousands of queries. Each evaluation requires an LLM API call, which "
            "adds cost and latency. Solutions include sampling, caching, or using cheaper models "
            "for initial filtering. Option A is incorrect - LLM-as-a-Judge is often more accurate "
            "for semantic relevance. Options C and D are false - it works for any search system."
        ),
        difficulty="intermediate",
        exam_domain="Evaluation and Tuning (13%)"
    ),
    
    QuizQuestion(
        question_id="m6_q7",
        question_text=(
            "In a hybrid search system, BM25 returns 'machine learning' documents highly ranked, "
            "while vector search ranks 'AI that learns from data' documents higher. What does this suggest?"
        ),
        question_type=QuestionType.SCENARIO,
        options=[
            "A) BM25 is broken and should be removed",
            "B) Vector search is broken and should be removed",
            "C) Both systems are working correctly - they capture different aspects",
            "D) The fusion algorithm needs adjustment"
        ],
        correct_answer="C",
        explanation=(
            "This is expected behavior! BM25 excels at exact keyword matching ('machine learning'), "
            "while vector search captures semantic relationships ('AI that learns from data' is "
            "semantically similar to 'machine learning'). Both are valuable - BM25 for exact matches, "
            "vector search for conceptual queries. A good fusion algorithm (option D) would combine "
            "these strengths, but neither system is 'broken' (options A and B are incorrect)."
        ),
        difficulty="advanced",
        exam_domain="Knowledge Integration and Data Handling (10%)"
    ),
    
    QuizQuestion(
        question_id="m6_q8",
        question_text=(
            "What is the primary benefit of evaluating legacy systems with modern LLM-based techniques?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) It makes legacy systems faster",
            "B) It provides semantic relevance assessment beyond keyword matching",
            "C) It eliminates the need for ground truth data",
            "D) It automatically fixes ranking issues"
        ],
        correct_answer="B",
        explanation=(
            "Modern LLM-based evaluation (LLM-as-a-Judge) can assess semantic relevance even for "
            "keyword-based systems like BM25. This reveals cases where BM25 misses semantically "
            "relevant documents due to vocabulary mismatch. Option A is incorrect - evaluation "
            "doesn't change system speed. Option C is false - ground truth is still valuable. "
            "Option D is incorrect - evaluation identifies issues but doesn't automatically fix them."
        ),
        difficulty="intermediate",
        exam_domain="Evaluation and Tuning (13%)"
    ),
    
    QuizQuestion(
        question_id="m6_q9",
        question_text=(
            "A company has a 10-year-old Elasticsearch (BM25) system with millions of documents. "
            "What's the best approach to improve it with modern techniques?"
        ),
        question_type=QuestionType.SCENARIO,
        options=[
            "A) Replace it entirely with a RAG system immediately",
            "B) Evaluate current performance, identify gaps, then augment with semantic search",
            "C) Keep it unchanged - old systems can't be improved",
            "D) Add more keywords to documents"
        ],
        correct_answer="B",
        explanation=(
            "The best approach is systematic: (1) Evaluate current BM25 performance with modern "
            "metrics, (2) Identify specific query types where it fails, (3) Augment with semantic "
            "search for those cases, (4) Build a hybrid system. Option A is risky and expensive - "
            "BM25 may work well for many queries. Option C is defeatist. Option D doesn't address "
            "semantic understanding issues. Gradual, data-driven improvement is the professional approach."
        ),
        difficulty="advanced",
        exam_domain="Agent Architecture and Design (15%)"
    ),
    
    QuizQuestion(
        question_id="m6_q10",
        question_text=(
            "What metric would best identify if a search system is retrieving irrelevant documents?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Context Recall",
            "B) Context Precision",
            "C) Context Relevance",
            "D) Faithfulness"
        ],
        correct_answer="C",
        explanation=(
            "Context Relevance measures what percentage of retrieved documents are actually relevant "
            "to the query. Low context relevance indicates many irrelevant documents are being retrieved. "
            "Context Precision (option B) measures ranking quality, not relevance. Context Recall "
            "(option A) measures coverage of relevant documents. Faithfulness (option D) only applies "
            "to generated responses, not search results."
        ),
        difficulty="intermediate",
        exam_domain="Evaluation and Tuning (13%)"
    )
]


def get_quiz() -> List[QuizQuestion]:
    """
    Get all quiz questions for Module 6.
    
    Returns:
        List of QuizQuestion objects
    """
    return QUIZ_QUESTIONS


def calculate_score(answers: dict) -> tuple:
    """
    Calculate quiz score.
    
    Args:
        answers: Dictionary mapping question_id to selected answer
    
    Returns:
        Tuple of (score, total, percentage)
    """
    correct = 0
    total = len(QUIZ_QUESTIONS)
    
    for question in QUIZ_QUESTIONS:
        if answers.get(question.question_id) == question.correct_answer:
            correct += 1
    
    percentage = (correct / total) * 100 if total > 0 else 0
    return correct, total, percentage


def display_quiz():
    """Display quiz questions with formatting."""
    print("=" * 80)
    print("MODULE 6 QUIZ: SEMANTIC SEARCH SYSTEM EVALUATION")
    print("=" * 80)
    print(f"\nTotal Questions: {len(QUIZ_QUESTIONS)}")
    print("Mix of conceptual and applied questions")
    print("\n" + "=" * 80)
    
    for i, question in enumerate(QUIZ_QUESTIONS, 1):
        print(f"\nQuestion {i} [{question.difficulty.upper()}]")
        print(f"Domain: {question.exam_domain}")
        print(f"\n{question.question_text}")
        print()
        for option in question.options:
            print(f"  {option}")
        print(f"\nCorrect Answer: {question.correct_answer}")
        print(f"\nExplanation: {question.explanation}")
        print("\n" + "-" * 80)


if __name__ == "__main__":
    display_quiz()
