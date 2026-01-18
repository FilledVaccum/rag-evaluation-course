"""
Module 7 Quiz: Production Deployment and Advanced Topics

This quiz tests understanding of production deployment considerations,
including temporal data handling, regulatory compliance, continuous evaluation,
performance profiling, A/B testing, monitoring, and multi-language challenges.

Total Questions: 10
Passing Score: 70%
Time Limit: 20 minutes
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QuizQuestion:
    """Quiz question with multiple choice options."""
    question_id: str
    question_text: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str
    exam_domain: str


# Module 7 Quiz Questions
MODULE_7_QUIZ = [
    QuizQuestion(
        question_id="m7_q1",
        question_text=(
            "You're deploying a RAG system for a news aggregation service. "
            "Recent articles should be prioritized over older ones. "
            "Which temporal data handling strategy is most appropriate?"
        ),
        options=[
            "A) Sliding window approach - only consider articles from last 30 days",
            "B) Exponential decay scoring - apply time-based decay to similarity scores",
            "C) Version-aware retrieval - track article versions",
            "D) No temporal handling - treat all articles equally"
        ],
        correct_answer="B",
        explanation=(
            "Exponential decay scoring is ideal for news aggregation because it:\n"
            "- Gradually reduces relevance of older articles\n"
            "- Doesn't completely exclude older articles (unlike sliding window)\n"
            "- Allows tuning decay rate (λ) based on content type\n"
            "- Balances semantic similarity with temporal relevance\n\n"
            "Formula: score_final = score_semantic * exp(-λ * age_days)\n\n"
            "Sliding window (A) would completely exclude older articles, which may still be relevant. "
            "Version-aware retrieval (C) is for document versioning, not time-based prioritization. "
            "No temporal handling (D) would treat 5-year-old news the same as today's news."
        ),
        difficulty="intermediate",
        exam_domain="Deployment and Scaling"
    ),
    
    QuizQuestion(
        question_id="m7_q2",
        question_text=(
            "Your healthcare RAG system must comply with HIPAA. "
            "Which of the following is NOT a required HIPAA compliance measure?"
        ),
        options=[
            "A) Encrypt Protected Health Information (PHI) at rest and in transit",
            "B) Implement audit trails logging all access to PHI",
            "C) Require user consent before processing any data",
            "D) Establish Business Associate Agreements with third-party services"
        ],
        correct_answer="C",
        explanation=(
            "User consent (C) is a GDPR requirement, not specifically a HIPAA requirement.\n\n"
            "HIPAA focuses on:\n"
            "- Security: Encryption, access controls (A)\n"
            "- Accountability: Audit trails, logging (B)\n"
            "- Business relationships: BAAs with vendors (D)\n"
            "- Privacy: De-identification, minimum necessary principle\n\n"
            "While consent may be required for some healthcare uses, HIPAA primarily focuses on "
            "protecting PHI through technical and administrative safeguards rather than consent management. "
            "GDPR, on the other hand, makes consent a central requirement for data processing."
        ),
        difficulty="intermediate",
        exam_domain="Safety, Ethics, and Compliance"
    ),
    
    QuizQuestion(
        question_id="m7_q3",
        question_text=(
            "You're implementing continuous evaluation for a production RAG system. "
            "Which metric would be LEAST useful for detecting system degradation?"
        ),
        options=[
            "A) Faithfulness score (from sampled queries)",
            "B) P95 query latency",
            "C) Total number of unique users",
            "D) Error rate (failed queries / total queries)"
        ],
        correct_answer="C",
        explanation=(
            "Total number of unique users (C) is a business metric, not a system health metric.\n\n"
            "For detecting system degradation, you need metrics that reflect:\n"
            "- Quality: Faithfulness, relevancy (A) - detects quality issues\n"
            "- Performance: Latency percentiles (B) - detects slowdowns\n"
            "- Reliability: Error rates (D) - detects failures\n\n"
            "User count may increase or decrease for business reasons unrelated to system health. "
            "A degraded system might still have many users (they just get poor results). "
            "Conversely, a healthy system might have fewer users due to seasonality or marketing changes.\n\n"
            "The key is to monitor metrics that directly reflect system behavior, not business outcomes."
        ),
        difficulty="intermediate",
        exam_domain="Run, Monitor, and Maintain"
    ),
    
    QuizQuestion(
        question_id="m7_q4",
        question_text=(
            "You're profiling a RAG pipeline and find:\n"
            "- Embedding: 100ms (7%)\n"
            "- Retrieval: 50ms (4%)\n"
            "- Augmentation: 10ms (1%)\n"
            "- Generation: 1200ms (88%)\n\n"
            "Which optimization would have the GREATEST impact on latency?"
        ),
        options=[
            "A) Use a smaller embedding model to reduce embedding time by 50%",
            "B) Implement response caching to avoid generation for common queries",
            "C) Optimize vector search to reduce retrieval time by 50%",
            "D) Pre-format context templates to reduce augmentation time by 50%"
        ],
        correct_answer="B",
        explanation=(
            "Response caching (B) has the greatest impact because:\n"
            "- Generation is 88% of total latency (1200ms)\n"
            "- Caching can eliminate generation entirely for cached queries\n"
            "- Even 20% cache hit rate saves 240ms average (20% * 1200ms)\n\n"
            "Other optimizations have limited impact:\n"
            "- (A) 50% embedding reduction = 50ms savings (only 4% improvement)\n"
            "- (C) 50% retrieval reduction = 25ms savings (only 2% improvement)\n"
            "- (D) 50% augmentation reduction = 5ms savings (only 0.4% improvement)\n\n"
            "This demonstrates the importance of profiling: optimize the bottleneck first. "
            "Amdahl's Law: speedup is limited by the fraction of time spent in the optimized component."
        ),
        difficulty="advanced",
        exam_domain="Deployment and Scaling"
    ),
    
    QuizQuestion(
        question_id="m7_q5",
        question_text=(
            "You're running an A/B test comparing two embedding models. "
            "After 1000 queries per variant, you find:\n"
            "- Variant A: context_precision = 0.82\n"
            "- Variant B: context_precision = 0.84\n"
            "- P-value = 0.12\n\n"
            "What should you conclude?"
        ),
        options=[
            "A) Variant B is better; promote it to production immediately",
            "B) The difference is not statistically significant; continue testing or keep variant A",
            "C) Variant A is better because it has lower variance",
            "D) The test is invalid because sample size is too small"
        ],
        correct_answer="B",
        explanation=(
            "The difference is not statistically significant (B) because p-value = 0.12 > 0.05.\n\n"
            "Statistical significance interpretation:\n"
            "- P-value < 0.05: Significant difference (reject null hypothesis)\n"
            "- P-value ≥ 0.05: Not significant (fail to reject null hypothesis)\n\n"
            "With p = 0.12, there's a 12% chance the observed difference is due to random variation. "
            "This is too high to confidently conclude variant B is better.\n\n"
            "Options:\n"
            "1. Continue testing to increase sample size and statistical power\n"
            "2. Keep variant A (baseline) since improvement isn't proven\n"
            "3. Consider practical significance: is 2.4% improvement worth the risk?\n\n"
            "(A) is wrong: don't promote without statistical significance\n"
            "(C) is wrong: we're not comparing variance\n"
            "(D) is wrong: 1000 samples per variant is typically sufficient"
        ),
        difficulty="advanced",
        exam_domain="Evaluation and Tuning"
    ),
    
    QuizQuestion(
        question_id="m7_q6",
        question_text=(
            "Your production RAG system costs $0.005 per query. "
            "You want to reduce costs to $0.003 per query while maintaining quality above 0.80. "
            "Which strategy is LEAST likely to achieve this?"
        ),
        options=[
            "A) Implement response caching for common queries",
            "B) Use query routing to send simple queries to smaller models",
            "C) Increase the number of retrieved documents (top_k) from 5 to 10",
            "D) Reduce context window size by filtering less relevant chunks"
        ],
        correct_answer="C",
        explanation=(
            "Increasing top_k from 5 to 10 (C) would INCREASE costs, not reduce them:\n"
            "- More documents = more tokens in context\n"
            "- More tokens = higher LLM generation cost\n"
            "- Typical cost increase: 40-50%\n\n"
            "Cost-reducing strategies:\n"
            "- (A) Caching: Eliminates generation cost for cached queries (20-40% savings)\n"
            "- (B) Query routing: Uses cheaper models for simple queries (15-30% savings)\n"
            "- (D) Smaller context: Fewer tokens = lower cost (10-20% savings)\n\n"
            "Cost optimization principles:\n"
            "1. Reduce LLM token usage (biggest cost driver)\n"
            "2. Cache when possible\n"
            "3. Use smaller models when appropriate\n"
            "4. Batch requests\n"
            "5. Monitor quality to ensure optimizations don't degrade performance"
        ),
        difficulty="intermediate",
        exam_domain="Deployment and Scaling"
    ),
    
    QuizQuestion(
        question_id="m7_q7",
        question_text=(
            "You're deploying a multi-language RAG system supporting English, Spanish, and Arabic. "
            "Which approach is most appropriate for handling Arabic queries?"
        ),
        options=[
            "A) Use the same English embedding model for all languages",
            "B) Translate Arabic queries to English, process, then translate back",
            "C) Use a multilingual embedding model or Arabic-specific model",
            "D) Reject Arabic queries and only support English and Spanish"
        ],
        correct_answer="C",
        explanation=(
            "Use multilingual or Arabic-specific embedding models (C) because:\n"
            "- Arabic has different script (right-to-left, non-Latin)\n"
            "- English models perform poorly on Arabic text\n"
            "- Multilingual models (e.g., LaBSE, multilingual-e5) support 100+ languages\n"
            "- Arabic-specific models (e.g., arabic-bert) optimize for Arabic nuances\n\n"
            "Why other options fail:\n"
            "(A) English models: Poor performance on non-Latin scripts\n"
            "(B) Translation: Loses nuance, adds latency and cost, translation errors compound\n"
            "(D) Rejecting queries: Unacceptable for multi-language requirements\n\n"
            "Best practices for multi-language RAG:\n"
            "1. Detect language early\n"
            "2. Route to language-appropriate pipeline\n"
            "3. Use multilingual models for cross-lingual retrieval\n"
            "4. Evaluate per-language performance separately\n"
            "5. Consider cultural context, not just language"
        ),
        difficulty="intermediate",
        exam_domain="Knowledge Integration and Data Handling"
    ),
    
    QuizQuestion(
        question_id="m7_q8",
        question_text=(
            "Your monitoring dashboard shows:\n"
            "- P95 latency: 2500ms (threshold: 2000ms)\n"
            "- Error rate: 0.03 (threshold: 0.05)\n"
            "- Faithfulness: 0.68 (threshold: 0.70)\n\n"
            "Which alert should be prioritized FIRST?"
        ),
        options=[
            "A) P95 latency exceeding threshold",
            "B) Error rate approaching threshold",
            "C) Faithfulness below threshold",
            "D) All alerts have equal priority"
        ],
        correct_answer="C",
        explanation=(
            "Faithfulness below threshold (C) should be prioritized because:\n"
            "- Quality issues directly impact user trust and safety\n"
            "- Low faithfulness means hallucinations (incorrect information)\n"
            "- In domains like healthcare or finance, this is critical\n"
            "- Users may not notice quality issues immediately (unlike latency)\n\n"
            "Alert priority framework:\n"
            "1. Quality/Safety: Faithfulness, accuracy (highest priority)\n"
            "2. Reliability: Error rates, availability\n"
            "3. Performance: Latency, throughput\n"
            "4. Cost: Budget overruns\n\n"
            "Reasoning:\n"
            "- (A) Latency: Users notice, but system still works\n"
            "- (B) Error rate: Still below threshold (3% < 5%)\n"
            "- (C) Faithfulness: Below threshold, users getting wrong information\n\n"
            "In production, a slow correct answer is better than a fast incorrect answer."
        ),
        difficulty="advanced",
        exam_domain="Run, Monitor, and Maintain"
    ),
    
    QuizQuestion(
        question_id="m7_q9",
        question_text=(
            "You're implementing GDPR 'right to be forgotten' for your RAG system. "
            "Which component requires the MOST complex implementation?"
        ),
        options=[
            "A) Deleting user data from the source database",
            "B) Removing user data from vector store embeddings",
            "C) Clearing user data from application logs",
            "D) Updating user consent records"
        ],
        correct_answer="B",
        explanation=(
            "Removing data from vector store embeddings (B) is most complex because:\n\n"
            "Challenges:\n"
            "1. Embeddings are high-dimensional vectors, not searchable by content\n"
            "2. Need to maintain metadata linking embeddings to source data\n"
            "3. Vector stores may not support efficient deletion\n"
            "4. May need to re-index entire collection\n"
            "5. Embeddings may be cached or replicated\n\n"
            "Implementation approaches:\n"
            "- Maintain document_id → embedding_id mapping\n"
            "- Implement soft deletion with filtering\n"
            "- Periodic re-indexing to remove deleted documents\n"
            "- Use vector stores with deletion support (e.g., Pinecone, Milvus)\n\n"
            "Other components are simpler:\n"
            "(A) Database deletion: Standard SQL DELETE\n"
            "(C) Log clearing: Log rotation or filtering\n"
            "(D) Consent updates: Database update\n\n"
            "This is why GDPR compliance requires careful architecture planning from the start."
        ),
        difficulty="advanced",
        exam_domain="Safety, Ethics, and Compliance"
    ),
    
    QuizQuestion(
        question_id="m7_q10",
        question_text=(
            "You're designing a feedback loop for continuous improvement. "
            "Which implicit feedback signal is MOST reliable for detecting poor responses?"
        ),
        options=[
            "A) User spent 30+ seconds reading the response",
            "B) User asked a follow-up question within 10 seconds",
            "C) User copied the response to clipboard",
            "D) User closed the application after receiving response"
        ],
        correct_answer="B",
        explanation=(
            "Quick follow-up questions (B) are the most reliable negative signal because:\n"
            "- Indicates user didn't get satisfactory answer\n"
            "- User needs clarification or different information\n"
            "- Strong correlation with dissatisfaction\n"
            "- Actionable: can analyze what was missing\n\n"
            "Other signals are ambiguous:\n"
            "(A) Long reading time: Could indicate interest OR confusion\n"
            "(C) Copying response: Usually positive (user found it useful)\n"
            "(D) Closing app: Could be satisfaction OR frustration\n\n"
            "Feedback signal reliability:\n"
            "1. Explicit feedback (thumbs up/down): Most reliable but low volume\n"
            "2. Quick follow-ups: Reliable negative signal\n"
            "3. Session abandonment: Moderate reliability\n"
            "4. Time on page: Low reliability (ambiguous)\n\n"
            "Best practice: Combine multiple signals with explicit feedback for training data."
        ),
        difficulty="intermediate",
        exam_domain="Run, Monitor, and Maintain"
    )
]


def calculate_score(answers: Dict[str, str]) -> Dict[str, Any]:
    """
    Calculate quiz score based on user answers.
    
    Args:
        answers: Dictionary mapping question_id to selected answer (A, B, C, or D)
        
    Returns:
        Dictionary with score, percentage, pass/fail, and feedback
    """
    correct = 0
    total = len(MODULE_7_QUIZ)
    results = []
    
    for question in MODULE_7_QUIZ:
        user_answer = answers.get(question.question_id, "")
        is_correct = user_answer == question.correct_answer
        
        if is_correct:
            correct += 1
        
        results.append({
            "question_id": question.question_id,
            "correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": question.correct_answer,
            "explanation": question.explanation
        })
    
    percentage = (correct / total) * 100
    passed = percentage >= 70
    
    return {
        "score": correct,
        "total": total,
        "percentage": percentage,
        "passed": passed,
        "results": results
    }


def print_quiz():
    """Print quiz questions for review."""
    print("=" * 80)
    print("MODULE 7 QUIZ: PRODUCTION DEPLOYMENT AND ADVANCED TOPICS")
    print("=" * 80)
    print(f"\nTotal Questions: {len(MODULE_7_QUIZ)}")
    print("Passing Score: 70%")
    print("Time Limit: 20 minutes")
    print("\n" + "=" * 80)
    
    for i, question in enumerate(MODULE_7_QUIZ, 1):
        print(f"\nQuestion {i} [{question.difficulty.upper()}]")
        print(f"Domain: {question.exam_domain}")
        print("-" * 80)
        print(question.question_text)
        print()
        for option in question.options:
            print(f"  {option}")
        print(f"\nCorrect Answer: {question.correct_answer}")
        print(f"\nExplanation:\n{question.explanation}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    print_quiz()
