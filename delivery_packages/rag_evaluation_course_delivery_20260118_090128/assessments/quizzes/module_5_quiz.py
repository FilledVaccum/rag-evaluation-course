"""
Module 5 Quiz: RAG Evaluation Metrics and Frameworks

This quiz assesses understanding of:
- LLM-as-a-Judge methodology
- Ragas framework architecture
- Generation and retrieval metrics
- Metric interpretation and optimization

Requirements: 13.1, 17.3
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class QuestionType(Enum):
    """Type of quiz question"""
    MULTIPLE_CHOICE = "multiple_choice"
    SCENARIO = "scenario"
    CONCEPTUAL = "conceptual"
    APPLIED = "applied"


@dataclass
class QuizQuestion:
    """Quiz question with answer and explanation"""
    question_id: str
    question_text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    explanation: str
    exam_domain: str = "Evaluation and Tuning (13%)"
    difficulty: str = "intermediate"


# Module 5 Quiz Questions
MODULE_5_QUIZ = [
    QuizQuestion(
        question_id="m5_q1",
        question_text=(
            "Why do traditional NLP metrics like BLEU and ROUGE fail for RAG evaluation?"
        ),
        question_type=QuestionType.CONCEPTUAL,
        options=[
            "A) They are too computationally expensive",
            "B) They only measure n-gram overlap and miss semantic equivalence",
            "C) They require ground truth answers which are hard to obtain",
            "D) They cannot handle multiple languages"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Traditional metrics like BLEU and ROUGE measure n-gram overlap between "
            "generated text and reference text. This fails for RAG because:\n"
            "- They miss semantic equivalence (different words, same meaning)\n"
            "- They can't assess faithfulness (whether claims are supported)\n"
            "- They don't evaluate relevance or context usage\n"
            "- They penalize valid paraphrasing\n\n"
            "Example: 'The capital is Paris' vs 'Paris is the capital' have different "
            "n-grams but identical meaning. BLEU would score this poorly, but semantically "
            "they're equivalent.\n\n"
            "Why other options are wrong:\n"
            "A) Computational cost is not the issue - these metrics are actually fast\n"
            "C) Ground truth is needed for many metrics, not unique to BLEU/ROUGE\n"
            "D) Language support is not the primary limitation"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q2",
        question_text=(
            "What is the primary advantage of LLM-as-a-Judge methodology?"
        ),
        question_type=QuestionType.CONCEPTUAL,
        options=[
            "A) It's cheaper than human evaluation",
            "B) It provides semantic understanding beyond exact word matching",
            "C) It never makes mistakes in evaluation",
            "D) It doesn't require any prompt engineering"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "The primary advantage of LLM-as-a-Judge is semantic understanding. "
            "LLMs can:\n"
            "- Understand paraphrasing and semantic equivalence\n"
            "- Assess complex criteria like faithfulness and relevance\n"
            "- Evaluate nuanced aspects like tone and appropriateness\n"
            "- Provide reasoning for their judgments\n\n"
            "Why other options are wrong:\n"
            "A) While often cheaper than humans, cost is not the PRIMARY advantage\n"
            "C) LLMs can make mistakes - they have limitations like bias and hallucination\n"
            "D) LLM-as-a-Judge requires careful prompt engineering for good results"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q3",
        question_text=(
            "Which limitation of LLM-as-a-Judge is most critical to address in production?"
        ),
        question_type=QuestionType.APPLIED,
        options=[
            "A) Position bias (favoring certain answer positions)",
            "B) Hallucination during evaluation",
            "C) Computational cost of API calls",
            "D) Difficulty in prompt engineering"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Hallucination during evaluation is the most critical limitation because:\n"
            "- The judge LLM might fabricate facts when evaluating\n"
            "- This leads to incorrect evaluation scores\n"
            "- It undermines trust in the evaluation system\n"
            "- It can cause wrong optimization decisions\n\n"
            "Mitigation strategies:\n"
            "- Use multiple judge LLMs and ensemble results\n"
            "- Validate judge outputs against ground truth samples\n"
            "- Use structured output formats (JSON) to reduce hallucination\n"
            "- Periodically audit judge decisions with human review\n\n"
            "Why other options are less critical:\n"
            "A) Position bias can be mitigated by randomizing answer order\n"
            "C) Cost is manageable with batching and caching\n"
            "D) Prompt engineering is a one-time effort, not ongoing risk"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q4",
        question_text=(
            "What does the Faithfulness metric measure in RAG evaluation?"
        ),
        question_type=QuestionType.CONCEPTUAL,
        options=[
            "A) Whether the response is relevant to the question",
            "B) Whether claims in the response are supported by retrieved context",
            "C) Whether the retrieved context is relevant to the question",
            "D) Whether the response matches the ground truth answer"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Faithfulness measures whether claims in the generated response are "
            "supported by the retrieved context. This is THE most critical metric "
            "for RAG systems because it prevents hallucination.\n\n"
            "How it works:\n"
            "1. Extract individual claims from the response\n"
            "2. Verify each claim against the retrieved context\n"
            "3. Score = (verified_claims / total_claims)\n\n"
            "Example:\n"
            "Context: 'Paris is the capital of France.'\n"
            "Response: 'Paris is the capital of France with 50 million people.'\n"
            "Claims: [Paris is capital ✓, Population 50M ✗]\n"
            "Faithfulness: 1/2 = 0.5\n\n"
            "Why other options are wrong:\n"
            "A) That's Answer Relevancy, not Faithfulness\n"
            "C) That's Context Relevance, not Faithfulness\n"
            "D) That's Answer Correctness, not Faithfulness"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q5",
        question_text=(
            "A RAG system has Faithfulness=0.95 but Answer Relevancy=0.55. "
            "What is the most likely diagnosis?"
        ),
        question_type=QuestionType.SCENARIO,
        options=[
            "A) The LLM is hallucinating information",
            "B) The retrieval system is getting irrelevant documents",
            "C) The LLM is being too conservative or retrieval is poor",
            "D) The evaluation metrics are misconfigured"
        ],
        correct_answer="C",
        explanation=(
            "Correct Answer: C\n\n"
            "High Faithfulness + Low Relevancy pattern indicates:\n"
            "- LLM is accurately using context (high faithfulness)\n"
            "- But responses don't address the question well (low relevancy)\n\n"
            "Root causes:\n"
            "1. LLM is being too conservative (only repeating context)\n"
            "2. Retrieved context is not relevant to the question\n"
            "3. LLM is not synthesizing information effectively\n\n"
            "Diagnostic steps:\n"
            "1. Check Context Relevance metric - is retrieval getting good docs?\n"
            "2. If retrieval is good, adjust generation prompt to be more direct\n"
            "3. If retrieval is poor, fix retrieval before generation\n\n"
            "Why other options are wrong:\n"
            "A) High faithfulness means NO hallucination\n"
            "B) This would cause low faithfulness too\n"
            "D) The pattern is consistent and meaningful, not a configuration error"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q6",
        question_text=(
            "What does Context Recall measure in RAG evaluation?"
        ),
        question_type=QuestionType.CONCEPTUAL,
        options=[
            "A) Percentage of retrieved documents that are relevant",
            "B) Whether retrieved context contains all necessary information from ground truth",
            "C) How well the LLM remembers previous context",
            "D) Ranking quality of retrieved documents"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Context Recall measures coverage - whether the retrieved documents "
            "contain ALL the information needed to answer the question (compared "
            "to ground truth).\n\n"
            "Formula: recall = (ground_truth_facts_in_context / total_ground_truth_facts)\n\n"
            "Example:\n"
            "Question: 'What are flu symptoms?'\n"
            "Ground Truth: 'Fever, cough, fatigue, body aches'\n"
            "Retrieved Context: 'Flu symptoms include fever and cough.'\n"
            "Recall: 2/4 = 0.5 (missing fatigue and body aches)\n\n"
            "Low recall means:\n"
            "- Missing information in retrieval\n"
            "- Incomplete answers likely\n"
            "- Need to increase k or improve retrieval\n\n"
            "Why other options are wrong:\n"
            "A) That's Context Relevance (filtering), not Recall (coverage)\n"
            "C) This is not about LLM memory, it's about retrieval completeness\n"
            "D) That's Context Precision, not Recall"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q7",
        question_text=(
            "A system has Context Precision=0.45 and Context Recall=0.85. "
            "What optimization should you prioritize?"
        ),
        question_type=QuestionType.SCENARIO,
        options=[
            "A) Increase k (number of retrieved documents)",
            "B) Add a re-ranking stage to filter irrelevant documents",
            "C) Improve the embedding model",
            "D) Adjust the generation prompt"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "Low Precision + High Recall pattern means:\n"
            "- Retrieval is getting all necessary information (high recall)\n"
            "- But also getting too much irrelevant content (low precision)\n\n"
            "This is a 'too broad' retrieval problem.\n\n"
            "Best solution: Add re-ranking stage\n"
            "- Re-ranker filters out irrelevant documents\n"
            "- Maintains high recall (keeps relevant docs)\n"
            "- Improves precision (removes irrelevant docs)\n"
            "- Cross-encoder models work well for re-ranking\n\n"
            "Why other options are wrong:\n"
            "A) Increasing k would make precision WORSE (more irrelevant docs)\n"
            "C) Better embeddings help, but re-ranking is more direct solution\n"
            "D) Generation prompt won't fix retrieval issues - fix retrieval first!"
        )
    ),
    
    QuizQuestion(
        question_id="m5_q8",
        question_text=(
            "When customizing a Ragas metric for a specific domain, what is the "
            "most important element to modify?"
        ),
        question_type=QuestionType.APPLIED,
        options=[
            "A) The metric name",
            "B) The prompt template with domain-specific criteria",
            "C) The scoring range (e.g., 0-10 instead of 0-1)",
            "D) The LLM model used for evaluation"
        ],
        correct_answer="B",
        explanation=(
            "Correct Answer: B\n\n"
            "The prompt template is the most important element to customize because:\n"
            "- It defines the evaluation criteria\n"
            "- It provides domain-specific context to the judge LLM\n"
            "- It determines what aspects are evaluated\n"
            "- It directly impacts evaluation quality\n\n"
            "Example customization for medical domain:\n"
            "Standard: 'Evaluate if response is supported by context'\n"
            "Medical: 'Evaluate if medical response is supported by clinical context. "
            "Consider medical terminology accuracy, clinical guidelines compliance, "
            "and evidence-based recommendations.'\n\n"
            "Best practices:\n"
            "- Add domain-specific terminology\n"
            "- Include relevant evaluation criteria\n"
            "- Provide examples of good/bad responses\n"
            "- Use clear scoring rubrics\n\n"
            "Why other options are less important:\n"
            "A) Name is just a label, doesn't affect evaluation\n"
            "C) Scoring range is convention, 0-1 is standard and works well\n"
            "D) Model matters, but prompt engineering has bigger impact"
        )
    ),
]


def get_quiz() -> List[QuizQuestion]:
    """Get all quiz questions for Module 5"""
    return MODULE_5_QUIZ


def print_quiz():
    """Print quiz in readable format"""
    print("="*80)
    print("MODULE 5 QUIZ: RAG EVALUATION METRICS AND FRAMEWORKS")
    print("="*80)
    print(f"\nTotal Questions: {len(MODULE_5_QUIZ)}")
    print("Exam Domain: Evaluation and Tuning (13%)")
    print("\n" + "="*80 + "\n")
    
    for i, q in enumerate(MODULE_5_QUIZ, 1):
        print(f"Question {i} ({q.question_type.value}):")
        print(f"{q.question_text}\n")
        
        for option in q.options:
            print(f"  {option}")
        
        print(f"\n{'='*80}\n")


def print_answer_key():
    """Print answer key with explanations"""
    print("="*80)
    print("MODULE 5 QUIZ - ANSWER KEY")
    print("="*80 + "\n")
    
    for i, q in enumerate(MODULE_5_QUIZ, 1):
        print(f"Question {i}: {q.correct_answer}")
        print(f"\n{q.explanation}")
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--answers":
        print_answer_key()
    else:
        print_quiz()
        print("\nTo see answers and explanations, run:")
        print("  python module_5_quiz.py --answers")
