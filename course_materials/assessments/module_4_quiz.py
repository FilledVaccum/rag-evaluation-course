"""
Module 4 Quiz: Synthetic Test Data Generation
Evaluating RAG and Semantic Search Systems Course

This quiz assesses understanding of synthetic data generation, prompt engineering,
and the 3-5 example optimal pattern.

Requirements Coverage: 13.1, 13.2, 17.2
Question Count: 8 questions (within 5-10 range)
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class QuestionType(Enum):
    """Types of assessment questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    SCENARIO = "scenario"
    APPLIED = "applied"


@dataclass
class QuizQuestion:
    """Represents a quiz question with answer and explanation."""
    question_id: str
    question_text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    explanation: str
    exam_domain: str
    difficulty: str


# Module 4 Quiz Questions
MODULE_4_QUIZ = [
    QuizQuestion(
        question_id="M4Q1",
        question_text="""
What is the optimal number of examples to include in a prompt for synthetic data generation?
        """.strip(),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) 1-2 examples",
            "B) 3-5 examples",
            "C) 6-8 examples",
            "D) 10+ examples"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) 3-5 examples**

The 3-5 example pattern is the "Goldilocks zone" for few-shot learning:

- **Too few (1-2)**: Insufficient pattern for the LLM to learn from, leading to generic outputs
- **Optimal (3-5)**: Provides enough pattern recognition without overfitting
- **Too many (6+)**: LLM overfits and generates near-copies of the examples

This is backed by research and practical experience across multiple domains.

**Why other options are incorrect:**
- A) 1-2 examples: Not enough steering, produces generic queries
- C) 6-8 examples: Leads to overfitting and repetitive outputs
- D) 10+ examples: Severe overfitting, LLM just copies example patterns

**Exam Domain**: Evaluation and Tuning (13%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Intermediate"
    ),
    
    QuizQuestion(
        question_id="M4Q2",
        question_text="""
You generate 100 synthetic queries for RAG evaluation, but 70% are over-generic 
(e.g., "What courses are available?"). What is the MOST likely cause?
        """.strip(),
        question_type=QuestionType.SCENARIO,
        options=[
            "A) Using the wrong LLM model",
            "B) Insufficient or vague prompt instructions",
            "C) Dataset is too small",
            "D) Temperature setting is too low"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) Insufficient or vague prompt instructions**

Over-generic queries are the #1 symptom of poor prompt engineering:

**Root causes:**
- Vague system instructions ("Generate questions about courses")
- No user persona definition
- Missing constraints on query style and focus
- No negative examples to prevent generic outputs
- Insufficient or poorly chosen examples

**Solution:** Apply prompt engineering best practices:
1. Define specific user persona
2. Add detailed constraints (length, style, focus)
3. Include 3-5 high-quality examples
4. Add explicit negative examples
5. Use extreme specificity in instructions

**Why other options are incorrect:**
- A) Model choice matters, but prompt quality is more critical
- C) Dataset size doesn't directly cause over-generic queries
- D) Temperature affects diversity, not specificity

**Exam Domain**: Evaluation and Tuning (13%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Intermediate"
    ),
    
    QuizQuestion(
        question_id="M4Q3",
        question_text="""
Which of the following is an example of an "explicit negative example" in prompt engineering?
        """.strip(),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) 'Generate questions about courses'",
            "B) 'DO NOT generate philosophical questions about education'",
            "C) 'Use 3-5 examples for optimal steering'",
            "D) 'Questions should be 10-25 words long'"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) 'DO NOT generate philosophical questions about education'**

Explicit negative examples tell the LLM what NOT to generate. They act as guardrails
to prevent common failure modes.

**Examples of negative examples:**
- "DO NOT generate philosophical or abstract questions"
- "DO NOT generate questions about admissions or tuition"
- "DO NOT generate questions answerable by course title alone"

**Why they're important:**
Without negative examples, LLMs often generate:
- Philosophical questions ("What is the meaning of education?")
- Off-topic questions ("How do I apply to USC?")
- Overly broad questions ("Tell me about computer science")

**Why other options are incorrect:**
- A) This is a system instruction, not a negative example
- C) This is a best practice guideline, not a negative example
- D) This is a constraint, not a negative example

**Exam Domain**: Evaluation and Tuning (13%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Beginner"
    ),
    
    QuizQuestion(
        question_id="M4Q4",
        question_text="""
You want to generate a balanced test set with both fact-seeking and reasoning queries.
What synthesizer mixing strategy should you use?
        """.strip(),
        question_type=QuestionType.APPLIED,
        options=[
            "A) 100% SpecificQuerySynthesizer",
            "B) 50% SpecificQuerySynthesizer, 50% ReasoningQuerySynthesizer",
            "C) 100% AbstractQuerySynthesizer",
            "D) 33% each of Specific, Abstract, and Reasoning"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) 50% SpecificQuerySynthesizer, 50% ReasoningQuerySynthesizer**

The 50-50 mixing strategy provides balanced coverage:

**SpecificQuerySynthesizer (50%):**
- Fact-seeking, detailed questions
- Tests simple retrieval accuracy
- Example: "What is the prerequisite for CSCI 567?"

**ReasoningQuerySynthesizer (50%):**
- Multi-hop reasoning questions
- Tests complex reasoning capability
- Example: "If I want to specialize in AI but I'm weak at math, what's my path?"

**Why this balance works:**
- Real users ask both types of questions
- Tests different RAG capabilities
- Reflects realistic query distribution

**Why other options are less optimal:**
- A) Only tests simple retrieval, misses reasoning
- C) Too abstract, doesn't test specific fact lookup
- D) Could work, but 50-50 is simpler and often sufficient

**Note:** Adjust ratios based on actual user query logs if available.

**Exam Domain**: Evaluation and Tuning (13%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Intermediate"
    ),
    
    QuizQuestion(
        question_id="M4Q5",
        question_text="""
What is the primary purpose of quality validation in synthetic data generation?
        """.strip(),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) To increase the number of generated queries",
            "B) To filter out low-quality queries that don't test RAG capabilities",
            "C) To make queries more complex",
            "D) To reduce generation time"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) To filter out low-quality queries that don't test RAG capabilities**

Quality validation ensures your test set contains realistic, answerable, diverse queries
that actually test your RAG system.

**Quality validation checks:**
1. **Domain Relevance**: Does query relate to your knowledge base?
2. **Answerability**: Can your knowledge base answer this?
3. **Specificity**: Is query specific enough to be useful?
4. **Diversity**: Are queries sufficiently different?
5. **Length**: Appropriate word count (5-50 words)
6. **Keyword Filtering**: Remove banned terms (off-topic)

**Typical filtering rate:** 10-30% of generated queries

**Why filtering is essential:**
- Poor quality data → false confidence in RAG system
- Generic queries → don't test actual capabilities
- Unanswerable queries → waste compute resources

**Why other options are incorrect:**
- A) Validation reduces query count (filters out bad ones)
- C) Validation doesn't change complexity, just filters
- D) Validation adds time, doesn't reduce it

**Exam Domain**: Evaluation and Tuning (13%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Beginner"
    ),
    
    QuizQuestion(
        question_id="M4Q6",
        question_text="""
A company wants to generate synthetic queries for their legal document RAG system.
Which prompt engineering principle is MOST important for domain-specific generation?
        """.strip(),
        question_type=QuestionType.SCENARIO,
        options=[
            "A) Using as many examples as possible (10+)",
            "B) Extreme specificity with clear user persona and domain context",
            "C) Using only abstract questions",
            "D) Avoiding negative examples to maximize diversity"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) Extreme specificity with clear user persona and domain context**

Domain-specific generation requires extreme specificity:

**For legal documents, specify:**
- **User Persona**: "Corporate lawyer researching contract law" vs. "Paralegal doing case research"
- **Domain Context**: Contract law, intellectual property, litigation, etc.
- **Query Style**: Formal legal language vs. practical questions
- **Constraints**: Reference legal concepts, cite precedents, etc.

**Example prompt for legal domain:**
```
You are simulating a corporate lawyer researching contract law.
Generate questions that:
- Reference specific legal concepts (force majeure, indemnification)
- Ask about precedents and case law
- Use formal legal terminology
- Are 15-30 words long

Examples:
1. "What are the key elements of a valid force majeure clause under New York law?"
2. "How have courts interpreted indemnification provisions in M&A agreements?"
3. "What disclosure requirements apply to material adverse change clauses?"

DO NOT generate:
- General questions about "what is a contract"
- Questions about legal career advice
- Questions about court procedures
```

**Why other options are incorrect:**
- A) 10+ examples leads to overfitting
- C) Abstract questions don't test domain-specific retrieval
- D) Negative examples are essential for domain focus

**Exam Domain**: Evaluation and Tuning (13%), Agent Development (15%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Advanced"
    ),
    
    QuizQuestion(
        question_id="M4Q7",
        question_text="""
Why is test-driven development (TDD) particularly important for RAG systems compared to traditional software?
        """.strip(),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) RAG systems are deterministic and easy to test",
            "B) RAG systems are non-deterministic and require comprehensive test sets to measure performance",
            "C) RAG systems don't need testing",
            "D) Traditional software doesn't use TDD"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) RAG systems are non-deterministic and require comprehensive test sets to measure performance**

RAG systems have unique testing challenges:

**Why TDD is critical for RAG:**

1. **Non-Deterministic Behavior**:
   - Same query can produce different responses
   - LLM outputs vary with temperature, context
   - Need statistical evaluation over many queries

2. **Context-Dependent**:
   - Performance depends on knowledge base
   - Retrieval quality affects generation
   - Need diverse test queries to cover edge cases

3. **No Ground Truth**:
   - Unlike traditional software (2+2=4)
   - RAG responses are subjective
   - Need test sets with expected behaviors

4. **Regression Detection**:
   - Changes to embeddings, chunking, prompts affect output
   - Without test sets, can't measure impact
   - Need baseline metrics to detect regressions

**TDD Workflow for RAG:**
1. Generate comprehensive test set (100+ queries)
2. Measure baseline performance
3. Make changes (new embeddings, chunking strategy)
4. Re-evaluate on same test set
5. Compare metrics to detect improvements/regressions

**Why other options are incorrect:**
- A) RAG is non-deterministic, not deterministic
- C) RAG absolutely needs testing
- D) Traditional software does use TDD

**Exam Domain**: Evaluation and Tuning (13%), Agent Development (15%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Intermediate"
    ),
    
    QuizQuestion(
        question_id="M4Q8",
        question_text="""
You generate 100 synthetic queries and find that 15 are near-duplicates. What is the recommended approach?
        """.strip(),
        question_type=QuestionType.APPLIED,
        options=[
            "A) Keep all queries to maximize test set size",
            "B) Remove near-duplicates using similarity threshold (e.g., 90%)",
            "C) Regenerate all 100 queries from scratch",
            "D) Manually rewrite each duplicate"
        ],
        correct_answer="B",
        explanation="""
**Correct Answer: B) Remove near-duplicates using similarity threshold (e.g., 90%)**

Duplicate removal is a standard quality validation step:

**Why remove duplicates:**
- Duplicates don't add test coverage
- Waste compute resources during evaluation
- Skew metrics (same query tested multiple times)
- Reduce test set diversity

**How to remove duplicates:**
1. Calculate similarity between all query pairs
2. Set threshold (typically 85-95%)
3. Remove queries above threshold
4. Keep most diverse set

**Similarity methods:**
- Word-based: Jaccard similarity on word sets
- Semantic: Cosine similarity on embeddings
- Edit distance: Levenshtein distance

**Example:**
```python
validator = QualityValidator(similarity_threshold=0.9)
unique_queries = validator.filter_duplicates(all_queries)
# 100 queries → 85 unique queries (15% duplicates removed)
```

**Why other options are less optimal:**
- A) Duplicates waste resources and skew metrics
- C) Too extreme, most queries are probably good
- D) Manual rewriting doesn't scale and is time-consuming

**Typical duplicate rate:** 5-15% with good prompts

**Exam Domain**: Evaluation and Tuning (13%)
        """.strip(),
        exam_domain="Evaluation and Tuning",
        difficulty="Intermediate"
    )
]


def get_quiz() -> List[QuizQuestion]:
    """Return all quiz questions for Module 4."""
    return MODULE_4_QUIZ


def print_quiz_summary():
    """Print summary of quiz structure."""
    print("Module 4 Quiz Summary")
    print("=" * 60)
    print(f"Total Questions: {len(MODULE_4_QUIZ)}")
    
    # Count by type
    type_counts = {}
    for q in MODULE_4_QUIZ:
        type_counts[q.question_type.value] = type_counts.get(q.question_type.value, 0) + 1
    
    print("\nQuestion Types:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count}")
    
    # Count by difficulty
    difficulty_counts = {}
    for q in MODULE_4_QUIZ:
        difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
    
    print("\nDifficulty Distribution:")
    for diff, count in difficulty_counts.items():
        print(f"  {diff}: {count}")
    
    print(f"\nExam Domain: {MODULE_4_QUIZ[0].exam_domain}")
    print("\n✅ All questions include detailed explanations")


if __name__ == "__main__":
    print_quiz_summary()
    
    print("\n" + "=" * 60)
    print("Sample Question:")
    print("=" * 60)
    
    sample = MODULE_4_QUIZ[0]
    print(f"\n{sample.question_text}\n")
    for option in sample.options:
        print(option)
    print(f"\nCorrect Answer: {sample.correct_answer}")
    print(f"\nExplanation:\n{sample.explanation}")
