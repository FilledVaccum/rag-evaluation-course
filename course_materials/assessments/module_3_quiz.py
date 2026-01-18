"""
Module 3 Quiz: RAG Architecture and Component Analysis

This quiz tests understanding of RAG pipeline architecture, component-level
debugging, and failure diagnosis techniques.

Requirements Coverage: 13.1, 13.3, 17.2
"""

from src.models.assessment import (
    Assessment, AssessmentType, Question, QuestionType,
    Difficulty, EvaluationRubric
)


# Quiz questions for Module 3
MODULE_3_QUESTIONS = [
    Question(
        question_id="m3_q1",
        question_text=(
            "What are the three stages of a RAG pipeline in order?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Generation → Retrieval → Augmentation",
            "B) Retrieval → Augmentation → Generation",
            "C) Augmentation → Retrieval → Generation",
            "D) Retrieval → Generation → Augmentation"
        ],
        correct_answer="B",
        explanation=(
            "The correct order is Retrieval → Augmentation → Generation. "
            "First, the system retrieves relevant documents from the vector store (Retrieval). "
            "Then, it formats the retrieved context with the query into a prompt (Augmentation). "
            "Finally, it sends the augmented prompt to the LLM to generate a response (Generation). "
            "This order is essential because you need to retrieve information before you can "
            "format it, and you need the formatted prompt before you can generate a response."
        ),
        points=1,
        exam_domain="Agent Architecture and Design",
        difficulty=Difficulty.BEGINNER
    ),
    
    Question(
        question_id="m3_q2",
        question_text=(
            "A RAG system is returning incorrect answers. You inspect the retrieved "
            "documents and find they contain the correct information, but the LLM's "
            "response doesn't use this information. What type of failure is this?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Retrieval failure",
            "B) Generation failure",
            "C) Augmentation failure",
            "D) Vector store failure"
        ],
        correct_answer="B",
        explanation=(
            "This is a Generation failure. The key diagnostic clue is that the retrieved "
            "documents contain the correct information, which means retrieval succeeded. "
            "However, the LLM is not using this information in its response, indicating "
            "a problem in the generation stage. Common causes include: poor prompt engineering, "
            "LLM ignoring context, or the LLM over-relying on its parametric knowledge instead "
            "of the provided context. The fix would involve improving prompts, using a better "
            "LLM, or adding explicit instructions to use only the provided context."
        ),
        points=1,
        exam_domain="Evaluation and Tuning",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m3_q3",
        question_text=(
            "According to the '80/20 rule' of RAG debugging, what percentage of RAG "
            "failures are typically retrieval problems rather than generation problems?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) 50%",
            "B) 60%",
            "C) 70%",
            "D) 80%"
        ],
        correct_answer="D",
        explanation=(
            "The correct answer is 80%. The '80/20 rule' states that approximately 80% of "
            "RAG failures are retrieval problems, not generation problems. This is "
            "counterintuitive because when you see a wrong answer, your first instinct is "
            "to blame the LLM. However, in most cases, the LLM never had access to the "
            "right information in the first place. This is why component-level debugging "
            "is so important - you need to verify retrieval quality before assuming the "
            "LLM is at fault. Always debug retrieval first!"
        ),
        points=1,
        exam_domain="Evaluation and Tuning",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m3_q4",
        question_text=(
            "What is Context Precision in RAG evaluation?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) The percentage of retrieved documents that are relevant",
            "B) The ranking quality of retrieved contexts (relevant docs at top)",
            "C) The coverage of ground truth information in retrieved contexts",
            "D) The semantic similarity between query and retrieved documents"
        ],
        correct_answer="B",
        explanation=(
            "Context Precision measures the ranking quality of retrieved contexts. "
            "It evaluates whether relevant documents appear at the top of the results. "
            "The formula is: precision@k = (relevant_docs_in_top_k) / k. "
            "Option A describes Context Relevance (percentage of relevant docs). "
            "Option C describes Context Recall (coverage of ground truth). "
            "Option D describes a proxy metric using embeddings. "
            "High precision means your ranking algorithm is working well and putting "
            "the most relevant information first, which is critical for RAG systems "
            "that may only use the top few results."
        ),
        points=1,
        exam_domain="Evaluation and Tuning",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m3_q5",
        question_text=(
            "What is Faithfulness in RAG evaluation?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Whether the response is relevant to the query",
            "B) Whether claims in the response are supported by the retrieved context",
            "C) Whether the retrieved context is relevant to the query",
            "D) Whether the response is grammatically correct"
        ],
        correct_answer="B",
        explanation=(
            "Faithfulness measures whether the claims in the response are supported by "
            "the retrieved context. It prevents hallucinations and ensures responses are "
            "grounded in actual retrieved information, not the LLM's parametric knowledge. "
            "The evaluation process involves: (1) extracting claims from the response, "
            "(2) verifying each claim against the retrieved context, and (3) calculating "
            "faithfulness = verified_claims / total_claims. "
            "Option A describes Answer Relevancy. Option C describes Context Relevance. "
            "Option D is not a standard RAG metric. Faithfulness is critical for "
            "trustworthy RAG systems, especially in domains like healthcare or legal "
            "where accuracy is paramount."
        ),
        points=1,
        exam_domain="Evaluation and Tuning",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m3_q6",
        question_text=(
            "You're debugging a RAG system that returns 'I don't know' even when the "
            "answer is clearly in the retrieved context. What is the most likely cause?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) The embedding model is not working correctly",
            "B) The vector store is not configured properly",
            "C) The LLM is ignoring the provided context",
            "D) The chunk size is too small"
        ],
        correct_answer="C",
        explanation=(
            "The most likely cause is that the LLM is ignoring the provided context. "
            "The key diagnostic clue is that the answer IS in the retrieved context, "
            "which means retrieval succeeded. The problem is in generation - the LLM "
            "is not using the information it was given. This is a common generation "
            "failure pattern called 'context ignorance'. "
            "Possible fixes include: (1) improving prompt templates to emphasize using "
            "the context, (2) adding explicit instructions like 'Answer using ONLY the "
            "provided context', (3) using a better LLM that follows instructions more "
            "reliably, or (4) adjusting the prompt format to make the context more "
            "prominent. Options A, B, and D would cause retrieval failures, not this "
            "specific symptom."
        ),
        points=1,
        exam_domain="Evaluation and Tuning",
        difficulty=Difficulty.ADVANCED
    ),
    
    Question(
        question_id="m3_q7",
        question_text=(
            "In a RAG pipeline, what is the purpose of the Augmentation stage?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) To search the vector store for relevant documents",
            "B) To combine the query with retrieved context into a formatted prompt",
            "C) To generate the final response using an LLM",
            "D) To embed the query into a vector representation"
        ],
        correct_answer="B",
        explanation=(
            "The Augmentation stage combines the user query with retrieved context into "
            "a formatted prompt for the LLM. This stage is responsible for: (1) formatting "
            "retrieved contexts in a way the LLM can understand, (2) combining the query "
            "with context, (3) applying prompt templates and instructions, and (4) preparing "
            "the final input for generation. "
            "Option A describes the Retrieval stage. Option C describes the Generation stage. "
            "Option D is part of the Retrieval stage (query embedding). "
            "The Augmentation stage is often overlooked but is critical - poor formatting "
            "can cause information loss or make it hard for the LLM to use the context "
            "effectively. This is why augmentation failures can occur even when retrieval "
            "and generation components are working correctly."
        ),
        points=1,
        exam_domain="Agent Architecture and Design",
        difficulty=Difficulty.INTERMEDIATE
    ),
    
    Question(
        question_id="m3_q8",
        question_text=(
            "What is the correct systematic workflow for debugging a RAG system that "
            "returns incorrect answers?"
        ),
        question_type=QuestionType.MULTIPLE_CHOICE,
        options=[
            "A) Debug generation first, then retrieval if needed",
            "B) Debug retrieval first, then generation if retrieval is correct",
            "C) Debug both simultaneously to save time",
            "D) Replace the LLM with a better model immediately"
        ],
        correct_answer="B",
        explanation=(
            "The correct workflow is to debug retrieval first, then generation if retrieval "
            "is correct. This systematic approach is based on the 80/20 rule - most failures "
            "are retrieval problems. The complete workflow is: "
            "(1) Run the query end-to-end, (2) If wrong, inspect retrieved documents, "
            "(3) Check if answer is in retrieved context, (4) If NOT in context → retrieval "
            "failure, fix embeddings/chunking/search, (5) If IN context → generation failure, "
            "fix prompts/LLM. "
            "Option A wastes time debugging generation when retrieval might be the issue. "
            "Option C is inefficient and makes it hard to isolate the problem. "
            "Option D is premature optimization - you need to diagnose first. "
            "Component-level debugging prevents wasted effort on the wrong optimization."
        ),
        points=1,
        exam_domain="Evaluation and Tuning",
        difficulty=Difficulty.ADVANCED
    ),
]


# Create the quiz assessment
MODULE_3_QUIZ = Assessment(
    assessment_id="module_3_quiz",
    assessment_type=AssessmentType.QUIZ,
    module_number=3,
    title="Module 3 Quiz: RAG Architecture and Component Analysis",
    description=(
        "Test your understanding of RAG pipeline architecture, component-level "
        "debugging, context relevance assessment, and response accuracy evaluation."
    ),
    questions=MODULE_3_QUESTIONS,
    rubric=EvaluationRubric(
        rubric_id="module_3_quiz_rubric",
        criteria={
            "rag_architecture": {
                "points": 2,
                "description": "Understanding of RAG pipeline stages and architecture"
            },
            "component_debugging": {
                "points": 3,
                "description": "Ability to diagnose retrieval vs generation failures"
            },
            "evaluation_metrics": {
                "points": 3,
                "description": "Knowledge of context relevance and faithfulness metrics"
            }
        },
        total_points=8,
        passing_score=6
    ),
    time_limit_minutes=20
)


def get_module_3_quiz() -> Assessment:
    """
    Returns the Module 3 quiz assessment.
    
    Returns:
        Assessment object with all quiz questions
    """
    return MODULE_3_QUIZ


if __name__ == "__main__":
    quiz = get_module_3_quiz()
    print(f"Module 3 Quiz: {quiz.title}")
    print(f"Questions: {len(quiz.questions)}")
    print(f"Time Limit: {quiz.time_limit_minutes} minutes")
    print(f"Passing Score: {quiz.rubric.passing_score}/{quiz.rubric.total_points}")
    print("\nQuestions:")
    for i, q in enumerate(quiz.questions, 1):
        print(f"  {i}. {q.question_text[:60]}...")
