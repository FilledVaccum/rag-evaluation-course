"""
Prompt Engineering Guidelines for RAG Systems

Comprehensive guidelines for effective prompt engineering in synthetic data
generation and LLM-as-a-Judge evaluation contexts.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class PromptType(Enum):
    """Types of prompts in RAG systems"""
    SYNTHETIC_DATA_GENERATION = "synthetic_data_generation"
    LLM_AS_JUDGE = "llm_as_judge"
    RAG_GENERATION = "rag_generation"
    QUERY_REWRITING = "query_rewriting"


@dataclass
class PromptGuideline:
    """Represents a prompt engineering guideline"""
    principle: str
    description: str
    rationale: str
    good_examples: List[str]
    bad_examples: List[str]
    implementation_tips: List[str]


# Guideline 1: Extreme Specificity Principle
GUIDELINE_EXTREME_SPECIFICITY = PromptGuideline(
    principle="Extreme Specificity Principle",
    description="""
    Write prompts with extreme specificity, as if explaining to a child who
    has no context or background knowledge. Define every term, specify every
    constraint, and leave nothing to interpretation.
    """,
    rationale="""
    LLMs are powerful but literal. They will interpret vague instructions in
    unpredictable ways. What seems "obvious" to humans is not obvious to LLMs.
    Extreme specificity eliminates ambiguity and ensures consistent behavior.
    """,
    good_examples=[
        """
# GOOD: Extremely specific
Generate questions that an undergraduate computer science student at a US university
would ask when searching for courses to register for the upcoming semester.

Requirements:
- Question length: 5-15 words (count every word)
- Language: Casual student English (e.g., "prereqs" not "prerequisites")
- Focus: Course logistics (schedule, prerequisites, units, instructor)
- Format: One complete question ending with "?"
- Avoid: Philosophical questions, curriculum design, education theory

Examples of target questions:
1. "What are the prereqs for CSCI 570?"
2. "Does CSCI 544 meet on Fridays?"
3. "How many units is CSCI 585?"
        """,
        """
# GOOD: Specific evaluation criteria
Evaluate whether the response is faithful to the provided context.

Definition of Faithfulness:
- Every factual claim in the response must be directly supported by the context
- "Directly supported" means the context explicitly states the information
- Do not consider general knowledge or reasoning
- Paraphrasing is acceptable if meaning is preserved

Scoring:
- 1.0: All claims supported by context
- 0.5: Some claims supported, some not
- 0.0: No claims supported by context

Output format: {"score": <float>, "explanation": "<string>"}
        """
    ],
    bad_examples=[
        """
# BAD: Vague and ambiguous
Generate realistic questions about courses.

Problems:
- "Realistic" is undefined (realistic to whom? in what context?)
- "About courses" is too broad (what aspects? what level of detail?)
- No format specification
- No length constraints
- No examples provided
        """,
        """
# BAD: Assumes context
Evaluate if the answer is good.

Problems:
- "Good" is subjective and undefined
- No evaluation criteria specified
- No scoring scale provided
- No output format specified
- Assumes evaluator knows what "good" means
        """
    ],
    implementation_tips=[
        "Define every term explicitly, even if it seems obvious",
        "Specify exact formats (length, structure, style)",
        "Include what NOT to do, not just what to do",
        "Provide concrete examples for every requirement",
        "Test prompts with someone unfamiliar with the domain",
        "If you can ask 'what does X mean?', add more specificity"
    ]
)


# Guideline 2: 3-5 Example Optimal Pattern
GUIDELINE_3_5_EXAMPLES = PromptGuideline(
    principle="3-5 Example Optimal Pattern",
    description="""
    Provide exactly 3-5 examples in your prompts. Fewer than 3 provides
    insufficient steering; more than 5 leads to overfitting and diminishing
    returns. This is the empirically optimal range for few-shot learning.
    """,
    rationale="""
    Research shows that 3-5 examples provide the optimal balance:
    - 1-2 examples: LLM doesn't recognize the pattern reliably
    - 3-5 examples: LLM learns the pattern without overfitting
    - 6+ examples: Diminishing returns, risk of overfitting to examples
    
    This applies to both synthetic data generation and evaluation prompts.
    """,
    good_examples=[
        """
# GOOD: Exactly 4 examples showing the pattern
Generate student questions about course prerequisites.

Examples of desired questions:
1. "What are the prereqs for CSCI 570?"
2. "Do I need CSCI 270 before taking CSCI 570?"
3. "Can I take CSCI 585 without CSCI 480?"
4. "What courses do I need before CSCI 544?"

Generate 10 more questions following this pattern.
        """,
        """
# GOOD: 5 examples with variation
Generate questions about course scheduling.

Examples showing desired style and variation:
1. "What time does CSCI 570 meet?" (time query)
2. "Does CSCI 544 have Friday classes?" (day query)
3. "Is CSCI 585 offered in the evening?" (time-of-day query)
4. "When is the final exam for CSCI 567?" (exam schedule)
5. "What days does CSCI 401 meet?" (days query)

Generate 15 more questions with similar variety.
        """
    ],
    bad_examples=[
        """
# BAD: Only 1 example (insufficient)
Generate questions about courses.

Example:
1. "What are the prerequisites for CSCI 570?"

Generate 20 more questions.

Problem: LLM doesn't have enough examples to understand the pattern
        """,
        """
# BAD: 10 examples (overfitting risk)
Generate questions about courses.

Examples:
1. "What are the prereqs for CSCI 570?"
2. "What are the prereqs for CSCI 544?"
3. "What are the prereqs for CSCI 585?"
4. "What are the prereqs for CSCI 567?"
5. "What are the prereqs for CSCI 401?"
6. "What are the prereqs for CSCI 480?"
7. "What are the prereqs for CSCI 530?"
8. "What are the prereqs for CSCI 550?"
9. "What are the prereqs for CSCI 560?"
10. "What are the prereqs for CSCI 580?"

Problem: Too many examples, all too similar, LLM will just copy the pattern
        """
    ],
    implementation_tips=[
        "Start with 3 examples, add up to 5 if needed for clarity",
        "Show variation within the pattern (different phrasings, structures)",
        "Don't make all examples too similar (avoid template copying)",
        "If you need more than 5, you probably need better instructions",
        "Test with 3 examples first, add more only if results are inconsistent",
        "For complex tasks, use 5 examples; for simple tasks, 3 is sufficient"
    ]
)


# Guideline 3: Explicit Negative Examples
GUIDELINE_NEGATIVE_EXAMPLES = PromptGuideline(
    principle="Explicit Negative Examples",
    description="""
    Include explicit examples of what NOT to generate or do. Negative examples
    clarify boundaries and prevent common failure modes that positive examples
    alone cannot address.
    """,
    rationale="""
    Positive examples show what to do, but don't clearly define boundaries.
    LLMs may generate outputs that technically match positive examples but
    violate implicit constraints. Negative examples make boundaries explicit.
    """,
    good_examples=[
        """
# GOOD: Both positive and negative examples
Generate practical student questions about courses.

GOOD examples (generate questions like these):
1. "What are the prereqs for CSCI 570?"
2. "Does CSCI 544 meet on Fridays?"
3. "How many units is CSCI 585?"

BAD examples (do NOT generate questions like these):
1. "What is the epistemological foundation of computer science?" (too philosophical)
2. "How does the curriculum reflect contemporary pedagogical theories?" (too academic)
3. "What are all the courses offered by the department?" (too broad)
4. "Why is education important for society?" (off-topic)

Generate 10 questions following the GOOD pattern, avoiding the BAD patterns.
        """,
        """
# GOOD: Negative examples for evaluation
Evaluate answer relevancy: Does the response address the question?

RELEVANT examples (score 1.0):
Q: "What are the prereqs for CSCI 570?"
A: "CSCI 570 requires CSCI 270 and CSCI 350 as prerequisites."

NOT RELEVANT examples (score 0.0):
Q: "What are the prereqs for CSCI 570?"
A: "CSCI 570 is a great course about algorithms." (doesn't answer question)
A: "Prerequisites are important for course planning." (generic, not specific)
A: "The computer science department offers many courses." (off-topic)

Evaluate the following response...
        """
    ],
    bad_examples=[
        """
# BAD: Only positive examples
Generate student questions about courses.

Examples:
1. "What are the prereqs for CSCI 570?"
2. "Does CSCI 544 meet on Fridays?"
3. "How many units is CSCI 585?"

Generate 10 more questions.

Problem: LLM doesn't know what to avoid, may generate philosophical or
off-topic questions that technically match the pattern
        """,
        """
# BAD: Vague negative instruction
Generate student questions. Don't make them too complicated.

Problem: "Too complicated" is subjective and undefined. What counts as
"too complicated"? Explicit negative examples would clarify.
        """
    ],
    implementation_tips=[
        "Include 2-3 negative examples for every 3-5 positive examples",
        "Choose negative examples that represent common failure modes",
        "Explain WHY each negative example is bad (in parentheses)",
        "Make negative examples clearly distinct from positive ones",
        "Test your prompt and add negative examples for any unwanted outputs",
        "Update negative examples as you discover new failure modes"
    ]
)


# Guideline 4: User Persona Specification
GUIDELINE_USER_PERSONA = PromptGuideline(
    principle="User Persona Specification",
    description="""
    Specify detailed user personas with demographics, role, context, and
    realistic scenarios. Use specific personas like "undergraduate computer
    science student" instead of generic terms like "user".
    """,
    rationale="""
    Generic terms like "user" or "person" don't provide enough context for
    LLMs to generate realistic, domain-appropriate content. Specific personas
    ground the generation in realistic scenarios and language patterns.
    """,
    good_examples=[
        """
# GOOD: Detailed persona
Generate questions that a 20-year-old undergraduate computer science student
at a US university would ask when planning their course schedule for next
semester. The student is:
- In their junior year (3rd year)
- Has completed introductory CS courses (CSCI 101, 102, 201)
- Planning to specialize in machine learning
- Concerned about workload balance (wants mix of hard and easier courses)
- Uses casual student language (e.g., "prereqs" not "prerequisites")

Context: The student is browsing the course catalog on their laptop during
registration period, trying to decide which courses to take.
        """,
        """
# GOOD: Professional persona for different context
Generate questions that a 35-year-old healthcare administrator with 10 years
of experience would ask when evaluating a new patient record system. The
administrator:
- Manages a team of 15 staff members
- Responsible for HIPAA compliance
- Has basic technical knowledge but not a developer
- Prioritizes patient safety and regulatory compliance
- Uses professional but non-technical language

Context: Evaluating vendor proposals for new EHR system during procurement.
        """
    ],
    bad_examples=[
        """
# BAD: Generic persona
Generate questions that a user would ask about courses.

Problems:
- "User" is too generic (student? professor? administrator?)
- No demographic details
- No context or scenario
- No language style specified
        """,
        """
# BAD: Vague persona
Generate questions that a student might ask.

Problems:
- What kind of student? (undergrad? grad? high school?)
- What field? (CS? biology? business?)
- What context? (registration? research? career planning?)
- What language style? (formal? casual? technical?)
        """
    ],
    implementation_tips=[
        "Include age, role, experience level, and context",
        "Specify language style (formal, casual, technical)",
        "Describe motivations and concerns",
        "Provide realistic scenario context",
        "Use specific titles, not generic terms",
        "Test with multiple personas to ensure appropriate variation"
    ]
)


# Guideline 5: LLM-as-a-Judge Scoring Rubrics
GUIDELINE_SCORING_RUBRICS = PromptGuideline(
    principle="Explicit Scoring Rubrics with 0-1 Scale",
    description="""
    For LLM-as-a-Judge evaluation, define explicit scoring rubrics using
    0-1 scale with clear criteria for each score level. Request structured
    JSON output for consistency.
    """,
    rationale="""
    Without explicit rubrics, LLM evaluators are inconsistent and subjective.
    0-1 scale is standard for metrics and allows easy aggregation. JSON output
    ensures parseable, structured results.
    """,
    good_examples=[
        """
# GOOD: Explicit rubric with clear criteria
Evaluate the faithfulness of the response to the provided context.

Definition: Faithfulness measures whether all factual claims in the response
are directly supported by the context.

Scoring Rubric:
- 1.0: All factual claims are directly supported by the context
  * Every statement can be traced to specific context sentences
  * No unsupported claims or hallucinations
  * Paraphrasing is acceptable if meaning preserved

- 0.7: Most claims supported, minor unsupported details
  * Core facts are supported
  * 1-2 minor details not in context
  * No major hallucinations

- 0.3: Some claims supported, significant unsupported content
  * Some facts from context
  * Multiple unsupported claims
  * May include hallucinations

- 0.0: No claims supported or major hallucinations
  * Response contradicts context
  * Most content not from context
  * Significant fabricated information

Output format:
{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation of score>",
  "unsupported_claims": ["<claim 1>", "<claim 2>", ...]
}
        """,
        """
# GOOD: Binary rubric with clear decision criteria
Evaluate whether the retrieved context is relevant to the question.

Definition: Context relevance measures whether the context contains information
that could help answer the question.

Scoring Rubric:
- 1.0: Context is relevant
  * Context contains information directly related to the question
  * Information in context could be used to answer the question
  * Context is on-topic

- 0.0: Context is not relevant
  * Context does not contain information related to the question
  * Context is off-topic or unrelated
  * Context could not help answer the question

Output format:
{
  "score": <0.0 or 1.0>,
  "explanation": "<brief explanation>",
  "relevant_sentences": ["<sentence 1>", "<sentence 2>", ...]
}
        """
    ],
    bad_examples=[
        """
# BAD: Vague rubric
Evaluate how good the response is.

Score from 0 to 1, where 1 is best.

Problems:
- "Good" is undefined
- No criteria for different score levels
- No output format specified
- Subjective and inconsistent
        """,
        """
# BAD: Arbitrary scale
Rate the response on a scale of 1-10.

Problems:
- 1-10 scale is not standard (use 0-1)
- No criteria for each level
- Too many levels (hard to distinguish 6 from 7)
- No output format
        """
    ],
    implementation_tips=[
        "Always use 0-1 scale for consistency with standard metrics",
        "Define 3-4 score levels with explicit criteria",
        "Include examples for each score level (calibration)",
        "Request JSON output for structured parsing",
        "Test rubric on sample data to ensure consistency",
        "Refine rubric based on evaluation variance"
    ]
)


# Guideline 6: Calibration Examples for Score Levels
GUIDELINE_CALIBRATION_EXAMPLES = PromptGuideline(
    principle="Calibration Examples for Each Score Level",
    description="""
    Provide concrete examples for each score level in your rubric to calibrate
    the LLM evaluator. This ensures consistent interpretation of score levels
    across different evaluations.
    """,
    rationale="""
    Score levels like "high", "medium", "low" are subjective without examples.
    Calibration examples anchor the LLM's understanding of what each score
    level means in practice, reducing variance.
    """,
    good_examples=[
        """
# GOOD: Calibration examples for each level
Evaluate answer relevancy: Does the response address the question?

Scoring Rubric with Calibration Examples:

Score 1.0 - Highly Relevant:
Example:
Q: "What are the prerequisites for CSCI 570?"
A: "CSCI 570 requires CSCI 270 and CSCI 350 as prerequisites."
Explanation: Directly answers the question with specific information.

Score 0.7 - Mostly Relevant:
Example:
Q: "What are the prerequisites for CSCI 570?"
A: "CSCI 570 is an algorithms course that requires some prior coursework in data structures."
Explanation: Addresses prerequisites but lacks specific course numbers.

Score 0.3 - Partially Relevant:
Example:
Q: "What are the prerequisites for CSCI 570?"
A: "CSCI 570 is a challenging course that covers algorithm analysis."
Explanation: Mentions the course but doesn't address prerequisites.

Score 0.0 - Not Relevant:
Example:
Q: "What are the prerequisites for CSCI 570?"
A: "The computer science department offers many great courses."
Explanation: Doesn't address the question at all.

Now evaluate the following response...
        """,
        """
# GOOD: Binary calibration with edge cases
Evaluate faithfulness: Are claims supported by context?

Score 1.0 - Faithful (Calibration Examples):
Context: "CSCI 570 requires CSCI 270 and CSCI 350."
Response: "You need CSCI 270 and CSCI 350 before taking CSCI 570."
Explanation: Paraphrased but all claims supported.

Context: "The course meets on Mondays and Wednesdays at 2:00 PM."
Response: "The course is on Monday and Wednesday afternoons."
Explanation: "Afternoons" is reasonable interpretation of 2:00 PM.

Score 0.0 - Not Faithful (Calibration Examples):
Context: "CSCI 570 requires CSCI 270 and CSCI 350."
Response: "CSCI 570 requires CSCI 270, CSCI 350, and CSCI 401."
Explanation: Added CSCI 401 which is not in context (hallucination).

Context: "The course meets on Mondays and Wednesdays."
Response: "The course meets on Tuesdays and Thursdays."
Explanation: Contradicts context.

Now evaluate the following response...
        """
    ],
    bad_examples=[
        """
# BAD: No calibration examples
Evaluate answer relevancy on a scale of 0 to 1.

1.0 = highly relevant
0.5 = somewhat relevant
0.0 = not relevant

Evaluate the following response...

Problem: No examples showing what each score level looks like in practice
        """,
        """
# BAD: Only one example
Evaluate faithfulness.

Example of faithful response:
Q: "What are the prereqs?"
A: "CSCI 270 and CSCI 350."

Evaluate the following...

Problem: Only shows one score level, no calibration for other levels
        """
    ],
    implementation_tips=[
        "Provide 1-2 examples for each score level",
        "Include edge cases that are hard to score",
        "Show examples with paraphrasing (acceptable) vs hallucination (not acceptable)",
        "Use examples from your actual domain",
        "Test calibration by having multiple evaluators score the same examples",
        "Update calibration examples based on evaluation inconsistencies"
    ]
)


# Guideline 7: Multi-Stage Evaluation Breakdown
GUIDELINE_MULTI_STAGE_EVALUATION = PromptGuideline(
    principle="Multi-Stage Evaluation Breakdown",
    description="""
    Break complex evaluations into multiple stages, where each stage performs
    a simpler sub-task. This improves accuracy and provides better debugging
    information than single-stage evaluation.
    """,
    rationale="""
    Complex evaluations (e.g., faithfulness) are hard for LLMs to perform
    accurately in one step. Breaking into stages (extract claims → verify
    claims → compute score) improves accuracy and provides interpretable
    intermediate results.
    """,
    good_examples=[
        """
# GOOD: Multi-stage faithfulness evaluation
Stage 1: Extract Claims
Task: Extract all factual claims from the response.
Input: Response text
Output: List of claims as JSON array

Example:
Response: "CSCI 570 requires CSCI 270 and CSCI 350. It meets on Mondays."
Output: {
  "claims": [
    "CSCI 570 requires CSCI 270",
    "CSCI 570 requires CSCI 350",
    "CSCI 570 meets on Mondays"
  ]
}

Stage 2: Verify Each Claim
Task: For each claim, determine if it is supported by the context.
Input: Claim + Context
Output: Boolean for each claim

Example:
Claim: "CSCI 570 requires CSCI 270"
Context: "CSCI 570 prerequisites: CSCI 270, CSCI 350"
Output: {"supported": true}

Stage 3: Compute Faithfulness Score
Task: Calculate the proportion of supported claims.
Formula: faithfulness = (supported_claims / total_claims)

Example:
Total claims: 3
Supported claims: 2
Faithfulness: 2/3 = 0.67
        """,
        """
# GOOD: Multi-stage context relevance evaluation
Stage 1: Identify Key Information Needs
Task: What information is needed to answer the question?
Input: Question
Output: List of information needs

Example:
Question: "What are the prerequisites for CSCI 570?"
Output: {
  "information_needs": [
    "Course code: CSCI 570",
    "Type of information: prerequisites",
    "Expected format: list of course codes"
  ]
}

Stage 2: Check Context for Each Need
Task: Does the context contain each piece of needed information?
Input: Information need + Context
Output: Boolean for each need

Example:
Need: "Course code: CSCI 570"
Context: "CSCI 570 is an algorithms course..."
Output: {"found": true}

Stage 3: Compute Relevance Score
Task: Calculate proportion of information needs met.
Formula: relevance = (needs_met / total_needs)

Example:
Total needs: 3
Needs met: 3
Relevance: 3/3 = 1.0
        """
    ],
    bad_examples=[
        """
# BAD: Single-stage complex evaluation
Evaluate faithfulness: Are all claims in the response supported by the context?

Input: Question, Context, Response
Output: Score from 0 to 1

Evaluate the following...

Problems:
- Too complex for single step
- No intermediate results for debugging
- Hard to understand why a particular score was given
- Less accurate than multi-stage approach
        """,
        """
# BAD: Vague multi-stage without clear tasks
Step 1: Analyze the response
Step 2: Check the context
Step 3: Give a score

Problems:
- Stages are vague and undefined
- No clear task for each stage
- No output format specified
- Not actually breaking down the complexity
        """
    ],
    implementation_tips=[
        "Identify natural sub-tasks in your evaluation",
        "Each stage should have a clear, simple task",
        "Specify input and output format for each stage",
        "Provide examples for each stage",
        "Test each stage independently",
        "Use intermediate results for debugging",
        "Typical stages: extract → verify → aggregate"
    ]
)


# Compile all guidelines
ALL_GUIDELINES = [
    GUIDELINE_EXTREME_SPECIFICITY,
    GUIDELINE_3_5_EXAMPLES,
    GUIDELINE_NEGATIVE_EXAMPLES,
    GUIDELINE_USER_PERSONA,
    GUIDELINE_SCORING_RUBRICS,
    GUIDELINE_CALIBRATION_EXAMPLES,
    GUIDELINE_MULTI_STAGE_EVALUATION
]


def get_guidelines_by_type(prompt_type: PromptType) -> List[PromptGuideline]:
    """Get relevant guidelines for a specific prompt type"""
    if prompt_type == PromptType.SYNTHETIC_DATA_GENERATION:
        return [
            GUIDELINE_EXTREME_SPECIFICITY,
            GUIDELINE_3_5_EXAMPLES,
            GUIDELINE_NEGATIVE_EXAMPLES,
            GUIDELINE_USER_PERSONA
        ]
    elif prompt_type == PromptType.LLM_AS_JUDGE:
        return [
            GUIDELINE_EXTREME_SPECIFICITY,
            GUIDELINE_SCORING_RUBRICS,
            GUIDELINE_CALIBRATION_EXAMPLES,
            GUIDELINE_MULTI_STAGE_EVALUATION
        ]
    else:
        return ALL_GUIDELINES


def generate_prompt_checklist() -> str:
    """Generate a checklist for prompt engineering"""
    checklist = "# Prompt Engineering Checklist\n\n"
    checklist += "Before deploying a prompt, verify:\n\n"
    
    checklist += "## Specificity\n"
    checklist += "- [ ] Every term is explicitly defined\n"
    checklist += "- [ ] Format and length are specified\n"
    checklist += "- [ ] Constraints are clear and unambiguous\n"
    checklist += "- [ ] No assumptions about context\n\n"
    
    checklist += "## Examples\n"
    checklist += "- [ ] 3-5 positive examples provided\n"
    checklist += "- [ ] Examples show variation within pattern\n"
    checklist += "- [ ] 2-3 negative examples provided\n"
    checklist += "- [ ] Negative examples cover common failure modes\n\n"
    
    checklist += "## Persona (for generation tasks)\n"
    checklist += "- [ ] User persona is specific (not 'user')\n"
    checklist += "- [ ] Demographics and role specified\n"
    checklist += "- [ ] Context and scenario described\n"
    checklist += "- [ ] Language style specified\n\n"
    
    checklist += "## Scoring (for evaluation tasks)\n"
    checklist += "- [ ] 0-1 scale used\n"
    checklist += "- [ ] Explicit criteria for each score level\n"
    checklist += "- [ ] Calibration examples for each level\n"
    checklist += "- [ ] JSON output format specified\n\n"
    
    checklist += "## Testing\n"
    checklist += "- [ ] Tested on small batch first\n"
    checklist += "- [ ] Results reviewed for consistency\n"
    checklist += "- [ ] Failure modes identified and addressed\n"
    checklist += "- [ ] Prompt refined based on results\n"
    
    return checklist


if __name__ == "__main__":
    # Print checklist
    print(generate_prompt_checklist())
    
    # Example: Get guidelines for synthetic data generation
    synth_guidelines = get_guidelines_by_type(PromptType.SYNTHETIC_DATA_GENERATION)
    print(f"\nFound {len(synth_guidelines)} guidelines for synthetic data generation")
