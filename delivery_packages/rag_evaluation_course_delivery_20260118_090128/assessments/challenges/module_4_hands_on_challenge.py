"""
Module 4 Hands-On Challenge: Custom Prompt Development
Evaluating RAG and Semantic Search Systems Course

This hands-on challenge requires students to develop a custom prompt for
synthetic data generation in a new domain.

Requirements Coverage: 13.2
Challenge Type: Open-ended with evaluation rubric
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ChallengeRubric:
    """Evaluation rubric for hands-on challenge."""
    criterion: str
    points: int
    description: str


# Module 4 Hands-On Challenge
CHALLENGE_DESCRIPTION = """
# Hands-On Challenge: Custom Prompt Development for Synthetic Data Generation

## Objective

Develop a custom prompt template for generating high-quality synthetic test data
for a RAG system in a domain of your choice (NOT USC courses).

## Challenge Requirements

### 1. Choose a Domain (10 points)

Select ONE of the following domains:
- **Healthcare**: Medical records, patient queries, clinical guidelines
- **Finance**: Investment research, financial reports, regulatory documents
- **Legal**: Contract law, case law, legal research
- **E-commerce**: Product catalogs, customer queries, reviews
- **Technical Documentation**: API docs, software manuals, troubleshooting guides

### 2. Define User Persona (15 points)

Create a detailed user persona that includes:
- Role/position (e.g., "Junior financial analyst")
- Experience level (e.g., "2 years experience")
- Goals and concerns (e.g., "Wants to learn about derivatives")
- Context (e.g., "Working at investment bank")

**Example:**
```
User Persona: Junior financial analyst at investment bank
- 2 years experience in equity research
- Wants to expand into derivatives and options
- Concerned about understanding complex financial instruments
- Needs to prepare research reports for clients
```

### 3. Create System Instruction (15 points)

Write a clear, specific system instruction that:
- Explains the simulation context
- Defines the task clearly
- Provides domain context
- Is 2-4 sentences long

**Example:**
```
You are simulating a junior financial analyst researching investment opportunities.
The analyst has access to financial reports, SEC filings, and market data.
Generate questions that reflect realistic research scenarios and information needs.
```

### 4. Define Constraints (20 points)

Create 4-6 specific constraints that define:
- Query length (word count range)
- Focus areas (what topics to cover)
- Language style (formal, casual, technical)
- Answerability (must use knowledge base)
- Any domain-specific requirements

**Example:**
```
Constraints:
1. Questions should be 12-30 words long
2. Focus on investment analysis, risk assessment, and market trends
3. Use professional financial terminology
4. Questions must require financial documents to answer
5. Reflect realistic analyst research workflows
6. Include both quantitative and qualitative questions
```

### 5. Provide 3-5 High-Quality Examples (25 points)

Create exactly 3-5 example queries that:
- Follow all constraints
- Demonstrate diversity in query types
- Are domain-specific and realistic
- Show the pattern you want the LLM to learn
- Are NOT generic or over-broad

**Example:**
```
Examples:
1. "What was the year-over-year revenue growth for tech sector companies in Q3 2024?"
2. "How do rising interest rates typically affect REIT valuations?"
3. "What are the key risk factors mentioned in Tesla's latest 10-K filing?"
4. "Which semiconductor companies have the highest R&D spending as % of revenue?"
```

### 6. Add Negative Examples (15 points)

Provide 3-5 explicit negative examples that state what NOT to generate:
- Off-topic queries
- Overly generic questions
- Unanswerable questions
- Wrong domain focus

**Example:**
```
DO NOT generate:
1. Questions about personal finance or budgeting
2. Questions about how to become a financial analyst
3. Philosophical questions about capitalism or economics
4. Questions answerable without financial documents
5. Questions about stock tips or investment advice
```

## Deliverables

Submit a complete prompt template in the following format:

```python
custom_prompt = {
    'domain': 'Your chosen domain',
    'system_instruction': 'Your system instruction',
    'user_persona': 'Your detailed user persona',
    'constraints': [
        'Constraint 1',
        'Constraint 2',
        # ... 4-6 total
    ],
    'examples': [
        'Example query 1',
        'Example query 2',
        'Example query 3',
        # 3-5 total (MUST be 3-5)
    ],
    'negative_examples': [
        'Negative example 1',
        'Negative example 2',
        'Negative example 3',
        # 3-5 total
    ]
}
```

## Evaluation Rubric

Your submission will be evaluated on:
"""

EVALUATION_RUBRIC = [
    ChallengeRubric(
        criterion="Domain Selection",
        points=10,
        description="""
- Clear domain choice (5 points)
- Appropriate for RAG evaluation (5 points)
        """.strip()
    ),
    ChallengeRubric(
        criterion="User Persona",
        points=15,
        description="""
- Specific role and experience level (5 points)
- Clear goals and concerns (5 points)
- Realistic context (5 points)
        """.strip()
    ),
    ChallengeRubric(
        criterion="System Instruction",
        points=15,
        description="""
- Clear and specific (5 points)
- Provides domain context (5 points)
- Appropriate length (2-4 sentences) (5 points)
        """.strip()
    ),
    ChallengeRubric(
        criterion="Constraints",
        points=20,
        description="""
- 4-6 constraints provided (5 points)
- Specific and measurable (5 points)
- Cover length, style, focus, answerability (5 points)
- Domain-appropriate (5 points)
        """.strip()
    ),
    ChallengeRubric(
        criterion="Example Queries",
        points=25,
        description="""
- Exactly 3-5 examples (5 points)
- Domain-specific and realistic (10 points)
- Diverse query types (5 points)
- Follow all constraints (5 points)
        """.strip()
    ),
    ChallengeRubric(
        criterion="Negative Examples",
        points=15,
        description="""
- 3-5 negative examples provided (5 points)
- Prevent common failure modes (5 points)
- Domain-appropriate (5 points)
        """.strip()
    )
]

BONUS_POINTS = """
## Bonus Points (Optional)

Earn up to 10 bonus points by:

1. **Testing Your Prompt (5 points)**:
   - Use your prompt with the SyntheticDataGenerator
   - Generate 20 sample queries
   - Analyze quality scores
   - Document results

2. **Iteration Documentation (5 points)**:
   - Show before/after prompt versions
   - Explain what you changed and why
   - Demonstrate quality improvement
"""

EXAMPLE_SUBMISSION = """
## Example Submission (Healthcare Domain)

```python
custom_prompt = {
    'domain': 'Healthcare - Clinical Guidelines',
    
    'system_instruction': '''
You are simulating a primary care physician looking up clinical guidelines
for patient treatment decisions. The physician has access to medical literature,
clinical practice guidelines, and treatment protocols.
    '''.strip(),
    
    'user_persona': '''
Primary care physician with 5 years experience
- Sees 20-30 patients per day in outpatient clinic
- Needs quick access to evidence-based treatment guidelines
- Concerned about staying current with best practices
- Must make treatment decisions during patient visits
    '''.strip(),
    
    'constraints': [
        'Questions should be 10-25 words long',
        'Focus on diagnosis, treatment protocols, and clinical guidelines',
        'Use medical terminology appropriately',
        'Questions must be answerable with clinical guidelines',
        'Reflect realistic clinical decision-making scenarios',
        'Include both common and complex conditions'
    ],
    
    'examples': [
        'What is the first-line treatment for uncomplicated hypertension in adults under 60?',
        'When should I refer a patient with suspected melanoma to dermatology?',
        'What are the diagnostic criteria for type 2 diabetes according to ADA guidelines?',
        'How do I manage a patient with both COPD and heart failure?'
    ],
    
    'negative_examples': [
        'Questions about medical school or residency training',
        'Questions about healthcare policy or insurance',
        'Philosophical questions about medical ethics',
        'Questions about rare diseases not in guidelines',
        'Questions about alternative medicine without evidence base'
    ]
}
```

**Quality Assessment:**
- Domain: Healthcare ✓
- Persona: Specific and realistic ✓
- System Instruction: Clear context ✓
- Constraints: 6 specific constraints ✓
- Examples: 4 high-quality examples ✓
- Negative Examples: 5 clear negatives ✓

**Expected Quality Score: 85-90%**
"""

SUBMISSION_TEMPLATE = """
## Submission Template

Copy and complete this template:

```python
# Module 4 Hands-On Challenge Submission
# Student Name: [Your Name]
# Domain: [Your Chosen Domain]

custom_prompt = {
    'domain': '',  # TODO: Fill in your domain
    
    'system_instruction': '''
    # TODO: Write your system instruction (2-4 sentences)
    '''.strip(),
    
    'user_persona': '''
    # TODO: Define your user persona
    # Include: role, experience, goals, concerns, context
    '''.strip(),
    
    'constraints': [
        # TODO: Add 4-6 specific constraints
    ],
    
    'examples': [
        # TODO: Add exactly 3-5 example queries
        # Make them domain-specific and realistic
    ],
    
    'negative_examples': [
        # TODO: Add 3-5 negative examples
        # State what NOT to generate
    ]
}

# Optional: Test your prompt
# from src.synthetic_data.generator import SyntheticDataGenerator
# generator = SyntheticDataGenerator(...)
# generator.customize_prompt(**custom_prompt)
# queries = generator.generate_questions(...)
```

## Submission Instructions

1. Complete the template above
2. Save as `module_4_challenge_[yourname].py`
3. Submit via course platform
4. Include optional testing results if completed

## Grading Timeline

- Submissions due: [Date]
- Feedback provided: Within 1 week
- Total possible points: 100 (+ 10 bonus)
"""


def print_challenge():
    """Print the complete challenge description."""
    print(CHALLENGE_DESCRIPTION)
    
    print("\n" + "=" * 60)
    print("EVALUATION RUBRIC (100 points total)")
    print("=" * 60)
    
    for rubric in EVALUATION_RUBRIC:
        print(f"\n{rubric.criterion} ({rubric.points} points)")
        print(rubric.description)
    
    print("\n" + BONUS_POINTS)
    print("\n" + EXAMPLE_SUBMISSION)
    print("\n" + SUBMISSION_TEMPLATE)


if __name__ == "__main__":
    print_challenge()
