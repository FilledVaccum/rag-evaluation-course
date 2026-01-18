"""
Module 5 Hands-On Challenge: Custom Metric Development

This challenge requires students to develop a custom evaluation metric
for a specific domain and use case.

Challenge: Create a custom metric for evaluating code generation in RAG systems

Requirements: 13.2
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from src.models.evaluation import MetricDefinition
from src.evaluation.framework import EvaluationFramework, TestSet


@dataclass
class ChallengeRequirements:
    """Requirements for the hands-on challenge"""
    title: str
    description: str
    learning_objectives: List[str]
    deliverables: List[str]
    evaluation_rubric: Dict[str, Dict[str, int]]


# Challenge Definition
CODE_GENERATION_CHALLENGE = ChallengeRequirements(
    title="Custom Metric for Code Generation Evaluation",
    description="""
    You are building a RAG system that helps developers by generating code snippets
    based on documentation and examples. Standard RAG metrics (faithfulness, relevancy)
    are not sufficient because they don't assess code quality.
    
    Your task is to create a custom evaluation metric called "Code Quality Score"
    that evaluates generated code on multiple dimensions specific to code generation.
    """,
    learning_objectives=[
        "Design domain-specific evaluation criteria",
        "Create custom metric definitions from scratch",
        "Implement LLM-as-a-Judge prompts for code evaluation",
        "Test and validate custom metrics on sample data",
        "Interpret metric results and provide optimization insights"
    ],
    deliverables=[
        "MetricDefinition for Code Quality Score",
        "Detailed prompt template with evaluation criteria",
        "Scoring rubric with clear score levels",
        "Test implementation on provided code samples",
        "Analysis report with findings and recommendations"
    ],
    evaluation_rubric={
        "Metric Design (30 points)": {
            "Comprehensive evaluation criteria (0-10)": 10,
            "Clear and specific criteria (0-10)": 10,
            "Appropriate for code generation (0-10)": 10
        },
        "Prompt Engineering (30 points)": {
            "Well-structured prompt template (0-10)": 10,
            "Clear instructions for LLM judge (0-10)": 10,
            "Includes examples and calibration (0-10)": 10
        },
        "Implementation (20 points)": {
            "Correct MetricDefinition structure (0-10)": 10,
            "Proper integration with framework (0-10)": 10
        },
        "Testing and Analysis (20 points)": {
            "Tests on provided samples (0-10)": 10,
            "Meaningful insights from results (0-10)": 10
        }
    }
)


# Sample test data for the challenge
SAMPLE_CODE_GENERATION_DATA = [
    {
        "question": "Write a Python function to calculate factorial",
        "context": [
            "Factorial is the product of all positive integers up to n.",
            "Example: factorial(5) = 5 * 4 * 3 * 2 * 1 = 120",
            "Base case: factorial(0) = 1"
        ],
        "response": """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
        """,
        "ground_truth": "Recursive implementation with base case"
    },
    {
        "question": "Create a function to validate email addresses",
        "context": [
            "Email format: username@domain.extension",
            "Username can contain letters, numbers, dots, underscores",
            "Domain must have at least one dot"
        ],
        "response": """
def validate_email(email):
    if '@' in email:
        return True
    return False
        """,
        "ground_truth": "Should use regex for proper validation"
    },
    {
        "question": "Implement binary search algorithm",
        "context": [
            "Binary search works on sorted arrays",
            "Compare middle element with target",
            "Recursively search left or right half",
            "Time complexity: O(log n)"
        ],
        "response": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
        """,
        "ground_truth": "Iterative binary search implementation"
    }
]


def get_challenge_instructions() -> str:
    """Get detailed challenge instructions"""
    return """
    ============================================================================
    HANDS-ON CHALLENGE: CUSTOM METRIC FOR CODE GENERATION
    ============================================================================
    
    SCENARIO:
    You're building a RAG system that generates code snippets for developers.
    Standard metrics don't capture code-specific quality aspects like:
    - Syntax correctness
    - Best practices adherence
    - Error handling
    - Code efficiency
    - Documentation quality
    
    YOUR TASK:
    Create a custom "Code Quality Score" metric that evaluates these aspects.
    
    STEP 1: DESIGN EVALUATION CRITERIA (30 minutes)
    -----------------------------------------------
    Define 4-6 specific criteria for evaluating code quality.
    
    Example criteria to consider:
    - Syntax Correctness: Is the code syntactically valid?
    - Functionality: Does it solve the stated problem?
    - Best Practices: Does it follow language conventions?
    - Error Handling: Does it handle edge cases?
    - Efficiency: Is the algorithm/approach efficient?
    - Readability: Is the code clear and well-structured?
    - Documentation: Are there helpful comments?
    
    For each criterion:
    - Define what "good" looks like
    - Define what "bad" looks like
    - Assign a weight or score range
    
    STEP 2: CREATE PROMPT TEMPLATE (30 minutes)
    -------------------------------------------
    Write a detailed prompt for the LLM judge that:
    - Explains the evaluation task clearly
    - Lists all evaluation criteria
    - Provides scoring guidance (0-1 scale)
    - Includes placeholders for {question}, {response}, {context}
    - Specifies output format (JSON with score and reasoning)
    
    Best practices:
    - Be extremely specific (as if explaining to a child)
    - Include 1-2 examples of good vs bad code
    - Break down scoring (e.g., 0.25 per criterion)
    - Request structured output for parsing
    
    STEP 3: IMPLEMENT METRIC DEFINITION (20 minutes)
    ------------------------------------------------
    Create a MetricDefinition object with:
    - name: "Code Quality Score"
    - description: Clear explanation of what it measures
    - evaluation_criteria: List of your criteria
    - prompt_template: Your detailed prompt
    - scoring_method: "llm"
    
    Example:
    ```python
    code_quality_def = MetricDefinition(
        name="Code Quality Score",
        description="Evaluates generated code on syntax, functionality, and best practices",
        evaluation_criteria=[
            "Syntax correctness",
            "Functional correctness",
            "Best practices adherence",
            "Error handling"
        ],
        prompt_template='''
        Evaluate the quality of this generated code...
        [Your detailed prompt here]
        ''',
        scoring_method="llm"
    )
    ```
    
    STEP 4: TEST YOUR METRIC (30 minutes)
    -------------------------------------
    1. Create your custom metric using EvaluationFramework
    2. Run it on the provided sample data (3 code examples)
    3. Analyze the scores - do they make sense?
    4. Identify which code sample scored highest/lowest
    5. Verify the reasoning aligns with your criteria
    
    Example:
    ```python
    framework = EvaluationFramework(llm_endpoint="nvidia/llama-3-70b")
    code_metric = framework.create_custom_metric(code_quality_def)
    
    # Test on samples
    test_set = TestSet(
        questions=[...],
        contexts=[...],
        responses=[...]
    )
    
    results = framework.evaluate_rag(test_set, metrics=["code_quality_score"])
    ```
    
    STEP 5: ANALYZE AND REPORT (20 minutes)
    ---------------------------------------
    Write a brief analysis report including:
    
    1. Metric Design Rationale
       - Why you chose these specific criteria
       - How they address code generation quality
    
    2. Test Results
       - Scores for each sample
       - Which sample scored highest/lowest and why
       - Whether scores align with your expectations
    
    3. Insights and Recommendations
       - What the metric reveals about code quality
       - How to use this metric for optimization
       - Potential improvements to the metric
    
    DELIVERABLES:
    -------------
    1. metric_definition.py - Your MetricDefinition code
    2. test_results.json - Evaluation results on sample data
    3. analysis_report.md - Your analysis and insights
    
    EVALUATION RUBRIC:
    ------------------
    - Metric Design (30 pts): Comprehensive, clear, appropriate criteria
    - Prompt Engineering (30 pts): Well-structured, clear, includes examples
    - Implementation (20 pts): Correct structure, proper integration
    - Testing & Analysis (20 pts): Thorough testing, meaningful insights
    
    BONUS CHALLENGES:
    -----------------
    1. Create a second metric for "Code Security Score"
    2. Implement multi-language support (Python, JavaScript, Java)
    3. Add automated syntax checking before LLM evaluation
    4. Create a composite metric combining multiple sub-metrics
    
    TIPS FOR SUCCESS:
    -----------------
    - Start simple, then refine based on test results
    - Use the 3-5 example pattern in your prompt
    - Test on edge cases (syntax errors, incomplete code)
    - Compare your metric with human judgment
    - Iterate on prompt based on LLM judge outputs
    
    TIME ESTIMATE: 2-2.5 hours
    
    Good luck! 🚀
    """


def create_sample_test_set() -> TestSet:
    """Create test set from sample data"""
    return TestSet(
        questions=[item["question"] for item in SAMPLE_CODE_GENERATION_DATA],
        contexts=[item["context"] for item in SAMPLE_CODE_GENERATION_DATA],
        responses=[item["response"] for item in SAMPLE_CODE_GENERATION_DATA],
        ground_truths=[item["ground_truth"] for item in SAMPLE_CODE_GENERATION_DATA],
        metadata={"challenge": "code_quality_metric", "num_samples": len(SAMPLE_CODE_GENERATION_DATA)}
    )


def example_solution() -> MetricDefinition:
    """
    Example solution for the challenge (for instructor reference).
    
    Students should NOT see this until after completing the challenge.
    """
    return MetricDefinition(
        name="Code Quality Score",
        description="Evaluates generated code on syntax, functionality, best practices, and error handling",
        evaluation_criteria=[
            "Syntax Correctness: Code is syntactically valid and runnable",
            "Functional Correctness: Code solves the stated problem correctly",
            "Best Practices: Follows language conventions and idioms",
            "Error Handling: Handles edge cases and potential errors",
            "Efficiency: Uses appropriate algorithms and data structures",
            "Readability: Code is clear, well-structured, and maintainable"
        ],
        prompt_template="""
You are evaluating the quality of generated code for a developer assistance system.

Question: {question}
Context: {context}
Generated Code:
{response}

Evaluate the code on these criteria (each worth 0-0.167 points, total 0-1):

1. SYNTAX CORRECTNESS (0-0.167):
   - Is the code syntactically valid?
   - Would it run without syntax errors?
   - Are all brackets, parentheses, and quotes properly closed?

2. FUNCTIONAL CORRECTNESS (0-0.167):
   - Does it solve the stated problem?
   - Does it handle the base cases correctly?
   - Would it produce correct output for valid inputs?

3. BEST PRACTICES (0-0.167):
   - Does it follow language conventions (PEP 8 for Python, etc.)?
   - Are variable names descriptive?
   - Is the code structure logical?

4. ERROR HANDLING (0-0.167):
   - Does it handle edge cases (empty input, None, etc.)?
   - Are there appropriate error checks?
   - Would it fail gracefully on invalid input?

5. EFFICIENCY (0-0.167):
   - Is the algorithm/approach efficient?
   - Are there obvious performance issues?
   - Does it use appropriate data structures?

6. READABILITY (0-0.167):
   - Is the code easy to understand?
   - Is it well-structured and organized?
   - Would another developer understand it quickly?

SCORING EXAMPLES:

Example 1 - High Quality (Score: 0.95):
```python
def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be non-negative integer")
    if n == 0:
        return 1
    return n * factorial(n - 1)
```
- Syntax: ✓ Valid
- Functionality: ✓ Correct with base case
- Best Practices: ✓ Docstring, type checking
- Error Handling: ✓ Validates input
- Efficiency: ✓ Standard recursive approach
- Readability: ✓ Clear and documented

Example 2 - Low Quality (Score: 0.35):
```python
def f(x):
    return x * f(x-1)
```
- Syntax: ✓ Valid
- Functionality: ✗ Missing base case (infinite recursion)
- Best Practices: ✗ Poor naming, no docstring
- Error Handling: ✗ No input validation
- Efficiency: ✗ Will crash
- Readability: ✗ Unclear purpose

Return your evaluation as JSON:
{{
    "score": 0.0-1.0,
    "reasoning": "Brief explanation of score",
    "breakdown": {{
        "syntax": 0.0-0.167,
        "functionality": 0.0-0.167,
        "best_practices": 0.0-0.167,
        "error_handling": 0.0-0.167,
        "efficiency": 0.0-0.167,
        "readability": 0.0-0.167
    }},
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"]
}}
        """,
        scoring_method="llm"
    )


def print_challenge():
    """Print challenge instructions"""
    print(get_challenge_instructions())
    
    print("\n" + "="*80)
    print("SAMPLE DATA FOR TESTING")
    print("="*80 + "\n")
    
    for i, sample in enumerate(SAMPLE_CODE_GENERATION_DATA, 1):
        print(f"Sample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Context: {sample['context']}")
        print(f"Generated Code:")
        print(sample['response'])
        print(f"Ground Truth: {sample['ground_truth']}")
        print("\n" + "-"*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--solution":
        print("="*80)
        print("EXAMPLE SOLUTION (Instructor Reference)")
        print("="*80 + "\n")
        solution = example_solution()
        print(f"Name: {solution.name}")
        print(f"Description: {solution.description}")
        print(f"\nCriteria:")
        for criterion in solution.evaluation_criteria:
            print(f"  - {criterion}")
        print(f"\nPrompt Template:")
        print(solution.prompt_template)
    else:
        print_challenge()
        print("\nTo see example solution (after completing challenge), run:")
        print("  python module_5_hands_on_challenge.py --solution")
