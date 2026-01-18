"""
Module 3 Debugging Exercise: Broken RAG Pipeline

This hands-on debugging exercise presents students with a broken RAG pipeline
and requires them to diagnose and fix component-level failures.

Requirements Coverage: 13.3
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DebugScenario:
    """A debugging scenario for students to solve."""
    scenario_id: str
    title: str
    description: str
    broken_code: str
    symptoms: List[str]
    hints: List[str]
    expected_fix: str
    learning_objectives: List[str]


# Debugging Exercise Scenarios
DEBUGGING_SCENARIOS = [
    DebugScenario(
        scenario_id="debug_1",
        title="Retrieval Returns Wrong Documents",
        description=(
            "A RAG system for course prerequisites is returning incorrect answers. "
            "When asked about CSCI 567 prerequisites, it returns information about "
            "CSCI 270 instead. The vector store search is not working correctly."
        ),
        broken_code="""
class BrokenVectorStore:
    def search(self, query: str, top_k: int = 3):
        # BUG: Always returns first k documents regardless of query
        return self.documents[:top_k]
""",
        symptoms=[
            "Same documents returned for different queries",
            "Answers are incorrect but plausible",
            "Retrieved documents don't match query semantically"
        ],
        hints=[
            "Inspect what documents are being retrieved for different queries",
            "Check if the search method is using the query parameter",
            "Consider how semantic similarity should work"
        ],
        expected_fix=(
            "Implement actual search logic that matches query to documents. "
            "Use keyword matching, embeddings, or semantic similarity to find "
            "relevant documents based on the query content."
        ),
        learning_objectives=[
            "Diagnose retrieval failures through inspection",
            "Understand importance of query-dependent search",
            "Implement basic relevance matching"
        ]
    ),
    
    DebugScenario(
        scenario_id="debug_2",
        title="LLM Ignores Retrieved Context",
        description=(
            "The RAG system retrieves correct documents, but the LLM generates "
            "generic responses like 'I don't have that information' even when "
            "the answer is clearly in the retrieved context."
        ),
        broken_code="""
class BrokenLLM:
    def generate(self, prompt: str):
        # BUG: Ignores prompt and returns generic response
        return "I don't have specific information about that."
""",
        symptoms=[
            "LLM says 'I don't know' when answer is in context",
            "Responses are generic and don't use retrieved information",
            "Retrieved documents are correct but unused"
        ],
        hints=[
            "Check if the LLM is actually processing the prompt",
            "Inspect the augmented prompt being sent to the LLM",
            "Consider whether the LLM is using the context or ignoring it"
        ],
        expected_fix=(
            "Implement LLM logic that actually processes the prompt and uses "
            "the context. Extract information from the prompt and generate "
            "responses based on the provided context, not generic fallbacks."
        ),
        learning_objectives=[
            "Diagnose generation failures vs retrieval failures",
            "Understand context utilization in LLMs",
            "Implement context-grounded response generation"
        ]
    ),
    
    DebugScenario(
        scenario_id="debug_3",
        title="Context Lost in Augmentation",
        description=(
            "The retrieval stage finds correct documents, but the augmentation "
            "stage truncates the context due to token limits, causing the LLM "
            "to miss critical information."
        ),
        broken_code="""
class BrokenAugmenter:
    def augment(self, query: str, contexts: List[str]):
        # BUG: Truncates context too aggressively
        context_str = contexts[0][:100]  # Only first 100 chars!
        return f"Context: {context_str}\\nQuestion: {query}\\nAnswer:"
""",
        symptoms=[
            "Partial or incomplete answers",
            "LLM says information is missing when it was retrieved",
            "Answers are correct for short contexts but fail for longer ones"
        ],
        hints=[
            "Check the augmented prompt length",
            "Verify all retrieved context is included in the prompt",
            "Consider token limits and context window sizes"
        ],
        expected_fix=(
            "Implement proper context formatting that includes all relevant "
            "information without aggressive truncation. Use appropriate token "
            "limits and consider summarization if needed."
        ),
        learning_objectives=[
            "Diagnose augmentation failures",
            "Understand information loss in prompt formatting",
            "Implement proper context management"
        ]
    ),
    
    DebugScenario(
        scenario_id="debug_4",
        title="Poor Ranking Quality",
        description=(
            "The vector store retrieves relevant documents, but they're ranked "
            "poorly with the most relevant information appearing last instead of "
            "first. This causes the LLM to miss key information."
        ),
        broken_code="""
class BrokenRanker:
    def rank_results(self, results: List[Tuple[str, float]]):
        # BUG: Sorts in ascending order instead of descending
        return sorted(results, key=lambda x: x[1])  # Wrong direction!
""",
        symptoms=[
            "Relevant documents retrieved but answers still wrong",
            "LLM uses less relevant information",
            "Context precision is low despite good recall"
        ],
        hints=[
            "Check the order of retrieved documents",
            "Verify relevance scores are sorted correctly",
            "Consider which documents the LLM sees first"
        ],
        expected_fix=(
            "Fix the ranking to sort in descending order (highest scores first). "
            "Ensure the most relevant documents appear at the top of the results "
            "where the LLM is most likely to use them."
        ),
        learning_objectives=[
            "Understand importance of ranking quality",
            "Diagnose context precision issues",
            "Implement proper result ranking"
        ]
    ),
]


# Debugging Exercise Assessment
DEBUGGING_EXERCISE = {
    "exercise_id": "module_3_debugging",
    "title": "Debugging Exercise: Broken RAG Pipeline",
    "description": (
        "You are given a RAG pipeline with multiple bugs. Your task is to "
        "systematically diagnose and fix each issue using the component-level "
        "debugging techniques learned in Module 3."
    ),
    "scenarios": DEBUGGING_SCENARIOS,
    "instructions": [
        "For each scenario, read the description and symptoms carefully",
        "Inspect the broken code to identify the bug",
        "Apply the systematic debugging workflow: retrieval → augmentation → generation",
        "Implement the fix and verify it resolves the issue",
        "Document your diagnosis process and reasoning"
    ],
    "evaluation_criteria": {
        "diagnosis": {
            "points": 10,
            "description": "Correctly identifies failure type (retrieval/augmentation/generation)"
        },
        "fix_implementation": {
            "points": 10,
            "description": "Implements working fix that resolves the issue"
        },
        "reasoning": {
            "points": 5,
            "description": "Explains diagnostic process and reasoning clearly"
        }
    },
    "total_points": 25,
    "time_limit_minutes": 60
}


def get_debugging_exercise() -> Dict:
    """
    Returns the Module 3 debugging exercise.
    
    Returns:
        Dictionary containing all debugging scenarios and instructions
    """
    return DEBUGGING_EXERCISE


def get_scenario(scenario_id: str) -> DebugScenario:
    """
    Get a specific debugging scenario by ID.
    
    Args:
        scenario_id: Scenario identifier
        
    Returns:
        DebugScenario object or None if not found
    """
    for scenario in DEBUGGING_SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    return None


if __name__ == "__main__":
    exercise = get_debugging_exercise()
    print(f"Debugging Exercise: {exercise['title']}")
    print(f"Total Points: {exercise['total_points']}")
    print(f"Time Limit: {exercise['time_limit_minutes']} minutes")
    print(f"\nScenarios: {len(exercise['scenarios'])}")
    for scenario in exercise['scenarios']:
        print(f"  - {scenario.title}")
        print(f"    Symptoms: {len(scenario.symptoms)}")
        print(f"    Hints: {len(scenario.hints)}")
