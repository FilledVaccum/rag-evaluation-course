"""
Module 4: Synthetic Test Data Generation - Slide Deck
Focus on LLM-based test data creation and prompt engineering
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Slide:
    """Represents a single slide in the presentation"""
    slide_number: int
    title: str
    content: List[str]
    mermaid_diagram: Optional[str] = None
    visual_notes: Optional[str] = None
    speaker_notes: Optional[str] = None


# Module 4 Slide Deck
MODULE_4_SLIDES = [
    Slide(
        slide_number=1,
        title="Module 4: Synthetic Test Data Generation",
        content=[
            "Creating High-Quality Test Sets with LLMs",
            "",
            "Learning Objectives:",
            "• Understand test-driven development for RAG",
            "• Generate synthetic test data with LLMs",
            "• Master prompt engineering for data steering",
            "• Validate and filter synthetic data quality",
            "• Create domain-specific test sets"
        ],
        visual_notes="Title slide with data generation workflow preview",
        speaker_notes="90-120 minute module. Critical for evaluation. Heavy on prompt engineering."
    ),
    
    Slide(
        slide_number=2,
        title="Why Synthetic Test Data?",
        content=[
            "The Testing Challenge for RAG Systems",
            "",
            "Problems with Manual Test Data:",
            "• Time-consuming to create",
            "• Limited coverage",
            "• Expensive to maintain",
            "• Doesn't scale",
            "",
            "Benefits of Synthetic Data:",
            "• Rapid generation (100s of questions in minutes)",
            "• Comprehensive coverage",
            "• Domain-specific customization",
            "• Continuous evaluation support",
            "• Cost-effective at scale"
        ],
        visual_notes="Comparison chart: manual vs synthetic data creation",
        speaker_notes="Synthetic data is not cheating - it's essential for production RAG systems."
    ),
    
    Slide(
        slide_number=3,
        title="Test-Driven Development for LLMs",
        content=[
            "Applying TDD Principles to RAG",
            "",
            "Traditional TDD:",
            "1. Write test",
            "2. Run test (fails)",
            "3. Write code",
            "4. Run test (passes)",
            "5. Refactor",
            "",
            "RAG TDD:",
            "1. Generate test questions",
            "2. Run RAG evaluation (baseline)",
            "3. Improve RAG components",
            "4. Run evaluation (improved)",
            "5. Iterate"
        ],
        mermaid_diagram="""
```mermaid
graph LR
    A[Generate Tests] --> B[Evaluate RAG]
    B --> C[Identify Failures]
    C --> D[Improve Components]
    D --> B
    
    style A fill:#e1f5ff
    style D fill:#e8f5e9
```
""",
        visual_notes="TDD cycle diagram adapted for RAG",
        speaker_notes="TDD mindset is crucial. Tests drive improvements, not just validate them."
    ),
    
    Slide(
        slide_number=4,
        title="LLM-Based Synthetic Data Generation",
        content=[
            "Using LLMs to Create Test Questions",
            "",
            "Process:",
            "1. Provide source documents to LLM",
            "2. Prompt LLM to generate questions",
            "3. LLM produces questions + answers",
            "4. Validate quality",
            "5. Filter and refine",
            "",
            "Key Models:",
            "• NVIDIA Nemotron-4-340B (recommended)",
            "• GPT-4, GPT-3.5",
            "• Claude, Llama",
            "",
            "Output Format: Question, Answer, Context, Metadata"
        ],
        visual_notes="Data generation pipeline diagram",
        speaker_notes="Nemotron-4-340B is optimized for this task. Part of NVIDIA ecosystem for certification."
    ),
    
    Slide(
        slide_number=5,
        title="The Over-Generalization Problem",
        content=[
            "Common Pitfall: Generic Questions",
            "",
            "Bad Example (Over-Generic):",
            "❌ 'What is the meaning of life?'",
            "❌ 'How does society function?'",
            "❌ 'What are the philosophical implications?'",
            "",
            "Good Example (Domain-Specific):",
            "✓ 'What are the prerequisites for CSCI 567?'",
            "✓ 'When does the Machine Learning course meet?'",
            "✓ 'How many units is CSCI 567 worth?'",
            "",
            "Root Cause: Insufficient prompt specificity"
        ],
        visual_notes="Side-by-side comparison with red X and green checkmark",
        speaker_notes="This is the #1 problem students face. Show real examples from USC dataset."
    ),
    
    Slide(
        slide_number=6,
        title="Prompt Engineering for Data Generation",
        content=[
            "The Art and Science of Steering LLMs",
            "",
            "Key Principles:",
            "1. Extreme Specificity: Write as if explaining to a child",
            "2. 3-5 Examples: Optimal number for steering",
            "3. Explicit Negatives: State what NOT to generate",
            "4. User Personas: Define realistic scenarios",
            "5. Output Format: Specify exact structure",
            "",
            "Example:",
            "❌ 'Generate questions about courses'",
            "✓ 'Generate 10 specific questions a USC student would ask",
            "   about course prerequisites, schedules, and units.",
            "   Do NOT generate philosophical or abstract questions.'"
        ],
        visual_notes="Before/after prompt examples with annotations",
        speaker_notes="This is critical content. Spend time on examples. Reference prompt engineering guidelines."
    ),
    
    Slide(
        slide_number=7,
        title="The 3-5 Example Rule",
        content=[
            "Optimal Number of Examples for Steering",
            "",
            "Research Findings:",
            "• 0-1 examples: Insufficient steering",
            "• 2 examples: Better, but still variable",
            "• 3-5 examples: Optimal balance",
            "• 6+ examples: Diminishing returns, overfitting risk",
            "",
            "Why 3-5?",
            "• Enough to establish pattern",
            "• Not so many that LLM just copies",
            "• Allows for variation and creativity",
            "• Practical for prompt engineering",
            "",
            "Best Practice: Start with 3, add up to 5 if needed"
        ],
        visual_notes="Graph showing quality vs number of examples",
        speaker_notes="This is empirically validated. Students should memorize this for certification."
    ),
    
    Slide(
        slide_number=8,
        title="Domain-Specific Query Generation",
        content=[
            "Tailoring Questions to Your Use Case",
            "",
            "Techniques:",
            "• Define user persona (student, researcher, customer)",
            "• Specify query types (factual, comparison, procedural)",
            "• Include domain terminology",
            "• Set complexity level",
            "• Add constraints (length, style)",
            "",
            "Example for USC Courses:",
            "Persona: 'Undergraduate CS student planning schedule'",
            "Query types: 'Prerequisites, schedules, workload'",
            "Constraints: 'Specific course codes, concrete details'",
            "",
            "Result: Student-focused, actionable questions"
        ],
        visual_notes="Persona cards with example queries",
        speaker_notes="Domain specificity is what makes synthetic data valuable. Generic data is useless."
    ),
    
    Slide(
        slide_number=9,
        title="Quality Validation and Filtering",
        content=[
            "Ensuring High-Quality Test Data",
            "",
            "Validation Checks:",
            "• Answerability: Can question be answered from context?",
            "• Specificity: Is question concrete and focused?",
            "• Diversity: Do questions cover different aspects?",
            "• Difficulty: Mix of easy, medium, hard questions",
            "• Format: Proper structure and grammar",
            "",
            "Filtering Strategies:",
            "• Automated: Regex, keyword matching, length checks",
            "• LLM-based: Use LLM to judge quality",
            "• Manual: Human review of samples",
            "• Hybrid: Combine automated + manual"
        ],
        visual_notes="Quality validation pipeline diagram",
        speaker_notes="Don't skip validation. Bad test data leads to misleading evaluation results."
    ),
    
    Slide(
        slide_number=10,
        title="Synthesizer Types and Mixing",
        content=[
            "Different Synthesizers for Different Needs",
            "",
            "Ragas Synthesizer Types:",
            "• Specific Query: Concrete, factual questions",
            "• Abstract Query: Conceptual, high-level questions",
            "• Reasoning Query: Multi-step, complex questions",
            "",
            "Mixing Strategy:",
            "• 50% Specific + 50% Abstract (balanced)",
            "• 70% Specific + 30% Reasoning (practical focus)",
            "• Custom ratios based on use case",
            "",
            "Implementation:",
            "```python",
            "mixed = mix_synthesizers([specific, abstract], [0.5, 0.5])",
            "```"
        ],
        visual_notes="Pie chart showing synthesizer mix ratios",
        speaker_notes="Different use cases need different mixes. Experiment to find what works."
    ),
    
    Slide(
        slide_number=11,
        title="Hands-On: Notebooks 3 & 4",
        content=[
            "Two-Part Exercise",
            "",
            "Notebook 3: Baseline Generation",
            "• Generate synthetic data out-of-the-box",
            "• Use USC course catalog",
            "• Observe over-generalization problems",
            "• Identify quality issues",
            "",
            "Notebook 4: Customized Generation",
            "• Modify prompts with 3-5 examples",
            "• Add domain-specific constraints",
            "• Create student-focused questions",
            "• Mix synthesizers",
            "• Compare before/after quality"
        ],
        visual_notes="Split screen showing both notebooks",
        speaker_notes="Give 40-45 minutes total. This is the core skill. Encourage experimentation."
    ),
    
    Slide(
        slide_number=12,
        title="Module 4 Summary",
        content=[
            "Key Takeaways:",
            "",
            "✓ Synthetic data enables scalable RAG testing",
            "✓ Over-generalization is the #1 problem",
            "✓ Use 3-5 examples for optimal steering",
            "✓ Be extremely specific in prompts",
            "✓ Always validate and filter generated data",
            "",
            "Next Module: RAG Evaluation Metrics and Frameworks",
            "• LLM-as-a-Judge methodology",
            "• Ragas framework deep dive",
            "• Custom metric development"
        ],
        visual_notes="Summary with prompt engineering checklist",
        speaker_notes="This module is critical for certification. Ensure understanding before moving on."
    )
]


def get_module_4_slides() -> List[Slide]:
    """Return all slides for Module 4"""
    return MODULE_4_SLIDES


def export_slides_to_markdown(slides: List[Slide]) -> str:
    """Export slides to markdown format"""
    markdown = "# Module 4: Synthetic Test Data Generation\n\n"
    
    for slide in slides:
        markdown += f"## Slide {slide.slide_number}: {slide.title}\n\n"
        
        for line in slide.content:
            markdown += f"{line}\n"
        markdown += "\n"
        
        if slide.mermaid_diagram:
            markdown += f"{slide.mermaid_diagram}\n\n"
        
        if slide.visual_notes:
            markdown += f"**Visual Notes:** {slide.visual_notes}\n\n"
        
        if slide.speaker_notes:
            markdown += f"**Speaker Notes:** {slide.speaker_notes}\n\n"
        
        markdown += "---\n\n"
    
    return markdown


if __name__ == "__main__":
    slides = get_module_4_slides()
    markdown_content = export_slides_to_markdown(slides)
    print(markdown_content)
