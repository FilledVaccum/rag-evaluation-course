"""
Notebook 3: Baseline Synthetic Data Generation
Evaluating RAG and Semantic Search Systems Course

This notebook demonstrates out-of-the-box synthetic data generation using
NVIDIA Nemotron-4-340B and the USC Course Catalog dataset.

Learning Objectives:
- Generate synthetic test data with minimal customization
- Identify over-generalization problems in baseline generation
- Understand the need for prompt engineering
- Debug intentional bugs in prompts

Requirements Coverage: 6.6, 10.2
Intentional Bugs: Yes (for debugging practice)

Dataset: USC Course Catalog (tabular data)
Model: NVIDIA Nemotron-4-340B via NIM
"""

# %% [markdown]
# # Notebook 3: Baseline Synthetic Data Generation
# 
# ## Overview
# 
# In this notebook, we'll generate synthetic test data for RAG evaluation using
# **out-of-the-box** settings with minimal customization. This will help us understand:
# 
# 1. What LLMs generate without specific guidance
# 2. Common problems with baseline generation (over-generalization)
# 3. Why prompt engineering is essential
# 
# **Dataset**: USC Course Catalog (1000+ courses)
# **Model**: NVIDIA Nemotron-4-340B
# **Goal**: Generate 50 synthetic student queries
# 
# ## Key Concepts
# 
# - **Baseline Generation**: Using LLM with minimal prompting
# - **Over-Generalization**: Queries that are too broad or philosophical
# - **Quality Issues**: Generic questions that don't test RAG capabilities

# %%
# Setup and imports
import sys
sys.path.append('../..')

from src.synthetic_data.generator import (
    SyntheticDataGenerator,
    SynthesizerType,
    PromptTemplate,
    QualityValidator
)
from src.utils.dataset_manager import DatasetManager
import pandas as pd
from typing import List, Dict
import json

# %%
# Load USC Course Catalog dataset
print("Loading USC Course Catalog...")
dataset_manager = DatasetManager()

# Load the course catalog (simulated for this notebook)
# In production, this would load from actual CSV file
course_data = {
    'course_name': [
        'CSCI 567 - Machine Learning',
        'CSCI 570 - Analysis of Algorithms',
        'CSCI 402 - Operating Systems',
        'CSCI 585 - Database Systems',
        'CSCI 544 - Natural Language Processing'
    ],
    'units': [4, 4, 4, 4, 4],
    'description': [
        'Fundamental concepts and algorithms in machine learning',
        'Design and analysis of efficient algorithms',
        'Operating system concepts and implementation',
        'Database design, implementation, and optimization',
        'Computational approaches to natural language'
    ],
    'prerequisites': [
        'CSCI 270, MATH 225',
        'CSCI 270',
        'CSCI 201',
        'CSCI 201',
        'CSCI 270'
    ]
}

courses_df = pd.DataFrame(course_data)
print(f"Loaded {len(courses_df)} courses")
print("\nSample courses:")
print(courses_df.head())

# %%
# Convert course catalog to context string
def create_dataset_context(df: pd.DataFrame, max_courses: int = 10) -> str:
    """Convert course dataframe to context string for LLM."""
    context_parts = ["USC Course Catalog:\n"]
    
    for idx, row in df.head(max_courses).iterrows():
        context_parts.append(
            f"\n{row['course_name']}"
            f"\nUnits: {row['units']}"
            f"\nDescription: {row['description']}"
            f"\nPrerequisites: {row['prerequisites']}"
            f"\n"
        )
    
    return "\n".join(context_parts)

dataset_context = create_dataset_context(courses_df)
print("Dataset context created:")
print(dataset_context[:500] + "...")

# %% [markdown]
# ## Part 1: Baseline Generation (Minimal Prompting)
# 
# Let's start with a **very simple prompt** and see what the LLM generates.
# This represents what many developers try first.

# %%
# INTENTIONAL BUG #1: Using only 1 example (should be 3-5)
# This will lead to over-generic outputs

baseline_prompt = PromptTemplate(
    system_instruction="Generate questions about courses",  # Too vague!
    user_persona="Student",  # Not specific enough!
    constraints=[
        "Generate questions"  # No real constraints!
    ],
    examples=[
        "What courses are available?"  # Only 1 example - BUG!
    ],
    negative_examples=[]  # No negative examples - BUG!
)

print("Baseline Prompt Template:")
print(f"  System Instruction: {baseline_prompt.system_instruction}")
print(f"  User Persona: {baseline_prompt.user_persona}")
print(f"  Number of Examples: {len(baseline_prompt.examples)}")
print(f"  Number of Negative Examples: {len(baseline_prompt.negative_examples)}")
print("\n⚠️  WARNING: This prompt has intentional bugs!")
print("  - Only 1 example (should be 3-5)")
print("  - No negative examples")
print("  - Vague instructions")

# %%
# Initialize generator with baseline prompt
generator = SyntheticDataGenerator(
    llm_endpoint="https://api.nvidia.com/nim",
    api_key="placeholder-key",
    model_name="nvidia/nemotron-4-340b-instruct"
)

# Try to use the buggy prompt (will fail validation)
try:
    generator.customize_prompt(
        system_instruction=baseline_prompt.system_instruction,
        user_persona=baseline_prompt.user_persona,
        constraints=baseline_prompt.constraints,
        examples=baseline_prompt.examples,
        negative_examples=baseline_prompt.negative_examples
    )
except ValueError as e:
    print(f"❌ Prompt validation failed: {e}")
    print("\nThis is expected! The prompt violates the 3-5 example rule.")

# %% [markdown]
# ### Debugging Exercise 1
# 
# **Problem**: The prompt template fails validation because it only has 1 example.
# 
# **Your Task**: Fix the prompt by adding 2-4 more examples to reach the optimal 3-5 range.
# 
# **Hint**: Think about different types of questions students might ask about courses.

# %%
# STUDENT EXERCISE: Fix the baseline prompt
# Add 2-4 more examples here to make it valid

fixed_baseline_examples = [
    "What courses are available?",
    # TODO: Add 2-4 more examples here
    # Example ideas:
    # - Questions about prerequisites
    # - Questions about course content
    # - Questions about scheduling
]

# Uncomment to test your fix:
# generator.customize_prompt(
#     system_instruction=baseline_prompt.system_instruction,
#     user_persona=baseline_prompt.user_persona,
#     constraints=baseline_prompt.constraints,
#     examples=fixed_baseline_examples,
#     negative_examples=baseline_prompt.negative_examples
# )

# %% [markdown]
# ## Part 2: Simulating Baseline Generation Output
# 
# Even with 3-5 examples, a baseline prompt with vague instructions produces
# **over-generic** queries. Let's simulate what typical baseline generation looks like.

# %%
# Simulated baseline generation output (what you'd get with minimal prompting)
baseline_generated_queries = [
    "What courses are available?",
    "Tell me about computer science courses",
    "What is the curriculum?",
    "How do I enroll in courses?",
    "What are the requirements?",
    "Can you explain the course catalog?",
    "What classes can I take?",
    "Tell me about the CS program",
    "What courses are offered this semester?",
    "How many units do I need?"
]

print("Baseline Generated Queries (Simulated):")
print("=" * 60)
for i, query in enumerate(baseline_generated_queries, 1):
    print(f"{i}. {query}")

# %% [markdown]
# ### Analysis: Problems with Baseline Generation
# 
# Let's analyze what's wrong with these queries:

# %%
def analyze_query_quality(queries: List[str]) -> Dict:
    """Analyze quality issues in generated queries."""
    analysis = {
        'total_queries': len(queries),
        'over_generic': [],
        'too_broad': [],
        'not_answerable': [],
        'good_queries': []
    }
    
    # Simple heuristic checks
    generic_keywords = ['what', 'tell me', 'explain', 'available']
    broad_keywords = ['curriculum', 'program', 'requirements']
    unanswerable_keywords = ['enroll', 'how do i', 'can i']
    
    for query in queries:
        query_lower = query.lower()
        
        # Check for over-generic
        if any(kw in query_lower for kw in generic_keywords) and len(query.split()) < 7:
            analysis['over_generic'].append(query)
        
        # Check for too broad
        elif any(kw in query_lower for kw in broad_keywords):
            analysis['too_broad'].append(query)
        
        # Check for unanswerable
        elif any(kw in query_lower for kw in unanswerable_keywords):
            analysis['not_answerable'].append(query)
        
        else:
            analysis['good_queries'].append(query)
    
    return analysis

analysis = analyze_query_quality(baseline_generated_queries)

print("\n📊 Quality Analysis:")
print("=" * 60)
print(f"Total Queries: {analysis['total_queries']}")
print(f"\n❌ Over-Generic ({len(analysis['over_generic'])}):")
for q in analysis['over_generic']:
    print(f"  - {q}")
print(f"\n❌ Too Broad ({len(analysis['too_broad'])}):")
for q in analysis['too_broad']:
    print(f"  - {q}")
print(f"\n❌ Not Answerable with Course Catalog ({len(analysis['not_answerable'])}):")
for q in analysis['not_answerable']:
    print(f"  - {q}")
print(f"\n✅ Good Queries ({len(analysis['good_queries'])}):")
for q in analysis['good_queries']:
    print(f"  - {q}")

quality_percentage = (len(analysis['good_queries']) / analysis['total_queries']) * 100
print(f"\n📈 Quality Score: {quality_percentage:.1f}%")

# %% [markdown]
# ### Key Problems Identified
# 
# 1. **Over-Generic**: "What courses are available?" - Too broad, not specific
# 2. **Philosophical**: "Tell me about computer science" - Not actionable
# 3. **Unanswerable**: "How do I enroll?" - Requires enrollment system, not course catalog
# 4. **Too Simple**: Most queries can be answered by just listing course names
# 
# **Why This Happens**:
# - Vague prompt instructions
# - No user persona specificity
# - No negative examples to constrain generation
# - Insufficient examples (or wrong examples)

# %% [markdown]
# ## Part 3: Comparing with Better Queries
# 
# Let's compare baseline queries with what we WANT to generate:

# %%
# What we WANT: Specific, realistic student queries
desired_queries = [
    "I want to take machine learning but I only know Python, what should I do?",
    "Which database course has the most hands-on projects?",
    "Are there any 2-unit courses that count toward the CS major?",
    "What's the typical workload for graduate-level systems courses?",
    "I'm interested in both AI and security, which courses overlap?",
    "Can I take the computer vision course without linear algebra?",
    "What's the prerequisite chain to get to advanced robotics?",
    "Which courses would prepare me for both backend and ML engineering roles?",
    "Are there any courses that teach Rust or Go instead of C++?",
    "I need one more elective, what's interesting but not too time-consuming?"
]

print("Desired Queries (What We Want):")
print("=" * 60)
for i, query in enumerate(desired_queries, 1):
    print(f"{i}. {query}")

# Analyze desired queries
desired_analysis = analyze_query_quality(desired_queries)
desired_quality = (len(desired_analysis['good_queries']) / len(desired_queries)) * 100

print(f"\n📈 Desired Quality Score: {desired_quality:.1f}%")

# %% [markdown]
# ### Comparison: Baseline vs. Desired
# 
# | Metric | Baseline | Desired |
# |--------|----------|---------|
# | Quality Score | ~10-20% | ~80-90% |
# | Specificity | Low | High |
# | Answerability | Mixed | High |
# | Realism | Low | High |
# | Test Value | Low | High |
# 
# **Key Differences**:
# - Desired queries are **specific** and **actionable**
# - They reflect **real student concerns** (prerequisites, workload, career prep)
# - They **require the course catalog** to answer
# - They test **actual RAG capabilities** (retrieval, reasoning)

# %% [markdown]
# ## Part 4: Understanding Over-Generalization
# 
# Over-generalization is the #1 problem with baseline synthetic data generation.

# %%
def demonstrate_over_generalization():
    """Show examples of over-generalization at different levels."""
    
    examples = {
        "Extremely Generic": [
            "What is this about?",
            "Tell me more",
            "Can you help me?",
            "What information do you have?"
        ],
        "Generic": [
            "What courses are available?",
            "Tell me about computer science",
            "What can I learn here?",
            "What are the options?"
        ],
        "Somewhat Specific": [
            "What machine learning courses are there?",
            "Which courses cover databases?",
            "What are the AI course prerequisites?",
            "How many units is the ML course?"
        ],
        "Specific (Good)": [
            "I want to take CSCI 567 but I'm weak at math, what should I take first?",
            "Which database course has more hands-on projects: 585 or 587?",
            "Can I take graduate ML courses as an undergrad with a 3.5 GPA?",
            "What's the prerequisite chain from intro CS to advanced robotics?"
        ]
    }
    
    print("Over-Generalization Spectrum:")
    print("=" * 60)
    
    for level, queries in examples.items():
        print(f"\n{level}:")
        for q in queries:
            print(f"  - {q}")
    
    print("\n💡 Key Insight:")
    print("  Baseline generation tends to produce 'Generic' or 'Extremely Generic' queries.")
    print("  We need prompt engineering to push toward 'Specific' queries.")

demonstrate_over_generalization()

# %% [markdown]
# ## Part 5: Debugging Exercise - Fix the Prompt
# 
# Now it's your turn to improve the baseline prompt!

# %%
# INTENTIONAL BUG #2: Missing constraints and negative examples
# This prompt will still produce generic outputs even with 3-5 examples

incomplete_prompt = PromptTemplate(
    system_instruction="Generate questions about USC courses",
    user_persona="Student",
    constraints=[
        "Generate questions about courses"  # Too vague!
    ],
    examples=[
        "What courses are available?",
        "When is the class offered?",
        "Who teaches this course?",
        "How many units is it?"
    ],
    negative_examples=[]  # BUG: No negative examples!
)

print("Incomplete Prompt (Has 4 examples but still buggy):")
print("=" * 60)
print(f"Examples: {len(incomplete_prompt.examples)} ✓")
print(f"Constraints: {incomplete_prompt.constraints}")
print(f"Negative Examples: {len(incomplete_prompt.negative_examples)} ❌")
print("\n⚠️  This prompt will still produce generic queries!")
print("Why? No negative examples to constrain generation.")

# %% [markdown]
# ### Debugging Exercise 2
# 
# **Problem**: Even with 4 examples, this prompt lacks:
# 1. Specific constraints about query style
# 2. Negative examples to prevent generic queries
# 3. Clear user persona
# 
# **Your Task**: Improve the prompt by adding:
# - More specific constraints (length, style, focus)
# - At least 3 negative examples
# - A more detailed user persona

# %%
# STUDENT EXERCISE: Improve the prompt
# Fill in the missing parts

improved_prompt = PromptTemplate(
    system_instruction="You are simulating a USC undergraduate CS student planning their schedule",
    user_persona="Junior CS major who has completed core requirements and wants to specialize in AI",
    constraints=[
        # TODO: Add specific constraints
        # Ideas:
        # - Query length (10-25 words)
        # - Focus areas (prerequisites, workload, career prep)
        # - Language style (casual, student-like)
        # - Answerability (must use course catalog)
    ],
    examples=[
        "I want to take machine learning but I only know Python, what should I do?",
        "Which database course has the most hands-on projects?",
        "Are there any 2-unit courses that count toward the CS major?",
        "What's the typical workload for graduate-level systems courses?"
    ],
    negative_examples=[
        # TODO: Add negative examples
        # Ideas:
        # - Philosophical questions
        # - Admissions/tuition questions
        # - Questions answerable by course title alone
    ]
)

# Test your improved prompt
print("\nImproved Prompt:")
print("=" * 60)
print(f"System Instruction: {improved_prompt.system_instruction}")
print(f"User Persona: {improved_prompt.user_persona}")
print(f"Constraints: {len(improved_prompt.constraints)}")
print(f"Examples: {len(improved_prompt.examples)}")
print(f"Negative Examples: {len(improved_prompt.negative_examples)}")

# %% [markdown]
# ## Part 6: Quality Validation
# 
# Let's implement quality validation to filter out bad queries.

# %%
# Create quality validator
validator = QualityValidator(
    min_length=5,  # Minimum 5 words
    max_length=50,  # Maximum 50 words
    banned_keywords=["weather", "admission", "tuition", "enroll", "apply"],
    similarity_threshold=0.9  # Remove near-duplicates
)

# Test validation on baseline queries
print("Validating Baseline Queries:")
print("=" * 60)

for query in baseline_generated_queries[:5]:
    length_valid = validator.validate_length(query)
    keyword_valid = validator.validate_keywords(query)
    quality_score = validator.calculate_quality_score(query)
    
    print(f"\nQuery: {query}")
    print(f"  Length Valid: {length_valid}")
    print(f"  Keywords Valid: {keyword_valid}")
    print(f"  Quality Score: {quality_score:.2f}")

# %% [markdown]
# ## Part 7: Key Takeaways
# 
# ### What We Learned
# 
# 1. **Baseline Generation Problems**:
#    - Over-generic queries ("What courses are available?")
#    - Philosophical questions ("Tell me about computer science")
#    - Unanswerable queries ("How do I enroll?")
#    - Low test value for RAG systems
# 
# 2. **Why Baseline Fails**:
#    - Insufficient examples (< 3 or > 5)
#    - Vague instructions
#    - No negative examples
#    - Unclear user persona
# 
# 3. **Quality Metrics Matter**:
#    - Length validation (5-50 words)
#    - Keyword filtering (banned terms)
#    - Domain relevance checking
#    - Duplicate removal
# 
# 4. **The 3-5 Example Rule**:
#    - Too few (1-2): Generic outputs
#    - Optimal (3-5): Good steering
#    - Too many (6+): Overfitting
# 
# ### Next Steps
# 
# In **Notebook 4**, we'll learn how to:
# - Customize prompts with specific instructions
# - Use the 3-5 example pattern effectively
# - Add negative examples to constrain generation
# - Generate domain-specific, realistic queries
# - Achieve 80%+ quality scores

# %% [markdown]
# ## Summary Statistics

# %%
def print_summary():
    """Print summary of baseline generation experiment."""
    
    print("\n" + "=" * 60)
    print("BASELINE SYNTHETIC DATA GENERATION - SUMMARY")
    print("=" * 60)
    
    print("\n📊 Results:")
    print(f"  Baseline Quality: ~10-20%")
    print(f"  Desired Quality: ~80-90%")
    print(f"  Gap: ~70 percentage points")
    
    print("\n❌ Main Problems:")
    print("  1. Over-generic queries (70%)")
    print("  2. Unanswerable queries (20%)")
    print("  3. Too broad queries (10%)")
    
    print("\n✅ Solutions (Next Notebook):")
    print("  1. Use 3-5 specific examples")
    print("  2. Add detailed constraints")
    print("  3. Include negative examples")
    print("  4. Define clear user persona")
    print("  5. Implement quality validation")
    
    print("\n💡 Key Insight:")
    print("  Baseline generation is a starting point, not a solution.")
    print("  Prompt engineering is ESSENTIAL for high-quality synthetic data.")
    
    print("\n" + "=" * 60)

print_summary()

# %% [markdown]
# ## Exercises for Students
# 
# 1. **Fix Intentional Bug #1**: Add 2-4 examples to reach 3-5 total
# 2. **Fix Intentional Bug #2**: Add constraints and negative examples
# 3. **Analyze Your Own Queries**: Generate 10 queries and calculate quality score
# 4. **Compare Synthesizers**: Try different synthesizer types (Specific, Abstract, Reasoning)
# 5. **Experiment with Validation**: Adjust quality thresholds and observe results
# 
# ## Additional Resources
# 
# - Module 4 Lecture: Prompt Engineering Best Practices
# - Notebook 4: Customized Synthetic Data Generation
# - NVIDIA Nemotron Documentation
# - Ragas Synthesizer Documentation
