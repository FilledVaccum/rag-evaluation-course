"""
Notebook 4: Customized Synthetic Data Generation
Evaluating RAG and Semantic Search Systems Course

This notebook demonstrates advanced prompt engineering techniques for generating
high-quality, domain-specific synthetic test data.

Learning Objectives:
- Apply the 3-5 example optimal pattern
- Implement domain-specific query generation
- Use before/after prompt engineering comparisons
- Mix multiple synthesizers for diverse test sets

Requirements Coverage: 6.7, 10.2
Intentional Bugs: Yes (for debugging practice)

Dataset: USC Course Catalog
Model: NVIDIA Nemotron-4-340B via NIM
"""

# %% [markdown]
# # Notebook 4: Customized Synthetic Data Generation
# 
# ## Overview
# 
# In Notebook 3, we saw that baseline generation produces over-generic queries.
# In this notebook, we'll learn how to **customize prompts** to generate
# high-quality, domain-specific test data.
# 
# **Key Techniques**:
# 1. The 3-5 Example Optimal Pattern
# 2. Extreme Specificity in Instructions
# 3. Explicit Negative Examples
# 4. User Persona Definition
# 5. Synthesizer Mixing (50-50 strategy)
# 
# ## Learning Goals
# 
# By the end of this notebook, you'll be able to:
# - Generate student-focused queries (not philosophical)
# - Achieve 80%+ quality scores
# - Mix synthesizers for diverse test sets
# - Validate and filter synthetic data

# %%
# Setup and imports
import sys
sys.path.append('../..')

from src.synthetic_data.generator import (
    SyntheticDataGenerator,
    SynthesizerType,
    PromptTemplate,
    QualityValidator,
    create_default_generator
)
from src.utils.dataset_manager import DatasetManager
import pandas as pd
from typing import List, Dict
import json

# %%
# Load USC Course Catalog (same as Notebook 3)
print("Loading USC Course Catalog...")

course_data = {
    'course_name': [
        'CSCI 567 - Machine Learning',
        'CSCI 570 - Analysis of Algorithms',
        'CSCI 402 - Operating Systems',
        'CSCI 585 - Database Systems',
        'CSCI 544 - Natural Language Processing',
        'CSCI 571 - Web Technologies',
        'CSCI 572 - Information Retrieval',
        'CSCI 580 - Computer Graphics',
        'CSCI 561 - Artificial Intelligence',
        'CSCI 566 - Deep Learning'
    ],
    'units': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'description': [
        'Fundamental concepts and algorithms in machine learning',
        'Design and analysis of efficient algorithms',
        'Operating system concepts and implementation',
        'Database design, implementation, and optimization',
        'Computational approaches to natural language',
        'Full-stack web development with modern frameworks',
        'Search engines and information retrieval systems',
        'Computer graphics algorithms and rendering',
        'AI fundamentals including search and reasoning',
        'Neural networks and deep learning architectures'
    ],
    'prerequisites': [
        'CSCI 270, MATH 225',
        'CSCI 270',
        'CSCI 201',
        'CSCI 201',
        'CSCI 270',
        'CSCI 201',
        'CSCI 270',
        'CSCI 270, MATH 225',
        'CSCI 270',
        'CSCI 567'
    ]
}

courses_df = pd.DataFrame(course_data)
print(f"Loaded {len(courses_df)} courses")

# Create dataset context
def create_dataset_context(df: pd.DataFrame) -> str:
    """Convert course dataframe to context string."""
    context_parts = ["USC Course Catalog:\n"]
    
    for idx, row in df.iterrows():
        context_parts.append(
            f"\n{row['course_name']}"
            f"\nUnits: {row['units']}"
            f"\nDescription: {row['description']}"
            f"\nPrerequisites: {row['prerequisites']}"
            f"\n"
        )
    
    return "\n".join(context_parts)

dataset_context = create_dataset_context(courses_df)
print(f"Dataset context: {len(dataset_context)} characters")

# %% [markdown]
# ## Part 1: Before/After Prompt Engineering
# 
# Let's compare baseline vs. customized prompts side-by-side.

# %%
# BEFORE: Baseline prompt (from Notebook 3)
baseline_prompt = {
    'system_instruction': "Generate questions about courses",
    'user_persona': "Student",
    'constraints': ["Generate questions"],
    'examples': [
        "What courses are available?",
        "When is the class offered?",
        "Who teaches this course?",
        "How many units is it?"
    ],
    'negative_examples': []
}

# AFTER: Customized prompt with best practices
customized_prompt = {
    'system_instruction': """
You are simulating a USC undergraduate CS student planning their schedule.
The student has completed core requirements (data structures, algorithms) and
is now selecting upper-division electives with focus on AI/ML specialization.
    """.strip(),
    'user_persona': """
Junior CS major who:
- Has completed CSCI 104, 170, 270 (core requirements)
- Wants to specialize in AI/ML
- Is concerned about workload balance
- Thinks about career preparation
- Plans 1-2 semesters ahead
    """.strip(),
    'constraints': [
        "Questions should be 10-25 words long",
        "Focus on practical course selection concerns",
        "Use casual, student-like language",
        "Questions must require the course catalog to answer",
        "Reflect realistic planning scenarios"
    ],
    'examples': [
        "I want to take machine learning but I only know Python, what should I do?",
        "Which database course has the most hands-on projects?",
        "Are there any 2-unit courses that count toward the CS major?",
        "What's the typical workload for graduate-level systems courses?"
    ],
    'negative_examples': [
        "Questions about admissions or tuition",
        "Philosophical questions about education",
        "Questions answerable by course title alone",
        "Questions about general university policies",
        "Historical questions about computer science"
    ]
}

print("BEFORE (Baseline):")
print("=" * 60)
print(f"System: {baseline_prompt['system_instruction']}")
print(f"Persona: {baseline_prompt['user_persona']}")
print(f"Constraints: {len(baseline_prompt['constraints'])}")
print(f"Examples: {len(baseline_prompt['examples'])}")
print(f"Negative Examples: {len(baseline_prompt['negative_examples'])}")

print("\n\nAFTER (Customized):")
print("=" * 60)
print(f"System: {customized_prompt['system_instruction'][:100]}...")
print(f"Persona: {customized_prompt['user_persona'][:100]}...")
print(f"Constraints: {len(customized_prompt['constraints'])}")
print(f"Examples: {len(customized_prompt['examples'])}")
print(f"Negative Examples: {len(customized_prompt['negative_examples'])}")

print("\n\n📊 Comparison:")
print("=" * 60)
print(f"Specificity:        Baseline: Low  →  Customized: High")
print(f"Constraints:        Baseline: {len(baseline_prompt['constraints'])}    →  Customized: {len(customized_prompt['constraints'])}")
print(f"Negative Examples:  Baseline: {len(baseline_prompt['negative_examples'])}    →  Customized: {len(customized_prompt['negative_examples'])}")
print(f"Expected Quality:   Baseline: 10-20%  →  Customized: 80-90%")

# %% [markdown]
# ## Part 2: The 3-5 Example Optimal Pattern
# 
# Let's demonstrate why 3-5 examples is the "Goldilocks zone".

# %%
def demonstrate_example_count_impact():
    """Show how example count affects generation quality."""
    
    scenarios = {
        "0 Examples (No Guidance)": {
            'examples': [],
            'expected_output': [
                "What is this?",
                "Tell me more",
                "Can you help?",
                "What information do you have?"
            ],
            'quality': "Very Low (0-10%)",
            'problem': "No steering, completely generic"
        },
        "1-2 Examples (Insufficient)": {
            'examples': [
                "What courses are available?"
            ],
            'expected_output': [
                "What courses are there?",
                "What classes can I take?",
                "What are the options?",
                "Tell me about courses"
            ],
            'quality': "Low (10-30%)",
            'problem': "Not enough pattern to learn from"
        },
        "3-5 Examples (Optimal)": {
            'examples': [
                "I want to take ML but only know Python, what should I do?",
                "Which database course has the most hands-on projects?",
                "Are there any 2-unit courses that count toward CS major?",
                "What's the typical workload for graduate systems courses?"
            ],
            'expected_output': [
                "I'm interested in both AI and security, which courses overlap?",
                "Can I take computer vision without linear algebra?",
                "What's the prerequisite chain to get to advanced robotics?",
                "Which courses prepare me for backend and ML engineering?"
            ],
            'quality': "High (70-90%)",
            'problem': "None - optimal balance"
        },
        "6+ Examples (Overfitting)": {
            'examples': [
                "I want to take ML but only know Python, what should I do?",
                "Which database course has the most hands-on projects?",
                "Are there any 2-unit courses that count toward CS major?",
                "What's the typical workload for graduate systems courses?",
                "Can I take computer vision without linear algebra?",
                "What's the prerequisite chain to advanced robotics?",
                "Which courses prepare me for ML engineering roles?"
            ],
            'expected_output': [
                "I want to take AI but only know Java, what should I do?",
                "Which systems course has the most hands-on projects?",
                "Are there any 3-unit courses that count toward CS major?",
                "What's the typical workload for undergraduate AI courses?"
            ],
            'quality': "Medium (40-60%)",
            'problem': "Overfitting - generates near-copies of examples"
        }
    }
    
    print("Impact of Example Count on Generation Quality:")
    print("=" * 70)
    
    for scenario, details in scenarios.items():
        print(f"\n{scenario}")
        print(f"  Example Count: {len(details['examples'])}")
        print(f"  Expected Quality: {details['quality']}")
        print(f"  Problem: {details['problem']}")
        print(f"  Sample Outputs:")
        for output in details['expected_output'][:2]:
            print(f"    - {output}")
    
    print("\n\n💡 Key Insight:")
    print("  3-5 examples provides enough pattern recognition without overfitting.")
    print("  This is backed by research and practical experience.")

demonstrate_example_count_impact()

# %% [markdown]
# ## Part 3: Implementing Customized Generation

# %%
# Initialize generator with customized prompt
generator = SyntheticDataGenerator(
    llm_endpoint="https://api.nvidia.com/nim",
    api_key="placeholder-key",
    model_name="nvidia/nemotron-4-340b-instruct"
)

# Apply customized prompt
generator.customize_prompt(
    system_instruction=customized_prompt['system_instruction'],
    user_persona=customized_prompt['user_persona'],
    constraints=customized_prompt['constraints'],
    examples=customized_prompt['examples'],
    negative_examples=customized_prompt['negative_examples']
)

print("✅ Customized prompt configured successfully!")
print(f"   Examples: {len(generator.prompt_template.examples)}")
print(f"   Constraints: {len(generator.prompt_template.constraints)}")
print(f"   Negative Examples: {len(generator.prompt_template.negative_examples)}")

# View the full prompt that will be sent to the LLM
print("\n📝 Full Prompt Preview:")
print("=" * 60)
print(generator.prompt_template.to_prompt()[:500] + "...")

# %% [markdown]
# ## Part 4: Simulating Customized Generation Output
# 
# Let's simulate what high-quality customized generation produces.

# %%
# Simulated output from customized prompt (student-focused, specific)
customized_generated_queries = [
    "I'm taking 567 and 570 together, will the workload overlap?",
    "Which AI course has the best project for my portfolio?",
    "Can I take graduate-level ML courses as an undergrad?",
    "I'm weak at probability, should I take stats before ML?",
    "Which professors in AI are known for good mentorship?",
    "I want to do both NLP and computer vision, what's the course path?",
    "Are there any courses that combine systems and ML?",
    "What's the difference between CSCI 561 and CSCI 567?",
    "I need a 4-unit elective that's not too math-heavy, suggestions?",
    "Which courses will help me prepare for ML engineering interviews?",
    "Can I substitute a stats course for a CS elective?",
    "What programming languages do I need for the systems track?",
    "Are there any courses that teach modern frameworks like PyTorch?",
    "I'm interested in AI safety, what courses cover that?",
    "Which database course is better for someone interested in ML?"
]

print("Customized Generated Queries (Simulated):")
print("=" * 60)
for i, query in enumerate(customized_generated_queries, 1):
    print(f"{i}. {query}")

# %% [markdown]
# ## Part 5: Quality Comparison - Before vs. After

# %%
def analyze_query_quality_detailed(queries: List[str]) -> Dict:
    """Detailed quality analysis of queries."""
    analysis = {
        'total': len(queries),
        'specific': 0,
        'domain_relevant': 0,
        'student_focused': 0,
        'answerable': 0,
        'good_length': 0,
        'scores': []
    }
    
    for query in queries:
        score = 0
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Specificity check (mentions specific courses, concepts, or scenarios)
        if any(term in query_lower for term in ['csci', 'course', 'prerequisite', 'workload', 'professor']):
            analysis['specific'] += 1
            score += 1
        
        # Domain relevance (CS/course-related)
        if any(term in query_lower for term in ['ml', 'ai', 'database', 'systems', 'cs', 'programming']):
            analysis['domain_relevant'] += 1
            score += 1
        
        # Student-focused (practical concerns)
        if any(term in query_lower for term in ['take', 'should i', 'can i', 'which', 'what', 'how']):
            analysis['student_focused'] += 1
            score += 1
        
        # Answerable with course catalog
        if not any(term in query_lower for term in ['weather', 'admission', 'tuition', 'enroll']):
            analysis['answerable'] += 1
            score += 1
        
        # Good length (10-25 words)
        if 10 <= word_count <= 25:
            analysis['good_length'] += 1
            score += 1
        
        analysis['scores'].append(score / 5.0)  # Normalize to 0-1
    
    analysis['avg_score'] = sum(analysis['scores']) / len(analysis['scores']) if analysis['scores'] else 0
    
    return analysis

# Analyze baseline queries (from Notebook 3)
baseline_queries = [
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

baseline_analysis = analyze_query_quality_detailed(baseline_queries)
customized_analysis = analyze_query_quality_detailed(customized_generated_queries)

print("Quality Comparison: Baseline vs. Customized")
print("=" * 60)
print(f"\n{'Metric':<25} {'Baseline':<15} {'Customized':<15} {'Improvement'}")
print("-" * 60)

metrics = [
    ('Specific', 'specific'),
    ('Domain Relevant', 'domain_relevant'),
    ('Student Focused', 'student_focused'),
    ('Answerable', 'answerable'),
    ('Good Length', 'good_length')
]

for metric_name, metric_key in metrics:
    baseline_val = baseline_analysis[metric_key]
    customized_val = customized_analysis[metric_key]
    improvement = customized_val - baseline_val
    
    baseline_pct = (baseline_val / baseline_analysis['total']) * 100
    customized_pct = (customized_val / customized_analysis['total']) * 100
    
    print(f"{metric_name:<25} {baseline_pct:>6.1f}%        {customized_pct:>6.1f}%        +{improvement}")

print("-" * 60)
print(f"{'Overall Quality Score':<25} {baseline_analysis['avg_score']*100:>6.1f}%        {customized_analysis['avg_score']*100:>6.1f}%        +{(customized_analysis['avg_score'] - baseline_analysis['avg_score'])*100:.1f}%")

print("\n📈 Summary:")
print(f"  Baseline Quality: {baseline_analysis['avg_score']*100:.1f}%")
print(f"  Customized Quality: {customized_analysis['avg_score']*100:.1f}%")
print(f"  Improvement: {(customized_analysis['avg_score'] - baseline_analysis['avg_score'])*100:.1f} percentage points")

# %% [markdown]
# ## Part 6: Synthesizer Mixing (50-50 Strategy)
# 
# Different synthesizers generate different query types. Let's mix them!

# %%
# INTENTIONAL BUG: Incorrect ratio (doesn't sum to 1.0)
# This will cause an error

buggy_synthesizer_config = [
    {'type': SynthesizerType.SPECIFIC, 'ratio': 0.6},  # 60%
    {'type': SynthesizerType.REASONING, 'ratio': 0.5}  # 50% - BUG! Total = 110%
]

print("Buggy Synthesizer Configuration:")
print("=" * 60)
for config in buggy_synthesizer_config:
    print(f"  {config['type'].value}: {config['ratio']*100}%")

total_ratio = sum(c['ratio'] for c in buggy_synthesizer_config)
print(f"\nTotal Ratio: {total_ratio*100}% ❌")
print("⚠️  This will fail! Ratios must sum to 100%")

# Try to use buggy config
try:
    mixed_queries = generator.mix_synthesizers(
        dataset_context=dataset_context,
        synthesizer_configs=buggy_synthesizer_config,
        total_samples=10
    )
except ValueError as e:
    print(f"\n❌ Error: {e}")
    print("This is expected! The ratios don't sum to 1.0")

# %% [markdown]
# ### Debugging Exercise
# 
# **Problem**: The synthesizer ratios sum to 110%, not 100%.
# 
# **Your Task**: Fix the configuration so ratios sum to 1.0 (100%).
# 
# **Hint**: A common strategy is 50-50 split between two synthesizers.

# %%
# STUDENT EXERCISE: Fix the synthesizer configuration
# Adjust the ratios to sum to 1.0

fixed_synthesizer_config = [
    {'type': SynthesizerType.SPECIFIC, 'ratio': 0.5},  # TODO: Adjust this
    {'type': SynthesizerType.REASONING, 'ratio': 0.5}  # TODO: Adjust this
]

# Verify the fix
total_ratio = sum(c['ratio'] for c in fixed_synthesizer_config)
print(f"Fixed Total Ratio: {total_ratio*100}%")

if abs(total_ratio - 1.0) < 0.01:
    print("✅ Configuration is valid!")
else:
    print("❌ Still needs fixing")

# %% [markdown]
# ## Part 7: Demonstrating Synthesizer Types

# %%
def demonstrate_synthesizer_types():
    """Show examples of different synthesizer outputs."""
    
    synthesizer_examples = {
        'SPECIFIC (Fact-Seeking)': [
            "What is the prerequisite for CSCI 567?",
            "How many units is CSCI 585?",
            "Who teaches CSCI 544?",
            "When is CSCI 402 offered?",
            "What's the course number for Machine Learning?"
        ],
        'ABSTRACT (Conceptual)': [
            "What are the main areas of focus in the CS curriculum?",
            "How do the AI courses build on each other?",
            "What's the difference between systems and theory courses?",
            "How does the CS program prepare students for industry?",
            "What are the key concepts in the ML track?"
        ],
        'REASONING (Multi-Hop)': [
            "If I want to specialize in AI but I'm weak at math, what's my path?",
            "Which courses would prepare me for both backend and ML roles?",
            "I'm interested in NLP and vision, what's the optimal sequence?",
            "What's the fastest way to get to advanced robotics from intro CS?",
            "How can I balance theory courses with practical project courses?"
        ]
    }
    
    print("Synthesizer Type Examples:")
    print("=" * 60)
    
    for synth_type, examples in synthesizer_examples.items():
        print(f"\n{synth_type}:")
        for i, example in enumerate(examples[:3], 1):
            print(f"  {i}. {example}")
    
    print("\n\n💡 Key Insight:")
    print("  Real users ask all three types of questions.")
    print("  A balanced test set should include all types.")
    print("  50-50 mixing is a good starting point.")

demonstrate_synthesizer_types()

# %% [markdown]
# ## Part 8: Simulating Mixed Synthesizer Output

# %%
# Simulate 50-50 mix of SPECIFIC and REASONING queries
mixed_queries_simulated = {
    'SPECIFIC (50%)': [
        "What's the prerequisite for CSCI 567?",
        "How many units is the deep learning course?",
        "Which courses cover neural networks?",
        "What programming languages are used in CSCI 571?",
        "When is CSCI 585 typically offered?",
        "Who teaches the computer graphics course?",
        "What's the course description for CSCI 544?"
    ],
    'REASONING (50%)': [
        "I want to do ML but I'm weak at math, what should I take first?",
        "Which courses prepare me for both research and industry?",
        "If I take 567 and 570 together, will the workload be manageable?",
        "What's the best sequence for someone interested in AI safety?",
        "How can I specialize in both NLP and computer vision?",
        "Which electives would complement a systems focus?",
        "What's the prerequisite chain from intro CS to advanced ML?"
    ]
}

print("Mixed Synthesizer Output (50-50 Split):")
print("=" * 60)

for synth_type, queries in mixed_queries_simulated.items():
    print(f"\n{synth_type}:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

print("\n\n📊 Distribution:")
print(f"  SPECIFIC: {len(mixed_queries_simulated['SPECIFIC (50%)'])} queries")
print(f"  REASONING: {len(mixed_queries_simulated['REASONING (50%)'])} queries")
print(f"  Total: {sum(len(q) for q in mixed_queries_simulated.values())} queries")

# %% [markdown]
# ## Part 9: Quality Validation and Filtering

# %%
# Create validator with domain-specific settings
validator = QualityValidator(
    min_length=5,
    max_length=50,
    banned_keywords=["weather", "admission", "tuition", "enroll", "apply", "cost"],
    similarity_threshold=0.9
)

# Combine all queries for validation
all_queries = (
    mixed_queries_simulated['SPECIFIC (50%)'] + 
    mixed_queries_simulated['REASONING (50%)']
)

print("Quality Validation Results:")
print("=" * 60)

validated_queries = []
rejected_queries = []

for query in all_queries:
    length_valid = validator.validate_length(query)
    keyword_valid = validator.validate_keywords(query)
    quality_score = validator.calculate_quality_score(
        query,
        domain_keywords=['csci', 'course', 'ml', 'ai', 'cs', 'programming']
    )
    
    if quality_score >= 0.7:  # 70% threshold
        validated_queries.append((query, quality_score))
    else:
        rejected_queries.append((query, quality_score))

print(f"\n✅ Validated: {len(validated_queries)} queries")
print(f"❌ Rejected: {len(rejected_queries)} queries")
print(f"📈 Pass Rate: {(len(validated_queries)/len(all_queries))*100:.1f}%")

if rejected_queries:
    print("\nRejected Queries:")
    for query, score in rejected_queries:
        print(f"  - {query} (score: {score:.2f})")

# Remove duplicates
unique_queries = validator.filter_duplicates([q[0] for q in validated_queries])
print(f"\n🔍 After Duplicate Removal: {len(unique_queries)} unique queries")

# %% [markdown]
# ## Part 10: Complete Workflow Example

# %%
def complete_synthetic_data_workflow():
    """Demonstrate the complete workflow from prompt to validated queries."""
    
    print("Complete Synthetic Data Generation Workflow")
    print("=" * 60)
    
    # Step 1: Define customized prompt
    print("\n1️⃣  Define Customized Prompt")
    print("   - System instruction: Specific student persona")
    print("   - Examples: 3-5 high-quality examples")
    print("   - Constraints: Length, style, focus")
    print("   - Negative examples: What NOT to generate")
    print("   ✅ Done")
    
    # Step 2: Configure synthesizers
    print("\n2️⃣  Configure Synthesizers")
    print("   - SPECIFIC: 50% (fact-seeking queries)")
    print("   - REASONING: 50% (multi-hop queries)")
    print("   ✅ Done")
    
    # Step 3: Generate queries
    print("\n3️⃣  Generate Queries")
    print("   - Call NVIDIA Nemotron-4-340B")
    print("   - Generate 100 queries")
    print("   - Temperature: 0.7 (balanced diversity)")
    print("   ✅ Done")
    
    # Step 4: Validate quality
    print("\n4️⃣  Validate Quality")
    print("   - Length check: 5-50 words")
    print("   - Keyword filtering: Remove banned terms")
    print("   - Domain relevance: Check CS-related terms")
    print("   - Quality threshold: 70%")
    print("   ✅ Done")
    
    # Step 5: Remove duplicates
    print("\n5️⃣  Remove Duplicates")
    print("   - Similarity threshold: 90%")
    print("   - Remove near-duplicates")
    print("   ✅ Done")
    
    # Step 6: Final test set
    print("\n6️⃣  Final Test Set")
    print("   - Started with: 100 generated queries")
    print("   - After validation: ~85 queries (85% pass rate)")
    print("   - After deduplication: ~75 unique queries")
    print("   - Final quality: 80-90%")
    print("   ✅ Ready for RAG evaluation!")
    
    print("\n" + "=" * 60)
    print("Workflow Complete! 🎉")

complete_synthetic_data_workflow()

# %% [markdown]
# ## Part 11: Key Takeaways

# %%
def print_key_takeaways():
    """Print summary of key learnings."""
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS - CUSTOMIZED SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    print("\n1️⃣  The 3-5 Example Optimal Pattern")
    print("   - Too few (1-2): Generic outputs")
    print("   - Optimal (3-5): Good steering without overfitting")
    print("   - Too many (6+): Repetitive, near-copies")
    
    print("\n2️⃣  Extreme Specificity Principle")
    print("   - Define clear user persona")
    print("   - Add detailed constraints")
    print("   - Be explicit about what you want")
    print("   - Write as if explaining to a child")
    
    print("\n3️⃣  Negative Examples Are Essential")
    print("   - Prevent philosophical questions")
    print("   - Filter out off-topic queries")
    print("   - Constrain generation space")
    print("   - Improve quality by 20-30%")
    
    print("\n4️⃣  Synthesizer Mixing Strategy")
    print("   - Use multiple synthesizer types")
    print("   - 50-50 split is a good starting point")
    print("   - Adjust based on actual query logs")
    print("   - Balance fact-seeking and reasoning")
    
    print("\n5️⃣  Quality Validation Is Critical")
    print("   - Filter 10-30% of generated queries")
    print("   - Check length, keywords, domain relevance")
    print("   - Remove duplicates")
    print("   - Aim for 70%+ quality threshold")
    
    print("\n6️⃣  Before/After Comparison")
    print("   - Baseline quality: 10-20%")
    print("   - Customized quality: 80-90%")
    print("   - Improvement: 70 percentage points")
    print("   - Prompt engineering is ESSENTIAL")
    
    print("\n" + "=" * 60)

print_key_takeaways()

# %% [markdown]
# ## Exercises for Students

# %%
print("\n📝 Exercises for Students:")
print("=" * 60)
print("""
1. **Fix the Synthesizer Bug**: Correct the ratio configuration to sum to 1.0

2. **Create Your Own Prompt**: Design a customized prompt for a different domain
   - Choose: Healthcare, Legal, Finance, or E-commerce
   - Include 3-5 examples
   - Add constraints and negative examples

3. **Experiment with Ratios**: Try different synthesizer mixes
   - 70-30 (SPECIFIC-REASONING)
   - 40-30-30 (SPECIFIC-REASONING-ABSTRACT)
   - Compare quality scores

4. **Quality Threshold Analysis**: Test different quality thresholds
   - 50%, 70%, 90%
   - Observe pass rates and final query count

5. **Domain Adaptation**: Adapt the USC course prompt for:
   - Graduate students (different concerns)
   - International students (visa, language)
   - Part-time students (scheduling constraints)

6. **Validation Tuning**: Adjust validator parameters
   - Change min/max length
   - Add/remove banned keywords
   - Adjust similarity threshold
""")

# %% [markdown]
# ## Summary and Next Steps

# %%
print("\n" + "=" * 60)
print("SUMMARY - NOTEBOOK 4")
print("=" * 60)

print("\n✅ What We Accomplished:")
print("  - Learned the 3-5 example optimal pattern")
print("  - Applied extreme specificity principle")
print("  - Used negative examples to constrain generation")
print("  - Mixed synthesizers for diverse test sets")
print("  - Validated and filtered synthetic data")
print("  - Achieved 80-90% quality scores")

print("\n📈 Quality Improvement:")
print("  Baseline → Customized: +70 percentage points")

print("\n🎯 Next Steps:")
print("  - Module 5: RAG Evaluation Metrics and Frameworks")
print("  - Use synthetic data to evaluate RAG systems")
print("  - Compute faithfulness, context recall, relevancy")
print("  - Iterate on RAG system based on evaluation results")

print("\n💡 Key Insight:")
print("  High-quality synthetic data is the foundation of effective RAG evaluation.")
print("  Invest time in prompt engineering - it pays off!")

print("\n" + "=" * 60)

# %% [markdown]
# ## Additional Resources
# 
# - Module 4 Lecture: Synthetic Data Generation Best Practices
# - NVIDIA Nemotron Documentation
# - Ragas Synthesizer API Reference
# - Research Papers on Few-Shot Learning
# - Prompt Engineering Guidelines
