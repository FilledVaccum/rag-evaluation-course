"""
Module 4: Synthetic Test Data Generation
Evaluating RAG and Semantic Search Systems Course

This module covers test-driven development for LLMs and RAG systems,
focusing on generating high-quality synthetic test data using NVIDIA Nemotron-4-340B.

Learning Objectives:
- Understand test-driven development principles for LLMs and RAG
- Master LLM-based synthetic data generation techniques
- Apply prompt engineering with the 3-5 example optimal pattern
- Generate domain-specific test data reflecting realistic user queries
- Validate and filter synthetic data for quality

Requirements Coverage: 6.1, 6.2, 6.3, 14.1
Certification Alignment: Evaluation and Tuning (13%), Agent Development (15%)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class PromptEngineeringPrinciple(Enum):
    """Core principles for effective prompt engineering."""
    EXTREME_SPECIFICITY = "Write prompts as if explaining to a child"
    OPTIMAL_EXAMPLES = "Use exactly 3-5 examples for steering"
    EXPLICIT_NEGATIVES = "State what NOT to generate"
    USER_PERSONAS = "Specify realistic user scenarios"
    ITERATIVE_VALIDATION = "Test prompts multiple times"


@dataclass
class SyntheticDataConcept:
    """Represents a key concept in synthetic data generation."""
    concept_name: str
    definition: str
    importance: str
    example: str
    common_pitfall: Optional[str] = None


# Module 4 Lecture Content
MODULE_4_LECTURE = {
    "module_number": 4,
    "title": "Synthetic Test Data Generation",
    "duration_minutes": 90,
    "learning_objectives": [
        "Understand test-driven development principles for LLMs and RAG systems",
        "Master LLM-based synthetic data generation using NVIDIA Nemotron-4-340B",
        "Apply prompt engineering techniques with 3-5 example optimal pattern",
        "Generate domain-specific queries reflecting realistic user behavior",
        "Validate and filter synthetic data for quality assurance"
    ],
    "exam_domains": {
        "Evaluation and Tuning": 0.13,
        "Agent Development": 0.15
    }
}


# Section 1: Test-Driven Development for LLMs and RAG
TDD_FOR_LLM_CONCEPTS = [
    SyntheticDataConcept(
        concept_name="Test-Driven Development for RAG",
        definition="""
        Test-driven development (TDD) for RAG systems means creating comprehensive 
        test datasets BEFORE deploying to production. Unlike traditional software where 
        you write tests for deterministic functions, RAG systems require test sets that 
        capture the diversity and complexity of real user queries.
        """,
        importance="""
        RAG systems are non-deterministic and context-dependent. Without proper test 
        data, you cannot:
        - Measure baseline performance
        - Detect regressions when making changes
        - Compare different retrieval or generation strategies
        - Validate that the system works for edge cases
        
        Production failures in RAG are expensive and damage user trust.
        """,
        example="""
        Example: A university RAG system for course information
        
        Without TDD:
        - Deploy system with general embeddings
        - Users complain about irrelevant results
        - No baseline to measure improvements
        - Trial-and-error optimization
        
        With TDD:
        - Generate 100+ synthetic student queries before deployment
        - Measure baseline retrieval accuracy: 65%
        - Test different embedding models
        - Validate improvement to 85% before production
        """,
        common_pitfall="Treating RAG evaluation as optional or post-deployment only"
    ),
    SyntheticDataConcept(
        concept_name="Why Manual Test Data is Insufficient",
        definition="""
        Manually creating test queries is time-consuming, biased toward obvious cases,
        and cannot scale to cover the diversity of real user behavior. A production RAG
        system may receive thousands of unique query patterns.
        """,
        importance="""
        Manual test data limitations:
        - Developer bias: You create queries you know the system can handle
        - Limited coverage: 10-20 manual queries vs. 1000+ real query patterns
        - Time-intensive: Hours to create what LLMs generate in minutes
        - Static: Doesn't evolve as your knowledge base changes
        """,
        example="""
        Manual approach:
        Query 1: "What is the prerequisite for CSCI 567?"
        Query 2: "When is CSCI 567 offered?"
        Query 3: "Who teaches CSCI 567?"
        
        Result: All queries follow same pattern, miss edge cases like:
        - "I want to take machine learning but I only know Python, what should I do?"
        - "Which AI courses don't require linear algebra?"
        - "What's the easiest way to fulfill my CS elective requirement?"
        """,
        common_pitfall="Creating test queries that are too similar to each other"
    ),
    
    SyntheticDataConcept(
        concept_name="LLM-Based Synthetic Data Generation",
        definition="""
        Using large language models (LLMs) to automatically generate diverse, realistic
        test queries based on your knowledge base. The LLM acts as a "user simulator"
        that creates questions a real user might ask.
        """,
        importance="""
        LLM-based generation provides:
        - Scale: Generate 100s-1000s of queries in minutes
        - Diversity: Different phrasings, complexity levels, query types
        - Domain adaptation: Queries reflect your specific knowledge base
        - Continuous generation: Create new test sets as data evolves
        """,
        example="""
        Input: USC Course Catalog (1000 courses)
        LLM: NVIDIA Nemotron-4-340B
        
        Generated queries (diverse patterns):
        1. "What programming languages do I need to know for the computer vision course?"
        2. "I'm interested in both AI and databases, which courses overlap?"
        3. "Are there any 2-unit courses that count toward the CS major?"
        4. "Which professors are known for being good at explaining difficult concepts?"
        5. "What's the typical workload for graduate-level systems courses?"
        
        Notice: Different complexity, specificity, and user intent
        """,
        common_pitfall="Using generic LLMs without domain-specific steering"
    )
]


# Section 2: Prompt Engineering for Synthetic Data
PROMPT_ENGINEERING_CONCEPTS = [
    SyntheticDataConcept(
        concept_name="The 3-5 Example Optimal Pattern",
        definition="""
        Research and practice show that providing exactly 3-5 examples in your prompt
        achieves the best balance between steering the LLM's behavior and avoiding
        overfitting to specific patterns.
        """,
        importance="""
        Why 3-5 examples?
        - Too few (0-2): LLM generates generic, unfocused queries
        - Optimal (3-5): LLM learns the pattern without memorizing
        - Too many (6+): LLM overfits, generates near-copies of examples
        
        This is the "Goldilocks zone" for few-shot learning.
        """,
        example="""
        BEFORE (0 examples - over-generic):
        Prompt: "Generate questions about USC courses"
        
        Output:
        - "What courses are available?"
        - "Tell me about computer science"
        - "What is the curriculum?"
        
        AFTER (4 examples - domain-specific):
        Prompt: "Generate questions like these:
        1. 'I need a course that covers both theory and implementation of neural networks'
        2. 'Which database course has the most hands-on projects?'
        3. 'Are there any courses that combine NLP and computer vision?'
        4. 'What's the prerequisite chain to get to advanced robotics?'
        
        Generate similar questions."
        
        Output:
        - "I want to learn reinforcement learning but I'm weak at math, what should I take first?"
        - "Which courses let me work with real industry datasets?"
        - "Are there any cross-listed courses between CS and business?"
        """,
        common_pitfall="Providing too many examples (6+) leading to repetitive outputs"
    ),
    
    SyntheticDataConcept(
        concept_name="Extreme Specificity Principle",
        definition="""
        Write prompts as if explaining to someone with no context about your domain.
        Be explicit about query style, length, complexity, user persona, and what NOT
        to generate.
        """,
        importance="""
        LLMs are powerful but need clear guidance. Vague prompts like "generate questions"
        produce vague results. Specific prompts produce targeted, useful test data.
        """,
        example="""
        VAGUE PROMPT:
        "Generate questions about courses"
        
        SPECIFIC PROMPT:
        '''
        You are simulating a USC undergraduate student planning their schedule.
        Generate questions that:
        - Are 10-25 words long
        - Focus on practical concerns (workload, prerequisites, career relevance)
        - Use casual, student-like language
        - Ask about course selection, not general university policies
        
        DO NOT generate:
        - Questions about admissions or tuition
        - Philosophical questions about education
        - Questions answerable by the course title alone
        
        Examples:
        [3-5 examples here]
        '''
        
        Result: Focused, realistic student queries
        """,
        common_pitfall="Assuming the LLM knows your domain context"
    ),
    SyntheticDataConcept(
        concept_name="Explicit Negative Examples",
        definition="""
        Explicitly state what you DON'T want the LLM to generate. This prevents
        common failure modes like overly generic or off-topic queries.
        """,
        importance="""
        Without negative examples, LLMs often generate:
        - Philosophical questions ("What is the meaning of education?")
        - Overly broad questions ("Tell me about computer science")
        - Off-domain questions ("How do I apply to USC?")
        
        Negative examples act as guardrails.
        """,
        example="""
        WITHOUT NEGATIVES:
        Generated: "What is the purpose of higher education?"
        Generated: "How has computer science evolved over time?"
        Generated: "What makes a good university?"
        
        WITH NEGATIVES:
        Prompt addition:
        '''
        DO NOT generate:
        - Philosophical or abstract questions
        - Historical questions about the field
        - Questions about university rankings or reputation
        - Questions that don't require the course catalog to answer
        '''
        
        Generated: "Which courses cover distributed systems and have group projects?"
        Generated: "I'm interested in AI safety, what's the course progression?"
        Generated: "Do any courses teach Rust or Go instead of just C++?"
        """,
        common_pitfall="Only providing positive examples without constraints"
    ),
    
    SyntheticDataConcept(
        concept_name="User Persona Specification",
        definition="""
        Define the specific type of user whose queries you're simulating. Different
        personas ask different types of questions.
        """,
        importance="""
        A prospective student asks different questions than:
        - Current undergraduate planning next semester
        - Graduate student looking for research opportunities
        - Industry professional seeking specific skills
        - Academic advisor helping students
        
        Persona clarity = query relevance
        """,
        example="""
        PERSONA 1: Undergraduate CS major (junior year)
        Generated queries:
        - "I need one more elective, what's interesting but not too time-consuming?"
        - "Which courses will help me prepare for software engineering interviews?"
        - "Can I take graduate courses as an undergrad?"
        
        PERSONA 2: Prospective graduate student
        Generated queries:
        - "Which faculty members work on computer vision and are taking students?"
        - "What's the typical course load for first-year PhD students?"
        - "Are there courses that combine machine learning with robotics?"
        
        Same knowledge base, different query patterns based on persona.
        """,
        common_pitfall="Mixing multiple personas in one generation run"
    )
]


# Section 3: NVIDIA Nemotron-4-340B for Synthetic Data
NEMOTRON_CONCEPTS = [
    SyntheticDataConcept(
        concept_name="NVIDIA Nemotron-4-340B Overview",
        definition="""
        Nemotron-4-340B is NVIDIA's large language model optimized for synthetic data
        generation. It's specifically trained to follow complex instructions and generate
        diverse, high-quality outputs for training and evaluation purposes.
        """,
        importance="""
        Why Nemotron for synthetic data:
        - Instruction following: Better at adhering to complex prompt constraints
        - Diversity: Generates more varied outputs than general-purpose LLMs
        - Quality: Produces coherent, realistic queries
        - Scale: Can generate thousands of queries efficiently
        - NVIDIA ecosystem: Integrates with NIM for easy deployment
        """,
        example="""
        Nemotron-4-340B via NVIDIA NIM:
        
        Input: Prompt with 4 examples of student queries + USC course catalog context
        Output: 100 diverse, realistic student questions in 2 minutes
        
        Quality metrics:
        - 95% queries are domain-relevant
        - 80% queries require course catalog to answer
        - 70% queries show realistic student concerns
        - Low repetition rate (< 5% near-duplicates)
        """,
        common_pitfall="Using general-purpose LLMs not optimized for data generation"
    ),
    SyntheticDataConcept(
        concept_name="Synthesizer Types in Ragas",
        definition="""
        Ragas provides different synthesizer types that generate different query patterns:
        - AbstractQuerySynthesizer: High-level, conceptual questions
        - SpecificQuerySynthesizer: Detailed, fact-seeking questions
        - ReasoningQuerySynthesizer: Multi-hop reasoning questions
        """,
        importance="""
        Real users ask different types of questions. Using multiple synthesizers ensures
        your test set covers:
        - Simple fact lookup ("When is CSCI 567 offered?")
        - Conceptual understanding ("What's the difference between ML and AI courses?")
        - Complex reasoning ("What's the fastest path to take advanced robotics?")
        """,
        example="""
        AbstractQuerySynthesizer:
        - "What are the main areas of focus in the CS curriculum?"
        - "How do the systems courses build on each other?"
        
        SpecificQuerySynthesizer:
        - "What is the prerequisite for CSCI 567?"
        - "How many units is CSCI 402?"
        
        ReasoningQuerySynthesizer:
        - "If I want to specialize in AI but I'm weak at math, what's my course path?"
        - "Which courses would prepare me for both backend and ML engineering roles?"
        """,
        common_pitfall="Using only one synthesizer type, missing query diversity"
    ),
    
    SyntheticDataConcept(
        concept_name="Mixing Synthesizers (50-50 Strategy)",
        definition="""
        Combining multiple synthesizers in equal proportions (e.g., 50% specific, 50%
        reasoning) creates a balanced test set that reflects real user query distribution.
        """,
        importance="""
        Production RAG systems face diverse query types. A test set with only simple
        queries won't catch failures on complex reasoning. A test set with only complex
        queries won't catch simple fact-lookup failures.
        
        50-50 mixing is a good starting point; adjust based on your actual query logs.
        """,
        example="""
        Test set generation strategy:
        
        Generate 100 queries:
        - 50 from SpecificQuerySynthesizer (fact lookup)
        - 50 from ReasoningQuerySynthesizer (multi-hop)
        
        Result: Balanced coverage
        - Tests simple retrieval accuracy
        - Tests complex reasoning capability
        - Reflects realistic query distribution
        
        Alternative: 40-30-30 (Specific-Reasoning-Abstract) based on query logs
        """,
        common_pitfall="Not analyzing actual user query distribution before mixing"
    )
]


# Section 4: Quality Validation and Filtering
QUALITY_VALIDATION_CONCEPTS = [
    SyntheticDataConcept(
        concept_name="Quality Metrics for Synthetic Data",
        definition="""
        Not all generated queries are useful. Quality validation ensures your test set
        contains realistic, answerable, diverse queries that actually test your RAG system.
        """,
        importance="""
        Poor quality synthetic data leads to:
        - False confidence (passing tests that don't reflect reality)
        - Missed bugs (not testing actual failure modes)
        - Wasted compute (evaluating on irrelevant queries)
        
        Quality validation is essential, not optional.
        """,
        example="""
        Quality metrics to check:
        
        1. Domain Relevance: Does query relate to your knowledge base?
           Good: "Which courses cover neural networks?"
           Bad: "What's the weather like in Los Angeles?"
        
        2. Answerability: Can your knowledge base answer this?
           Good: "What are the prerequisites for CSCI 567?"
           Bad: "What will future AI courses look like in 2030?"
        
        3. Specificity: Is query specific enough to be useful?
           Good: "Which database course has the most hands-on projects?"
           Bad: "Tell me about courses"
        
        4. Diversity: Are queries sufficiently different?
           Good: Mix of fact-lookup, comparison, reasoning
           Bad: All queries follow same template
        """,
        common_pitfall="Skipping quality validation and using all generated queries"
    ),
    SyntheticDataConcept(
        concept_name="Filtering Strategies",
        definition="""
        Automated and manual filtering to remove low-quality queries from your test set.
        Typical filtering removes 10-30% of generated queries.
        """,
        importance="""
        Filtering improves test set quality:
        - Removes off-topic queries
        - Eliminates unanswerable questions
        - Reduces near-duplicates
        - Ensures minimum complexity threshold
        """,
        example="""
        Filtering pipeline:
        
        1. Keyword filtering: Remove queries with banned terms
           Remove: "weather", "admission", "tuition" (off-topic)
        
        2. Length filtering: Remove too short or too long
           Remove: < 5 words or > 50 words
        
        3. Similarity filtering: Remove near-duplicates
           Remove: Queries with > 90% semantic similarity
        
        4. Answerability check: Use LLM to verify
           Prompt: "Can this question be answered using course catalog? Yes/No"
           Remove: Questions marked "No"
        
        Result: 100 generated → 75 high-quality queries after filtering
        """,
        common_pitfall="Over-filtering and losing valuable edge cases"
    )
]


# Section 5: Before/After Prompt Engineering Examples
PROMPT_ENGINEERING_EXAMPLES = {
    "example_1": {
        "title": "From Generic to Domain-Specific",
        "before": {
            "prompt": "Generate questions about courses",
            "output": [
                "What courses are available?",
                "Tell me about computer science",
                "What is the curriculum?",
                "How do I enroll?",
                "What are the requirements?"
            ],
            "problems": [
                "Too generic and broad",
                "Not specific to USC course catalog",
                "Mix of different question types",
                "Some unanswerable with course data"
            ]
        },
        "after": {
            "prompt": """
You are simulating a USC undergraduate CS student planning their schedule.

Generate questions that:
- Focus on course selection, prerequisites, and scheduling
- Use casual student language
- Are 10-25 words long
- Require the course catalog to answer

DO NOT generate questions about:
- Admissions or tuition
- General university policies
- Abstract educational philosophy

Examples:
1. "I want to take machine learning but I only know Python, what should I do?"
2. "Which database course has the most hands-on projects?"
3. "Are there any 2-unit courses that count toward the CS major?"
4. "What's the typical workload for graduate-level systems courses?"

Generate 5 similar questions.
            """,
            "output": [
                "I'm interested in both AI and security, which courses overlap?",
                "Which professors are known for being good at explaining difficult concepts?",
                "Can I take the computer vision course without linear algebra?",
                "What's the prerequisite chain to get to advanced robotics?",
                "Are there any courses that teach Rust or Go instead of C++?"
            ],
            "improvements": [
                "Domain-specific (USC CS courses)",
                "Realistic student concerns",
                "Appropriate length and complexity",
                "All answerable with course catalog"
            ]
        }
    },
    "example_2": {
        "title": "From Philosophical to Practical",
        "before": {
            "prompt": "Generate questions a student might ask about education",
            "output": [
                "What is the purpose of higher education?",
                "How has computer science evolved over time?",
                "What makes a good university?",
                "Why is learning important?",
                "What are the benefits of a CS degree?"
            ],
            "problems": [
                "Philosophical, not practical",
                "Don't require course catalog",
                "Too abstract for RAG system",
                "Not actionable for students"
            ]
        },
        "after": {
            "prompt": """
You are simulating a USC CS student making practical course decisions.

Generate questions about:
- Specific courses and their content
- Prerequisites and course sequences
- Workload and difficulty
- Career preparation

DO NOT generate:
- Philosophical questions about education
- Historical questions about CS
- Questions about university reputation

Examples:
1. "Which courses will help me prepare for software engineering interviews?"
2. "I need one more elective, what's interesting but not too time-consuming?"
3. "Do any courses teach modern web frameworks like React?"
4. "What's the difference between CSCI 570 and CSCI 670?"

Generate 5 similar questions.
            """,
            "output": [
                "Which courses cover distributed systems and have group projects?",
                "I'm interested in AI safety, what's the course progression?",
                "Are there any courses that combine NLP and computer vision?",
                "What programming languages do I need to know for the systems track?",
                "Which courses have the best industry connections for internships?"
            ],
            "improvements": [
                "Practical, actionable questions",
                "Require specific course information",
                "Reflect real student decision-making",
                "Testable with RAG system"
            ]
        }
    },
    
    "example_3": {
        "title": "Adding Persona Specificity",
        "before": {
            "prompt": "Generate questions about USC courses with these examples:\n1. What courses are available?\n2. When is the class offered?\n3. Who teaches this course?",
            "output": [
                "What courses are available in computer science?",
                "When is CSCI 567 offered?",
                "Who teaches CSCI 402?",
                "What courses are available in fall?",
                "When is the machine learning class?"
            ],
            "problems": [
                "All follow same template",
                "No diversity in query types",
                "Missing student context",
                "Too simple, not realistic"
            ]
        },
        "after": {
            "prompt": """
You are simulating a USC junior CS major who:
- Has completed core requirements (data structures, algorithms)
- Is planning their senior year
- Wants to specialize in AI/ML
- Is concerned about workload balance

Generate questions that reflect this student's planning process.

Examples:
1. "I want to take 3 AI courses next semester, is that too much?"
2. "Which ML course should I take first: 567 or 570?"
3. "Can I substitute a stats course for a CS elective?"
4. "Are there any AI courses that don't require heavy math?"

Generate 5 similar questions.
            """,
            "output": [
                "I'm taking 567 and 570 together, will the workload overlap?",
                "Which AI course has the best project for my portfolio?",
                "Can I take graduate-level ML courses as an undergrad?",
                "I'm weak at probability, should I take a stats course before ML?",
                "Which professors in AI are known for good mentorship?"
            ],
            "improvements": [
                "Persona-driven queries",
                "Realistic planning concerns",
                "Diverse question types",
                "Context-aware (junior, AI focus)"
            ]
        }
    }
}


# Module Summary
MODULE_4_SUMMARY = """
Module 4: Synthetic Test Data Generation - Key Takeaways

1. Test-Driven Development for RAG:
   - Create test datasets BEFORE production deployment
   - Manual test data is insufficient for scale and diversity
   - LLM-based generation provides scalable, diverse test sets

2. The 3-5 Example Optimal Pattern:
   - 3-5 examples achieve best steering without overfitting
   - Too few = generic outputs, too many = repetitive outputs
   - This is the "Goldilocks zone" for few-shot learning

3. Prompt Engineering Principles:
   - Extreme Specificity: Write as if explaining to a child
   - Explicit Negatives: State what NOT to generate
   - User Personas: Define who you're simulating
   - Iterative Validation: Test and refine prompts

4. NVIDIA Nemotron-4-340B:
   - Optimized for synthetic data generation
   - Better instruction following than general LLMs
   - Integrates with NVIDIA NIM ecosystem

5. Quality Validation:
   - Not all generated queries are useful
   - Filter for domain relevance, answerability, specificity
   - Typical filtering removes 10-30% of generated queries
   - Quality > Quantity for test sets

6. Synthesizer Mixing:
   - Use multiple synthesizer types (Specific, Reasoning, Abstract)
   - 50-50 mixing is a good starting point
   - Adjust based on actual user query distribution

Common Pitfalls to Avoid:
- Over-generic prompts without domain context
- Using only 1-2 examples or 6+ examples
- Skipping quality validation
- Using single synthesizer type
- Not defining user personas
- Treating evaluation as optional

Next Steps:
- Notebook 1: Generate baseline synthetic data (out-of-the-box)
- Notebook 2: Customize prompts for domain-specific queries
- Practice: Iterate on prompts to improve quality
"""


def get_lecture_content() -> Dict:
    """Return complete lecture content for Module 4."""
    return {
        "module_info": MODULE_4_LECTURE,
        "tdd_concepts": TDD_FOR_LLM_CONCEPTS,
        "prompt_engineering": PROMPT_ENGINEERING_CONCEPTS,
        "nemotron_concepts": NEMOTRON_CONCEPTS,
        "quality_validation": QUALITY_VALIDATION_CONCEPTS,
        "before_after_examples": PROMPT_ENGINEERING_EXAMPLES,
        "summary": MODULE_4_SUMMARY
    }


if __name__ == "__main__":
    # Display module overview
    content = get_lecture_content()
    print(f"Module {content['module_info']['module_number']}: {content['module_info']['title']}")
    print(f"Duration: {content['module_info']['duration_minutes']} minutes")
    print(f"\nLearning Objectives:")
    for obj in content['module_info']['learning_objectives']:
        print(f"  - {obj}")
    print(f"\nCertification Alignment:")
    for domain, weight in content['module_info']['exam_domains'].items():
        print(f"  - {domain}: {weight*100}%")
