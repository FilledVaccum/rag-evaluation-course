"""
Before/After Optimization Examples for RAG Evaluation Course
Demonstrating improvements in prompt engineering, chunking, embeddings, and metrics
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class OptimizationExample:
    """Represents a before/after optimization example"""
    category: str
    title: str
    before: Dict[str, str]
    after: Dict[str, str]
    improvement_metrics: Dict[str, str]
    explanation: str
    key_takeaways: List[str]


# Prompt Engineering Optimization Examples
PROMPT_ENGINEERING_EXAMPLES = [
    OptimizationExample(
        category="Prompt Engineering",
        title="Synthetic Data Generation - Fixing Over-Generic Questions",
        before={
            "prompt": """Generate questions about the USC course catalog.""",
            "sample_output": """
1. What is the meaning of education?
2. How does learning shape society?
3. What are the philosophical implications of knowledge?
4. Why do universities exist?
5. What is the purpose of higher education?
""",
            "quality_score": "2/10 - Completely generic, not actionable"
        },
        after={
            "prompt": """You are generating questions that USC undergraduate students would ask when planning their course schedule.

Generate 10 specific, actionable questions about course prerequisites, schedules, units, and instructors.

Examples of good questions:
1. "What are the prerequisites for CSCI 567 Machine Learning?"
2. "When does CSCI 270 meet this semester?"
3. "How many units is MATH 225 Linear Algebra?"
4. "Who is teaching CSCI 360 this fall?"
5. "Does CSCI 567 have a lab component?"

Do NOT generate:
- Philosophical questions about education
- Abstract questions about learning
- Questions about university policies
- Questions not specific to courses

Generate questions now:""",
            "sample_output": """
1. What are the prerequisites for CSCI 567?
2. When does CSCI 270 meet on Tuesdays?
3. How many units is MATH 225 worth?
4. Is CSCI 360 offered in the spring semester?
5. What time does CSCI 567 start?
6. Does MATH 225 require calculus as a prerequisite?
7. Who is the instructor for CSCI 270?
8. Is there a lab section for CSCI 360?
9. How many hours per week is CSCI 567?
10. What building is MATH 225 taught in?
""",
            "quality_score": "9/10 - Specific, actionable, student-focused"
        },
        improvement_metrics={
            "specificity": "Improved from 10% to 95% specific questions",
            "actionability": "Improved from 0% to 100% actionable questions",
            "domain_relevance": "Improved from 20% to 98% domain-relevant",
            "usability": "Test set now usable for evaluation (was unusable before)"
        },
        explanation="""
The key improvements:
1. Added user persona: "USC undergraduate students planning schedule"
2. Provided 5 concrete examples (3-5 example rule)
3. Explicitly stated what NOT to generate (negative examples)
4. Specified question types: prerequisites, schedules, units, instructors
5. Used imperative language: "Generate questions now"

This demonstrates the power of prompt engineering. Same LLM, dramatically different output.
""",
        key_takeaways=[
            "Be extremely specific about user persona and use case",
            "Use 3-5 concrete examples to establish pattern",
            "Explicitly state what NOT to generate (negative examples)",
            "Specify output format and constraints clearly",
            "Iterate: first attempt won't be perfect"
        ]
    ),
    
    OptimizationExample(
        category="Prompt Engineering",
        title="LLM-as-a-Judge - Improving Faithfulness Evaluation",
        before={
            "prompt": """Is this response faithful to the context?

Context: {context}
Response: {response}

Answer: Yes or No""",
            "sample_output": """Yes""",
            "quality_score": "3/10 - No explanation, inconsistent, not actionable"
        },
        after={
            "prompt": """You are evaluating whether a response is faithful to the provided context.

A response is faithful if:
1. All claims in the response are supported by the context
2. No information is added that isn't in the context
3. No claims contradict the context

Task: Extract claims from the response, then verify each claim against the context.

Context: {context}

Response: {response}

Evaluation Process:
1. Extract claims from response (list each claim)
2. For each claim, check if supported by context (Yes/No)
3. Calculate faithfulness score: (supported_claims / total_claims)

Output format (JSON):
{{
  "claims": ["claim 1", "claim 2", ...],
  "supported": [true, false, ...],
  "faithfulness_score": 0.0-1.0,
  "explanation": "Brief explanation"
}}

Evaluate now:""",
            "sample_output": """{
  "claims": [
    "CSCI 567 is a 4-unit course",
    "The course meets on Mondays and Wednesdays",
    "Prerequisites are CSCI 270 and MATH 225"
  ],
  "supported": [true, false, true],
  "faithfulness_score": 0.67,
  "explanation": "Claim 2 is not supported - context doesn't mention meeting days"
}""",
            "quality_score": "9/10 - Structured, explainable, actionable"
        },
        improvement_metrics={
            "consistency": "Improved from 60% to 95% consistent scoring",
            "explainability": "Now provides detailed explanation (was binary yes/no)",
            "actionability": "Can identify specific unsupported claims",
            "correlation_with_human": "Improved from 0.65 to 0.92 correlation"
        },
        explanation="""
The key improvements:
1. Clear definition of faithfulness (3 criteria)
2. Multi-stage evaluation process (extract claims → verify → score)
3. Structured JSON output (parseable, consistent)
4. Explanation field (debugging and trust)
5. Explicit scoring formula

This demonstrates how prompt engineering improves LLM-as-a-Judge reliability.
""",
        key_takeaways=[
            "Define evaluation criteria explicitly",
            "Break complex evaluation into steps",
            "Request structured output (JSON) for consistency",
            "Include explanation field for debugging",
            "Provide scoring rubric and formula"
        ]
    )
]


# Chunking Strategy Optimization Examples
CHUNKING_OPTIMIZATION_EXAMPLES = [
    OptimizationExample(
        category="Chunking Strategy",
        title="Document Chunking - Optimizing for Retrieval Quality",
        before={
            "strategy": "Fixed 256-token chunks, no overlap",
            "example_chunk": """...end of previous section. The course covers machine learning fundamentals including supervised learning, unsupervised learning, and reinforcement learning. Topics include linear regression, logistic regression, decision trees, neural networks, and deep learning. Prerequisites are CSCI 270 and MATH 225. The course meets...""",
            "retrieval_quality": "Context Precision: 0.62, Context Recall: 0.58",
            "problems": [
                "Chunks cut mid-sentence",
                "Context split across chunks",
                "Missing information in retrieval",
                "Low relevance scores"
            ]
        },
        after={
            "strategy": "Semantic chunking with 512 tokens, 50-token overlap, preserve paragraph boundaries",
            "example_chunk": """Course Description: CSCI 567 - Machine Learning

The course covers machine learning fundamentals including supervised learning, unsupervised learning, and reinforcement learning. Topics include linear regression, logistic regression, decision trees, neural networks, and deep learning.

Prerequisites: CSCI 270 (Algorithms) and MATH 225 (Linear Algebra)

Course Details:
- Units: 4
- Meeting Time: MW 2:00-3:20 PM
- Instructor: Dr. Smith
- Location: KAP 140""",
            "retrieval_quality": "Context Precision: 0.89, Context Recall: 0.85",
            "improvements": [
                "Chunks respect paragraph boundaries",
                "Complete semantic units",
                "Overlap ensures no information loss",
                "Higher relevance scores"
            ]
        },
        improvement_metrics={
            "context_precision": "Improved from 0.62 to 0.89 (+44%)",
            "context_recall": "Improved from 0.58 to 0.85 (+47%)",
            "answer_quality": "Improved from 3.2/5 to 4.6/5 rating",
            "retrieval_failures": "Reduced from 18% to 4%"
        },
        explanation="""
The key improvements:
1. Increased chunk size from 256 to 512 tokens (more context)
2. Added 50-token overlap (prevents information loss at boundaries)
3. Preserved paragraph boundaries (semantic coherence)
4. Structured format (clear sections)

Larger chunks with overlap provide more context while ensuring no information is lost at chunk boundaries.
""",
        key_takeaways=[
            "Chunk size matters: too small loses context, too large dilutes relevance",
            "Overlap prevents information loss at boundaries (10-20% overlap recommended)",
            "Respect semantic boundaries (paragraphs, sections)",
            "Experiment with your specific data - no universal optimal size",
            "Measure impact with retrieval metrics (precision, recall)"
        ]
    ),
    
    OptimizationExample(
        category="Chunking Strategy",
        title="Tabular Data Chunking - USC Course Catalog",
        before={
            "strategy": "Concatenate all columns without labels",
            "example_chunk": """CSCI 567 4 Machine Learning fundamentals including supervised learning unsupervised learning reinforcement learning MW 2:00-3:20 PM Dr. Smith CSCI 270 MATH 225""",
            "retrieval_quality": "Context Relevance: 0.54, Answer Quality: 2.8/5",
            "problems": [
                "No context for what each value means",
                "Hard to understand without table structure",
                "Embedding model confused by format",
                "Low retrieval quality"
            ]
        },
        after={
            "strategy": "Row-based with descriptive labels",
            "example_chunk": """Course: CSCI 567 - Machine Learning

Description: The course covers machine learning fundamentals including supervised learning, unsupervised learning, and reinforcement learning. Topics include linear regression, logistic regression, decision trees, neural networks, and deep learning.

Units: 4 units

Schedule: Meets Monday and Wednesday, 2:00-3:20 PM

Instructor: Dr. Smith

Prerequisites: CSCI 270 (Algorithms) and MATH 225 (Linear Algebra)

Location: KAP 140""",
            "retrieval_quality": "Context Relevance: 0.91, Answer Quality: 4.5/5",
            "improvements": [
                "Self-descriptive (understandable without table)",
                "Clear labels for each field",
                "Natural language format",
                "High retrieval quality"
            ]
        },
        improvement_metrics={
            "context_relevance": "Improved from 0.54 to 0.91 (+69%)",
            "answer_quality": "Improved from 2.8/5 to 4.5/5 (+61%)",
            "retrieval_success_rate": "Improved from 62% to 94%",
            "user_satisfaction": "Improved from 3.1/5 to 4.6/5"
        },
        explanation="""
The key improvements:
1. Added descriptive labels ("Course:", "Units:", "Schedule:")
2. Expanded abbreviations (MW → Monday and Wednesday)
3. Natural language format (readable without table structure)
4. Structured sections (clear organization)

Tabular data needs to be transformed into self-descriptive text for effective embedding and retrieval.
""",
        key_takeaways=[
            "Add descriptive labels to make data self-descriptive",
            "Expand abbreviations and codes",
            "Use natural language format",
            "Each row should be understandable independently",
            "Test retrieval quality - measure the impact"
        ]
    )
]


# Embedding Model Selection Optimization Examples
EMBEDDING_MODEL_OPTIMIZATION_EXAMPLES = [
    OptimizationExample(
        category="Embedding Model Selection",
        title="Financial Documents - General vs Domain-Specific Embeddings",
        before={
            "model": "OpenAI text-embedding-ada-002 (general purpose)",
            "retrieval_quality": "Context Precision: 0.68, Context Recall: 0.64",
            "example_query": "What is the company's EBITDA margin?",
            "retrieved_docs": [
                "Document about company margins (relevant)",
                "Document about profit margins (somewhat relevant)",
                "Document about EBIT (not EBITDA, less relevant)",
                "Document about revenue (not relevant)",
                "Document about market share (not relevant)"
            ],
            "problems": [
                "Confuses similar financial terms (EBIT vs EBITDA)",
                "Doesn't understand financial jargon",
                "Low precision for domain-specific queries"
            ]
        },
        after={
            "model": "FinBERT (finance-specific embeddings)",
            "retrieval_quality": "Context Precision: 0.89, Context Recall: 0.86",
            "example_query": "What is the company's EBITDA margin?",
            "retrieved_docs": [
                "Document with EBITDA calculation (highly relevant)",
                "Document with EBITDA margin analysis (highly relevant)",
                "Document with EBITDA trends (relevant)",
                "Document with operating margin (somewhat relevant)",
                "Document with financial ratios (somewhat relevant)"
            ],
            "improvements": [
                "Understands financial terminology precisely",
                "Distinguishes between similar terms (EBIT vs EBITDA)",
                "Higher precision for domain queries"
            ]
        },
        improvement_metrics={
            "context_precision": "Improved from 0.68 to 0.89 (+31%)",
            "context_recall": "Improved from 0.64 to 0.86 (+34%)",
            "answer_accuracy": "Improved from 72% to 94%",
            "user_satisfaction": "Improved from 3.4/5 to 4.7/5"
        },
        explanation="""
The key improvement:
Switched from general-purpose embeddings to domain-specific FinBERT embeddings.

FinBERT was pre-trained on financial documents and understands:
- Financial terminology (EBITDA, P/E ratio, etc.)
- Relationships between financial concepts
- Nuances in financial language

This demonstrates the importance of domain-specific embeddings for specialized domains.
""",
        key_takeaways=[
            "Domain-specific embeddings significantly outperform general embeddings for specialized domains",
            "Financial, medical, legal domains benefit most from specialized models",
            "Performance improvement: typically 20-40% for domain-specific models",
            "Consider fine-tuning if no pre-trained domain model exists",
            "Measure impact with retrieval metrics before/after"
        ]
    ),
    
    OptimizationExample(
        category="Embedding Model Selection",
        title="Multi-Language Support - General vs Multilingual Embeddings",
        before={
            "model": "English-only embedding model",
            "retrieval_quality": "English: 0.85, Spanish: 0.42, Arabic: 0.18",
            "example_query_spanish": "¿Cuáles son los requisitos previos para CSCI 567?",
            "retrieved_docs": [
                "English document about prerequisites (language mismatch)",
                "English document about CSCI 567 (language mismatch)",
                "Spanish document about different course (wrong course)",
                "English document about requirements (language mismatch)",
                "Irrelevant document"
            ],
            "problems": [
                "Can't match queries to documents in different languages",
                "Low retrieval quality for non-English queries",
                "Poor user experience for international users"
            ]
        },
        after={
            "model": "Multilingual embedding model (mBERT or XLM-RoBERTa)",
            "retrieval_quality": "English: 0.84, Spanish: 0.81, Arabic: 0.76",
            "example_query_spanish": "¿Cuáles son los requisitos previos para CSCI 567?",
            "retrieved_docs": [
                "English document about CSCI 567 prerequisites (relevant, cross-lingual)",
                "Spanish document about prerequisites (relevant, same language)",
                "English document about CSCI 567 details (relevant, cross-lingual)",
                "Spanish document about course requirements (relevant)",
                "English document about related courses (somewhat relevant)"
            ],
            "improvements": [
                "Cross-lingual retrieval works",
                "Consistent quality across languages",
                "Better user experience for international users"
            ]
        },
        improvement_metrics={
            "spanish_retrieval": "Improved from 0.42 to 0.81 (+93%)",
            "arabic_retrieval": "Improved from 0.18 to 0.76 (+322%)",
            "cross_lingual_success": "Improved from 15% to 87%",
            "international_user_satisfaction": "Improved from 2.1/5 to 4.3/5"
        },
        explanation="""
The key improvement:
Switched from English-only to multilingual embedding model.

Multilingual models:
- Embed text from multiple languages into shared vector space
- Enable cross-lingual retrieval (query in Spanish, retrieve English docs)
- Maintain quality across languages

This demonstrates the importance of multilingual embeddings for global applications.
""",
        key_takeaways=[
            "Multilingual embeddings enable cross-lingual retrieval",
            "Quality is consistent across languages (unlike translation-based approaches)",
            "Essential for global applications and international users",
            "Models: mBERT, XLM-RoBERTa, multilingual E5",
            "Test with queries in all target languages"
        ]
    )
]


# Evaluation Metric Customization Examples
EVALUATION_METRIC_CUSTOMIZATION_EXAMPLES = [
    OptimizationExample(
        category="Evaluation Metric Customization",
        title="Faithfulness Metric - Adding Domain-Specific Verification",
        before={
            "metric": "Standard Ragas faithfulness metric",
            "prompt": "Generic claim verification prompt",
            "performance": "Correlation with human judgment: 0.78",
            "problems": [
                "Doesn't understand domain-specific terminology",
                "Misses nuanced claims",
                "False positives on technical details"
            ]
        },
        after={
            "metric": "Customized faithfulness metric for medical domain",
            "prompt": """You are evaluating medical claims for faithfulness to clinical context.

Medical-specific considerations:
1. Dosage and medication names must be exact
2. Contraindications must be explicitly stated in context
3. Diagnostic criteria must match established guidelines
4. Statistical claims must have supporting data

Evaluate each claim with medical precision...""",
            "performance": "Correlation with human judgment: 0.94",
            "improvements": [
                "Understands medical terminology",
                "Stricter verification for critical claims",
                "Fewer false positives"
            ]
        },
        improvement_metrics={
            "correlation_with_experts": "Improved from 0.78 to 0.94 (+21%)",
            "false_positive_rate": "Reduced from 12% to 3%",
            "false_negative_rate": "Reduced from 8% to 2%",
            "physician_trust": "Improved from 3.6/5 to 4.8/5"
        },
        explanation="""
The key improvements:
1. Added domain-specific evaluation criteria (medical precision)
2. Stricter verification for critical claims (dosages, contraindications)
3. Reference to established guidelines
4. Medical terminology understanding

This demonstrates how metric customization improves evaluation quality for specialized domains.
""",
        key_takeaways=[
            "Customize metrics for domain-specific requirements",
            "Add stricter verification for critical claims",
            "Include domain terminology in prompts",
            "Validate with expert human evaluation",
            "Measure correlation with human judgment"
        ]
    )
]


# Collection of all optimization examples
ALL_OPTIMIZATION_EXAMPLES = (
    PROMPT_ENGINEERING_EXAMPLES +
    CHUNKING_OPTIMIZATION_EXAMPLES +
    EMBEDDING_MODEL_OPTIMIZATION_EXAMPLES +
    EVALUATION_METRIC_CUSTOMIZATION_EXAMPLES
)


def get_examples_by_category(category: str) -> List[OptimizationExample]:
    """Get all optimization examples for a specific category"""
    return [ex for ex in ALL_OPTIMIZATION_EXAMPLES if ex.category == category]


def export_optimization_examples_to_markdown() -> str:
    """Export all optimization examples to markdown format"""
    md = "# Before/After Optimization Examples: RAG Evaluation Course\n\n"
    md += "## Demonstrating Improvements in Prompt Engineering, Chunking, Embeddings, and Metrics\n\n"
    md += "---\n\n"
    
    categories = ["Prompt Engineering", "Chunking Strategy", "Embedding Model Selection", "Evaluation Metric Customization"]
    
    for category in categories:
        md += f"## {category}\n\n"
        examples = get_examples_by_category(category)
        
        for example in examples:
            md += f"### {example.title}\n\n"
            
            md += "#### Before Optimization\n\n"
            for key, value in example.before.items():
                md += f"**{key.replace('_', ' ').title()}:**\n```\n{value}\n```\n\n"
            
            md += "#### After Optimization\n\n"
            for key, value in example.after.items():
                md += f"**{key.replace('_', ' ').title()}:**\n```\n{value}\n```\n\n"
            
            md += "#### Improvement Metrics\n\n"
            for key, value in example.improvement_metrics.items():
                md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            md += "\n"
            
            md += "#### Explanation\n\n"
            md += f"{example.explanation}\n\n"
            
            md += "#### Key Takeaways\n\n"
            for takeaway in example.key_takeaways:
                md += f"- {takeaway}\n"
            md += "\n"
            
            md += "---\n\n"
    
    return md


if __name__ == "__main__":
    # Export optimization examples
    optimization_md = export_optimization_examples_to_markdown()
    print(optimization_md)
