"""
Instructor Guide for RAG Evaluation Course
Comprehensive speaker notes, timing guidance, common questions, and troubleshooting tips
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class InstructorTip:
    """Represents an instructor tip for a specific module or topic"""
    module: str
    topic: str
    tip_type: str  # timing, common_question, troubleshooting, teaching_strategy
    content: str


# Timing Guidance for Each Module
TIMING_GUIDANCE = {
    "module_1": {
        "total_minutes": 45,
        "breakdown": {
            "introduction": 5,
            "classic_search": 8,
            "semantic_search": 10,
            "hybrid_systems": 10,
            "notebook_0": 25,
            "wrap_up": 5
        },
        "notes": "Module 1 is foundational. Don't rush. Students need solid understanding before moving forward."
    },
    "module_2": {
        "total_minutes": 60,
        "breakdown": {
            "introduction": 5,
            "embeddings_fundamentals": 12,
            "domain_specific_models": 10,
            "vector_stores": 10,
            "chunking_strategies": 10,
            "notebook_1": 30,
            "wrap_up": 5
        },
        "notes": "Technical content. Expect questions about high-dimensional spaces. Use analogies."
    },
    "module_3": {
        "total_minutes": 90,
        "breakdown": {
            "introduction": 5,
            "rag_architecture": 15,
            "component_debugging": 20,
            "failure_diagnosis": 15,
            "notebook_2": 35,
            "wrap_up": 10
        },
        "notes": "Core module. Emphasize debugging workflow. This is the most important practical skill."
    },
    "module_4": {
        "total_minutes": 120,
        "breakdown": {
            "introduction": 5,
            "synthetic_data_basics": 15,
            "prompt_engineering": 25,
            "quality_validation": 15,
            "notebook_3": 25,
            "notebook_4": 25,
            "wrap_up": 10
        },
        "notes": "Longest module. Prompt engineering is critical. Show many examples."
    },
    "module_5": {
        "total_minutes": 150,
        "breakdown": {
            "introduction": 5,
            "llm_as_judge": 15,
            "ragas_framework": 20,
            "generation_metrics": 25,
            "retrieval_metrics": 25,
            "customization": 20,
            "notebook_5": 40,
            "wrap_up": 10
        },
        "notes": "Heart of the course. Primary certification content. Take time to ensure understanding."
    },
    "module_6": {
        "total_minutes": 105,
        "breakdown": {
            "introduction": 5,
            "legacy_systems": 15,
            "hybrid_evaluation": 20,
            "ranking_assessment": 15,
            "notebook_6": 30,
            "wrap_up": 10
        },
        "notes": "Enterprise-focused. Resonates with students working in large organizations."
    },
    "module_7": {
        "total_minutes": 75,
        "breakdown": {
            "introduction": 5,
            "production_considerations": 20,
            "monitoring_observability": 15,
            "advanced_topics": 10,
            "notebook_7": 25,
            "course_wrap_up": 15
        },
        "notes": "Final module. Tie everything together. Discuss certification preparation."
    }
}


# Common Student Questions and Answers
COMMON_QUESTIONS = [
    InstructorTip(
        module="module_1",
        topic="BM25 vs Semantic Search",
        tip_type="common_question",
        content="""
Q: "Should we completely replace BM25 with semantic search?"
A: No! BM25 is still valuable for exact keyword matching. Most production systems use hybrid approaches combining both. BM25 is faster and cheaper for many use cases. The key is knowing when to use each approach.

Teaching Strategy: Show concrete examples where BM25 outperforms semantic search (e.g., product codes, legal citations).
"""
    ),
    InstructorTip(
        module="module_2",
        topic="High-Dimensional Embeddings",
        tip_type="common_question",
        content="""
Q: "How can I visualize 768-dimensional embeddings?"
A: You can't directly visualize 768 dimensions. Use dimensionality reduction techniques like t-SNE or UMAP to project to 2D/3D for visualization. But remember: the full 768 dimensions contain the real information. 2D plots are just approximations for human understanding.

Teaching Strategy: Use GPS coordinates analogy - we can't see all dimensions, but they work mathematically.
"""
    ),
    InstructorTip(
        module="module_2",
        topic="Chunking Strategy",
        tip_type="common_question",
        content="""
Q: "What's the optimal chunk size?"
A: There's no universal answer. It depends on your data, queries, and use case. Typical ranges: 256-1024 tokens. Too small loses context, too large dilutes relevance. You must experiment with your specific data.

Teaching Strategy: Show examples with different chunk sizes and their impact on retrieval quality.
"""
    ),
    InstructorTip(
        module="module_3",
        topic="Retrieval vs Generation Failures",
        tip_type="common_question",
        content="""
Q: "How do I know if the problem is retrieval or generation?"
A: Always check retrieval first! Inspect the top-3 retrieved documents. If they don't contain the answer, it's a retrieval problem. If they do contain the answer but the response is wrong, it's a generation problem. 80% of failures are retrieval.

Teaching Strategy: Walk through real debugging example step-by-step. Show the systematic approach.
"""
    ),
    InstructorTip(
        module="module_4",
        topic="Over-Generic Synthetic Data",
        tip_type="common_question",
        content="""
Q: "Why are my generated questions so philosophical and abstract?"
A: This is the #1 problem with synthetic data. The LLM defaults to generic questions without specific guidance. Solution: Be extremely specific in your prompts. Use 3-5 concrete examples. Explicitly state what NOT to generate.

Teaching Strategy: Show before/after examples. Demonstrate the dramatic improvement from prompt engineering.
"""
    ),
    InstructorTip(
        module="module_4",
        topic="3-5 Example Rule",
        tip_type="common_question",
        content="""
Q: "Why exactly 3-5 examples? Why not more?"
A: Research shows 3-5 is optimal. Fewer than 3: insufficient pattern. More than 5: diminishing returns and risk of overfitting (LLM just copies examples). This is empirically validated across many use cases.

Teaching Strategy: This is a key certification fact. Students should memorize this.
"""
    ),
    InstructorTip(
        module="module_5",
        topic="Faithfulness vs Accuracy",
        tip_type="common_question",
        content="""
Q: "What's the difference between faithfulness and accuracy?"
A: Faithfulness: Are claims supported by retrieved context? (context-grounded)
Accuracy: Are claims factually correct? (ground-truth comparison)
A response can be faithful but inaccurate if the retrieved context is wrong.

Teaching Strategy: Use concrete example to illustrate the distinction.
"""
    ),
    InstructorTip(
        module="module_5",
        topic="LLM-as-a-Judge Reliability",
        tip_type="common_question",
        content="""
Q: "Can we trust LLM-as-a-Judge? Isn't it biased?"
A: LLM-as-a-Judge has limitations (cost, latency, potential biases), but it's currently the best scalable method for semantic evaluation. Traditional metrics (BLEU, F1) are worse for RAG. Best practice: Use LLM-as-a-Judge with well-engineered prompts and validate with human evaluation on samples.

Teaching Strategy: Acknowledge limitations but emphasize it's the industry standard.
"""
    ),
    InstructorTip(
        module="module_6",
        topic="Legacy System Migration",
        tip_type="common_question",
        content="""
Q: "How long does it take to migrate from BM25 to semantic search?"
A: Depends on scale and complexity, but typically 6-12 months for large enterprises. It's not a big-bang replacement. Use gradual migration: baseline → pilot → A/B test → hybrid → optimize → scale. Maintain BM25 as fallback.

Teaching Strategy: Emphasize risk mitigation and gradual approach. This resonates with enterprise students.
"""
    ),
    InstructorTip(
        module="module_7",
        topic="Production Costs",
        tip_type="common_question",
        content="""
Q: "How much does it cost to run RAG in production?"
A: Highly variable. Depends on: query volume, model choices, caching strategy. Rough estimates: $0.001-$0.10 per query. Embedding APIs are cheap, LLM inference is expensive. Optimization strategies: caching, smaller models for simple queries, batch processing.

Teaching Strategy: Show cost breakdown and optimization strategies. This is a real concern for production.
"""
    )
]


# Troubleshooting Tips for Live Demos
TROUBLESHOOTING_TIPS = [
    InstructorTip(
        module="module_1",
        topic="Notebook 0 - Search Comparison",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. API Key Not Set: Check environment variables. Have backup key ready.
2. Rate Limiting: Use time.sleep() between API calls. Have pre-computed results as backup.
3. Dataset Not Loading: Verify file paths. Have local copy ready.

Backup Plan: If live demo fails, show pre-recorded results. Focus on interpretation, not execution.
"""
    ),
    InstructorTip(
        module="module_2",
        topic="Notebook 1 - Embeddings",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. NVIDIA NIM API Timeout: Network issues. Have fallback to OpenAI embeddings.
2. Memory Error with Large Dataset: Reduce batch size. Process in chunks.
3. Vector Store Connection: Check credentials. Have local FAISS as backup.

Backup Plan: Use smaller dataset subset if performance issues arise.
"""
    ),
    InstructorTip(
        module="module_3",
        topic="Notebook 2 - RAG Debugging",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. Intentional Bugs Too Hard: Provide hints after 10 minutes. Don't let students get stuck.
2. LLM Generating Unexpected Output: Temperature too high. Reduce to 0.1 for consistency.
3. Retrieval Returns Empty: Check index was built correctly. Verify query embedding.

Teaching Tip: Walk through first bug together as a class. Let students find remaining bugs independently.
"""
    ),
    InstructorTip(
        module="module_4",
        topic="Notebooks 3 & 4 - Synthetic Data",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. Generated Questions Still Generic: Prompt needs more specificity. Show example improvements.
2. Nemotron API Slow: Expected. Set realistic expectations. Have pre-generated data as backup.
3. Quality Validation Failing: Adjust thresholds. Not all generated data will be perfect.

Teaching Tip: Emphasize iteration. First attempt won't be perfect. Show improvement process.
"""
    ),
    InstructorTip(
        module="module_5",
        topic="Notebook 5 - Ragas Evaluation",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. Ragas Installation Issues: Use pip install ragas --upgrade. Check Python version (3.8+).
2. LLM API Errors: Rate limiting or quota exceeded. Use smaller test set or add delays.
3. Metric Computation Slow: Expected for LLM-as-a-Judge. Set expectations. Use smaller sample for demo.

Backup Plan: Have pre-computed metric results ready. Focus on interpretation if computation fails.
"""
    ),
    InstructorTip(
        module="module_6",
        topic="Notebook 6 - Semantic Search Eval",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. BM25 Library Not Installed: pip install rank-bm25. Have requirements.txt ready.
2. Comparison Results Unexpected: Good teaching moment! Discuss why results differ.
3. NDCG Calculation Confusion: Walk through formula step-by-step. Use simple example.

Teaching Tip: Unexpected results are learning opportunities. Discuss implications.
"""
    ),
    InstructorTip(
        module="module_7",
        topic="Notebook 7 - Production Monitoring",
        tip_type="troubleshooting",
        content="""
Common Issues:
1. Monitoring Dashboard Not Rendering: Browser compatibility. Try different browser.
2. A/B Test Sample Size Too Small: Explain statistical significance requirements.
3. Cost Calculations Confusing: Walk through example with real numbers.

Teaching Tip: Production topics are abstract. Use concrete examples and real-world scenarios.
"""
    )
]


# Teaching Strategies by Module
TEACHING_STRATEGIES = [
    InstructorTip(
        module="module_1",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 1 Teaching Strategy:
- Start with familiar concepts (Google search) before introducing new concepts
- Use analogies: BM25 = library card catalog, Semantic search = asking a librarian
- Show side-by-side comparisons with same query
- Emphasize: this isn't either/or, it's about combining approaches
- Set expectations: RAG builds on these foundations

Engagement: Ask students about their experience with search systems. Build on their knowledge.
"""
    ),
    InstructorTip(
        module="module_2",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 2 Teaching Strategy:
- High-dimensional spaces are abstract. Use concrete analogies (GPS coordinates, color spaces)
- Show visualizations early and often (t-SNE plots, similarity heatmaps)
- Hands-on experimentation is key. Encourage students to try different parameters
- Domain-specific models: show performance differences with real examples
- Chunking: demonstrate impact with before/after retrieval quality

Engagement: Poll students on their domain. Discuss relevant embedding models for their use cases.
"""
    ),
    InstructorTip(
        module="module_3",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 3 Teaching Strategy:
- This is THE critical module. Take time to ensure understanding
- Emphasize systematic debugging: always check retrieval first
- Use real failure examples. Show actual broken RAG systems
- Walk through debugging process step-by-step multiple times
- Component isolation is key skill. Practice with multiple examples

Engagement: Have students share their RAG debugging experiences. Learn from each other.
"""
    ),
    InstructorTip(
        module="module_4",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 4 Teaching Strategy:
- Prompt engineering is both art and science. Show many examples
- Before/after comparisons are powerful. Show dramatic improvements
- 3-5 example rule: emphasize this repeatedly. It's a key certification fact
- Let students struggle a bit with over-generic data. Then show solution
- Iteration is normal. First prompts won't be perfect

Engagement: Have students share their prompt improvements. Celebrate good examples.
"""
    ),
    InstructorTip(
        module="module_5",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 5 Teaching Strategy:
- This is the heart of the course. Primary certification content
- Start with why traditional metrics fail. Build motivation
- Each metric: definition → formula → example → interpretation
- Emphasize metric interpretation leads to action
- Customization is powerful but advanced. Show one clear example
- LLM-as-a-Judge: acknowledge limitations but emphasize it's the standard

Engagement: Use real evaluation results. Discuss what actions to take based on scores.
"""
    ),
    InstructorTip(
        module="module_6",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 6 Teaching Strategy:
- Enterprise focus. Acknowledge real-world constraints
- Legacy systems aren't bad. They're reality. Show respect
- Gradual migration is smart, not cowardly. Emphasize risk management
- Hybrid approaches are sophisticated, not compromises
- Show how modern techniques apply to old systems

Engagement: Students in enterprises will relate. Let them share migration challenges.
"""
    ),
    InstructorTip(
        module="module_7",
        topic="General Strategy",
        tip_type="teaching_strategy",
        content="""
Module 7 Teaching Strategy:
- Production is different from development. Set expectations
- Monitoring and observability are not optional. They're essential
- Cost optimization is real concern. Show concrete strategies
- Compliance is non-negotiable for regulated industries
- Continuous evaluation prevents quality degradation

Engagement: Discuss production war stories. What went wrong? What went right?
"""
    )
]


# Pacing and Break Recommendations
PACING_RECOMMENDATIONS = """
Overall Course Pacing (6-8 hours total):

Session 1 (2.5 hours):
- Module 1: Evolution of Search (45 min)
- 10-minute break
- Module 2: Embeddings and Vector Stores (60 min)
- 10-minute break
- Module 3: RAG Architecture (first 45 min)

Session 2 (3 hours):
- Module 3: RAG Architecture (remaining 45 min)
- 10-minute break
- Module 4: Synthetic Data Generation (120 min)
- 15-minute break

Session 3 (2.5 hours):
- Module 5: Evaluation Metrics (150 min)
- 15-minute break

Session 4 (2 hours):
- Module 6: Semantic Search Evaluation (105 min)
- 10-minute break
- Module 7: Production Deployment (75 min)
- Course wrap-up and Q&A (15 min)

Break Guidelines:
- 10 minutes after every 60-90 minutes
- 15 minutes after intensive modules (4, 5)
- Encourage students to stretch, hydrate, check messages
- Use breaks to reset energy and refocus attention

Flexibility:
- Adjust timing based on student engagement and questions
- If running behind, can reduce some hands-on time
- If running ahead, add more discussion and Q&A
- Priority: ensure understanding over covering all content
"""


def get_instructor_tips_by_module(module: str) -> List[InstructorTip]:
    """Get all instructor tips for a specific module"""
    return [tip for tip in COMMON_QUESTIONS + TROUBLESHOOTING_TIPS + TEACHING_STRATEGIES 
            if tip.module == module]


def get_timing_for_module(module: str) -> Dict:
    """Get timing guidance for a specific module"""
    return TIMING_GUIDANCE.get(module, {})


def export_instructor_guide_to_markdown() -> str:
    """Export complete instructor guide to markdown"""
    md = "# Instructor Guide: RAG Evaluation Course\n\n"
    md += "## Complete Teaching Resource with Timing, Tips, and Troubleshooting\n\n"
    md += "---\n\n"
    
    # Timing Guidance
    md += "## Timing Guidance\n\n"
    for module, timing in TIMING_GUIDANCE.items():
        md += f"### {module.replace('_', ' ').title()}\n\n"
        md += f"**Total Time:** {timing['total_minutes']} minutes\n\n"
        md += "**Breakdown:**\n"
        for section, minutes in timing['breakdown'].items():
            md += f"- {section.replace('_', ' ').title()}: {minutes} minutes\n"
        md += f"\n**Notes:** {timing['notes']}\n\n"
        md += "---\n\n"
    
    # Pacing Recommendations
    md += "## Pacing and Break Recommendations\n\n"
    md += PACING_RECOMMENDATIONS
    md += "\n\n---\n\n"
    
    # Common Questions
    md += "## Common Student Questions and Answers\n\n"
    for tip in COMMON_QUESTIONS:
        md += f"### {tip.module.replace('_', ' ').title()} - {tip.topic}\n\n"
        md += f"{tip.content}\n\n"
        md += "---\n\n"
    
    # Troubleshooting Tips
    md += "## Troubleshooting Tips for Live Demos\n\n"
    for tip in TROUBLESHOOTING_TIPS:
        md += f"### {tip.module.replace('_', ' ').title()} - {tip.topic}\n\n"
        md += f"{tip.content}\n\n"
        md += "---\n\n"
    
    # Teaching Strategies
    md += "## Teaching Strategies by Module\n\n"
    for tip in TEACHING_STRATEGIES:
        md += f"### {tip.module.replace('_', ' ').title()}\n\n"
        md += f"{tip.content}\n\n"
        md += "---\n\n"
    
    return md


if __name__ == "__main__":
    # Export instructor guide
    guide = export_instructor_guide_to_markdown()
    print(guide)
