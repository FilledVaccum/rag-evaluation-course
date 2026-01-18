"""
Module 4 Concept Summary: Synthetic Test Data Generation
Evaluating RAG and Semantic Search Systems Course

One-page summary of key concepts for Module 4.

Requirements Coverage: 17.2
Format: Concise, visual, exam-focused
"""

MODULE_4_CONCEPT_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODULE 4: SYNTHETIC TEST DATA GENERATION                  ║
║                     One-Page Concept Summary for NCP-AAI                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. WHY SYNTHETIC DATA FOR RAG?                                               │
└──────────────────────────────────────────────────────────────────────────────┘

✓ Test-Driven Development: Create test sets BEFORE production deployment
✓ Scale: Generate 100s-1000s of queries in minutes (vs. manual creation)
✓ Diversity: Cover edge cases and query patterns humans miss
✓ Continuous Evaluation: Regenerate as knowledge base evolves

Problem: Manual test data is insufficient (biased, limited, time-intensive)
Solution: LLM-based generation with NVIDIA Nemotron-4-340B

┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. THE 3-5 EXAMPLE OPTIMAL PATTERN ⭐ CRITICAL                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┬──────────────┬────────────────────────────────────────────────┐
│ # Examples  │ Quality      │ Problem                                        │
├─────────────┼──────────────┼────────────────────────────────────────────────┤
│ 0-2         │ 0-30%        │ Too generic, no pattern to learn               │
│ 3-5 ✓       │ 70-90%       │ OPTIMAL - Good steering without overfitting    │
│ 6+          │ 40-60%       │ Overfitting - generates near-copies            │
└─────────────┴──────────────┴────────────────────────────────────────────────┘

💡 Key Insight: 3-5 is the "Goldilocks zone" for few-shot learning

┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. PROMPT ENGINEERING BEST PRACTICES                                         │
└──────────────────────────────────────────────────────────────────────────────┘

① EXTREME SPECIFICITY
   • Write as if explaining to a child
   • Define clear user persona (role, experience, goals)
   • Be explicit about query style, length, focus

② 3-5 HIGH-QUALITY EXAMPLES
   • Show the pattern you want LLM to learn
   • Diverse query types (not all same template)
   • Domain-specific and realistic

③ EXPLICIT NEGATIVE EXAMPLES
   • State what NOT to generate
   • Prevent philosophical/generic queries
   • Filter off-topic questions

④ DETAILED CONSTRAINTS
   • Length: 10-25 words
   • Style: Casual, formal, technical
   • Focus: Specific topics/concerns
   • Answerability: Must use knowledge base

⑤ ITERATIVE VALIDATION
   • Test prompts multiple times
   • Measure quality scores
   • Refine based on results

┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. SYNTHESIZER TYPES & MIXING                                                │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬────────────────────────────────────────────────────────┐
│ Synthesizer Type │ Query Pattern                                          │
├──────────────────┼────────────────────────────────────────────────────────┤
│ SPECIFIC         │ Fact-seeking: "What is the prerequisite for CSCI 567?"│
│ ABSTRACT         │ Conceptual: "What are the main AI course themes?"     │
│ REASONING        │ Multi-hop: "If I'm weak at math, what's my AI path?"  │
└──────────────────┴────────────────────────────────────────────────────────┘

Mixing Strategy: 50% SPECIFIC + 50% REASONING (balanced coverage)
Alternative: 40-30-30 (Specific-Reasoning-Abstract) based on query logs

┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. QUALITY VALIDATION & FILTERING                                            │
└──────────────────────────────────────────────────────────────────────────────┘

Quality Checks:
✓ Length: 5-50 words
✓ Keywords: Remove banned terms (admission, tuition, weather)
✓ Domain Relevance: Check for domain-specific terms
✓ Answerability: Can knowledge base answer this?
✓ Duplicates: Remove queries with >90% similarity

Typical Filtering Rate: 10-30% of generated queries
Quality Threshold: 70%+ to keep query

┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. COMMON PITFALLS & SOLUTIONS                                               │
└──────────────────────────────────────────────────────────────────────────────┘

❌ PITFALL                          ✓ SOLUTION
─────────────────────────────────────────────────────────────────────────────
Over-generic queries                Add specific constraints & negatives
("What courses are available?")     Use 3-5 domain-specific examples

Philosophical questions             Explicit negative examples
("What is education?")              Define practical user persona

Unanswerable queries                Constraint: "Must use knowledge base"
("How do I enroll?")                Test answerability

All queries same template           Diverse examples (3-5 different patterns)
                                    Mix synthesizer types

Low quality (10-20%)                Apply all best practices
                                    Achieve 80-90% quality

┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. COMPLETE WORKFLOW                                                         │
└──────────────────────────────────────────────────────────────────────────────┘

1. Define Customized Prompt
   └─> System instruction + User persona + Constraints + Examples + Negatives

2. Configure Synthesizers
   └─> 50% Specific + 50% Reasoning (or custom mix)

3. Generate Queries
   └─> Call NVIDIA Nemotron-4-340B (100+ queries)

4. Validate Quality
   └─> Length, keywords, domain relevance, answerability

5. Remove Duplicates
   └─> Similarity threshold 90%

6. Final Test Set
   └─> 70-85 high-quality unique queries (80-90% quality)

┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. NVIDIA PLATFORM INTEGRATION                                               │
└──────────────────────────────────────────────────────────────────────────────┘

NVIDIA Nemotron-4-340B:
• Optimized for synthetic data generation
• Better instruction following than general LLMs
• Generates diverse, high-quality outputs
• Integrates with NVIDIA NIM ecosystem

Access: NVIDIA NIM endpoint + API key

┌──────────────────────────────────────────────────────────────────────────────┐
│ 9. EXAM FOCUS AREAS (NCP-AAI)                                                │
└──────────────────────────────────────────────────────────────────────────────┘

Evaluation and Tuning (13%):
✓ Why synthetic data is essential for RAG evaluation
✓ The 3-5 example optimal pattern (memorize this!)
✓ Prompt engineering best practices
✓ Quality validation metrics
✓ Synthesizer mixing strategies

Agent Development (15%):
✓ Test-driven development for LLMs
✓ Continuous evaluation workflows
✓ Domain-specific data generation

┌──────────────────────────────────────────────────────────────────────────────┐
│ 10. KEY FORMULAS & METRICS                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Quality Score = (Valid Checks / Total Checks)
• Length valid: 5-50 words
• Keywords valid: No banned terms
• Domain relevant: Contains domain keywords
• Typical threshold: 0.7 (70%)

Similarity Score = |Q1 ∩ Q2| / |Q1 ∪ Q2|  (Jaccard)
• Threshold: 0.9 (90%)
• Remove queries above threshold

Expected Quality Improvement:
• Baseline (no prompt engineering): 10-20%
• Customized (with best practices): 80-90%
• Improvement: +70 percentage points

┌──────────────────────────────────────────────────────────────────────────────┐
│ QUICK REFERENCE: PROMPT TEMPLATE                                             │
└──────────────────────────────────────────────────────────────────────────────┘

```
System: [Specific simulation context]
Persona: [Role, experience, goals, concerns]
Constraints: [4-6 specific requirements]
Examples: [3-5 high-quality examples]  ⭐ MUST BE 3-5
Negatives: [3-5 explicit "DO NOT" statements]
```

┌──────────────────────────────────────────────────────────────────────────────┐
│ REMEMBER FOR EXAM                                                            │
└──────────────────────────────────────────────────────────────────────────────┘

1. 3-5 examples is OPTIMAL (not 1-2, not 6+)
2. Negative examples are ESSENTIAL (not optional)
3. Quality validation filters 10-30% of queries
4. 50-50 synthesizer mixing is standard starting point
5. Baseline quality: 10-20%, Target: 80-90%
6. NVIDIA Nemotron-4-340B is optimized for synthetic data
7. Test-driven development is CRITICAL for RAG (not optional)

═══════════════════════════════════════════════════════════════════════════════
                        END OF MODULE 4 CONCEPT SUMMARY
═══════════════════════════════════════════════════════════════════════════════
"""


def print_summary():
    """Print the one-page concept summary."""
    print(MODULE_4_CONCEPT_SUMMARY)


def get_key_concepts() -> dict:
    """Return key concepts as structured data."""
    return {
        "module_number": 4,
        "title": "Synthetic Test Data Generation",
        "exam_domains": {
            "Evaluation and Tuning": 0.13,
            "Agent Development": 0.15
        },
        "critical_concepts": [
            "3-5 Example Optimal Pattern",
            "Extreme Specificity Principle",
            "Explicit Negative Examples",
            "Synthesizer Mixing (50-50)",
            "Quality Validation (70% threshold)"
        ],
        "key_formulas": {
            "quality_score": "(valid_checks / total_checks)",
            "similarity_score": "|Q1 ∩ Q2| / |Q1 ∪ Q2|",
            "quality_improvement": "Baseline (10-20%) → Customized (80-90%)"
        },
        "common_pitfalls": [
            "Using 1-2 or 6+ examples (not 3-5)",
            "Skipping negative examples",
            "Vague system instructions",
            "No quality validation",
            "Single synthesizer type"
        ],
        "nvidia_tools": [
            "NVIDIA Nemotron-4-340B (synthetic data generation)",
            "NVIDIA NIM (endpoint access)"
        ]
    }


if __name__ == "__main__":
    print_summary()
    
    print("\n" + "=" * 80)
    print("Key Concepts for Quick Review:")
    print("=" * 80)
    
    concepts = get_key_concepts()
    print(f"\nModule: {concepts['title']}")
    print(f"\nExam Domains:")
    for domain, weight in concepts['exam_domains'].items():
        print(f"  - {domain}: {weight*100}%")
    
    print(f"\nCritical Concepts to Memorize:")
    for i, concept in enumerate(concepts['critical_concepts'], 1):
        print(f"  {i}. {concept}")
    
    print(f"\nCommon Pitfalls to Avoid:")
    for pitfall in concepts['common_pitfalls']:
        print(f"  ❌ {pitfall}")
