"""
Module 5 Concept Summary: RAG Evaluation Metrics and Frameworks

One-page summary of key concepts, formulas, and decision frameworks
for quick review and exam preparation.

Requirements: 17.2
"""

MODULE_5_CONCEPT_SUMMARY = """
================================================================================
MODULE 5: RAG EVALUATION METRICS AND FRAMEWORKS - CONCEPT SUMMARY
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. WHY TRADITIONAL METRICS FAIL FOR RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ BLEU, ROUGE, F1 Score:
   - Only measure n-gram overlap (exact word matching)
   - Miss semantic equivalence ("capital is Paris" vs "Paris is capital")
   - Can't assess faithfulness, relevance, or context usage
   - Penalize valid paraphrasing

✓ LLM-as-a-Judge:
   - Semantic understanding beyond word matching
   - Can evaluate complex criteria (faithfulness, tone, appropriateness)
   - Scalable and cost-effective vs human evaluation
   - Customizable for domain-specific needs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. LLM-AS-A-JUDGE: ADVANTAGES & LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADVANTAGES:
✓ Semantic understanding        ✓ Scalable automation
✓ Flexible criteria             ✓ Consistent evaluation
✓ Cost-effective vs humans      ✓ Customizable prompts

LIMITATIONS:
⚠ Bias inheritance              ⚠ Hallucination risk
⚠ Prompt sensitivity            ⚠ API costs at scale
⚠ Position/length bias          ⚠ Calibration issues

BEST PRACTICES:
→ Clear scoring rubrics (0-1 scale, JSON output)
→ Calibration examples for each score level
→ Multi-stage evaluation (break complex metrics into steps)
→ Multiple judges for robustness (ensemble)
→ Periodic human validation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. RAGAS FRAMEWORK ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ragas = Retrieval-Augmented Generation Assessment

COMPONENTS:
┌─────────────────────────────────────────────────────────────┐
│  Generation Metrics    │  Retrieval Metrics                 │
│  - Faithfulness        │  - Context Precision               │
│  - Answer Relevancy    │  - Context Recall                  │
│  - Context Utilization │  - Context Relevance               │
└─────────────────────────────────────────────────────────────┘
                    ↓
         LLM-as-a-Judge Engine
         (NVIDIA NIM / OpenAI)
                    ↓
         Evaluation Results & Insights

REQUIRED DATA FORMAT:
- question (user_input)
- contexts (retrieved_context) - List of documents
- response (answer) - Generated response
- ground_truth (optional) - Reference answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. GENERATION METRICS (Evaluate LLM Output Quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│ FAITHFULNESS ⭐ MOST CRITICAL                                           │
│ Question: Are response claims supported by retrieved context?          │
│ Formula: faithfulness = verified_claims / total_claims                  │
│ Range: 0.0 (no support) → 1.0 (fully supported)                        │
│ Target: >0.85 (general), >0.95 (high-stakes)                           │
│                                                                          │
│ How it works:                                                           │
│ 1. Extract claims from response                                        │
│ 2. Verify each claim against context                                   │
│ 3. Calculate score                                                     │
│                                                                          │
│ Low score (<0.7) → LLM is hallucinating                                │
│ Fix: Stronger prompts, lower temperature, different LLM                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ANSWER RELEVANCY                                                        │
│ Question: Is response relevant to the user's question?                 │
│ Formula: cosine_similarity(embed(question), embed(response))           │
│ Range: 0.0 (off-topic) → 1.0 (highly relevant)                         │
│ Target: >0.80                                                           │
│                                                                          │
│ Low score (<0.6) → Response is off-topic                               │
│ Fix: Improve retrieval, strengthen question in prompt                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ CONTEXT UTILIZATION                                                     │
│ Question: Does response actually use retrieved context?                │
│ Formula: overlap(response, context) / length(response)                 │
│ Range: 0.0 (ignores context) → 1.0 (fully uses context)                │
│ Target: >0.70                                                           │
│                                                                          │
│ Low score (<0.6) → LLM ignoring context                                │
│ Fix: Strengthen prompt to emphasize context usage                      │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. RETRIEVAL METRICS (Evaluate Document Retrieval Quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────┐
│ CONTEXT PRECISION (Ranking Quality)                                    │
│ Question: Are relevant docs ranked higher?                             │
│ Formula: precision@k = relevant_in_top_k / k                            │
│ Range: 0.0 (poor ranking) → 1.0 (perfect ranking)                      │
│ Target: >0.70                                                           │
│                                                                          │
│ Why it matters: LLMs pay more attention to earlier context             │
│                                                                          │
│ Low score (<0.5) → Poor ranking                                        │
│ Fix: Re-ranking, better embeddings, hybrid search                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ CONTEXT RECALL (Information Coverage)                                  │
│ Question: Does context contain ALL necessary information?              │
│ Formula: recall = ground_truth_covered / total_ground_truth            │
│ Range: 0.0 (missing info) → 1.0 (complete coverage)                    │
│ Target: >0.75                                                           │
│                                                                          │
│ Low score (<0.6) → Missing information                                 │
│ Fix: Increase k, lower threshold, improve chunking                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ CONTEXT RELEVANCE (Document Filtering)                                 │
│ Question: What % of retrieved docs are relevant?                       │
│ Formula: relevance = relevant_chunks / total_chunks                    │
│ Range: 0.0 (all irrelevant) → 1.0 (all relevant)                       │
│ Target: >0.70                                                           │
│                                                                          │
│ Low score (<0.5) → Too much noise                                      │
│ Fix: Higher threshold, fewer docs (k), better embeddings               │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. METRIC PATTERNS & DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN                          DIAGNOSIS                    FIX
─────────────────────────────────────────────────────────────────────────
High Faithfulness (>0.8)         LLM being too               Adjust generation
+ Low Relevancy (<0.6)           conservative                prompt

Low Faithfulness (<0.7)          LLM hallucinating           Stronger constraints,
+ High Relevancy (>0.8)                                      lower temp, new LLM

High Precision (>0.7)            Retrieval too               Increase k,
+ Low Recall (<0.6)              selective                   lower threshold

Low Precision (<0.6)             Retrieval too               Add re-ranking,
+ High Recall (>0.7)             broad                       raise threshold

Low Context Utilization          LLM ignoring                Strengthen prompt
(<0.6)                           context                     to use context

All Metrics Low                  Fundamental                 Start with retrieval
                                 issues                      evaluation first

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. OPTIMIZATION WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BASELINE → Run evaluation, record all metrics

2. DIAGNOSE RETRIEVAL FIRST
   Check: Context Precision, Recall, Relevance
   If low → Fix retrieval before generation
   Fixes: Embedding model, chunking, k, threshold

3. DIAGNOSE GENERATION SECOND
   Check: Faithfulness, Relevancy, Utilization
   Only after retrieval is good (precision >0.7, recall >0.7)
   Fixes: Prompt engineering, temperature, LLM selection

4. ITERATE
   - Make ONE change at a time
   - Re-run evaluation after each change
   - Track metrics (spreadsheet/dashboard)
   - Look for trade-offs

5. VALIDATE
   - Test on held-out dataset
   - Get human feedback
   - Monitor in production

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. METRIC CUSTOMIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUSTOMIZE EXISTING METRIC:
→ Modify prompt template for domain-specific evaluation
→ Add domain terminology and criteria
→ Provide calibration examples
→ Keep same metric structure

CREATE CUSTOM METRIC FROM SCRATCH:
→ Define evaluation criteria (4-6 specific criteria)
→ Build detailed prompt template
→ Create scoring rubric (0-1 scale)
→ Test on sample data
→ Iterate based on results

PROMPT ENGINEERING BEST PRACTICES:
✓ Extreme specificity (explain like to a child)
✓ 3-5 examples (optimal for steering)
✓ Explicit negatives (what NOT to do)
✓ User personas and scenarios
✓ Structured output (JSON)
✓ Multi-stage evaluation for complex metrics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. TARGET SCORES BY USE CASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE CASE              Faithfulness  Relevancy  Precision  Recall  Relevance
─────────────────────────────────────────────────────────────────────────
High-Stakes            >0.95        >0.85      >0.80      >0.85    >0.80
(Healthcare, Legal)

General Enterprise     >0.85        >0.80      >0.70      >0.75    >0.70
(Support, Internal KB)

Exploratory            >0.75        >0.75      >0.60      >0.80    >0.65
(Research, Discovery)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. KEY EXAM TAKEAWAYS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Traditional NLP metrics fail for RAG - use LLM-as-a-Judge
✓ Faithfulness is THE most critical metric (prevents hallucination)
✓ Always diagnose retrieval BEFORE generation
✓ Ragas provides 6 core metrics (3 generation + 3 retrieval)
✓ Metric patterns reveal specific failure modes
✓ Customize metrics with domain-specific prompts
✓ Set targets based on use case (high-stakes vs exploratory)
✓ Iterate: baseline → diagnose → fix → re-evaluate → validate

NVIDIA PLATFORM INTEGRATION:
→ NVIDIA NIM for LLM-as-a-Judge endpoints
→ NV-Embed-v2 for embedding-based metrics
→ Ragas framework with NVIDIA endpoints

================================================================================
"""


def print_concept_summary():
    """Print the concept summary"""
    print(MODULE_5_CONCEPT_SUMMARY)


def save_concept_summary_to_file(filename: str = "module_5_summary.txt"):
    """Save concept summary to a text file"""
    with open(filename, 'w') as f:
        f.write(MODULE_5_CONCEPT_SUMMARY)
    print(f"✓ Concept summary saved to {filename}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        filename = sys.argv[2] if len(sys.argv) > 2 else "module_5_summary.txt"
        save_concept_summary_to_file(filename)
    else:
        print_concept_summary()
        print("\nTo save to file, run:")
        print("  python module_5_concept_summary.py --save [filename]")
