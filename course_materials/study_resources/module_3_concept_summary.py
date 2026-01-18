"""
Module 3 Concept Summary: RAG Architecture and Component Analysis

One-page summary of key concepts, formulas, and decision frameworks for
Module 3: RAG Architecture and Component Analysis.

Requirements Coverage: 17.2
"""

MODULE_3_CONCEPT_SUMMARY = """
# Module 3: RAG Architecture and Component Analysis - Concept Summary

## Core Architecture

### Three-Stage RAG Pipeline
```
Query → [Retrieval] → Context → [Augmentation] → Prompt → [Generation] → Response
```

**Stage 1: Retrieval**
- Embed query using embedding model
- Search vector store for top-k similar documents
- Return ranked list of relevant contexts

**Stage 2: Augmentation**
- Format retrieved contexts into prompt template
- Combine query with context
- Apply prompt engineering techniques

**Stage 3: Generation**
- Send augmented prompt to LLM
- Generate response grounded in context
- Apply post-processing

## Component-Level Debugging

### The 80/20 Rule
**80% of RAG failures are retrieval problems, not generation problems**

### Systematic Debugging Workflow
1. Run query end-to-end
2. If wrong → Inspect retrieved documents
3. Check if answer is in retrieved context
4. If NOT in context → **Retrieval Failure**
   - Fix: Improve embeddings, chunking, or search
5. If IN context → **Generation Failure**
   - Fix: Improve prompts, use better LLM

### Failure Types

**Retrieval Failures:**
- Wrong documents retrieved
- Relevant docs ranked too low
- Semantic mismatch (query vs docs)
- Chunk boundary issues

**Generation Failures:**
- LLM ignores context (context ignorance)
- Hallucinations (claims not in context)
- Partial answers (misses key info)
- Over-reliance on parametric knowledge

**Augmentation Failures:**
- Context truncation
- Poor formatting
- Information loss
- Token limit issues

## Evaluation Metrics

### Retrieval Metrics

**Context Precision** (Ranking Quality)
```
precision@k = (relevant_docs_in_top_k) / k
```
- Measures if relevant docs appear at top
- High precision = good ranking

**Context Recall** (Coverage)
```
recall = (ground_truth_covered) / (total_ground_truth)
```
- Measures if all necessary info retrieved
- High recall = complete information

**Context Relevance** (Binary Classification)
```
relevance = (relevant_chunks) / (total_chunks)
```
- Measures percentage of relevant chunks
- 1.0 = all relevant, 0.0 = no relevant info

### Generation Metrics

**Faithfulness** (Context Grounding)
```
faithfulness = (verified_claims) / (total_claims)
```
- Multi-stage: Extract claims → Verify → Score
- Prevents hallucinations
- Ensures response grounded in context

**Answer Relevancy** (Query Alignment)
```
relevancy = cosine_sim(embed(query), embed(response))
```
- Measures semantic similarity to query
- Ensures response answers the question

**Context Utilization**
- Does response actually use retrieved context?
- Distinct from faithfulness
- Identifies when LLM ignores available info

## Decision Frameworks

### When to Debug Retrieval
- [ ] Retrieved docs don't contain answer
- [ ] Wrong documents retrieved
- [ ] Relevant docs ranked low
- [ ] Semantic mismatch between query and docs

### When to Debug Generation
- [ ] Retrieved docs contain answer
- [ ] LLM response doesn't use context
- [ ] Hallucinations present
- [ ] Generic "I don't know" responses

### When to Debug Augmentation
- [ ] Context present but not in prompt
- [ ] Information loss during formatting
- [ ] Truncation issues
- [ ] Poor prompt structure

## Assessment Techniques

### Manual Assessment (Gold Standard)
- Human reviewers judge relevance
- Time-consuming but accurate
- Use for validation and calibration

### LLM-as-a-Judge (Scalable)
- Use LLM to assess relevance/faithfulness
- Fast and scalable
- Requires validation against manual assessment

### Embedding Similarity (Proxy)
- Cosine similarity between embeddings
- Fast but imperfect
- Use as preliminary filter

## Common Pitfalls

❌ **Assuming LLM is the problem** → Debug retrieval first!
❌ **Not inspecting retrieved docs** → Always verify retrieval quality
❌ **Debugging both stages simultaneously** → Isolate components
❌ **Ignoring ranking quality** → Top results matter most
❌ **Skipping augmentation check** → Context can be lost in formatting

## Key Formulas

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Precision@k | relevant_in_top_k / k | Ranking quality |
| Recall | covered / total_ground_truth | Information coverage |
| Faithfulness | verified_claims / total_claims | Context grounding |
| Relevancy | cosine_sim(query, response) | Query alignment |

## Exam Tips

**Agent Architecture (15%):**
- Know the three RAG stages in order
- Understand component independence
- Explain orchestration patterns

**Evaluation and Tuning (13%):**
- Master component-level debugging
- Know when to use each metric
- Understand failure diagnosis workflows

**Agent Development (15%):**
- Implement retrieval and generation independently
- Debug component failures systematically
- Apply evaluation metrics correctly

## Quick Reference

**Retrieval Stage:**
- Input: Query
- Output: Ranked contexts
- Key Metric: Context Precision, Recall

**Augmentation Stage:**
- Input: Query + Contexts
- Output: Formatted prompt
- Key Metric: Information preservation

**Generation Stage:**
- Input: Augmented prompt
- Output: Response
- Key Metric: Faithfulness, Relevancy

## Next Steps

- **Module 4:** Generate synthetic test data for evaluation
- **Module 5:** Master Ragas framework and custom metrics
- **Module 6:** Evaluate legacy semantic search systems
- **Module 7:** Deploy RAG systems to production

---

**Remember:** Component-level debugging is essential. Always verify retrieval
before blaming generation. The 80/20 rule saves time and effort!
"""


def get_module_3_summary() -> str:
    """
    Returns the Module 3 concept summary.
    
    Returns:
        String containing the complete one-page summary
    """
    return MODULE_3_CONCEPT_SUMMARY


def print_summary():
    """Print the concept summary to console."""
    print(MODULE_3_CONCEPT_SUMMARY)


if __name__ == "__main__":
    print_summary()
