"""
Module 3: RAG Architecture and Component Analysis

This module provides comprehensive lecture materials on RAG pipeline architecture,
component-level failure diagnosis, context relevance assessment, and response
accuracy evaluation.

Learning Objectives:
- Understand the three-stage RAG pipeline (Retrieval → Augmentation → Generation)
- Master component-level failure diagnosis techniques
- Implement context relevance assessment
- Evaluate response accuracy and faithfulness
- Debug RAG systems at each pipeline stage

Requirements Coverage: 5.1, 5.2, 5.3, 5.4, 18.4
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class PipelineStage(Enum):
    """RAG pipeline stages for component-level analysis"""
    RETRIEVAL = "retrieval"
    AUGMENTATION = "augmentation"
    GENERATION = "generation"


class FailureType(Enum):
    """Types of failures in RAG systems"""
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    AUGMENTATION_FAILURE = "augmentation_failure"
    ORCHESTRATION_FAILURE = "orchestration_failure"


@dataclass
class LectureMaterial:
    """Lecture material structure for Module 3"""
    section_title: str
    content: str
    diagrams: List[str]
    key_concepts: List[str]
    examples: List[str]


# ============================================================================
# SECTION 1: RAG Pipeline Architecture Overview
# ============================================================================

RAG_PIPELINE_OVERVIEW = LectureMaterial(
    section_title="RAG Pipeline Architecture: Three-Stage Flow",
    content="""
## Understanding RAG: Retrieval-Augmented Generation

RAG (Retrieval-Augmented Generation) is a powerful architecture that combines
the strengths of information retrieval systems with large language models (LLMs).
Unlike pure LLM approaches that rely solely on parametric knowledge, RAG systems
dynamically retrieve relevant information from external knowledge bases to augment
the generation process.

### The Three-Stage RAG Pipeline

Every RAG system follows a three-stage pipeline:

1. **Retrieval Stage**: Query → Relevant Documents
   - Convert user query into embeddings
   - Search vector store for semantically similar documents
   - Rank and select top-k most relevant chunks
   - Return context passages for augmentation

2. **Augmentation Stage**: Query + Context → Augmented Prompt
   - Combine user query with retrieved context
   - Format context for optimal LLM consumption
   - Apply prompt templates and instructions
   - Prepare final input for generation

3. **Generation Stage**: Augmented Prompt → Response
   - Send augmented prompt to LLM
   - Generate response grounded in retrieved context
   - Apply post-processing and formatting
   - Return final answer to user

### Why Three Stages Matter

Understanding these distinct stages is critical for:
- **Debugging**: Identify which stage is causing failures
- **Optimization**: Tune each component independently
- **Evaluation**: Measure performance at each stage
- **Maintenance**: Update components without full system rewrites

### Key Insight: Component Independence

Each stage can fail independently. A perfect retrieval system can be undermined
by poor generation, and vice versa. This is why component-level evaluation is
essential for production RAG systems.
""",
    diagrams=[
        """
```mermaid
graph LR
    A[User Query] --> B[Retrieval Stage]
    B --> C[Retrieved Context]
    C --> D[Augmentation Stage]
    A --> D
    D --> E[Augmented Prompt]
    E --> F[Generation Stage]
    F --> G[Final Response]
    
    style B fill:#e1f5ff
    style D fill:#fff4e1
    style F fill:#e8f5e9
```
        """,
        """
```mermaid
sequenceDiagram
    participant User
    participant RAG System
    participant Vector Store
    participant LLM
    
    User->>RAG System: Submit Query
    RAG System->>Vector Store: Search for relevant docs
    Vector Store-->>RAG System: Return top-k chunks
    RAG System->>RAG System: Augment query with context
    RAG System->>LLM: Send augmented prompt
    LLM-->>RAG System: Generate response
    RAG System-->>User: Return final answer
```
        """
    ],
    key_concepts=[
        "Three-stage pipeline: Retrieval → Augmentation → Generation",
        "Component independence enables targeted debugging",
        "Each stage has distinct failure modes",
        "Evaluation must address all three stages",
        "Optimization requires stage-specific metrics"
    ],
    examples=[
        "Query: 'What are the prerequisites for CSCI 567?' → Retrieval finds course catalog entries → Augmentation formats context → Generation produces answer",
        "Retrieval failure: Wrong documents retrieved → Generation produces hallucinated answer",
        "Generation failure: Correct documents retrieved → LLM ignores context and hallucinates"
    ]
)


# ============================================================================
# SECTION 2: Component-Level Failure Diagnosis
# ============================================================================

COMPONENT_FAILURE_DIAGNOSIS = LectureMaterial(
    section_title="Component-Level Failure Diagnosis: Debugging RAG Systems",
    content="""
## The Critical Skill: Isolating Failure Points

One of the most common mistakes in RAG development is misdiagnosing failures.
Developers often blame the LLM (generation stage) when the real problem is in
retrieval. This wastes time and resources on the wrong optimization.

### Failure Diagnosis Workflow

Follow this systematic approach to identify failure points:

**Step 1: Verify Retrieval Quality**
- Inspect retrieved documents manually
- Check if relevant information is present
- Verify ranking quality (best docs at top)
- Assess chunk boundaries and completeness

**Step 2: Evaluate Augmentation**
- Review the augmented prompt sent to LLM
- Check if context is properly formatted
- Verify prompt template effectiveness
- Ensure no information loss during formatting

**Step 3: Assess Generation**
- Compare LLM output to retrieved context
- Check if LLM is using provided context
- Identify hallucinations or fabrications
- Verify response relevance to query

### Common Failure Patterns

#### Retrieval Failures
- **Symptom**: LLM generates plausible but incorrect answers
- **Root Cause**: Wrong documents retrieved, relevant info missing
- **Diagnosis**: Retrieved chunks don't contain answer
- **Solution**: Improve embeddings, chunking, or query reformulation

#### Generation Failures
- **Symptom**: LLM ignores retrieved context
- **Root Cause**: Poor prompt engineering or LLM limitations
- **Diagnosis**: Retrieved chunks contain answer, but LLM doesn't use it
- **Solution**: Improve prompts, use better LLM, or add instructions

#### Augmentation Failures
- **Symptom**: Information loss between retrieval and generation
- **Root Cause**: Poor context formatting or truncation
- **Diagnosis**: Context present but not accessible to LLM
- **Solution**: Improve prompt templates, increase context window

### The 80/20 Rule of RAG Debugging

**80% of RAG failures are retrieval problems, not generation problems.**

This is counterintuitive because generation is more visible. When you see a
wrong answer, your first instinct is to blame the LLM. But in most cases, the
LLM never had access to the right information in the first place.

### Diagnostic Questions to Ask

1. **Did retrieval find the right documents?**
   - Manually inspect top-k retrieved chunks
   - Check if answer is present in retrieved context

2. **Is the context properly formatted?**
   - Review the augmented prompt
   - Verify no truncation or information loss

3. **Is the LLM using the context?**
   - Compare response to retrieved context
   - Check for hallucinations vs. context-grounded answers

4. **Is the query well-formed?**
   - Consider query reformulation
   - Test with alternative phrasings
""",
    diagrams=[
        """
```mermaid
flowchart TD
    A[RAG System Failure] --> B{Inspect Retrieved Docs}
    B -->|Relevant info missing| C[Retrieval Failure]
    B -->|Relevant info present| D{Check Augmented Prompt}
    D -->|Context lost/malformed| E[Augmentation Failure]
    D -->|Context properly formatted| F{Analyze LLM Response}
    F -->|LLM ignores context| G[Generation Failure]
    F -->|LLM uses context incorrectly| G
    
    C --> H[Fix: Improve embeddings, chunking, or query]
    E --> I[Fix: Improve prompt templates]
    G --> J[Fix: Better prompts or stronger LLM]
    
    style C fill:#ffcdd2
    style E fill:#fff9c4
    style G fill:#c8e6c9
```
        """,
        """
```mermaid
graph TB
    subgraph "Retrieval Stage Debug"
        R1[Query Embedding]
        R2[Vector Search]
        R3[Retrieved Chunks]
        R4{Contains Answer?}
    end
    
    subgraph "Augmentation Stage Debug"
        A1[Context Formatting]
        A2[Prompt Template]
        A3[Augmented Prompt]
        A4{Info Preserved?}
    end
    
    subgraph "Generation Stage Debug"
        G1[LLM Processing]
        G2[Response Generation]
        G3{Uses Context?}
    end
    
    R1 --> R2 --> R3 --> R4
    R4 -->|Yes| A1
    R4 -->|No| RF[Retrieval Failure]
    A1 --> A2 --> A3 --> A4
    A4 -->|Yes| G1
    A4 -->|No| AF[Augmentation Failure]
    G1 --> G2 --> G3
    G3 -->|No| GF[Generation Failure]
    G3 -->|Yes| OK[System Working]
```
        """
    ],
    key_concepts=[
        "80% of RAG failures are retrieval problems",
        "Systematic diagnosis prevents wasted optimization effort",
        "Each stage has distinct failure signatures",
        "Manual inspection is essential for diagnosis",
        "Component-level evaluation enables targeted fixes"
    ],
    examples=[
        "Retrieval failure: Query 'machine learning course' retrieves biology courses → Fix embeddings",
        "Generation failure: Retrieved context has answer, LLM says 'I don't know' → Fix prompt",
        "Augmentation failure: Context truncated due to token limits → Increase window or summarize"
    ]
)


# ============================================================================
# SECTION 3: Context Relevance Assessment
# ============================================================================

CONTEXT_RELEVANCE_ASSESSMENT = LectureMaterial(
    section_title="Context Relevance Assessment: Evaluating Retrieval Quality",
    content="""
## Measuring Retrieval Effectiveness

Context relevance is the foundation of RAG system performance. If retrieval
fails to find relevant information, no amount of prompt engineering or LLM
sophistication can compensate.

### What is Context Relevance?

**Context Relevance** measures whether the retrieved documents are actually
relevant to the user's query. It answers the question: "Did we retrieve the
right information?"

### Key Metrics for Context Relevance

#### 1. Context Precision
**Definition**: Measures the ranking quality of retrieved contexts.

**Formula**: `precision@k = (relevant_docs_in_top_k) / k`

**Interpretation**:
- High precision: Relevant docs appear at the top
- Low precision: Relevant docs buried in results or missing

**Example**:
- Query: "What are CSCI 567 prerequisites?"
- Top 5 results: [CSCI 567 page, CSCI 570 page, CSCI 567 page, Math 225 page, CSCI 567 page]
- Relevant: 3 out of 5
- Precision@5 = 0.6

#### 2. Context Recall
**Definition**: Measures coverage of ground truth information in retrieved contexts.

**Formula**: `recall = (ground_truth_info_covered) / (total_ground_truth_info)`

**Interpretation**:
- High recall: All necessary information retrieved
- Low recall: Missing critical information

**Example**:
- Ground truth: "CSCI 567 requires CSCI 270 and Math 225"
- Retrieved context mentions: "CSCI 270"
- Recall = 0.5 (only 1 of 2 prerequisites found)

#### 3. Context Relevance Score
**Definition**: Binary classification of each retrieved chunk.

**Formula**: `relevance = (relevant_chunks) / (total_chunks)`

**Interpretation**:
- 1.0: All chunks relevant
- 0.5: Half the chunks are noise
- 0.0: No relevant information retrieved

### Assessment Techniques

#### Manual Assessment (Gold Standard)
- Human reviewers judge relevance
- Time-consuming but accurate
- Use for validation and calibration

#### LLM-as-a-Judge (Scalable)
- Use LLM to assess relevance
- Prompt: "Is this context relevant to answering the query?"
- Fast and scalable, but requires validation

#### Embedding Similarity (Proxy Metric)
- Cosine similarity between query and context embeddings
- Fast but imperfect (semantic similarity ≠ relevance)
- Use as preliminary filter

### Practical Assessment Workflow

1. **Sample Test Queries**: Create representative query set
2. **Retrieve Contexts**: Run retrieval for each query
3. **Manual Inspection**: Review top-k results for subset
4. **Automated Evaluation**: Use LLM-as-a-Judge for scale
5. **Analyze Patterns**: Identify systematic retrieval failures
6. **Iterate**: Improve embeddings, chunking, or ranking

### Common Relevance Issues

**Issue 1: Semantic Mismatch**
- Query uses different terminology than documents
- Solution: Query expansion, synonym handling, or better embeddings

**Issue 2: Chunk Boundaries**
- Relevant info split across multiple chunks
- Solution: Adjust chunk size, add overlap, or use hierarchical retrieval

**Issue 3: Ranking Problems**
- Relevant docs retrieved but ranked low
- Solution: Improve ranking algorithm or add re-ranking stage

**Issue 4: Domain Mismatch**
- General embeddings fail on specialized domains
- Solution: Use domain-specific embeddings or fine-tune
""",
    diagrams=[
        """
```mermaid
graph TB
    A[Query] --> B[Retrieval System]
    B --> C[Retrieved Chunks]
    
    C --> D[Context Precision]
    C --> E[Context Recall]
    C --> F[Context Relevance]
    
    D --> G{High Precision?}
    E --> H{High Recall?}
    F --> I{High Relevance?}
    
    G -->|No| J[Improve Ranking]
    H -->|No| K[Retrieve More/Better]
    I -->|No| L[Fix Embeddings]
    
    G -->|Yes| M[Good Retrieval]
    H -->|Yes| M
    I -->|Yes| M
    
    style M fill:#c8e6c9
    style J fill:#ffcdd2
    style K fill:#ffcdd2
    style L fill:#ffcdd2
```
        """,
        """
```mermaid
flowchart LR
    subgraph "Context Relevance Pipeline"
        A[Query] --> B[Embed Query]
        B --> C[Vector Search]
        C --> D[Top-K Chunks]
        D --> E[Relevance Assessment]
        E --> F[Precision Score]
        E --> G[Recall Score]
        E --> H[Relevance Score]
    end
    
    F --> I[Ranking Quality]
    G --> J[Coverage Quality]
    H --> K[Overall Quality]
    
    I --> L{Acceptable?}
    J --> L
    K --> L
    
    L -->|Yes| M[Proceed to Generation]
    L -->|No| N[Debug Retrieval]
```
        """
    ],
    key_concepts=[
        "Context relevance is the foundation of RAG performance",
        "Precision measures ranking quality",
        "Recall measures information coverage",
        "LLM-as-a-Judge enables scalable assessment",
        "Manual inspection validates automated metrics"
    ],
    examples=[
        "High precision, low recall: Top results relevant but incomplete",
        "Low precision, high recall: All info present but buried in noise",
        "Semantic mismatch: Query 'ML course' doesn't match 'Machine Learning' in docs"
    ]
)


# ============================================================================
# SECTION 4: Response Accuracy and Faithfulness Evaluation
# ============================================================================

RESPONSE_ACCURACY_FAITHFULNESS = LectureMaterial(
    section_title="Response Accuracy and Faithfulness: Evaluating Generation Quality",
    content="""
## Ensuring Trustworthy Responses

Once retrieval succeeds, we must ensure the LLM generates accurate, faithful
responses that are grounded in the retrieved context. This is where generation
evaluation becomes critical.

### Two Dimensions of Generation Quality

#### 1. Faithfulness (Context Grounding)
**Question**: Are the claims in the response supported by the retrieved context?

**Why It Matters**: Prevents hallucinations and ensures responses are grounded
in actual retrieved information, not LLM's parametric knowledge.

**Evaluation Approach**:
- Extract claims from response
- Verify each claim against retrieved context
- Calculate: `faithfulness = verified_claims / total_claims`

**Example**:
- Context: "CSCI 567 requires CSCI 270 and Math 225"
- Response: "CSCI 567 requires CSCI 270, Math 225, and CSCI 104"
- Claims: [CSCI 270 ✓, Math 225 ✓, CSCI 104 ✗]
- Faithfulness: 2/3 = 0.67

#### 2. Answer Relevancy (Query Alignment)
**Question**: Is the response relevant to the user's query?

**Why It Matters**: Ensures the LLM actually answers the question asked, not
a different question or provides tangential information.

**Evaluation Approach**:
- Measure semantic similarity between query and response
- Use embedding cosine similarity
- Calculate: `relevancy = cosine_sim(embed(query), embed(response))`

**Example**:
- Query: "What are the prerequisites for CSCI 567?"
- Good Response: "CSCI 567 requires CSCI 270 and Math 225" (High relevancy)
- Poor Response: "CSCI 567 is a machine learning course" (Low relevancy)

### Multi-Stage Faithfulness Evaluation

Faithfulness evaluation is complex and requires multiple steps:

**Stage 1: Claim Extraction**
- Parse response into atomic claims
- Each claim should be independently verifiable
- Use LLM to extract claims systematically

**Stage 2: Claim Verification**
- For each claim, check if supported by context
- Use LLM-as-a-Judge with explicit instructions
- Binary classification: supported or not supported

**Stage 3: Score Normalization**
- Calculate percentage of verified claims
- Normalize to 0-1 scale
- Report both score and detailed breakdown

### Context Utilization

**Definition**: Does the response actually use the retrieved context?

This is distinct from faithfulness. A response can be faithful (no false claims)
but still fail to use the context (e.g., "I don't know" when answer is in context).

**Evaluation**:
- Check if response references context information
- Measure information overlap between context and response
- Identify when LLM ignores available context

### Practical Evaluation Workflow

1. **Generate Response**: Run RAG pipeline on test query
2. **Extract Claims**: Parse response into verifiable statements
3. **Verify Faithfulness**: Check each claim against context
4. **Measure Relevancy**: Compare response to query semantically
5. **Assess Utilization**: Verify context was actually used
6. **Aggregate Scores**: Combine metrics for overall quality

### Common Generation Issues

**Issue 1: Hallucination**
- LLM adds information not in context
- Solution: Stronger faithfulness prompts, better LLM, or post-processing

**Issue 2: Context Ignorance**
- LLM ignores retrieved context entirely
- Solution: Improve prompt templates, emphasize context importance

**Issue 3: Partial Answers**
- LLM uses some context but misses key information
- Solution: Better context formatting or explicit instructions

**Issue 4: Over-Reliance on Parametric Knowledge**
- LLM prefers its training data over retrieved context
- Solution: Explicit instructions to prioritize context

### The Faithfulness-Relevancy Trade-off

High faithfulness with low relevancy: Response is accurate but doesn't answer
the question (e.g., provides related but not requested information).

High relevancy with low faithfulness: Response answers the question but includes
unsupported claims (hallucinations).

**Goal**: Maximize both faithfulness AND relevancy simultaneously.
""",
    diagrams=[
        """
```mermaid
graph TB
    A[Generated Response] --> B[Claim Extraction]
    B --> C[Claim 1]
    B --> D[Claim 2]
    B --> E[Claim N]
    
    F[Retrieved Context] --> G[Verification]
    C --> G
    D --> G
    E --> G
    
    G --> H{Claim Supported?}
    H -->|Yes| I[Verified Claims]
    H -->|No| J[Unverified Claims]
    
    I --> K[Faithfulness Score]
    J --> K
    
    A --> L[Embedding]
    M[Query] --> N[Embedding]
    L --> O[Cosine Similarity]
    N --> O
    O --> P[Relevancy Score]
    
    style I fill:#c8e6c9
    style J fill:#ffcdd2
```
        """,
        """
```mermaid
flowchart TD
    subgraph "Generation Evaluation Pipeline"
        A[Query + Context] --> B[LLM Generation]
        B --> C[Response]
        
        C --> D[Faithfulness Check]
        C --> E[Relevancy Check]
        C --> F[Utilization Check]
        
        D --> G{All Claims Verified?}
        E --> H{Semantically Similar?}
        F --> I{Context Used?}
        
        G -->|Yes| J[High Faithfulness]
        G -->|No| K[Hallucination Risk]
        
        H -->|Yes| L[High Relevancy]
        H -->|No| M[Off-Topic Response]
        
        I -->|Yes| N[Good Utilization]
        I -->|No| O[Context Ignored]
    end
    
    J --> P[Quality Response]
    L --> P
    N --> P
    
    K --> Q[Debug Generation]
    M --> Q
    O --> Q
    
    style P fill:#c8e6c9
    style Q fill:#ffcdd2
```
        """
    ],
    key_concepts=[
        "Faithfulness ensures responses are grounded in context",
        "Relevancy ensures responses answer the actual query",
        "Multi-stage evaluation: extract claims → verify → score",
        "Context utilization measures if LLM uses retrieved info",
        "Balance faithfulness and relevancy for quality responses"
    ],
    examples=[
        "High faithfulness, low relevancy: Accurate but off-topic response",
        "Low faithfulness, high relevancy: On-topic but hallucinated response",
        "Hallucination: LLM adds prerequisites not mentioned in context",
        "Context ignorance: LLM says 'I don't know' when answer is in context"
    ]
)


# ============================================================================
# MODULE 3 COMPLETE LECTURE CONTENT
# ============================================================================

MODULE_3_LECTURE_CONTENT = {
    "module_number": 3,
    "title": "RAG Architecture and Component Analysis",
    "duration_minutes": 75,
    "sections": [
        RAG_PIPELINE_OVERVIEW,
        COMPONENT_FAILURE_DIAGNOSIS,
        CONTEXT_RELEVANCE_ASSESSMENT,
        RESPONSE_ACCURACY_FAITHFULNESS
    ],
    "learning_objectives": [
        "Understand the three-stage RAG pipeline architecture",
        "Master component-level failure diagnosis techniques",
        "Implement context relevance assessment methods",
        "Evaluate response accuracy and faithfulness",
        "Debug RAG systems at each pipeline stage"
    ],
    "prerequisites": [
        "Module 1: Evolution of Search to RAG",
        "Module 2: Embeddings and Vector Stores"
    ],
    "certification_alignment": {
        "Agent Architecture and Design": 0.15,
        "Agent Development": 0.15,
        "Evaluation and Tuning": 0.13
    }
}


def get_module_3_content() -> Dict:
    """
    Returns complete Module 3 lecture content.
    
    Returns:
        Dictionary containing all lecture materials, diagrams, and metadata
    """
    return MODULE_3_LECTURE_CONTENT


def get_section_content(section_name: str) -> Optional[LectureMaterial]:
    """
    Retrieve specific section content by name.
    
    Args:
        section_name: Name of the section to retrieve
        
    Returns:
        LectureMaterial object or None if not found
    """
    sections = {
        "overview": RAG_PIPELINE_OVERVIEW,
        "diagnosis": COMPONENT_FAILURE_DIAGNOSIS,
        "relevance": CONTEXT_RELEVANCE_ASSESSMENT,
        "accuracy": RESPONSE_ACCURACY_FAITHFULNESS
    }
    return sections.get(section_name.lower())


if __name__ == "__main__":
    # Display module overview
    content = get_module_3_content()
    print(f"Module {content['module_number']}: {content['title']}")
    print(f"Duration: {content['duration_minutes']} minutes")
    print(f"\nLearning Objectives:")
    for obj in content['learning_objectives']:
        print(f"  - {obj}")
    print(f"\nSections: {len(content['sections'])}")
    for section in content['sections']:
        print(f"  - {section.section_title}")
