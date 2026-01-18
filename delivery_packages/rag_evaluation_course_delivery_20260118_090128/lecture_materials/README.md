# Lecture Materials and Visual Content

This directory contains comprehensive lecture materials for the "Evaluating RAG and Semantic Search Systems" course, including slide decks, instructor guides, case studies, and optimization examples.

## Contents

### Slide Decks (Modules 1-7)

Complete slide decks for all seven course modules with:
- Visual diagrams and Mermaid architecture diagrams
- Speaker notes for each slide
- Timing guidance
- Visual design notes

**Files:**
- `module_1_slides.py` - Evolution of Search to RAG Systems (11 slides)
- `module_2_slides.py` - Embeddings and Vector Stores (13 slides)
- `module_3_slides.py` - RAG Architecture and Component Analysis (13 slides)
- `module_4_slides.py` - Synthetic Test Data Generation (12 slides)
- `module_5_slides.py` - RAG Evaluation Metrics and Frameworks (15 slides)
- `module_6_slides.py` - Semantic Search System Evaluation (11 slides)
- `module_7_slides.py` - Production Deployment and Advanced Topics (13 slides)

**Total:** 88 slides across 7 modules

### Instructor Guide

Comprehensive teaching resource with:
- Timing guidance for each module (breakdown by section)
- Common student questions and answers
- Troubleshooting tips for live demos
- Teaching strategies by module
- Pacing and break recommendations

**File:** `instructor_guide.py`

### Case Studies

Real-world domain-specific case studies demonstrating RAG implementation:

1. **Finance Domain** - Investment Research RAG System for Wealth Management Firm
   - Challenge: Domain-specific language, temporal sensitivity, regulatory compliance
   - Solution: FinBERT embeddings, hybrid search, time-weighted retrieval
   - Results: 92% faithfulness, $0.05/query, 85% adoption

2. **Healthcare Domain** - Clinical Decision Support RAG System for Hospital Network
   - Challenge: Medical terminology, HIPAA compliance, life-critical accuracy
   - Solution: BioBERT/PubMedBERT, on-premise deployment, expert validation
   - Results: 94% medical accuracy, 25min → 5min research time

3. **Legal Domain** - Legal Research RAG System for Law Firm
   - Challenge: Legal language, precedent importance, exact citations required
   - Solution: LegalBERT, hybrid search with citation graph, Bluebook formatting
   - Results: 99.2% citation accuracy, 12hrs → 4hrs research time/week

4. **E-commerce Domain** - Product Search and Recommendation RAG System
   - Challenge: Scale (10M products), latency (<100ms), multi-modal data
   - Solution: Hybrid search, image embeddings (CLIP), aggressive caching
   - Results: 18% conversion increase, $45M additional revenue

**File:** `case_studies.py`

### Optimization Examples

Before/after examples demonstrating improvements in:

1. **Prompt Engineering**
   - Synthetic data generation (generic → specific questions)
   - LLM-as-a-Judge (binary → structured evaluation)
   - Improvements: 10% → 95% specificity, 60% → 95% consistency

2. **Chunking Strategy**
   - Document chunking (256 tokens → 512 tokens with overlap)
   - Tabular data (concatenated → labeled format)
   - Improvements: +44% precision, +47% recall

3. **Embedding Model Selection**
   - Financial documents (general → FinBERT)
   - Multi-language (English-only → multilingual)
   - Improvements: +31% precision, +93% Spanish retrieval

4. **Evaluation Metric Customization**
   - Faithfulness metric (generic → medical-specific)
   - Improvements: 0.78 → 0.94 correlation with experts

**File:** `optimization_examples.py`

## Usage

### Accessing Slide Content

```python
from course_materials.slides import get_module_1_slides, get_all_slides

# Get slides for a specific module
module_1 = get_module_1_slides()

# Get all slides
all_slides = get_all_slides()

# Export to markdown
from course_materials.slides import export_all_slides_to_markdown
markdown_content = export_all_slides_to_markdown()
```

### Accessing Instructor Guide

```python
from course_materials.slides.instructor_guide import (
    get_instructor_tips_by_module,
    get_timing_for_module,
    export_instructor_guide_to_markdown
)

# Get tips for a specific module
module_3_tips = get_instructor_tips_by_module("module_3")

# Get timing guidance
module_5_timing = get_timing_for_module("module_5")

# Export full guide
guide_md = export_instructor_guide_to_markdown()
```

### Accessing Case Studies

```python
from course_materials.slides.case_studies import (
    get_case_study_by_domain,
    export_case_studies_to_markdown
)

# Get specific case study
finance_case = get_case_study_by_domain("Finance")

# Export all case studies
case_studies_md = export_case_studies_to_markdown()
```

### Accessing Optimization Examples

```python
from course_materials.slides.optimization_examples import (
    get_examples_by_category,
    export_optimization_examples_to_markdown
)

# Get examples by category
prompt_examples = get_examples_by_category("Prompt Engineering")

# Export all examples
optimization_md = export_optimization_examples_to_markdown()
```

## Design Principles

### Visual Consistency
- All slides follow consistent formatting
- Mermaid diagrams for architecture and data flow
- Color-coded components (retrieval, augmentation, generation)
- Clear visual hierarchy

### Speaker Notes
- Every slide includes speaker notes
- Timing guidance for each section
- Common student questions anticipated
- Troubleshooting tips for demos

### Pedagogical Approach
- Build from fundamentals to advanced topics
- Use concrete examples before abstractions
- Show before/after comparisons
- Emphasize practical skills over theory

### Real-World Focus
- Domain-specific case studies
- Enterprise considerations
- Production deployment challenges
- Cost and performance trade-offs

## Course Delivery

### Recommended Timing
- **Module 1:** 45 minutes
- **Module 2:** 60 minutes
- **Module 3:** 90 minutes
- **Module 4:** 120 minutes
- **Module 5:** 150 minutes
- **Module 6:** 105 minutes
- **Module 7:** 75 minutes

**Total:** 645 minutes (10.75 hours) including hands-on exercises

### Break Schedule
- 10-minute break after every 60-90 minutes
- 15-minute break after intensive modules (4, 5)
- Flexibility based on student engagement

## Certification Alignment

All materials align with NVIDIA-Certified Professional: Agentic AI (NCP-AAI) exam:
- **Primary Focus:** Evaluation and Tuning (13%)
- **Core Coverage:** Knowledge Integration (10%)
- **Supporting:** Agent Development (15%), Architecture (15%)

## Export Formats

All materials can be exported to:
- **Markdown:** For documentation and review
- **Python objects:** For programmatic access
- **JSON:** For integration with other tools

## Maintenance

When updating materials:
1. Update slide content in respective module files
2. Update speaker notes in instructor_guide.py
3. Add new case studies to case_studies.py
4. Add new optimization examples to optimization_examples.py
5. Update this README with any structural changes

## License

These materials are part of the RAG Evaluation Course curriculum and are intended for educational use in preparing for the NVIDIA NCP-AAI certification exam.
