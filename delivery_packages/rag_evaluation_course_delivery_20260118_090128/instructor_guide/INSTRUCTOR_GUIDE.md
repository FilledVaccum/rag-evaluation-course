# Instructor Guide: Evaluating RAG and Semantic Search Systems

## Course Overview

**Version:** 1.0.0
**Total Duration:** 7.8 hours (470 minutes)
**Target Certification:** NVIDIA-Certified Professional: Agentic AI (NCP-AAI)
**Target Audience:** AI/ML professionals with 1-2 years of experience

## Course Objectives

This course prepares students for the NCP-AAI certification exam with primary focus on:
- Evaluation and Tuning (13% of exam)
- Knowledge Integration and Data Handling (10% of exam)
- Supporting coverage of Agent Development and Architecture

## Module Structure


### Module 1: Evolution of Search to RAG Systems

**Duration:** 40 minutes (0.7 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion

**Learning Objectives:**
- Explain the progression from traditional search to RAG
- Compare BM25 keyword search with semantic search paradigms
- Understand enterprise hybrid systems combining multiple approaches
- Make informed decisions about when to use each search approach

**Exam Domain Mapping:**
- Agent Architecture and Design (15.0%)
- Knowledge Integration and Data Handling (10.0%)

**Delivery Tips:**
- Lecture/Demo: 16 minutes
- Hands-On Practice: 20 minutes
- Discussion/Q&A: 4 minutes

**Cross-References:**
- Referenced by other modules:
  - Module 6: Module 6 evaluates BM25 systems introduced in Module 1

---

### Module 2: Embeddings and Vector Stores

**Duration:** 50 minutes (0.8 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion (Prerequisites: Modules 1)

**Learning Objectives:**
- Understand embedding fundamentals and multi-dimensional similarity
- Select appropriate domain-specific embedding models
- Configure and optimize vector stores for retrieval
- Handle diverse data types including tabular data in RAG systems
- Apply effective chunking strategies

**Exam Domain Mapping:**
- Knowledge Integration and Data Handling (10.0%)
- NVIDIA Platform Implementation (7.0%)

**Delivery Tips:**
- Lecture/Demo: 20 minutes
- Hands-On Practice: 25 minutes
- Discussion/Q&A: 5 minutes

**Cross-References:**
- Referenced by other modules:
  - Module 3: RAG retrieval uses vector stores configured in Module 2
  - Module 4: Synthetic data generation should match embedding model domain

---

### Module 3: RAG Architecture and Component Analysis

**Duration:** 70 minutes (1.2 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion (Prerequisites: Modules 2)

**Learning Objectives:**
- Architect end-to-end RAG systems with three-stage pipeline
- Diagnose failures at specific pipeline stages
- Evaluate retrieval and generation independently
- Implement error handling and graceful failure recovery
- Understand orchestration patterns and multi-step reasoning

**Exam Domain Mapping:**
- Agent Architecture and Design (15.0%)
- Agent Development (15.0%)

**Delivery Tips:**
- Lecture/Demo: 28 minutes
- Hands-On Practice: 35 minutes
- Discussion/Q&A: 7 minutes

**Cross-References:**
- References to other modules:
  - Module 2: RAG retrieval uses vector stores configured in Module 2
- Referenced by other modules:
  - Module 5: Faithfulness metric helps identify generation stage failures
  - Module 5: Context recall metric helps identify retrieval stage failures

---

### Module 4: Synthetic Test Data Generation

**Duration:** 80 minutes (1.3 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion (Prerequisites: Modules 3)

**Learning Objectives:**
- Generate robust test sets for RAG evaluation
- Apply prompt engineering techniques for data steering
- Customize prompts to reflect authentic user queries
- Validate synthetic data quality
- Support continuous evaluation workflows

**Exam Domain Mapping:**
- Evaluation and Tuning (13.0%)
- Agent Development (15.0%)

**Delivery Tips:**
- Lecture/Demo: 32 minutes
- Hands-On Practice: 40 minutes
- Discussion/Q&A: 8 minutes

**Cross-References:**
- References to other modules:
  - Module 5: Synthetic data from Module 4 is used for evaluation in Module 5
  - Module 2: Synthetic data generation should match embedding model domain
- Referenced by other modules:
  - Module 7: A/B testing requires synthetic test data for comparison

---

### Module 5: RAG Evaluation Metrics and Frameworks

**Duration:** 100 minutes (1.7 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion (Prerequisites: Modules 4, 3)

**Learning Objectives:**
- Implement comprehensive evaluation pipelines using Ragas
- Select appropriate metrics for different RAG components
- Customize metrics for domain-specific requirements
- Interpret evaluation results to guide optimization
- Create custom metrics from scratch

**Exam Domain Mapping:**
- Evaluation and Tuning (13.0%)

**Delivery Tips:**
- Lecture/Demo: 40 minutes
- Hands-On Practice: 50 minutes
- Discussion/Q&A: 10 minutes

**Cross-References:**
- References to other modules:
  - Module 3: Faithfulness metric helps identify generation stage failures
  - Module 3: Context recall metric helps identify retrieval stage failures
- Referenced by other modules:
  - Module 4: Synthetic data from Module 4 is used for evaluation in Module 5
  - Module 7: Production monitoring uses evaluation pipelines from Module 5

---

### Module 6: Semantic Search System Evaluation

**Duration:** 70 minutes (1.2 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion (Prerequisites: Modules 5)

**Learning Objectives:**
- Evaluate legacy search systems with modern LLM techniques
- Apply Ragas to non-RAG search systems
- Bridge traditional and modern evaluation approaches
- Support enterprise hybrid system requirements
- Optimize ranking algorithms

**Exam Domain Mapping:**
- Evaluation and Tuning (13.0%)
- Knowledge Integration and Data Handling (10.0%)

**Delivery Tips:**
- Lecture/Demo: 28 minutes
- Hands-On Practice: 35 minutes
- Discussion/Q&A: 7 minutes

**Cross-References:**
- References to other modules:
  - Module 1: Module 6 evaluates BM25 systems introduced in Module 1

---

### Module 7: Production Deployment and Advanced Topics

**Duration:** 60 minutes (1.0 hours)
**Time Allocation:** 40% Lecture / 50% Hands-On / 10% Discussion (Prerequisites: Modules 5, 6)

**Learning Objectives:**
- Deploy evaluation systems at production scale
- Monitor RAG performance continuously
- Handle enterprise-specific requirements
- Implement feedback loops for improvement
- Balance cost-efficiency with accuracy

**Exam Domain Mapping:**
- Deployment and Scaling (13.0%)
- Run, Monitor, and Maintain (5.0%)

**Delivery Tips:**
- Lecture/Demo: 24 minutes
- Hands-On Practice: 30 minutes
- Discussion/Q&A: 6 minutes

**Cross-References:**
- References to other modules:
  - Module 5: Production monitoring uses evaluation pipelines from Module 5
  - Module 4: A/B testing requires synthetic test data for comparison

---

## Learning Paths

The course supports multiple learning paths for different student needs:


### Complete Certification Path

**Target Audience:** Certification candidates seeking complete coverage
**Estimated Duration:** 7.5 hours
**Module Sequence:** 1 → 2 → 3 → 4 → 5 → 6 → 7

Full course path for comprehensive NCP-AAI certification preparation


### Evaluation Focus Path

**Target Audience:** Students focusing on Evaluation and Tuning domain
**Estimated Duration:** 5.5 hours
**Module Sequence:** 1 → 3 → 4 → 5 → 7

Focused path emphasizing evaluation techniques (primary exam domain)


### Technical Implementation Path

**Target Audience:** Engineers implementing RAG systems
**Estimated Duration:** 5.0 hours
**Module Sequence:** 2 → 3 → 4 → 5

Technical deep-dive into RAG implementation and evaluation


### Enterprise Deployment Path

**Target Audience:** Enterprise architects and deployment engineers
**Estimated Duration:** 5.0 hours
**Module Sequence:** 1 → 3 → 5 → 6 → 7

Path focusing on enterprise considerations and production deployment


## Key Terminology

Students should master these terms throughout the course:


**Module 1:**
- **RAG**: Retrieval-Augmented Generation - a system that combines information retrieval with language model generation
  - Also known as: Retrieval-Augmented Generation, RAG System
- **BM25**: Best Match 25 - a keyword-based ranking algorithm used in traditional search systems
  - Also known as: Best Match 25, Okapi BM25

**Module 2:**
- **Embedding**: A numerical vector representation of text that captures semantic meaning in high-dimensional space
  - Also known as: Vector Embedding, Text Embedding
- **Vector Store**: A database optimized for storing and retrieving high-dimensional embeddings
  - Also known as: Vector Database, Embedding Database
- **Chunking**: The process of dividing documents into optimal-sized segments for retrieval
  - Also known as: Text Chunking, Document Segmentation
- **NVIDIA NIM**: NVIDIA Inference Microservices for embedding and LLM inference
  - Also known as: NIM, NVIDIA Inference Microservices

**Module 4:**
- **Synthetic Data**: Artificially generated test data created using LLMs to reflect realistic user queries
  - Also known as: Synthetic Test Data, Generated Test Sets
- **Nemotron**: NVIDIA's family of large language models, including Nemotron-4-340B for synthetic data generation
  - Also known as: Nemotron-4-340B, NVIDIA Nemotron

**Module 5:**
- **Context Relevance**: A metric measuring whether retrieved context is relevant to a query
  - Also known as: Retrieval Relevance
- **Faithfulness**: A metric measuring whether generated responses are supported by retrieved context
  - Also known as: Answer Faithfulness, Factual Consistency
- **LLM-as-a-Judge**: A methodology using language models to evaluate other language model outputs
  - Also known as: LLM Judge, Model-Based Evaluation
- **Ragas**: Retrieval-Augmented Generation Assessment framework for evaluating RAG systems
  - Also known as: RAG Assessment Framework

## Delivery Recommendations

### Time Management

- **Lecture/Demo (40%):** Focus on concepts, demonstrations, and examples
- **Hands-On Practice (50%):** Students work through notebooks and exercises
- **Discussion/Q&A (10%):** Address questions, clarify concepts, share insights

### Teaching Strategies

1. **Start with Motivation:** Begin each module with real-world problems
2. **Hands-On First:** Let students experiment before explaining theory
3. **Intentional Bugs:** Notebooks include bugs for debugging practice
4. **Open-Ended Exercises:** Multiple valid solutions encourage exploration
5. **Certification Focus:** Regularly reference exam domains and requirements

### Common Student Questions

**Module 1-2:**
- "When should I use BM25 vs. semantic search?"
- "How do I choose the right embedding model?"
- "What chunk size should I use?"

**Module 3-4:**
- "How do I know if retrieval or generation is failing?"
- "How many examples should I include in prompts?"
- "How do I validate synthetic data quality?"

**Module 5-6:**
- "Why can't I use BLEU or F1 scores for RAG?"
- "How do I interpret faithfulness scores?"
- "Can I apply Ragas to non-RAG systems?"

**Module 7:**
- "How do I monitor RAG systems in production?"
- "What's the cost-accuracy trade-off?"
- "How do I handle regulatory compliance?"

### Technical Setup

Ensure students have:
- Python 3.10+ installed
- JupyterLab environment configured
- NVIDIA API keys (provide instructions)
- All dependencies installed (requirements.txt)
- Datasets pre-loaded

### Troubleshooting

Common issues and solutions:
- **API Rate Limits:** Use retry logic with exponential backoff
- **Memory Issues:** Reduce batch sizes, use streaming
- **Slow Embeddings:** Cache embeddings, use smaller models for testing
- **Missing Dependencies:** Provide requirements.txt and setup script

## Assessment Guidelines

### Module Quizzes
- 5-10 questions per module
- Mix of conceptual and applied questions
- Immediate feedback with explanations

### Hands-On Challenges
- Open-ended exercises
- Evaluation rubrics provided
- Multiple valid solutions

### Mock Certification Exam
- 60-70 questions
- 120-minute time limit
- Mirrors actual NCP-AAI exam format
- Detailed explanations for all answers

## Resources

### For Instructors
- Slide decks with speaker notes
- Case studies and examples
- Timing guidance for each section
- Common student questions and answers

### For Students
- Jupyter notebooks with exercises
- Datasets (USC Course Catalog, Amnesty Q&A)
- Study guides and concept summaries
- Practice questions and mock exam
- Recommended reading list

## Support

For questions or issues:
- Technical support: [support email]
- Course updates: [GitHub repository]
- Community forum: [forum link]
- NVIDIA certification: https://www.nvidia.com/en-us/training/certification/

---

**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}
**Course Version:** {course.course_version}
