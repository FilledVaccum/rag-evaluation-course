# Requirements Document

## Introduction

This document specifies the requirements for developing a production-ready course titled "Evaluating RAG and Semantic Search Systems" designed to prepare candidates for the NVIDIA-Certified Professional: Agentic AI (NCP-AAI) certification exam. The course focuses on the Evaluation and Tuning domain (13% of exam weight) while providing comprehensive coverage of RAG architecture, synthetic data generation, evaluation frameworks, and production deployment considerations.

## Glossary

- **RAG**: Retrieval-Augmented Generation - a system that combines information retrieval with language model generation
- **NCP-AAI**: NVIDIA-Certified Professional: Agentic AI certification
- **Course_System**: The complete course delivery system including materials, notebooks, assessments, and resources
- **Evaluation_Framework**: Tools and methodologies for assessing RAG system performance (e.g., Ragas)
- **LLM-as-a-Judge**: Methodology using language models to evaluate other language model outputs
- **Synthetic_Data_Generator**: System for creating test datasets using LLMs
- **BM25**: Best Match 25 - a keyword-based ranking algorithm
- **Vector_Store**: Database optimized for storing and retrieving high-dimensional embeddings
- **NVIDIA_NIM**: NVIDIA Inference Microservices for embedding and LLM inference
- **Ragas**: Retrieval-Augmented Generation Assessment framework
- **Jupyter_Notebook**: Interactive computational environment for course exercises
- **Embedding_Model**: Neural network that converts text into numerical vector representations
- **Context_Relevance**: Metric measuring whether retrieved context is relevant to a query
- **Faithfulness**: Metric measuring whether generated responses are supported by retrieved context
- **Chunking_Strategy**: Method for dividing documents into optimal-sized segments for retrieval

## Requirements

### Requirement 1: Course Architecture and Module Structure

**User Story:** As a course developer, I want to create a 7-module progressive learning path, so that students build foundational knowledge before advancing to complex evaluation techniques.

#### Acceptance Criteria

1. THE Course_System SHALL include exactly seven modules covering: Evolution of Search to RAG, Embeddings and Vector Stores, RAG Architecture, Synthetic Test Data Generation, RAG Evaluation Metrics, Semantic Search Evaluation, and Production Deployment
2. WHEN modules are sequenced, THE Course_System SHALL ensure each module builds upon concepts from previous modules
3. THE Course_System SHALL allocate 40% time to lecture/demo, 50% to hands-on practice, and 10% to discussion/Q&A
4. THE Course_System SHALL provide a total duration of 6-8 hours of content
5. WHERE a module introduces technical concepts, THE Course_System SHALL include corresponding hands-on Jupyter notebooks

### Requirement 2: Certification Exam Alignment

**User Story:** As a certification candidate, I want course content mapped to NCP-AAI exam objectives, so that I can effectively prepare for the certification exam.

#### Acceptance Criteria

1. THE Course_System SHALL map every module to specific NCP-AAI exam domains with explicit weight percentages
2. THE Course_System SHALL provide primary coverage (⭐⭐⭐) for Evaluation and Tuning (13% exam weight)
3. THE Course_System SHALL provide core coverage (⭐⭐⭐) for Knowledge Integration and Data Handling (10% exam weight)
4. THE Course_System SHALL include 60-70 practice questions mirroring the exam format (120 minutes, professional level)
5. WHEN content addresses exam topics, THE Course_System SHALL explicitly reference the corresponding exam domain and weight

### Requirement 3: Module 1 - Evolution of Search to RAG Systems

**User Story:** As a student, I want to understand the progression from traditional search to RAG systems, so that I can make informed decisions about when to use each approach.

#### Acceptance Criteria

1. THE Course_System SHALL explain classic search architecture including crawling, indexing, and ranking
2. THE Course_System SHALL compare BM25 keyword search with semantic search paradigms
3. THE Course_System SHALL describe enterprise hybrid systems combining BM25, vector search, and re-ranking
4. THE Course_System SHALL provide a decision framework for selecting appropriate search approaches
5. THE Course_System SHALL include Notebook 0 comparing search paradigms on identical queries

### Requirement 4: Module 2 - Embeddings and Vector Stores

**User Story:** As a student, I want to understand embedding fundamentals and vector store configuration, so that I can optimize retrieval performance for specific domains.

#### Acceptance Criteria

1. THE Course_System SHALL explain multi-dimensional similarity and embedding fundamentals
2. THE Course_System SHALL describe domain-specific embedding models for code, finance, healthcare, and multilingual applications
3. THE Course_System SHALL demonstrate NVIDIA NIM embedding models including NV-Embed-v2
4. THE Course_System SHALL explain vector store configuration and optimization strategies
5. THE Course_System SHALL describe chunking strategies for different data types including text, tabular data, and code
6. THE Course_System SHALL include hands-on exercises implementing embeddings with NVIDIA NIM and experimenting with chunking

### Requirement 5: Module 3 - RAG Architecture and Component Analysis

**User Story:** As a student, I want to understand RAG pipeline architecture and component-level debugging, so that I can diagnose failures at specific pipeline stages.

#### Acceptance Criteria

1. THE Course_System SHALL describe the three-stage RAG pipeline: Retrieval, Augmentation, and Generation
2. THE Course_System SHALL explain component-level failure diagnosis distinguishing retrieval from generation failures
3. THE Course_System SHALL describe context relevance assessment techniques
4. THE Course_System SHALL explain response accuracy and faithfulness evaluation methods
5. THE Course_System SHALL describe orchestration patterns and multi-step reasoning
6. THE Course_System SHALL include hands-on exercises building end-to-end RAG pipelines and debugging component failures

### Requirement 6: Module 4 - Synthetic Test Data Generation

**User Story:** As a student, I want to generate high-quality synthetic test data for RAG evaluation, so that I can create domain-specific test sets reflecting realistic user queries.

#### Acceptance Criteria

1. THE Course_System SHALL explain test-driven development principles for LLMs and RAG systems
2. THE Course_System SHALL demonstrate LLM-based synthetic data generation using NVIDIA Nemotron-4-340B
3. THE Course_System SHALL teach prompt engineering techniques for data steering using the 3-5 example optimal pattern
4. THE Course_System SHALL explain domain-specific query generation techniques
5. THE Course_System SHALL describe quality validation and filtering strategies for synthetic data
6. THE Course_System SHALL include Notebook 1 for generating baseline synthetic data using USC course catalog
7. THE Course_System SHALL include Notebook 2 for customizing prompts to fix over-generalization and create student-focused queries

### Requirement 7: Module 5 - RAG Evaluation Metrics and Frameworks

**User Story:** As a student, I want to master RAG evaluation metrics and frameworks, so that I can implement comprehensive evaluation pipelines and interpret results to guide optimization.

#### Acceptance Criteria

1. THE Course_System SHALL explain LLM-as-a-Judge methodology and its limitations
2. THE Course_System SHALL provide deep dive into Ragas framework including architecture, metrics, and customization
3. THE Course_System SHALL describe generation metrics including Faithfulness, Answer Relevancy, and Context Utilization
4. THE Course_System SHALL describe retrieval metrics including Context Precision, Context Recall, and Context Relevance
5. THE Course_System SHALL demonstrate custom metric development from scratch
6. THE Course_System SHALL explain metric interpretation and actionable optimization insights
7. THE Course_System SHALL include Notebook 3 implementing Ragas on Amnesty Q&A dataset, computing faithfulness and recall, customizing metrics, and creating custom evaluators

### Requirement 8: Module 6 - Semantic Search System Evaluation

**User Story:** As a student, I want to evaluate legacy semantic search systems with modern techniques, so that I can support enterprise hybrid system requirements.

#### Acceptance Criteria

1. THE Course_System SHALL explain evaluation of legacy BM25 systems with modern LLM techniques
2. THE Course_System SHALL demonstrate applying Ragas to non-RAG search systems
3. THE Course_System SHALL describe hybrid evaluation strategies combining RAG and semantic search
4. THE Course_System SHALL explain ranking algorithm assessment and optimization
5. THE Course_System SHALL describe integration with existing enterprise systems
6. THE Course_System SHALL include Notebook 4 evaluating semantic search with LLM-as-a-Judge and comparing RAG versus traditional search

### Requirement 9: Module 7 - Production Deployment and Advanced Topics

**User Story:** As a student, I want to understand production deployment considerations, so that I can deploy evaluation systems at production scale with appropriate monitoring and compliance.

#### Acceptance Criteria

1. THE Course_System SHALL explain temporal data handling and time-weighted retrieval
2. THE Course_System SHALL describe regulatory compliance requirements including GDPR and HIPAA
3. THE Course_System SHALL explain continuous evaluation in production environments
4. THE Course_System SHALL describe performance profiling and cost-efficiency trade-offs
5. THE Course_System SHALL explain A/B testing frameworks for RAG systems
6. THE Course_System SHALL describe monitoring, observability, and feedback loops
7. THE Course_System SHALL address multi-language and low-resource language challenges
8. THE Course_System SHALL include hands-on exercises implementing production monitoring pipelines and designing A/B tests

### Requirement 10: Jupyter Notebook Development

**User Story:** As a student, I want interactive Jupyter notebooks with intentional bugs to fix, so that I can develop practical debugging skills through hands-on experimentation.

#### Acceptance Criteria

1. THE Course_System SHALL provide a minimum of four Jupyter notebooks covering search evolution, synthetic data generation (baseline and customized), Ragas evaluation, and semantic search evaluation
2. WHEN notebooks are created, THE Course_System SHALL include intentional bugs in prompts for students to debug
3. THE Course_System SHALL design all exercises as open-ended with multiple valid solutions
4. THE Course_System SHALL provide comparison frameworks to evaluate different approaches quantitatively
5. THE Course_System SHALL include code walkthroughs with architecture diagrams and data flow visualizations

### Requirement 11: Dataset Integration

**User Story:** As a student, I want access to curated datasets for hands-on practice, so that I can apply evaluation techniques to realistic data.

#### Acceptance Criteria

1. THE Course_System SHALL provide USC Course Catalog as primary dataset for tabular data and chunking practice
2. THE Course_System SHALL provide Amnesty Q&A dataset pre-formatted for Ragas with no preprocessing required
3. THE Course_System SHALL support optional student-provided domain-specific datasets
4. WHEN datasets are provided, THE Course_System SHALL include preprocessing scripts and data loading utilities

### Requirement 12: NVIDIA Platform Integration

**User Story:** As a certification candidate, I want comprehensive coverage of NVIDIA tools and platforms, so that I can demonstrate proficiency with NVIDIA ecosystem required for NCP-AAI certification.

#### Acceptance Criteria

1. THE Course_System SHALL demonstrate NVIDIA NIM for embeddings using NV-Embed-v2 and domain-specific models
2. THE Course_System SHALL demonstrate NVIDIA NIMs for LLM inference using Llama, Mistral, and custom models
3. THE Course_System SHALL demonstrate NVIDIA Nemotron-4-340B for synthetic data generation
4. THE Course_System SHALL reference NVIDIA Triton Inference Server for production deployment
5. THE Course_System SHALL reference NVIDIA NeMo for advanced agent development
6. THE Course_System SHALL reference NVIDIA Agent Intelligence Toolkit throughout course materials

### Requirement 13: Assessment and Knowledge Validation

**User Story:** As a student, I want comprehensive assessments throughout the course, so that I can validate my understanding and identify knowledge gaps before the certification exam.

#### Acceptance Criteria

1. THE Course_System SHALL provide module-end quizzes with 5-10 questions per module mixing conceptual and applied questions
2. THE Course_System SHALL include hands-on challenges as open-ended exercises with evaluation rubrics
3. THE Course_System SHALL provide debugging exercises with intentionally broken RAG pipelines
4. THE Course_System SHALL include design challenges requiring students to architect RAG systems for specific requirements
5. THE Course_System SHALL provide a capstone project requiring students to build and evaluate a complete RAG system for a custom domain
6. THE Course_System SHALL include a mock certification exam with 60-70 questions in a 120-minute timed simulation

### Requirement 14: Critical Pitfalls and Best Practices

**User Story:** As a student, I want to learn common pitfalls and best practices, so that I can avoid mistakes and implement effective RAG evaluation systems.

#### Acceptance Criteria

1. THE Course_System SHALL include dedicated sections addressing over-generic synthetic data with before/after prompt engineering examples
2. THE Course_System SHALL explain retrieval versus generation failure misdiagnosis with component-level debugging workflows
3. THE Course_System SHALL provide a decision matrix for wrong embedding model selection across domains
4. THE Course_System SHALL describe inappropriate chunk size issues with an experimentation framework using metrics
5. THE Course_System SHALL explain why traditional NLP metrics (BLEU, F1) fail for RAG and what to use instead
6. THE Course_System SHALL document prompt engineering mistakes including the 3-5 example rule and specificity requirements
7. THE Course_System SHALL describe production monitoring gaps including what to track and alerting strategies

### Requirement 15: Prompt Engineering Guidelines

**User Story:** As a student, I want to master prompt engineering for synthetic data generation and LLM-as-a-Judge evaluation, so that I can create effective prompts without extensive trial and error.

#### Acceptance Criteria

1. WHEN teaching synthetic data generation, THE Course_System SHALL explain writing prompts with extreme specificity as if explaining to a child
2. THE Course_System SHALL teach using exactly 3-5 examples for optimal steering without overfitting
3. THE Course_System SHALL demonstrate including explicit negative examples specifying what NOT to generate
4. THE Course_System SHALL explain specifying user personas and realistic scenarios
5. WHEN teaching LLM-as-a-Judge, THE Course_System SHALL explain defining explicit scoring rubrics with 0-1 scales and JSON output
6. THE Course_System SHALL demonstrate providing calibration examples for each score level
7. THE Course_System SHALL explain breaking complex metrics into multi-stage evaluations

### Requirement 16: Technical Setup and Environment

**User Story:** As a student, I want a pre-configured technical environment with clear setup instructions, so that I can focus on learning rather than troubleshooting installation issues.

#### Acceptance Criteria

1. THE Course_System SHALL provide a technical setup guide with JupyterLab environment configuration instructions
2. THE Course_System SHALL include NVIDIA API key setup instructions
3. THE Course_System SHALL provide dependency installation instructions with requirements.txt file
4. THE Course_System SHALL include troubleshooting guide for common setup issues
5. THE Course_System SHALL pre-load sample datasets in the environment
6. THE Course_System SHALL pre-install all required dependencies

### Requirement 17: Study Guide and Post-Course Resources

**User Story:** As a certification candidate, I want comprehensive study materials and post-course resources, so that I can continue learning and prepare effectively for the NCP-AAI exam.

#### Acceptance Criteria

1. THE Course_System SHALL provide a study guide with certification exam topic mapping
2. THE Course_System SHALL include key concepts summary as one-pagers per module
3. THE Course_System SHALL provide practice questions with detailed explanations
4. THE Course_System SHALL include a recommended reading list from external reference links
5. THE Course_System SHALL provide downloadable notebooks and datasets for continued practice
6. THE Course_System SHALL include links to NVIDIA AI Catalog for API access
7. THE Course_System SHALL provide community forum links and study group information
8. THE Course_System SHALL include additional NVIDIA course recommendations for comprehensive NCP-AAI preparation
9. THE Course_System SHALL provide a GitHub repository with course materials and updates

### Requirement 18: Lecture Materials and Visual Content

**User Story:** As an instructor, I want comprehensive lecture materials with visual diagrams, so that I can effectively deliver course content and explain complex concepts.

#### Acceptance Criteria

1. THE Course_System SHALL provide slide decks with visual diagrams for architecture, data flow, and metrics
2. THE Course_System SHALL include speaker notes with instructor tips extracted from course transcripts
3. THE Course_System SHALL provide real-world examples and case studies from finance, healthcare, legal, and e-commerce domains
4. WHEN explaining architectures, THE Course_System SHALL use Mermaid diagrams or equivalent visualization tools
5. THE Course_System SHALL include before/after examples demonstrating optimization techniques

### Requirement 19: Success Metrics and Learning Outcomes

**User Story:** As a course administrator, I want clearly defined success metrics and learning outcomes, so that I can measure course effectiveness and student readiness for certification.

#### Acceptance Criteria

1. WHEN a student completes the course, THE Course_System SHALL enable them to implement end-to-end RAG evaluation pipelines from scratch
2. WHEN a student completes the course, THE Course_System SHALL enable them to generate domain-specific synthetic test data with custom prompts
3. WHEN a student completes the course, THE Course_System SHALL enable them to select appropriate evaluation metrics for different RAG components
4. WHEN a student completes the course, THE Course_System SHALL enable them to diagnose retrieval versus generation failures independently
5. WHEN a student completes the course, THE Course_System SHALL enable them to customize Ragas metrics and create custom evaluators
6. WHEN a student completes the course, THE Course_System SHALL enable them to evaluate legacy semantic search systems with modern techniques
7. WHEN a student completes the course, THE Course_System SHALL enable them to deploy production-ready RAG systems with monitoring
8. THE Course_System SHALL prepare students to pass the NCP-AAI certification exam Evaluation and Tuning section

### Requirement 20: Content Depth and Technical Rigor

**User Story:** As a student with 1-2 years of AI/ML experience, I want technically rigorous content at professional level, so that I can develop production-ready skills beyond basic tutorials.

#### Acceptance Criteria

1. WHEN modules present concepts, THE Course_System SHALL include conceptual foundations with clear definitions and visual diagrams
2. THE Course_System SHALL provide historical context and evolution of techniques
3. THE Course_System SHALL document industry best practices and anti-patterns
4. WHEN modules present implementations, THE Course_System SHALL include step-by-step code walkthroughs
5. THE Course_System SHALL provide architecture diagrams with data flow
6. THE Course_System SHALL include configuration examples and parameter tuning guidance
7. WHEN modules present applications, THE Course_System SHALL include real-world use cases from multiple industries
8. THE Course_System SHALL document common pitfalls and debugging strategies
9. THE Course_System SHALL explain performance optimization techniques
10. THE Course_System SHALL include research paper references from RAG blog articles
11. THE Course_System SHALL describe cutting-edge techniques from 2024-2026 developments
