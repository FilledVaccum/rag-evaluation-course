---
inclusion: always
---

# Course Development Guide: Evaluating RAG and Semantic Search Systems
## Preparing for NVIDIA-Certified Professional: Agentic AI (NCP-AAI) Certification

## Course Overview

This steering document guides the creation of a comprehensive course on "Evaluating RAG and Semantic Search Systems" with the primary goal of preparing candidates to pass the NVIDIA-Certified Professional: Agentic AI (NCP-AAI) certification exam.

### Target Certification Details
- **Certification Level**: Professional (Intermediate)
- **Exam Duration**: 120 minutes
- **Number of Questions**: 60-70 questions
- **Prerequisites**: 1-2 years of AI/ML experience with production-level agentic AI projects
- **Focus Areas**: Agent development, architecture, orchestration, multi-agent frameworks, evaluation, observability, deployment, and reliability guardrails

## Certification Exam Blueprint Alignment

### Primary Topic Coverage (Direct Alignment)
1. **Evaluation and Tuning** (13% of exam) - PRIMARY FOCUS
2. **Knowledge Integration and Data Handling** (10% of exam) - CORE CONTENT
3. **Agent Development** (15% of exam) - SUPPORTING CONTENT
4. **Agent Architecture and Design** (15% of exam) - CONTEXTUAL FOUNDATION

### Secondary Topic Coverage (Indirect Support)
- Deployment and Scaling (13%)
- Run, Monitor, and Maintain (5%)
- NVIDIA Platform Implementation (7%)

## Course Learning Objectives

### Core Competencies to Develop

1. **RAG System Evaluation Fundamentals**
   - Understand the evolution from keyword-based search (BM25) to modern vector stores and semantic retrieval
   - Master the unique challenges of deploying RAG in enterprise settings
   - Integrate legacy and modern retrieval systems effectively

2. **Evaluation Framework Mastery**
   - Implement evaluation pipelines using Ragas and similar frameworks
   - Apply LLM-as-a-Judge techniques for scalable metric-based and qualitative assessment
   - Compare agent performance across tasks and datasets
   - Analyze evaluation results to guide targeted optimization

3. **Synthetic Test Data Generation**
   - Generate, customize, and validate synthetic test sets reflecting realistic user queries
   - Apply prompt engineering techniques to steer synthetic data generation
   - Create domain-specific test data that reflects authentic business scenarios
   - Support test-driven development for RAG systems

4. **Component-Level Evaluation**
   - Independently evaluate retrieval and generation stages of RAG pipelines
   - Use standard and custom metrics to identify performance bottlenecks
   - Debug specific components within the RAG flow
   - Implement structured user feedback collection for iterative improvements

5. **Enterprise-Grade Considerations**
   - Handle temporal data and evolving information needs
   - Address domain-specific language requirements
   - Ensure compliance with regulatory standards
   - Manage data quality, augmentation, and preprocessing

6. **Advanced Evaluation Techniques**
   - Tune model parameters for accuracy and latency-efficiency trade-offs
   - Implement retrieval pipelines (RAG, embedded search, hybrid approaches)
   - Configure and optimize vector databases for fast retrieval
   - Build ETL pipelines to integrate enterprise data sources

## Course Structure and Content Modules

### Module 1: Evolution of Search and RAG Systems (Foundation)
**Duration**: 30-45 minutes  
**Certification Alignment**: Knowledge Integration (10%), Agent Architecture (15%)

#### Key Topics:
- Classic search flow: crawling, analysis, indexing, ranking
- Keyword-based search systems and BM25 ranking
- Limitations of closed-source ranking algorithms
- Introduction to semantic search systems
- Transition from search to RAG architecture
- Enterprise hybrid systems (BM25 + Vector Search + Re-ranking)

#### Hands-On Activities:
- **Notebook 0**: Evolution from classic search to RAG
  - Run same query across different search paradigms
  - Compare keyword search vs. semantic search results
  - Understand ranking algorithm control and customization

#### Learning Outcomes:
- Explain the progression from traditional search to RAG
- Identify when to use keyword vs. semantic search
- Understand enterprise requirements for hybrid systems

### Module 2: Embeddings and Vector Stores (Technical Foundation)
**Duration**: 45-60 minutes  
**Certification Alignment**: Knowledge Integration (10%), NVIDIA Platform (7%)

#### Key Topics:
- Embedding fundamentals and multi-dimensional similarity
- Domain-specific embedding models (code, finance, healthcare, multilingual)
- NVIDIA NIM embedding models and alternatives
- Vector store configuration and optimization
- Handling tabular data in embedding space
- Chunking strategies and optimal text length

#### Hands-On Activities:
- Implement embeddings using NVIDIA NIM
- Experiment with different embedding models
- Transform tabular data (USC course catalog) into embeddings
- Test various chunking strategies

#### Learning Outcomes:
- Select appropriate embedding models for specific domains
- Configure vector stores for optimal retrieval
- Handle diverse data types in RAG systems

### Module 3: RAG Architecture and Component Analysis (Core)
**Duration**: 60-90 minutes  
**Certification Alignment**: Agent Architecture (15%), Agent Development (15%)

#### Key Topics:
- RAG pipeline architecture: retrieval → augmentation → generation
- Component-level failure analysis
- Retrieval stage evaluation
- Generation stage evaluation
- Context relevance assessment
- Response accuracy and faithfulness
- Orchestration and multi-step reasoning

#### Hands-On Activities:
- Build complete RAG pipeline
- Identify failure points in retrieval vs. generation
- Analyze context relevance independently
- Debug component-specific issues

#### Learning Outcomes:
- Architect end-to-end RAG systems
- Diagnose failures at specific pipeline stages
- Implement error handling and graceful failure recovery

### Module 4: Synthetic Test Data Generation (Critical Skill)
**Duration**: 90-120 minutes  
**Certification Alignment**: Evaluation and Tuning (13%), Agent Development (15%)

#### Key Topics:
- Importance of test-driven development for LLMs and RAG
- Synthetic data generation fundamentals
- LLM-based test data creation
- Prompt engineering for data generation
- Steering synthetic data with custom prompts
- Question style, length, and persona customization
- Domain-specific query generation
- NVIDIA Nemotron-4-340B synthetic data models
- Quality validation and filtering

#### Hands-On Activities:
- **Notebook 1**: Generate synthetic test data (out-of-the-box)
  - Use USC course catalog dataset
  - Generate initial synthetic questions
  - Identify over-generalization problems
  
- **Notebook 2**: Customize synthetic data generation
  - Modify prompts to steer data generation
  - Add domain-specific examples (3-5 examples optimal)
  - Create student-focused questions vs. philosophical queries
  - Experiment with different synthesizers
  - Combine multiple synthesizers (50-50 mix)

#### Learning Outcomes:
- Generate robust test sets for RAG evaluation
- Customize prompts to reflect authentic user queries
- Validate synthetic data quality
- Support continuous evaluation workflows

### Module 5: RAG Evaluation Metrics and Frameworks (Core Competency)
**Duration**: 120-150 minutes  
**Certification Alignment**: Evaluation and Tuning (13% - PRIMARY)

#### Key Topics:
- Limitations of traditional NLP metrics (BLEU, F1) for RAG
- LLM-as-a-Judge methodology
- Ragas framework and alternatives
- Generation metrics: Faithfulness, Answer Relevancy
- Retrieval metrics: Context Precision, Context Recall
- Multi-stage metric computation
- Custom metric development
- Metric interpretation and actionable insights

#### Specialized Metrics:
1. **Context Relevance**: Is retrieved context relevant to the question?
2. **Context Utilization**: Does response use retrieved context?
3. **Answer Accuracy**: Is response accurate and relevant to question?
4. **Faithfulness**: Are claims in response supported by context?
5. **Context Precision**: Ranking quality of retrieved contexts
6. **Context Recall**: Coverage of ground truth in retrieved contexts

#### Hands-On Activities:
- **Notebook 3**: Implement evaluation metrics
  - Use Amnesty Q&A dataset (pre-formatted for Ragas)
  - Compute faithfulness and context recall metrics
  - Analyze metric outputs
  - Customize existing metrics with prompt modifications
  - Create custom metrics from scratch
  - Compare metric performance

#### Advanced Exercises:
- Modify faithfulness prompts for domain-specific evaluation
- Create custom metrics for enterprise requirements
- Implement multi-stage evaluation pipelines
- Tune metric parameters for specific use cases

#### Learning Outcomes:
- Implement comprehensive evaluation pipelines
- Select appropriate metrics for different RAG components
- Customize metrics for domain-specific requirements
- Interpret evaluation results to guide optimization

### Module 6: Semantic Search System Evaluation (Enterprise Focus)
**Duration**: 90-120 minutes  
**Certification Alignment**: Evaluation and Tuning (13%), Knowledge Integration (10%)

#### Key Topics:
- Legacy semantic search systems in enterprises
- Applying modern LLM evaluation to traditional search
- Hybrid evaluation strategies (RAG + Semantic Search)
- BM25 system evaluation with LLM-as-a-Judge
- Ranking algorithm assessment
- Search result relevance metrics
- Integration with existing enterprise systems

#### Hands-On Activities:
- **Notebook 4+**: Evaluate semantic search systems
  - Apply Ragas to non-RAG search systems
  - Customize evaluation for keyword search
  - Compare RAG vs. semantic search performance
  - Implement hybrid evaluation workflows

#### Learning Outcomes:
- Evaluate legacy search systems with modern techniques
- Bridge traditional and modern evaluation approaches
- Support enterprise hybrid system requirements

### Module 7: Advanced Topics and Production Considerations (Professional Level)
**Duration**: 60-90 minutes  
**Certification Alignment**: Deployment (13%), Run/Monitor/Maintain (5%)

#### Key Topics:
- Temporal data handling and time-weighted retrieval
- Regulatory compliance and data quality checks
- Continuous evaluation in production
- Performance profiling and optimization
- Cost-efficiency vs. accuracy trade-offs
- Monitoring and observability
- A/B testing for RAG systems
- Iterative improvement workflows

#### Enterprise Challenges:
- Multi-language support and low-resource languages
- Domain-specific terminology (finance, healthcare, legal)
- Data privacy and security considerations
- Scalability and latency requirements
- Integration with existing MLOps pipelines

#### Learning Outcomes:
- Deploy evaluation systems at production scale
- Monitor RAG performance continuously
- Handle enterprise-specific requirements
- Implement feedback loops for improvement

## Practical Implementation Guidelines

### Dataset Selection Strategy

1. **Primary Dataset: USC Course Catalog**
   - **Purpose**: Focused, domain-specific evaluation
   - **Format**: Tabular data (good for chunking exercises)
   - **Use Cases**: Student-focused queries, factual retrieval
   - **Challenges**: Handling structured data in embeddings

2. **Secondary Dataset: Amnesty Q&A**
   - **Purpose**: Pre-formatted for Ragas (user_input, retrieved_context, response, ground_truth)
   - **Use Cases**: Metric computation, faithfulness evaluation
   - **Advantages**: No preprocessing required

3. **Custom Datasets**: Encourage students to bring domain-specific data

### Tool and Framework Stack

#### Core Frameworks:
- **Ragas**: Primary evaluation framework (open-source, large community)
- **NVIDIA NIM**: Embedding models and LLM endpoints
- **LangChain**: Optional for orchestration
- **Vector Stores**: Milvus, Pinecone, or alternatives

#### NVIDIA Platform Integration:
- NVIDIA NIM for embeddings (NV-Embed models)
- NVIDIA NIMs for LLM inference
- NVIDIA Triton for production deployment
- NVIDIA NeMo for advanced agent development
- NVIDIA Nemotron models for synthetic data generation

### Prompt Engineering Best Practices

#### For Synthetic Data Generation:
1. **Be Extremely Specific**: Write prompts as if explaining to a child
2. **Use 3-5 Examples**: Optimal number for steering behavior
3. **Include Domain Context**: Specify user personas and scenarios
4. **Avoid Over-Generalization**: Explicitly state what NOT to generate
5. **Iterate and Validate**: Test prompts multiple times for consistency

#### For LLM-as-a-Judge:
1. **Clear Evaluation Criteria**: Define scoring rubrics explicitly
2. **Structured Output**: Request specific formats (0-1 scores, JSON)
3. **Multi-Stage Evaluation**: Break complex metrics into steps
4. **Example-Driven**: Provide scoring examples for calibration

### Common Pitfalls and Solutions

#### Pitfall 1: Over-Generic Synthetic Data
**Problem**: Generated questions are too philosophical or broad  
**Solution**: Add specific examples, constrain question types, use domain-specific synthesizers

#### Pitfall 2: Retrieval Failure Misdiagnosed as Generation Failure
**Problem**: Blaming LLM when embedding model is the issue  
**Solution**: Evaluate retrieval and generation independently

#### Pitfall 3: Wrong Embedding Model for Domain
**Problem**: Using general embeddings for specialized domains (Hebrew, Arabic, medical, legal)  
**Solution**: Select domain-specific or train custom embedding models

#### Pitfall 4: Inappropriate Chunk Size
**Problem**: Missing context due to too-small chunks or inefficiency from too-large chunks  
**Solution**: Experiment with chunking strategies, use overlap, validate with metrics

#### Pitfall 5: Relying on Traditional NLP Metrics
**Problem**: Using BLEU, F1 scores for RAG evaluation  
**Solution**: Implement RAG-specific metrics (faithfulness, context relevance)

## Assessment and Certification Preparation

### Knowledge Check Questions (Throughout Course)

#### Module 1-2: Foundations
- Explain the difference between BM25 and semantic search
- When would you use a hybrid search system?
- How do embedding models impact RAG performance?
- What are the challenges of handling tabular data in RAG?

#### Module 3-4: RAG Architecture and Testing
- Describe the three main stages of a RAG pipeline
- How do you diagnose retrieval vs. generation failures?
- What makes synthetic test data "high quality" for RAG?
- How many prompt examples are typically sufficient for steering?

#### Module 5-6: Evaluation Mastery
- What is LLM-as-a-Judge and when is it appropriate?
- Explain the difference between faithfulness and answer relevancy
- How do you evaluate a legacy BM25 system with modern techniques?
- What metrics would you use to evaluate retrieval quality?

#### Module 7: Production and Enterprise
- How do you handle temporal data in RAG systems?
- What are key considerations for production RAG monitoring?
- How do you balance cost and accuracy in RAG deployments?
- Describe continuous evaluation workflows

### Hands-On Exercises (Open-Ended)

All exercises should be open-ended with no single "correct" answer, allowing students to:
- Experiment with different approaches
- Compare results quantitatively
- Develop critical thinking about trade-offs
- Build practical skills for real-world scenarios

### Certification Exam Preparation Tips

1. **Focus on Practical Application**: Exam tests real-world scenarios, not just theory
2. **Understand Component Interactions**: Know how retrieval, augmentation, and generation work together
3. **Master Evaluation Metrics**: Be able to select and interpret appropriate metrics
4. **Know NVIDIA Tools**: Familiarity with NIM, NeMo, Triton, and related platforms
5. **Enterprise Considerations**: Understand production deployment, monitoring, and compliance
6. **Multi-Agent Context**: RAG evaluation in the context of agentic AI systems

### Study Resources Alignment

#### NVIDIA Courses (Recommended):
- Building RAG Agents With LLMs
- Building Agentic AI Applications with LLMs
- Evaluating RAG and Semantic Search Systems (this course)
- Deploying RAG Pipelines for Production at Scale

#### Key Reading Materials:
- NVIDIA Agent Intelligence Toolkit documentation
- Ragas documentation and tutorials
- NVIDIA NIM and NeMo documentation
- Research papers on RAG evaluation methodologies

## Course Delivery Recommendations

### Time Allocation
- **Total Course Duration**: 6-8 hours (can be split into 2-3 sessions)
- **Lecture/Demo**: 40% of time
- **Hands-On Practice**: 50% of time
- **Discussion/Q&A**: 10% of time

### Pacing Strategy
- Start with foundational concepts (Modules 1-2): 25%
- Deep dive into core skills (Modules 3-5): 50%
- Advanced topics and integration (Modules 6-7): 25%

### Student Support
- Provide downloadable notebooks for continued learning
- Include intentional "bugs" in prompts for students to fix
- Encourage experimentation beyond course time
- Offer optional advanced exercises for fast learners

### Technical Setup
- Pre-configured JupyterLab environment
- NVIDIA API keys provided (or instructions for obtaining)
- Sample datasets pre-loaded
- All dependencies pre-installed

## Success Metrics for Course

### Student Outcomes:
- Ability to implement end-to-end RAG evaluation pipeline
- Proficiency in generating domain-specific synthetic test data
- Competence in selecting and customizing evaluation metrics
- Understanding of enterprise RAG deployment considerations
- Readiness to tackle NCP-AAI certification exam (Evaluation section)

### Course Quality Indicators:
- Students can complete exercises independently
- Generated synthetic data shows clear improvement after customization
- Students can explain trade-offs in different evaluation approaches
- Positive feedback on practical applicability
- High certification exam pass rate (for Evaluation and Tuning section)

## Post-Course Resources

### Continued Learning:
- Download all notebooks and datasets
- Access to NVIDIA AI Catalog for API keys
- Links to NVIDIA documentation and tutorials
- Community forums and support channels
- Additional reading materials and research papers

### Certification Preparation:
- Review all 10 certification topic areas
- Take additional NVIDIA courses for comprehensive coverage
- Practice with NVIDIA Agent Intelligence Toolkit
- Build portfolio projects demonstrating RAG evaluation skills
- Join study groups and certification preparation communities

## Key Takeaways for Course Developers

1. **Test-Driven Development is Critical**: Emphasize that RAG evaluation is not optional but essential for production systems

2. **Component-Level Debugging**: Teach students to evaluate retrieval and generation independently to avoid wasting time on wrong optimizations

3. **Prompt Engineering is Powerful**: Show that sophisticated evaluation doesn't always require training new models—good prompts can achieve excellent results

4. **Enterprise Reality**: Address the fact that organizations have legacy systems and need hybrid approaches

5. **Practical Over Theoretical**: Focus on hands-on skills that students can immediately apply in their work

6. **Certification Context**: Frame all content in the context of the broader NCP-AAI certification requirements

7. **Open-Ended Learning**: Encourage experimentation and critical thinking rather than memorization

8. **NVIDIA Ecosystem**: Integrate NVIDIA tools and platforms throughout to prepare students for certification and real-world use

---

## Quick Reference: Certification Exam Coverage

| Exam Topic | Weight | Course Coverage | Key Modules |
|------------|--------|-----------------|-------------|
| Evaluation and Tuning | 13% | ⭐⭐⭐ PRIMARY | 4, 5, 6 |
| Knowledge Integration | 10% | ⭐⭐⭐ CORE | 2, 3, 6 |
| Agent Development | 15% | ⭐⭐ SUPPORTING | 3, 4 |
| Agent Architecture | 15% | ⭐⭐ FOUNDATION | 1, 3 |
| Deployment and Scaling | 13% | ⭐ CONTEXTUAL | 7 |
| Run, Monitor, Maintain | 5% | ⭐ CONTEXTUAL | 7 |
| NVIDIA Platform | 7% | ⭐⭐ INTEGRATED | All |
| Cognition, Planning, Memory | 10% | ⭐ REFERENCED | 3 |
| Safety, Ethics, Compliance | 5% | ⭐ REFERENCED | 7 |
| Human-AI Interaction | 5% | ⭐ REFERENCED | 7 |

**Legend**: ⭐⭐⭐ Primary Focus | ⭐⭐ Significant Coverage | ⭐ Supporting/Contextual

---

*This steering document should be used as the authoritative guide for developing course materials, exercises, assessments, and delivery strategies for the "Evaluating RAG and Semantic Search Systems" course with NCP-AAI certification preparation as the primary objective.*
