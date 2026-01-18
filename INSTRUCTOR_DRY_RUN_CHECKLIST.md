# Instructor Dry Run Checklist
## RAG Evaluation Course - Comprehensive Validation

**Date:** January 18, 2026  
**Course:** Evaluating RAG and Semantic Search Systems  
**Target:** NVIDIA NCP-AAI Certification Preparation

---

## Pre-Dry Run Validation Status

### ✅ Completed Automated Validations

#### 1. Property-Based Tests (26 tests)
- ✅ All 26 property tests passed
- ✅ 100+ iterations per test
- ✅ No edge cases discovered that violate properties
- **Coverage:** Time allocation, module structure, exam mapping, content quality

#### 2. Unit Tests (98 tests)
- ✅ All 98 unit tests passed
- ✅ Dataset utilities validated
- ✅ Environment setup validated
- ✅ Data models validated
- ✅ NVIDIA API client validated
- **Code Coverage:** 52% (src/), 32% (overall including content)

#### 3. End-to-End Course Validation
- ✅ Datasets loaded successfully (USC Catalog: 20 courses, Amnesty Q&A: 5 records)
- ✅ All 7 module quizzes validated (8-10 questions each)
- ✅ Mock certification exam validated (65 questions, 120 minutes)
- ✅ All 8 notebooks validated and executable
- ✅ All assessments importable and functional

---

## Instructor Dry Run Objectives

### Primary Goals
1. **Content Delivery Validation**: Verify all materials can be delivered within time allocations
2. **Technical Setup Validation**: Ensure environment setup works smoothly for students
3. **Pedagogical Flow Validation**: Confirm learning progression is logical and effective
4. **Engagement Assessment**: Gauge student interest and comprehension
5. **Timing Calibration**: Validate 40/50/10 split (lecture/hands-on/discussion)

### Secondary Goals
1. Identify confusing explanations or concepts
2. Discover missing prerequisites or background knowledge
3. Test hands-on exercises for appropriate difficulty
4. Validate intentional bugs are discoverable
5. Assess certification exam preparation effectiveness

---

## Module-by-Module Dry Run Checklist

### Module 1: Evolution of Search to RAG (30-45 min)

#### Pre-Delivery Checklist
- [ ] Slides loaded and tested
- [ ] Mermaid diagrams render correctly
- [ ] Notebook 0 opens without errors
- [ ] Sample queries prepared

#### During Delivery - Observe
- [ ] Students understand BM25 vs semantic search distinction
- [ ] Hybrid system architecture is clear
- [ ] Decision framework is actionable
- [ ] Notebook 0 runs without technical issues
- [ ] Intentional bugs are discoverable

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 30-45 minutes sufficient?
- [ ] Clarity: Were concepts explained clearly?
- [ ] Engagement: Did students ask questions?
- [ ] Difficulty: Was the material too easy/hard?

**Quiz Validation:**
- [ ] 8 questions completed in reasonable time
- [ ] Questions test understanding, not memorization
- [ ] Explanations are helpful

---

### Module 2: Embeddings and Vector Stores (45-60 min)

#### Pre-Delivery Checklist
- [ ] NVIDIA NIM API keys configured
- [ ] Embedding models accessible
- [ ] USC Course Catalog dataset loaded
- [ ] Notebook 1 tested with live API calls

#### During Delivery - Observe
- [ ] Students grasp multi-dimensional similarity
- [ ] Domain-specific embedding selection is clear
- [ ] Chunking strategies make sense
- [ ] Tabular data transformation is understood
- [ ] API calls work reliably

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 45-60 minutes sufficient?
- [ ] Technical Issues: Any API failures or errors?
- [ ] Hands-On: Were exercises engaging?
- [ ] Difficulty: Appropriate for target audience?

**Quiz Validation:**
- [ ] 10 questions cover key concepts
- [ ] Applied questions test practical skills

---

### Module 3: RAG Architecture (60-90 min)

#### Pre-Delivery Checklist
- [ ] RAG pipeline diagrams ready
- [ ] Notebook 2 debugging exercises tested
- [ ] Intentional bugs documented
- [ ] Component-level debugging workflow clear

#### During Delivery - Observe
- [ ] Three-stage pipeline is understood
- [ ] Students can distinguish retrieval vs generation failures
- [ ] Debugging workflow is logical
- [ ] Intentional bugs are found and fixed
- [ ] Context relevance assessment is clear

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 60-90 minutes sufficient?
- [ ] Debugging: Were exercises helpful?
- [ ] Clarity: Was component separation clear?
- [ ] Engagement: Did students enjoy debugging?

**Debugging Exercise Validation:**
- [ ] Students identify retrieval failures
- [ ] Students identify generation failures
- [ ] Solutions are reasonable

---

### Module 4: Synthetic Data Generation (90-120 min)

#### Pre-Delivery Checklist
- [ ] NVIDIA Nemotron-4-340B accessible
- [ ] Notebooks 3 & 4 tested
- [ ] Baseline vs customized comparison ready
- [ ] 3-5 example pattern explained

#### During Delivery - Observe
- [ ] Students understand over-generalization problem
- [ ] Prompt engineering principles are clear
- [ ] 3-5 example pattern makes sense
- [ ] Negative examples concept is understood
- [ ] Quality validation is practical

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 90-120 minutes sufficient?
- [ ] Prompt Engineering: Was it intuitive?
- [ ] Hands-On: Were exercises valuable?
- [ ] Quality Improvement: Was it measurable?

**Hands-On Challenge Validation:**
- [ ] Students create custom prompts
- [ ] Quality scores improve measurably
- [ ] Students understand trade-offs

---

### Module 5: RAG Evaluation Metrics (120-150 min)

#### Pre-Delivery Checklist
- [ ] Ragas framework installed
- [ ] Amnesty Q&A dataset loaded
- [ ] Notebook 5 tested with all metrics
- [ ] LLM-as-a-Judge examples ready

#### During Delivery - Observe
- [ ] LLM-as-a-Judge methodology is clear
- [ ] Ragas framework is understood
- [ ] Metric differences are clear (faithfulness vs relevancy)
- [ ] Custom metric creation is feasible
- [ ] Interpretation leads to actionable insights

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 120-150 minutes sufficient?
- [ ] Complexity: Was Ragas overwhelming?
- [ ] Metrics: Were distinctions clear?
- [ ] Customization: Was it too advanced?

**Design Challenge Validation:**
- [ ] Students design evaluation pipelines
- [ ] Metric selection is justified
- [ ] Architectures are reasonable

---

### Module 6: Semantic Search Evaluation (90-120 min)

#### Pre-Delivery Checklist
- [ ] Legacy system examples ready
- [ ] Notebook 6 tested
- [ ] BM25 evaluation examples prepared
- [ ] Hybrid evaluation strategies clear

#### During Delivery - Observe
- [ ] Legacy system evaluation is understood
- [ ] Ragas adaptation to non-RAG is clear
- [ ] Hybrid strategies make sense
- [ ] Enterprise context is appreciated

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 90-120 minutes sufficient?
- [ ] Relevance: Was legacy focus valuable?
- [ ] Clarity: Were adaptations clear?
- [ ] Practicality: Is this applicable?

**Hands-On Challenge Validation:**
- [ ] Students evaluate semantic search
- [ ] Comparisons are meaningful
- [ ] Insights are actionable

---

### Module 7: Production Deployment (60-90 min)

#### Pre-Delivery Checklist
- [ ] Production architecture diagrams ready
- [ ] Notebook 7 monitoring examples tested
- [ ] Compliance examples prepared
- [ ] A/B testing framework explained

#### During Delivery - Observe
- [ ] Production considerations are understood
- [ ] Monitoring strategies are clear
- [ ] Compliance requirements are appreciated
- [ ] A/B testing is practical
- [ ] Cost-efficiency trade-offs make sense

#### Post-Delivery - Collect Feedback
- [ ] Timing: Was 60-90 minutes sufficient?
- [ ] Relevance: Was production focus valuable?
- [ ] Depth: Was coverage sufficient?
- [ ] Practicality: Can students apply this?

**Design Challenge Validation:**
- [ ] Students design production architectures
- [ ] Monitoring plans are comprehensive
- [ ] Compliance is addressed

---

## Overall Course Validation

### Time Allocation Validation

| Module | Target Time | Actual Time | Variance | Notes |
|--------|-------------|-------------|----------|-------|
| Module 1 | 30-45 min | ___ min | ___ | |
| Module 2 | 45-60 min | ___ min | ___ | |
| Module 3 | 60-90 min | ___ min | ___ | |
| Module 4 | 90-120 min | ___ min | ___ | |
| Module 5 | 120-150 min | ___ min | ___ | |
| Module 6 | 90-120 min | ___ min | ___ | |
| Module 7 | 60-90 min | ___ min | ___ | |
| **Total** | **6-8 hours** | **___ hours** | **___** | |

### 40/50/10 Split Validation

| Module | Lecture % | Hands-On % | Discussion % | Compliant? |
|--------|-----------|------------|--------------|------------|
| Module 1 | ___ | ___ | ___ | [ ] |
| Module 2 | ___ | ___ | ___ | [ ] |
| Module 3 | ___ | ___ | ___ | [ ] |
| Module 4 | ___ | ___ | ___ | [ ] |
| Module 5 | ___ | ___ | ___ | [ ] |
| Module 6 | ___ | ___ | ___ | [ ] |
| Module 7 | ___ | ___ | ___ | [ ] |

---

## Technical Issues Log

### Environment Setup Issues
- [ ] No issues
- [ ] Issues encountered:
  - Issue: ___
  - Resolution: ___
  - Time Lost: ___

### API Integration Issues
- [ ] No issues
- [ ] Issues encountered:
  - Issue: ___
  - Resolution: ___
  - Time Lost: ___

### Dataset Loading Issues
- [ ] No issues
- [ ] Issues encountered:
  - Issue: ___
  - Resolution: ___
  - Time Lost: ___

### Notebook Execution Issues
- [ ] No issues
- [ ] Issues encountered:
  - Notebook: ___
  - Issue: ___
  - Resolution: ___
  - Time Lost: ___

---

## Student Feedback Collection

### Comprehension Assessment (1-5 scale)

| Topic | Understanding | Confidence | Interest |
|-------|---------------|------------|----------|
| Search Evolution | ___ | ___ | ___ |
| Embeddings | ___ | ___ | ___ |
| RAG Architecture | ___ | ___ | ___ |
| Synthetic Data | ___ | ___ | ___ |
| Evaluation Metrics | ___ | ___ | ___ |
| Semantic Search | ___ | ___ | ___ |
| Production | ___ | ___ | ___ |

### Open-Ended Feedback

**What worked well?**
- ___
- ___
- ___

**What was confusing?**
- ___
- ___
- ___

**What should be added?**
- ___
- ___
- ___

**What should be removed?**
- ___
- ___
- ___

**Certification Preparation:**
- Do you feel prepared for the NCP-AAI Evaluation & Tuning section? (Yes/No/Partially)
- What additional preparation would help?

---

## Certification Alignment Validation

### Exam Domain Coverage Assessment

| Domain | Weight | Coverage | Student Confidence |
|--------|--------|----------|-------------------|
| Evaluation & Tuning | 13% | ⭐⭐⭐ | ___ |
| Knowledge Integration | 10% | ⭐⭐⭐ | ___ |
| Agent Development | 15% | ⭐⭐ | ___ |
| Agent Architecture | 15% | ⭐⭐ | ___ |
| Deployment | 13% | ⭐ | ___ |
| Run/Monitor/Maintain | 5% | ⭐ | ___ |
| NVIDIA Platform | 7% | ⭐⭐ | ___ |

### Mock Exam Performance
- [ ] Mock exam administered
- Average Score: ___%
- Pass Rate (70%+): ___%
- Time Completion: ___ minutes (target: 120)
- Feedback on difficulty: ___

---

## Refinement Recommendations

### Content Adjustments
1. **Add:**
   - ___
   - ___

2. **Remove:**
   - ___
   - ___

3. **Clarify:**
   - ___
   - ___

4. **Expand:**
   - ___
   - ___

### Timing Adjustments
- Module ___ needs +/- ___ minutes
- Overall pacing: Too Fast / Just Right / Too Slow

### Technical Improvements
- ___
- ___

### Pedagogical Improvements
- ___
- ___

---

## Final Readiness Assessment

### Course Delivery Readiness
- [ ] All materials tested and working
- [ ] Timing validated and adjusted
- [ ] Technical issues resolved
- [ ] Content clarity confirmed
- [ ] Student engagement verified

### Instructor Preparedness
- [ ] Comfortable with all content
- [ ] Familiar with all tools and platforms
- [ ] Prepared for common questions
- [ ] Backup plans for technical issues
- [ ] Assessment rubrics understood

### Student Readiness Indicators
- [ ] Students can complete exercises independently
- [ ] Students understand key concepts
- [ ] Students feel prepared for certification
- [ ] Students provide positive feedback
- [ ] Students recommend course to others

---

## Sign-Off

**Instructor:** ___________________  
**Date:** ___________________  
**Overall Assessment:** Ready / Needs Minor Adjustments / Needs Major Revision  

**Key Takeaways:**
1. ___
2. ___
3. ___

**Action Items Before Production:**
1. ___
2. ___
3. ___

---

## Appendix: Automated Validation Results

### Property-Based Tests Summary
- Total Tests: 26
- Passed: 26
- Failed: 0
- Edge Cases: None discovered
- Execution Time: 2.86s

### Unit Tests Summary
- Total Tests: 98
- Passed: 98
- Failed: 0
- Code Coverage: 52% (src/), 32% (overall)
- Execution Time: 1.16s

### End-to-End Validation Summary
- Datasets: ✅ Loaded successfully
- Notebooks: ✅ All 8 validated
- Assessments: ✅ All 7 quizzes + mock exam validated
- NVIDIA APIs: ⚠️ Requires live testing during dry run

---

**Document Version:** 1.0  
**Last Updated:** January 18, 2026  
**Next Review:** After instructor dry run completion
