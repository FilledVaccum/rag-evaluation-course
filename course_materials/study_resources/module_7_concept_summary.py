"""
Module 7 Concept Summary: Production Deployment and Advanced Topics

One-page reference guide for production RAG deployment considerations.
"""

MODULE_7_CONCEPT_SUMMARY = """
# Module 7: Production Deployment and Advanced Topics - Concept Summary

## 1. Temporal Data Handling

### Time-Weighted Retrieval Strategies
| Strategy | Formula | Use Case |
|----------|---------|----------|
| **Exponential Decay** | score = semantic × e^(-λ × age) | News, trends |
| **Sliding Window** | Filter by date range | Current docs only |
| **Hybrid Scoring** | α × semantic + (1-α) × temporal | Balanced approach |
| **Version-Aware** | Track and prioritize versions | Documentation |

**Key Parameters**:
- λ (lambda): Decay rate (0.01 = slow, 0.1 = fast)
- α (alpha): Semantic vs temporal weight (0-1)
- Window size: Days to include

---

## 2. Regulatory Compliance

### GDPR (EU Data Protection)
✓ Right to be forgotten (delete user data)
✓ Data minimization (store only necessary)
✓ Consent management
✓ Transparency (explain data usage)

### HIPAA (US Healthcare)
✓ Encrypt PHI at rest and in transit
✓ Access controls and authentication
✓ Audit trails (log all access)
✓ Business Associate Agreements

### CCPA (California Privacy)
✓ Right to know (disclose data collection)
✓ Right to delete
✓ Right to opt-out

**Implementation Checklist**:
- [ ] Data classification system
- [ ] Encryption (AES-256)
- [ ] Audit logging (10-year retention)
- [ ] Access controls (RBAC)
- [ ] Retention policies (automated)
- [ ] Deletion workflows
- [ ] Regular compliance audits

---

## 3. Continuous Evaluation

### Evaluation Strategies
1. **Synthetic Query Evaluation** (Proactive)
   - Generate test queries daily
   - Run through pipeline
   - Track metrics over time
   
2. **Production Query Sampling** (Reactive)
   - Sample 1-5% of real queries
   - Annotate with ground truth
   - Evaluate quality

3. **A/B Testing** (Comparative)
   - Split traffic between variants
   - Collect metrics
   - Statistical significance testing

### Key Metrics to Monitor
| Category | Metrics | Threshold Example |
|----------|---------|-------------------|
| **Quality** | Faithfulness, Relevancy | > 0.70 |
| **Performance** | P95 Latency, QPS | < 2000ms |
| **Reliability** | Error Rate | < 5% |
| **Cost** | Cost per Query | < $0.01 |

### Alert Severity Levels
- **Critical**: Quality < threshold, Error rate > 5%
- **High**: Latency > threshold, Cost overrun
- **Medium**: Approaching thresholds
- **Low**: Informational

---

## 4. Performance Profiling

### RAG Pipeline Stages (Typical Latency)
```
Embedding:     100ms  (7%)   ← Cache, smaller model
Retrieval:      50ms  (4%)   ← ANN algorithms, GPU
Augmentation:   10ms  (1%)   ← Pre-format templates
Generation:   1200ms (88%)   ← Biggest bottleneck!
```

### Optimization Strategies
1. **Caching** (20-40% cost savings)
   - Cache common queries
   - TTL based on content freshness
   
2. **Query Routing** (15-30% cost savings)
   - Simple queries → Small model
   - Complex queries → Large model
   
3. **Batch Processing** (10-20% cost savings)
   - Batch embeddings
   - Batch LLM requests
   
4. **Context Optimization** (5-15% cost savings)
   - Reduce context window
   - Filter irrelevant chunks

### Cost-Quality Trade-offs
| Optimization | Cost Savings | Quality Impact |
|--------------|--------------|----------------|
| Smaller model | 40-60% | -5 to -10% |
| Caching | 20-40% | 0% (same quality) |
| Reduce context | 10-20% | -2 to -5% |
| Query routing | 15-30% | 0% (adaptive) |

---

## 5. A/B Testing

### A/B Test Framework
```
1. Define hypothesis (e.g., "Model B improves precision")
2. Split traffic (50/50 or 90/10 for canary)
3. Collect metrics (minimum 1000 samples per variant)
4. Statistical test (t-test, p-value < 0.05)
5. Make decision (promote winner or keep baseline)
```

### Statistical Significance
- **P-value < 0.05**: Significant (reject null hypothesis)
- **P-value ≥ 0.05**: Not significant (keep baseline)
- **Effect size**: Practical significance (Cohen's d)

### Common A/B Tests
- Embedding model comparison
- Chunk size optimization
- LLM model selection
- Prompt template variations
- Re-ranking strategies

---

## 6. Monitoring and Observability

### Three Pillars of Observability
1. **Metrics**: What is happening? (latency, error rate)
2. **Logs**: What happened? (query text, errors)
3. **Traces**: How did it happen? (request flow)

### Monitoring Dashboard
```
┌─────────────────────────────────────┐
│ Real-Time Metrics                   │
│ QPS: 125  P95: 850ms  Errors: 0.2% │
├─────────────────────────────────────┤
│ Quality Metrics (1h window)         │
│ Faithfulness: 0.89  Relevancy: 0.92 │
├─────────────────────────────────────┤
│ Cost Metrics                        │
│ Cost/Query: $0.0035  Daily: $437   │
├─────────────────────────────────────┤
│ Alerts (2)                          │
│ ⚠️  P95 latency approaching limit   │
│ ℹ️  Cache hit rate below target     │
└─────────────────────────────────────┘
```

### Feedback Loop Types
- **Explicit**: Thumbs up/down, ratings, corrections
- **Implicit**: Follow-up queries, time on page, abandonment

---

## 7. Multi-Language Challenges

### Challenges
- Embedding models are English-centric
- Poor performance on non-Latin scripts (Arabic, Hebrew, Chinese)
- Low-resource languages lack training data
- Code-switching (mixing languages)

### Solutions
1. **Multilingual Models**
   - multilingual-e5-large (100+ languages)
   - LaBSE (Language-agnostic BERT)
   - Domain-specific (arabic-bert, chinese-roberta)

2. **Language Detection & Routing**
   - Detect language early
   - Route to language-specific pipeline
   - Use appropriate embedding model

3. **Cross-Lingual Retrieval**
   - Option 1: Translate query to doc language
   - Option 2: Use multilingual embedding space

4. **Low-Resource Languages**
   - Translate to high-resource language
   - Use multilingual model with few-shot examples
   - Hybrid approach

---

## Key Takeaways

1. **Temporal Data**: Use exponential decay for time-sensitive content
2. **Compliance**: Plan for GDPR/HIPAA from day one (hard to retrofit)
3. **Continuous Evaluation**: Monitor quality continuously, not just at launch
4. **Performance**: Profile first, optimize bottlenecks (usually generation)
5. **A/B Testing**: Require statistical significance (p < 0.05) before promoting
6. **Monitoring**: Track quality, performance, reliability, and cost
7. **Multi-Language**: Use multilingual models, detect language early

---

## Production Readiness Checklist

### Before Launch
- [ ] Performance profiling complete
- [ ] Compliance requirements met
- [ ] Monitoring and alerting configured
- [ ] Incident response plan documented
- [ ] Cost optimization implemented
- [ ] A/B testing framework ready
- [ ] Continuous evaluation pipeline active
- [ ] Feedback loop implemented
- [ ] Documentation complete
- [ ] Team training complete

### After Launch
- [ ] Monitor metrics daily
- [ ] Review alerts and incidents
- [ ] Analyze user feedback
- [ ] Run weekly synthetic evaluations
- [ ] Conduct monthly A/B tests
- [ ] Quarterly compliance audits
- [ ] Continuous cost optimization
- [ ] Regular performance reviews

---

## Exam Focus Areas

**Deployment and Scaling (13%)**:
- Temporal data handling strategies
- Performance profiling and optimization
- Cost-efficiency trade-offs
- Scalability patterns

**Run, Monitor, and Maintain (5%)**:
- Monitoring and observability
- Alert thresholds and incident response
- Continuous evaluation pipelines
- Feedback loops

**Safety, Ethics, and Compliance (5%)**:
- GDPR, HIPAA, CCPA requirements
- Data protection and encryption
- Audit logging and access controls
- Right to deletion implementation

---

*This summary covers the essential concepts for Module 7. For detailed implementation
examples, refer to the lecture materials and hands-on notebooks.*
"""


def print_summary():
    """Print the concept summary."""
    print(MODULE_7_CONCEPT_summary)


if __name__ == "__main__":
    print_summary()
