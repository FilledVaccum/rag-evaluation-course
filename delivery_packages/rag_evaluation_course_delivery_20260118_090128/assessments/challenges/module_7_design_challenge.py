"""
Module 7 Design Challenge: Production RAG Architecture

This design challenge requires students to architect a production-ready RAG system
with comprehensive monitoring, compliance, and optimization considerations.

Challenge Type: System Design
Duration: 60-90 minutes
Difficulty: Advanced
"""

from typing import List, Dict, Any
from dataclasses import dataclass


DESIGN_CHALLENGE_PROMPT = """
# Design Challenge: Production RAG System for Healthcare

## Scenario

You are the lead architect for a healthcare company deploying a RAG system to help
doctors quickly find relevant medical research and treatment guidelines. The system
must handle 10,000 queries per day from 500 doctors across multiple hospitals.

## Requirements

### Functional Requirements
1. Support medical literature search (100,000+ research papers)
2. Provide treatment guideline recommendations
3. Handle queries in English and Spanish
4. Return responses within 2 seconds (P95)
5. Maintain high accuracy (faithfulness > 0.90)

### Non-Functional Requirements
1. HIPAA compliance (Protected Health Information)
2. 99.9% uptime (< 45 minutes downtime per month)
3. Cost budget: $500/day maximum
4. Audit trail for all queries and responses
5. Continuous quality monitoring

### Constraints
1. Cannot use public cloud services for PHI storage
2. Must support offline operation during network outages
3. Must integrate with existing hospital authentication system
4. Regulatory audit every 6 months

## Your Task

Design a complete production RAG architecture that addresses:

### Part 1: System Architecture (30 points)
Design the overall system architecture including:
- RAG pipeline components (embedding, retrieval, generation)
- Infrastructure (servers, load balancers, databases)
- Data flow and component interactions
- Scalability considerations

Deliverable: Architecture diagram with component descriptions

### Part 2: Compliance and Security (25 points)
Design compliance and security measures:
- HIPAA compliance implementation
- Data encryption strategy
- Access control and authentication
- Audit logging system
- Data retention and deletion policies

Deliverable: Compliance checklist and security architecture

### Part 3: Monitoring and Observability (20 points)
Design monitoring and alerting system:
- Key metrics to track
- Alert thresholds and escalation
- Dashboard design
- Incident response workflow

Deliverable: Monitoring strategy document

### Part 4: Cost Optimization (15 points)
Design cost optimization strategy:
- Cost breakdown by component
- Optimization opportunities
- Trade-offs between cost and quality
- Budget allocation

Deliverable: Cost analysis and optimization plan

### Part 5: Continuous Evaluation (10 points)
Design continuous evaluation pipeline:
- Evaluation metrics and frequency
- Synthetic test data generation
- A/B testing framework
- Feedback loop implementation

Deliverable: Evaluation pipeline design

## Evaluation Rubric

### System Architecture (30 points)
- **Excellent (25-30)**: Comprehensive architecture with clear component separation,
  scalability considerations, and failure handling. Includes detailed data flow.
- **Good (18-24)**: Solid architecture with main components identified. Some scalability
  considerations. Basic failure handling.
- **Adequate (12-17)**: Basic architecture with essential components. Limited scalability
  or failure handling considerations.
- **Needs Improvement (0-11)**: Incomplete architecture or missing critical components.

### Compliance and Security (25 points)
- **Excellent (21-25)**: Comprehensive HIPAA compliance with detailed security measures.
  Clear audit trail and data protection strategies. Addresses all regulatory requirements.
- **Good (16-20)**: Good compliance coverage with main security measures. Some gaps in
  audit trail or data protection.
- **Adequate (10-15)**: Basic compliance measures. Missing some security considerations
  or audit requirements.
- **Needs Improvement (0-9)**: Incomplete compliance or significant security gaps.

### Monitoring and Observability (20 points)
- **Excellent (17-20)**: Comprehensive monitoring with appropriate metrics, thresholds,
  and alerting. Clear incident response workflow.
- **Good (13-16)**: Good monitoring coverage with main metrics. Basic alerting and
  incident response.
- **Adequate (8-12)**: Basic monitoring with essential metrics. Limited alerting or
  incident response.
- **Needs Improvement (0-7)**: Incomplete monitoring or missing critical metrics.

### Cost Optimization (15 points)
- **Excellent (13-15)**: Detailed cost analysis with realistic estimates. Multiple
  optimization strategies with clear trade-offs.
- **Good (10-12)**: Good cost analysis with main components. Some optimization strategies.
- **Adequate (6-9)**: Basic cost analysis. Limited optimization strategies.
- **Needs Improvement (0-5)**: Incomplete cost analysis or unrealistic estimates.

### Continuous Evaluation (10 points)
- **Excellent (9-10)**: Comprehensive evaluation pipeline with multiple metrics,
  automated testing, and feedback loops.
- **Good (7-8)**: Good evaluation approach with main metrics and some automation.
- **Adequate (5-6)**: Basic evaluation with essential metrics.
- **Needs Improvement (0-4)**: Incomplete evaluation or missing key components.

## Starter Template

Use this template to structure your design:
"""


DESIGN_TEMPLATE = """
# Production RAG System Design: Healthcare Medical Research Assistant

## Executive Summary
[Brief overview of your design approach and key decisions]

## Part 1: System Architecture

### High-Level Architecture
```
[Insert architecture diagram here using Mermaid or ASCII art]
```

### Component Descriptions

#### 1. RAG Pipeline
- **Embedding Model**: [Model choice and rationale]
- **Vector Store**: [Vector store choice and configuration]
- **LLM**: [LLM choice and rationale]
- **Retrieval Strategy**: [Top-k, re-ranking, etc.]

#### 2. Infrastructure
- **Compute**: [Server specifications and scaling]
- **Storage**: [Database and vector store infrastructure]
- **Networking**: [Load balancing and routing]
- **Caching**: [Caching strategy]

#### 3. Data Flow
[Describe the flow from query to response]

#### 4. Scalability
[How system scales to handle increased load]

#### 5. Failure Handling
[How system handles component failures]

---

## Part 2: Compliance and Security

### HIPAA Compliance Checklist
- [ ] Data encryption at rest
- [ ] Data encryption in transit
- [ ] Access controls and authentication
- [ ] Audit logging
- [ ] Data retention policies
- [ ] Business Associate Agreements
- [ ] Incident response plan
- [ ] Regular security audits

### Security Architecture
[Describe security measures in detail]

### Audit Logging
[What is logged and how]

### Data Protection
[Encryption, access control, de-identification]

---

## Part 3: Monitoring and Observability

### Key Metrics

#### Performance Metrics
- Query latency (P50, P95, P99)
- Throughput (QPS)
- Error rate
- Cache hit rate

#### Quality Metrics
- Faithfulness (target: > 0.90)
- Answer relevancy
- Context precision

#### System Health Metrics
- CPU/Memory utilization
- Disk I/O
- Network latency

### Alert Thresholds
| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| P95 Latency | > 1.5s | > 2.0s | [Action] |
| Error Rate | > 2% | > 5% | [Action] |
| Faithfulness | < 0.85 | < 0.80 | [Action] |

### Dashboard Design
[Describe monitoring dashboard layout and key visualizations]

### Incident Response
[Workflow for handling alerts and incidents]

---

## Part 4: Cost Optimization

### Cost Breakdown (per day)

| Component | Cost | Percentage | Optimization Opportunity |
|-----------|------|------------|--------------------------|
| LLM Generation | $X | Y% | [Strategy] |
| Embedding | $X | Y% | [Strategy] |
| Vector Store | $X | Y% | [Strategy] |
| Infrastructure | $X | Y% | [Strategy] |
| **Total** | **$500** | **100%** | |

### Optimization Strategies
1. [Strategy 1 with expected savings]
2. [Strategy 2 with expected savings]
3. [Strategy 3 with expected savings]

### Cost-Quality Trade-offs
[Discuss trade-offs between cost reduction and quality maintenance]

---

## Part 5: Continuous Evaluation

### Evaluation Metrics
- Faithfulness (daily)
- Answer relevancy (daily)
- Context precision (weekly)
- User satisfaction (continuous)

### Synthetic Test Data
[How synthetic test data is generated and used]

### A/B Testing Framework
[How new system variants are tested]

### Feedback Loop
[How user feedback is collected and incorporated]

---

## Risk Analysis

### Technical Risks
1. [Risk 1 and mitigation]
2. [Risk 2 and mitigation]

### Compliance Risks
1. [Risk 1 and mitigation]
2. [Risk 2 and mitigation]

### Operational Risks
1. [Risk 1 and mitigation]
2. [Risk 2 and mitigation]

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1-4)
- [Tasks]

### Phase 2: Core Features (Weeks 5-8)
- [Tasks]

### Phase 3: Production Hardening (Weeks 9-12)
- [Tasks]

### Phase 4: Launch and Monitoring (Week 13+)
- [Tasks]

---

## Conclusion
[Summary of key design decisions and expected outcomes]
"""


SAMPLE_SOLUTION_HIGHLIGHTS = """
# Sample Solution Highlights

## Key Design Decisions

### 1. Hybrid Deployment Model
- On-premise vector store and database (HIPAA compliance)
- Cloud-based LLM inference with data anonymization
- Offline fallback mode with cached responses

### 2. Multi-Tier Caching Strategy
- L1: In-memory cache for common queries (5-minute TTL)
- L2: Redis cache for recent queries (1-hour TTL)
- L3: Disk cache for historical queries (24-hour TTL)
- Expected cache hit rate: 35-40%
- Cost savings: ~$150/day

### 3. Query Routing
- Simple queries → Smaller model (7B parameters)
- Complex queries → Larger model (70B parameters)
- Medical terminology detection → Specialized medical LLM
- Expected cost savings: ~$100/day

### 4. Continuous Evaluation
- Daily synthetic test set (100 queries)
- Weekly manual review (50 queries)
- Real-time faithfulness sampling (1% of queries)
- Monthly A/B tests for improvements

### 5. Compliance Architecture
- All PHI encrypted with AES-256
- Role-based access control (RBAC)
- Comprehensive audit logging (10-year retention)
- Automated compliance reporting
- Quarterly security audits

## Cost Breakdown

| Component | Daily Cost | Optimization |
|-----------|------------|--------------|
| LLM Generation | $250 | Caching + routing |
| Embedding | $50 | Batch processing |
| Vector Store | $80 | Optimized indexing |
| Infrastructure | $70 | Auto-scaling |
| Monitoring | $30 | Sampling |
| Compliance | $20 | Automated |
| **Total** | **$500** | |

## Expected Performance

- P95 Latency: 1.8 seconds
- Faithfulness: 0.92
- Uptime: 99.95%
- Cost per query: $0.05
- Cache hit rate: 38%

## Key Innovations

1. **Medical Entity Recognition**: Pre-process queries to identify medical terms
2. **Temporal Weighting**: Recent research papers weighted higher
3. **Multi-Language Support**: English and Spanish with language detection
4. **Offline Mode**: Cached responses during network outages
5. **Federated Learning**: Improve model without centralizing PHI
"""


def evaluate_design(submission: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate student design submission.
    
    Args:
        submission: Dictionary with student's design components
        
    Returns:
        Evaluation results with scores and feedback
    """
    scores = {
        "architecture": 0,
        "compliance": 0,
        "monitoring": 0,
        "cost": 0,
        "evaluation": 0
    }
    
    feedback = {
        "architecture": [],
        "compliance": [],
        "monitoring": [],
        "cost": [],
        "evaluation": []
    }
    
    # Evaluate architecture (30 points)
    if "architecture" in submission:
        arch = submission["architecture"]
        
        # Check for key components
        if "embedding_model" in arch:
            scores["architecture"] += 5
        else:
            feedback["architecture"].append("Missing embedding model specification")
        
        if "vector_store" in arch:
            scores["architecture"] += 5
        else:
            feedback["architecture"].append("Missing vector store design")
        
        if "llm" in arch:
            scores["architecture"] += 5
        else:
            feedback["architecture"].append("Missing LLM specification")
        
        if "scalability" in arch:
            scores["architecture"] += 8
        else:
            feedback["architecture"].append("Missing scalability considerations")
        
        if "failure_handling" in arch:
            scores["architecture"] += 7
        else:
            feedback["architecture"].append("Missing failure handling strategy")
    
    # Evaluate compliance (25 points)
    if "compliance" in submission:
        comp = submission["compliance"]
        
        required_checks = [
            "encryption_at_rest",
            "encryption_in_transit",
            "access_controls",
            "audit_logging",
            "data_retention"
        ]
        
        for check in required_checks:
            if check in comp:
                scores["compliance"] += 5
            else:
                feedback["compliance"].append(f"Missing {check.replace('_', ' ')}")
    
    # Evaluate monitoring (20 points)
    if "monitoring" in submission:
        mon = submission["monitoring"]
        
        if "performance_metrics" in mon:
            scores["monitoring"] += 7
        else:
            feedback["monitoring"].append("Missing performance metrics")
        
        if "quality_metrics" in mon:
            scores["monitoring"] += 7
        else:
            feedback["monitoring"].append("Missing quality metrics")
        
        if "alert_thresholds" in mon:
            scores["monitoring"] += 6
        else:
            feedback["monitoring"].append("Missing alert thresholds")
    
    # Evaluate cost (15 points)
    if "cost" in submission:
        cost = submission["cost"]
        
        if "breakdown" in cost:
            scores["cost"] += 8
        else:
            feedback["cost"].append("Missing cost breakdown")
        
        if "optimizations" in cost:
            scores["cost"] += 7
        else:
            feedback["cost"].append("Missing optimization strategies")
    
    # Evaluate continuous evaluation (10 points)
    if "evaluation" in submission:
        eval_comp = submission["evaluation"]
        
        if "metrics" in eval_comp:
            scores["evaluation"] += 4
        else:
            feedback["evaluation"].append("Missing evaluation metrics")
        
        if "synthetic_data" in eval_comp:
            scores["evaluation"] += 3
        else:
            feedback["evaluation"].append("Missing synthetic data strategy")
        
        if "feedback_loop" in eval_comp:
            scores["evaluation"] += 3
        else:
            feedback["evaluation"].append("Missing feedback loop")
    
    total_score = sum(scores.values())
    max_score = 100
    percentage = (total_score / max_score) * 100
    
    return {
        "scores": scores,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": percentage,
        "feedback": feedback,
        "grade": get_letter_grade(percentage)
    }


def get_letter_grade(percentage: float) -> str:
    """Convert percentage to letter grade."""
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"


if __name__ == "__main__":
    print("=" * 80)
    print("MODULE 7 DESIGN CHALLENGE")
    print("=" * 80)
    print(DESIGN_CHALLENGE_PROMPT)
    print("\n" + "=" * 80)
    print("DESIGN TEMPLATE")
    print("=" * 80)
    print(DESIGN_TEMPLATE)
    print("\n" + "=" * 80)
    print("SAMPLE SOLUTION HIGHLIGHTS")
    print("=" * 80)
    print(SAMPLE_SOLUTION_HIGHLIGHTS)
