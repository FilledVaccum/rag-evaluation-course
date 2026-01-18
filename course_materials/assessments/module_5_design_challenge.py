"""
Module 5 Design Challenge: Evaluation Pipeline Architecture

This challenge requires students to design a comprehensive evaluation
pipeline for a production RAG system.

Requirements: 13.4
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DesignChallengeRequirements:
    """Requirements for the design challenge"""
    title: str
    scenario: str
    requirements: List[str]
    constraints: List[str]
    deliverables: List[str]
    evaluation_criteria: Dict[str, int]


# Design Challenge Definition
EVALUATION_PIPELINE_CHALLENGE = DesignChallengeRequirements(
    title="Design a Production RAG Evaluation Pipeline",
    scenario="""
    You are the ML engineer at a healthcare company building a RAG system
    that answers medical questions for healthcare professionals. The system
    retrieves information from medical literature and clinical guidelines.
    
    Due to the high-stakes nature of healthcare, evaluation is critical:
    - Incorrect information could harm patients
    - Regulatory compliance (HIPAA) is required
    - Medical terminology must be accurate
    - Citations to source documents are mandatory
    
    Your task is to design a comprehensive evaluation pipeline that ensures
    the RAG system meets quality and safety standards before deployment.
    """,
    requirements=[
        "Evaluate both retrieval and generation components independently",
        "Include domain-specific metrics for medical accuracy",
        "Implement continuous evaluation in production",
        "Support A/B testing of system variants",
        "Provide actionable insights for optimization",
        "Handle evaluation failures gracefully",
        "Log all evaluations for audit trails",
        "Scale to evaluate 10,000+ queries per day"
    ],
    constraints=[
        "Budget: $500/month for LLM API calls",
        "Latency: Evaluation must complete within 30 seconds",
        "Storage: Limited to 100GB for evaluation data",
        "Compliance: Must maintain HIPAA compliance",
        "Team: 2 ML engineers, 1 medical expert for validation"
    ],
    deliverables=[
        "Architecture diagram showing all components",
        "Metric selection with justification",
        "Evaluation workflow (when/how often to evaluate)",
        "Error handling and fallback strategies",
        "Cost estimation and optimization plan",
        "Monitoring and alerting strategy"
    ],
    evaluation_criteria={
        "Architecture Design (25 points)": 25,
        "Metric Selection (20 points)": 20,
        "Workflow Design (20 points)": 20,
        "Error Handling (15 points)": 15,
        "Cost Optimization (10 points)": 10,
        "Monitoring Strategy (10 points)": 10
    }
)


def get_design_challenge_instructions() -> str:
    """Get detailed design challenge instructions"""
    return """
    ============================================================================
    DESIGN CHALLENGE: PRODUCTION RAG EVALUATION PIPELINE
    ============================================================================
    
    SCENARIO:
    Healthcare RAG system for medical professionals
    - High-stakes: Patient safety depends on accuracy
    - Regulated: HIPAA compliance required
    - Complex: Medical terminology and citations
    - Scale: 10,000+ queries per day
    
    YOUR TASK:
    Design a comprehensive evaluation pipeline that ensures quality and safety.
    
    ============================================================================
    PART 1: ARCHITECTURE DESIGN (45 minutes)
    ============================================================================
    
    Design the overall architecture including:
    
    1. COMPONENTS:
       - Evaluation orchestrator
       - Metric computation engines
       - Data storage (test sets, results)
       - Monitoring and alerting
       - Reporting dashboard
    
    2. DATA FLOW:
       - How does data flow through the pipeline?
       - Where are evaluation results stored?
       - How are insights surfaced to the team?
    
    3. INTEGRATION POINTS:
       - How does evaluation integrate with RAG system?
       - When does evaluation run (pre-deployment, production, both)?
       - How are results fed back to improve the system?
    
    Create a diagram (Mermaid or similar) showing:
    - All components and their relationships
    - Data flow between components
    - External dependencies (LLM APIs, databases)
    
    ============================================================================
    PART 2: METRIC SELECTION (30 minutes)
    ============================================================================
    
    Select and justify metrics for:
    
    1. GENERATION METRICS:
       - Which standard metrics? (faithfulness, relevancy, etc.)
       - Which custom metrics for medical domain?
       - Why are these metrics appropriate?
    
    2. RETRIEVAL METRICS:
       - Which standard metrics? (precision, recall, relevance)
       - How to measure citation accuracy?
       - How to validate medical terminology?
    
    3. CUSTOM MEDICAL METRICS:
       Design 2-3 custom metrics such as:
       - Medical Terminology Accuracy
       - Clinical Guideline Compliance
       - Citation Completeness
       - Contraindication Detection
    
    For each metric, specify:
    - What it measures
    - Why it's important for healthcare
    - How it will be computed
    - Target score threshold
    
    ============================================================================
    PART 3: EVALUATION WORKFLOW (30 minutes)
    ============================================================================
    
    Design the evaluation workflow:
    
    1. PRE-DEPLOYMENT EVALUATION:
       - When: Before each release
       - What: Full test suite on curated test set
       - Criteria: All metrics must meet thresholds
       - Action: Block deployment if criteria not met
    
    2. PRODUCTION EVALUATION:
       - When: Continuous (sample of live queries)
       - What: Subset of metrics for speed
       - Criteria: Alert if metrics degrade
       - Action: Trigger investigation and rollback if needed
    
    3. A/B TESTING:
       - When: Testing new models or prompts
       - What: Compare metrics between variants
       - Criteria: Statistical significance required
       - Action: Promote winner to production
    
    4. PERIODIC REVIEW:
       - When: Weekly/monthly
       - What: Deep dive analysis of trends
       - Criteria: Identify optimization opportunities
       - Action: Update test sets and metrics
    
    Create a flowchart showing:
    - Decision points
    - Evaluation triggers
    - Pass/fail criteria
    - Actions taken based on results
    
    ============================================================================
    PART 4: ERROR HANDLING (20 minutes)
    ============================================================================
    
    Design error handling for:
    
    1. LLM API FAILURES:
       - What if judge LLM is unavailable?
       - Fallback strategy?
       - How to maintain evaluation continuity?
    
    2. INVALID SAMPLES:
       - How to detect invalid test samples?
       - Should they be skipped or fixed?
       - How to prevent invalid samples in future?
    
    3. METRIC COMPUTATION ERRORS:
       - What if a metric fails to compute?
       - Partial results or full failure?
       - How to debug metric issues?
    
    4. PERFORMANCE ISSUES:
       - What if evaluation takes too long?
       - How to optimize for speed?
       - Trade-offs between speed and accuracy?
    
    ============================================================================
    PART 5: COST OPTIMIZATION (20 minutes)
    ============================================================================
    
    Budget: $500/month for LLM API calls
    
    Optimize costs by:
    
    1. SAMPLING STRATEGY:
       - Don't evaluate every query in production
       - Sample X% of queries (what percentage?)
       - Stratified sampling by query type?
    
    2. METRIC SELECTION:
       - Some metrics are expensive (multi-stage LLM calls)
       - Which metrics are essential vs nice-to-have?
       - Can some metrics run less frequently?
    
    3. CACHING:
       - Cache evaluation results for repeated queries
       - Cache LLM judge responses
       - How long to keep cache?
    
    4. BATCHING:
       - Batch evaluation requests
       - Reduce API call overhead
       - Balance latency vs cost
    
    Calculate estimated costs:
    - Queries per day: 10,000
    - Sampling rate: ?%
    - Metrics per query: ?
    - LLM calls per metric: ?
    - Cost per LLM call: $0.001
    - Total monthly cost: ?
    
    ============================================================================
    PART 6: MONITORING AND ALERTING (20 minutes)
    ============================================================================
    
    Design monitoring strategy:
    
    1. METRICS TO MONITOR:
       - Evaluation scores (faithfulness, relevancy, etc.)
       - Evaluation latency
       - Evaluation failure rate
       - Cost per evaluation
    
    2. ALERT CONDITIONS:
       - When should alerts fire?
       - What thresholds trigger alerts?
       - Who gets notified?
    
    3. DASHBOARDS:
       - What visualizations are most useful?
       - Real-time vs historical views?
       - Drill-down capabilities?
    
    4. AUDIT TRAIL:
       - What needs to be logged for HIPAA compliance?
       - How long to retain logs?
       - How to make logs searchable?
    
    ============================================================================
    DELIVERABLES
    ============================================================================
    
    Submit the following:
    
    1. architecture_diagram.png/mermaid
       - Complete system architecture
       - All components and data flows
    
    2. metric_selection.md
       - List of all metrics with justifications
       - Custom metric definitions
       - Target thresholds
    
    3. evaluation_workflow.md
       - Detailed workflow description
       - Flowchart of evaluation process
       - Decision criteria
    
    4. error_handling_plan.md
       - Error scenarios and responses
       - Fallback strategies
       - Recovery procedures
    
    5. cost_analysis.xlsx/md
       - Cost breakdown
       - Optimization strategies
       - Monthly budget projection
    
    6. monitoring_plan.md
       - Metrics to monitor
       - Alert conditions
       - Dashboard mockups
    
    ============================================================================
    EVALUATION RUBRIC
    ============================================================================
    
    Architecture Design (25 points):
    - Comprehensive component design (10 pts)
    - Clear data flow (8 pts)
    - Proper integration points (7 pts)
    
    Metric Selection (20 points):
    - Appropriate standard metrics (8 pts)
    - Well-designed custom metrics (8 pts)
    - Clear justifications (4 pts)
    
    Workflow Design (20 points):
    - Complete workflow coverage (8 pts)
    - Clear decision criteria (7 pts)
    - Practical and implementable (5 pts)
    
    Error Handling (15 points):
    - Comprehensive error scenarios (7 pts)
    - Effective fallback strategies (5 pts)
    - Recovery procedures (3 pts)
    
    Cost Optimization (10 points):
    - Realistic cost estimates (5 pts)
    - Effective optimization strategies (5 pts)
    
    Monitoring Strategy (10 points):
    - Appropriate metrics to monitor (5 pts)
    - Effective alerting strategy (5 pts)
    
    ============================================================================
    TIPS FOR SUCCESS
    ============================================================================
    
    1. Start with requirements - what MUST the system do?
    2. Consider trade-offs - speed vs accuracy, cost vs coverage
    3. Think about failure modes - what can go wrong?
    4. Design for scale - 10,000 queries/day is significant
    5. Remember compliance - HIPAA audit trails are mandatory
    6. Be specific - vague designs score poorly
    7. Justify decisions - explain WHY you made each choice
    8. Consider the team - 2 engineers + 1 medical expert
    
    TIME ESTIMATE: 3-4 hours
    
    Good luck! 🏥
    """


def print_design_challenge():
    """Print design challenge"""
    print("="*80)
    print("MODULE 5 DESIGN CHALLENGE")
    print("="*80)
    print(f"\nTitle: {EVALUATION_PIPELINE_CHALLENGE.title}")
    print(f"\nScenario:\n{EVALUATION_PIPELINE_CHALLENGE.scenario}")
    print(f"\nRequirements:")
    for req in EVALUATION_PIPELINE_CHALLENGE.requirements:
        print(f"  - {req}")
    print(f"\nConstraints:")
    for constraint in EVALUATION_PIPELINE_CHALLENGE.constraints:
        print(f"  - {constraint}")
    print(f"\nDeliverables:")
    for deliverable in EVALUATION_PIPELINE_CHALLENGE.deliverables:
        print(f"  - {deliverable}")
    print(f"\nEvaluation Criteria:")
    for criterion, points in EVALUATION_PIPELINE_CHALLENGE.evaluation_criteria.items():
        print(f"  - {criterion}: {points} points")
    
    print("\n" + "="*80)
    print("DETAILED INSTRUCTIONS")
    print("="*80)
    print(get_design_challenge_instructions())


if __name__ == "__main__":
    print_design_challenge()
