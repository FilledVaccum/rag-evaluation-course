"""
Domain-Specific Case Studies for RAG Evaluation Course
Real-world examples from finance, healthcare, legal, and e-commerce domains
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CaseStudy:
    """Represents a domain-specific case study"""
    domain: str
    title: str
    background: str
    challenge: str
    solution: str
    implementation_details: Dict[str, str]
    results: Dict[str, str]
    lessons_learned: List[str]
    relevant_modules: List[str]


# Finance Domain Case Study
FINANCE_CASE_STUDY = CaseStudy(
    domain="Finance",
    title="Investment Research RAG System for Wealth Management Firm",
    background="""
A mid-sized wealth management firm manages $50B in assets across 10,000 clients. 
Investment advisors need to quickly access research reports, market analysis, and 
regulatory filings to make informed recommendations. The firm has 15 years of 
proprietary research reports (50,000+ documents) plus external sources (SEC filings, 
analyst reports, market data).

Traditional keyword search was failing because:
- Financial terminology is nuanced (e.g., "bearish" vs "negative outlook")
- Advisors ask natural language questions, not keyword queries
- Need to synthesize information across multiple documents
- Regulatory compliance requires accurate citations
""",
    challenge="""
Key Challenges:
1. Domain-Specific Language: Financial jargon, acronyms (EBITDA, P/E ratio, etc.)
2. Temporal Sensitivity: Market data becomes stale quickly
3. Regulatory Compliance: Must cite sources accurately (SEC requirements)
4. High Accuracy Requirements: Wrong information = bad investment decisions
5. Multi-Document Synthesis: Answers often require multiple sources
6. Performance: Advisors need sub-second response times
""",
    solution="""
Implemented hybrid RAG system with:

1. Embedding Strategy:
   - FinBERT embeddings for financial documents
   - Custom fine-tuned model on proprietary research
   - Separate indices for different document types (research, filings, news)

2. Retrieval Architecture:
   - Hybrid search: BM25 for exact terms (ticker symbols, company names)
   - Vector search for semantic understanding
   - Time-weighted retrieval (boost recent documents)
   - Metadata filtering (document type, date range, sector)

3. Generation:
   - GPT-4 for complex synthesis
   - GPT-3.5 for simple factual queries (cost optimization)
   - Strict prompt: "Only use provided context. Cite sources."
   - Structured output format with citations

4. Evaluation:
   - Synthetic test data: 500 questions from historical advisor queries
   - Metrics: Faithfulness (critical), Context Precision, Answer Relevancy
   - Human evaluation: 10% sample reviewed by senior advisors
   - Continuous monitoring: Daily evaluation runs
""",
    implementation_details={
        "embedding_model": "FinBERT + custom fine-tuned model",
        "vector_store": "Pinecone (managed, scalable)",
        "llm": "GPT-4 for complex, GPT-3.5 for simple (routing logic)",
        "chunking": "512 tokens with 50 token overlap, preserve section boundaries",
        "retrieval": "Hybrid (BM25 + vector), top-10 candidates, re-rank to top-3",
        "evaluation_framework": "Ragas with custom faithfulness metric",
        "monitoring": "Grafana dashboard, PagerDuty alerts"
    },
    results={
        "accuracy": "92% faithfulness score (vs 78% baseline)",
        "advisor_satisfaction": "4.6/5 rating (vs 3.2/5 for old system)",
        "time_savings": "Average query time reduced from 15 min to 30 sec",
        "cost": "$0.05 per query (80% use GPT-3.5, 20% use GPT-4)",
        "compliance": "100% citation accuracy (audited by compliance team)",
        "adoption": "85% of advisors use daily (vs 40% for old system)"
    },
    lessons_learned=[
        "Domain-specific embeddings (FinBERT) improved retrieval by 25% vs general embeddings",
        "Time-weighted retrieval was critical - market data older than 1 week is often irrelevant",
        "Query routing (simple → GPT-3.5, complex → GPT-4) reduced costs by 60% with minimal quality impact",
        "Faithfulness metric was most important - advisors need to trust the system",
        "Human evaluation on 10% sample caught edge cases that automated metrics missed",
        "Continuous evaluation detected quality degradation when market volatility increased"
    ],
    relevant_modules=["module_2", "module_3", "module_5", "module_6", "module_7"]
)


# Healthcare Domain Case Study
HEALTHCARE_CASE_STUDY = CaseStudy(
    domain="Healthcare",
    title="Clinical Decision Support RAG System for Hospital Network",
    background="""
A hospital network with 12 facilities and 5,000 physicians needed a clinical decision 
support system to help doctors access medical literature, treatment guidelines, and 
patient case histories. The system needed to handle:
- 100,000+ medical research papers
- Clinical practice guidelines from medical societies
- De-identified patient case histories (HIPAA compliant)
- Drug interaction databases
- Diagnostic criteria and treatment protocols

Doctors were spending 20-30 minutes per complex case searching for relevant information.
""",
    challenge="""
Key Challenges:
1. Medical Terminology: Highly specialized language, Latin terms, abbreviations
2. HIPAA Compliance: Strict data privacy requirements, no PII in embeddings
3. Life-Critical Accuracy: Wrong information could harm patients
4. Multi-Modal Data: Text, images (X-rays, MRIs), lab results
5. Rapid Updates: Medical knowledge evolves quickly, guidelines change
6. Liability: System recommendations must be defensible
7. Integration: Must work with existing EHR systems
""",
    solution="""
Implemented specialized medical RAG system with:

1. Embedding Strategy:
   - BioBERT and PubMedBERT for medical literature
   - Separate embeddings for different content types (research, guidelines, cases)
   - Privacy-preserving embeddings (no PII in vector representations)

2. Retrieval Architecture:
   - Semantic search for symptom-disease matching
   - Metadata filtering (patient age, gender, comorbidities)
   - Recency weighting for treatment guidelines
   - Cross-reference with drug interaction database

3. Generation:
   - GPT-4 with medical fine-tuning
   - Strict prompt: "Provide evidence-based recommendations with citations"
   - Confidence scoring for each recommendation
   - Disclaimer: "This is decision support, not a diagnosis"

4. Evaluation:
   - Test set: 1,000 de-identified real cases with expert annotations
   - Metrics: Faithfulness (critical), Medical Accuracy (custom metric)
   - Expert review: Board-certified physicians evaluate 20% of outputs
   - A/B testing: Compare with physician-only decisions
   - Continuous monitoring: Track patient outcomes (with IRB approval)

5. Compliance:
   - HIPAA-compliant infrastructure (encrypted at rest and in transit)
   - Audit logs for all queries and responses
   - No patient data in training or embeddings
   - Regular security audits
""",
    implementation_details={
        "embedding_model": "BioBERT + PubMedBERT ensemble",
        "vector_store": "Milvus (on-premise for HIPAA compliance)",
        "llm": "GPT-4 with medical fine-tuning (Azure OpenAI for compliance)",
        "chunking": "Paragraph-based (medical papers), preserve section structure",
        "retrieval": "Semantic search with metadata filtering, top-5 results",
        "evaluation_framework": "Ragas + custom medical accuracy metric",
        "monitoring": "Real-time dashboard, physician feedback loop",
        "compliance": "HIPAA-compliant infrastructure, SOC 2 Type II certified"
    },
    results={
        "accuracy": "94% medical accuracy (validated by expert physicians)",
        "time_savings": "Average case research time reduced from 25 min to 5 min",
        "physician_satisfaction": "4.7/5 rating, 78% daily usage",
        "patient_outcomes": "No adverse events attributed to system (12-month study)",
        "cost_savings": "$2.5M annually in physician time savings",
        "literature_coverage": "Recommendations cite average of 4.2 sources per case"
    },
    lessons_learned=[
        "Medical domain embeddings (BioBERT, PubMedBERT) were essential - general embeddings failed on medical terminology",
        "HIPAA compliance required on-premise deployment - cloud solutions were not acceptable",
        "Expert physician review was critical for validation - automated metrics alone were insufficient",
        "Confidence scoring helped physicians know when to seek additional consultation",
        "Integration with EHR was complex but necessary for adoption",
        "Continuous monitoring of patient outcomes provided ultimate validation",
        "Liability concerns required extensive legal review and disclaimers"
    ],
    relevant_modules=["module_2", "module_3", "module_5", "module_7"]
)


# Legal Domain Case Study
LEGAL_CASE_STUDY = CaseStudy(
    domain="Legal",
    title="Legal Research RAG System for Law Firm",
    background="""
A large law firm with 500 attorneys needed to modernize their legal research system. 
The firm has:
- 30 years of case law and legal precedents (200,000+ documents)
- Internal memos and briefs from past cases
- Regulatory filings and compliance documents
- Contract templates and clause libraries

Associates were spending 10-15 hours per week on legal research. The firm wanted to:
- Reduce research time
- Improve consistency in legal arguments
- Find relevant precedents more effectively
- Ensure compliance with evolving regulations
""",
    challenge="""
Key Challenges:
1. Legal Language: Highly formal, archaic terms, Latin phrases (habeas corpus, etc.)
2. Precedent Importance: Exact citations and case law hierarchy matter
3. Jurisdiction Specificity: Laws vary by state, federal vs state law
4. Temporal Sensitivity: Laws change, old precedents may be overturned
5. Exactness Requirements: Misquoting case law is malpractice
6. Confidentiality: Client information must remain confidential
7. Billable Hours: System must track time for client billing
""",
    solution="""
Implemented legal-specific RAG system with:

1. Embedding Strategy:
   - LegalBERT for case law and legal documents
   - Separate indices for different document types (cases, statutes, memos)
   - Jurisdiction-aware embeddings (federal, state, local)

2. Retrieval Architecture:
   - Hybrid search: BM25 for exact case citations (e.g., "Brown v. Board")
   - Vector search for conceptual similarity (e.g., "discrimination cases")
   - Metadata filtering (jurisdiction, date, case outcome)
   - Citation graph traversal (find related cases)

3. Generation:
   - GPT-4 for legal analysis and argument generation
   - Strict citation format (Bluebook style)
   - Confidence scoring for each legal argument
   - Separate mode for contract review vs legal research

4. Evaluation:
   - Test set: 300 legal questions from past cases (anonymized)
   - Metrics: Citation Accuracy (custom), Faithfulness, Relevancy
   - Attorney review: Senior partners evaluate 15% of outputs
   - Comparison with Westlaw and LexisNexis results
   - Track: Did system find precedents that attorneys missed?

5. Compliance:
   - Client confidentiality: Separate indices per client (access control)
   - Audit logs: Track all queries for malpractice defense
   - No client data in training
   - Regular security audits and penetration testing
""",
    implementation_details={
        "embedding_model": "LegalBERT + custom fine-tuned model",
        "vector_store": "Weaviate (GraphQL interface, citation graph support)",
        "llm": "GPT-4 (Azure OpenAI for enterprise compliance)",
        "chunking": "Section-based (legal documents have clear structure)",
        "retrieval": "Hybrid (BM25 + vector) with citation graph, top-10 results",
        "evaluation_framework": "Ragas + custom citation accuracy metric",
        "monitoring": "Usage analytics, attorney feedback, quality metrics",
        "compliance": "SOC 2 Type II, attorney-client privilege protection"
    },
    results={
        "time_savings": "Legal research time reduced from 12 hours/week to 4 hours/week per associate",
        "cost_savings": "$5M annually in associate time (billable hours)",
        "precedent_discovery": "System found relevant precedents missed by attorneys in 23% of cases",
        "citation_accuracy": "99.2% citation accuracy (Bluebook format)",
        "attorney_satisfaction": "4.5/5 rating, 82% daily usage",
        "client_satisfaction": "Faster turnaround on legal research, lower bills"
    },
    lessons_learned=[
        "Legal domain embeddings (LegalBERT) were critical for understanding legal terminology",
        "Hybrid search was essential - exact case citations (BM25) + conceptual similarity (vector)",
        "Citation accuracy was the most important metric - misquoting case law is unacceptable",
        "Jurisdiction filtering was critical - federal vs state law distinctions matter",
        "Citation graph traversal found related cases that pure similarity search missed",
        "Attorney review was necessary for validation - legal reasoning is nuanced",
        "Client confidentiality required strict access controls and separate indices",
        "Integration with existing legal research tools (Westlaw) was important for adoption"
    ],
    relevant_modules=["module_2", "module_3", "module_5", "module_6", "module_7"]
)


# E-commerce Domain Case Study
ECOMMERCE_CASE_STUDY = CaseStudy(
    domain="E-commerce",
    title="Product Search and Recommendation RAG System for Online Retailer",
    background="""
A large online retailer with 10M products and 50M monthly visitors needed to improve 
product search and recommendations. The existing keyword search was failing because:
- Customers use natural language ("comfortable running shoes for flat feet")
- Product descriptions vary in quality and completeness
- Need to understand synonyms and related concepts
- Want to provide personalized recommendations
- Need to handle multi-modal data (text, images, reviews)

The retailer wanted to:
- Improve search relevance (reduce "no results found")
- Increase conversion rate
- Provide better product recommendations
- Answer customer questions about products
""",
    challenge="""
Key Challenges:
1. Scale: 10M products, 50M monthly visitors, 100K queries/hour at peak
2. Latency: Sub-100ms response time required for good UX
3. Multi-Modal: Text descriptions, images, customer reviews, specifications
4. Personalization: Different customers want different things
5. Inventory Changes: Products go in/out of stock constantly
6. Query Diversity: From specific ("iPhone 15 Pro Max 256GB") to vague ("gift for mom")
7. Cost: Need to optimize for cost at scale
""",
    solution="""
Implemented scalable e-commerce RAG system with:

1. Embedding Strategy:
   - Product embeddings: Combine title, description, specs, reviews
   - Image embeddings: CLIP for visual similarity
   - Query embeddings: User intent understanding
   - Personalization: User history embeddings

2. Retrieval Architecture:
   - Hybrid search: BM25 for exact product names/SKUs
   - Vector search for semantic understanding
   - Image search for visual similarity
   - Collaborative filtering for personalization
   - Real-time inventory filtering

3. Generation:
   - GPT-3.5 for product Q&A (cost-optimized)
   - GPT-4 for complex queries (gift recommendations)
   - Structured output: Product list with explanations
   - Personalization: Consider user history and preferences

4. Evaluation:
   - Test set: 10,000 real customer queries with click-through data
   - Metrics: Click-Through Rate (CTR), Conversion Rate, NDCG
   - A/B testing: Compare with baseline search
   - User satisfaction surveys
   - Revenue impact analysis

5. Optimization:
   - Caching: Cache popular queries (80% hit rate)
   - Query routing: Simple → BM25, complex → RAG
   - Batch processing: Update embeddings nightly
   - CDN: Distribute vector search globally
""",
    implementation_details={
        "embedding_model": "E5-large (general) + CLIP (images)",
        "vector_store": "Pinecone (managed, globally distributed)",
        "llm": "GPT-3.5 (80% of queries), GPT-4 (20% complex queries)",
        "chunking": "Product-level (each product is one chunk)",
        "retrieval": "Hybrid (BM25 + vector + image + collaborative filtering)",
        "evaluation_framework": "Custom metrics (CTR, conversion, revenue)",
        "monitoring": "Real-time dashboard, A/B testing platform",
        "optimization": "Aggressive caching, query routing, CDN"
    },
    results={
        "search_relevance": "NDCG improved from 0.72 to 0.89",
        "conversion_rate": "Increased by 18% (from 3.2% to 3.8%)",
        "revenue_impact": "$45M additional annual revenue",
        "customer_satisfaction": "Search satisfaction score: 4.4/5 (vs 3.6/5 baseline)",
        "latency": "P95 latency: 85ms (within 100ms target)",
        "cost": "$0.002 per query (aggressive caching and routing)",
        "no_results_rate": "Reduced from 12% to 3%"
    },
    lessons_learned=[
        "Hybrid search was essential - exact product names (BM25) + semantic understanding (vector)",
        "Multi-modal embeddings (text + images) improved relevance significantly",
        "Caching was critical for cost and latency at scale - 80% cache hit rate",
        "Query routing (simple → BM25, complex → RAG) optimized cost without sacrificing quality",
        "A/B testing was the gold standard for validation - revenue impact was the ultimate metric",
        "Real-time inventory filtering was necessary - showing out-of-stock products frustrated customers",
        "Personalization improved conversion but required careful privacy considerations",
        "Image search was surprisingly effective for fashion and home decor categories"
    ],
    relevant_modules=["module_1", "module_2", "module_3", "module_5", "module_6", "module_7"]
)


# Collection of all case studies
ALL_CASE_STUDIES = [
    FINANCE_CASE_STUDY,
    HEALTHCARE_CASE_STUDY,
    LEGAL_CASE_STUDY,
    ECOMMERCE_CASE_STUDY
]


def get_case_study_by_domain(domain: str) -> Optional[CaseStudy]:
    """Get case study for a specific domain"""
    for case_study in ALL_CASE_STUDIES:
        if case_study.domain.lower() == domain.lower():
            return case_study
    return None


def export_case_studies_to_markdown() -> str:
    """Export all case studies to markdown format"""
    md = "# Domain-Specific Case Studies: RAG Evaluation Course\n\n"
    md += "## Real-World Examples from Finance, Healthcare, Legal, and E-commerce\n\n"
    md += "---\n\n"
    
    for case_study in ALL_CASE_STUDIES:
        md += f"## {case_study.domain} Domain: {case_study.title}\n\n"
        
        md += "### Background\n\n"
        md += f"{case_study.background}\n\n"
        
        md += "### Challenge\n\n"
        md += f"{case_study.challenge}\n\n"
        
        md += "### Solution\n\n"
        md += f"{case_study.solution}\n\n"
        
        md += "### Implementation Details\n\n"
        for key, value in case_study.implementation_details.items():
            md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        md += "\n"
        
        md += "### Results\n\n"
        for key, value in case_study.results.items():
            md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        md += "\n"
        
        md += "### Lessons Learned\n\n"
        for lesson in case_study.lessons_learned:
            md += f"- {lesson}\n"
        md += "\n"
        
        md += f"### Relevant Course Modules\n\n"
        md += f"{', '.join(case_study.relevant_modules)}\n\n"
        
        md += "---\n\n"
    
    return md


if __name__ == "__main__":
    # Export case studies
    case_studies_md = export_case_studies_to_markdown()
    print(case_studies_md)
