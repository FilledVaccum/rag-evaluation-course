"""
Capstone Project Specification for RAG Evaluation Course.

This module defines the requirements, evaluation rubric, and domain selection
guidance for the course capstone project.

The capstone requires students to build and evaluate a complete RAG system
for a custom domain, demonstrating mastery of all course concepts.
"""

from typing import List, Dict
from src.models.assessment import (
    Assessment,
    AssessmentType,
    Question,
    QuestionType,
    Difficulty,
    EvaluationRubric
)


# Capstone Project Specification
CAPSTONE_PROJECT_SPEC = {
    "title": "End-to-End RAG System Evaluation Capstone Project",
    "description": """
Build and evaluate a complete RAG system for a domain of your choice.
This capstone project demonstrates your mastery of RAG evaluation concepts,
synthetic data generation, metric implementation, and production considerations.
    """,
    
    "learning_objectives": [
        "Design and implement an end-to-end RAG system for a specific domain",
        "Generate domain-specific synthetic test data using prompt engineering",
        "Implement comprehensive evaluation pipeline with Ragas and custom metrics",
        "Analyze evaluation results and provide actionable optimization insights",
        "Address production deployment considerations including monitoring and compliance"
    ],
    
    "requirements": {
        "1_domain_selection": {
            "title": "Domain Selection and Problem Definition",
            "description": "Select a domain and define the RAG use case",
            "deliverables": [
                "Domain description (e.g., healthcare, finance, legal, education)",
                "Problem statement: What questions will the RAG system answer?",
                "User persona definition: Who will use this system?",
                "Success criteria: What makes a good answer in this domain?",
                "Dataset description: What knowledge base will you use?"
            ],
            "points": 10
        },
        
        "2_rag_implementation": {
            "title": "RAG System Implementation",
            "description": "Build a complete RAG pipeline",
            "deliverables": [
                "Retrieval component with appropriate embedding model selection",
                "Vector store configuration optimized for your domain",
                "Chunking strategy with justification for chunk size",
                "Augmentation logic for context preparation",
                "Generation component using NVIDIA NIM or equivalent",
                "End-to-end pipeline integration"
            ],
            "points": 25
        },
        
        "3_synthetic_data_generation": {
            "title": "Synthetic Test Data Generation",
            "description": "Generate domain-specific test data",
            "deliverables": [
                "Baseline synthetic data generation using Nemotron or equivalent",
                "Custom prompt engineering with 3-5 examples",
                "Domain-specific query generation reflecting realistic user needs",
                "Quality validation and filtering strategy",
                "Test set of at least 50 question-answer pairs",
                "Before/after comparison showing prompt engineering improvements"
            ],
            "points": 20
        },
        
        "4_evaluation_implementation": {
            "title": "Comprehensive Evaluation Pipeline",
            "description": "Implement evaluation metrics and analysis",
            "deliverables": [
                "Ragas evaluation implementation with standard metrics",
                "At least one customized metric with modified prompts",
                "At least one custom metric created from scratch",
                "Component-level evaluation (retrieval vs generation)",
                "Evaluation results with detailed analysis",
                "Actionable optimization insights based on metrics"
            ],
            "points": 25
        },
        
        "5_production_considerations": {
            "title": "Production Deployment Considerations",
            "description": "Address production-ready requirements",
            "deliverables": [
                "Error handling and graceful degradation strategy",
                "Performance profiling (latency, throughput)",
                "Cost-efficiency analysis",
                "Monitoring and observability plan",
                "Compliance considerations (if applicable: GDPR, HIPAA, etc.)",
                "Continuous evaluation strategy"
            ],
            "points": 10
        },
        
        "6_documentation": {
            "title": "Documentation and Presentation",
            "description": "Document your work and present findings",
            "deliverables": [
                "README with project overview and setup instructions",
                "Architecture diagram showing RAG pipeline components",
                "Evaluation results summary with visualizations",
                "Lessons learned and optimization recommendations",
                "Code documentation and comments",
                "5-10 minute presentation (slides or video)"
            ],
            "points": 10
        }
    },
    
    "domain_selection_guidance": {
        "recommended_domains": [
            {
                "domain": "Healthcare",
                "description": "Medical literature search, diagnosis support, patient education",
                "considerations": [
                    "HIPAA compliance required",
                    "Domain-specific medical embeddings recommended",
                    "High accuracy requirements",
                    "Specialized terminology"
                ],
                "example_use_cases": [
                    "Medical literature Q&A for clinicians",
                    "Patient education chatbot",
                    "Clinical trial matching"
                ]
            },
            {
                "domain": "Finance",
                "description": "Financial document analysis, investment research, compliance",
                "considerations": [
                    "Regulatory compliance (SEC, FINRA)",
                    "Temporal data handling (time-sensitive information)",
                    "High precision requirements",
                    "Financial terminology and calculations"
                ],
                "example_use_cases": [
                    "Earnings report analysis",
                    "Investment research assistant",
                    "Regulatory compliance checker"
                ]
            },
            {
                "domain": "Legal",
                "description": "Legal document search, case law research, contract analysis",
                "considerations": [
                    "Citation accuracy critical",
                    "Complex document structures",
                    "Precedent and temporal relationships",
                    "Legal terminology"
                ],
                "example_use_cases": [
                    "Case law research assistant",
                    "Contract review and analysis",
                    "Legal precedent finder"
                ]
            },
            {
                "domain": "Education",
                "description": "Course materials, tutoring, curriculum development",
                "considerations": [
                    "Age-appropriate language",
                    "Pedagogical considerations",
                    "Multiple difficulty levels",
                    "Learning progression"
                ],
                "example_use_cases": [
                    "Intelligent tutoring system",
                    "Course material Q&A",
                    "Homework help assistant"
                ]
            },
            {
                "domain": "E-commerce",
                "description": "Product search, recommendations, customer support",
                "considerations": [
                    "Product catalog structure",
                    "User intent understanding",
                    "Personalization",
                    "Real-time inventory"
                ],
                "example_use_cases": [
                    "Product recommendation engine",
                    "Customer support chatbot",
                    "Product comparison assistant"
                ]
            },
            {
                "domain": "Software Development",
                "description": "Code search, documentation, debugging assistance",
                "considerations": [
                    "Code-specific embeddings",
                    "Syntax highlighting and formatting",
                    "Version control integration",
                    "Multiple programming languages"
                ],
                "example_use_cases": [
                    "Code documentation search",
                    "API usage examples",
                    "Debugging assistant"
                ]
            }
        ],
        
        "selection_criteria": [
            "Choose a domain you're familiar with or interested in learning",
            "Ensure you have access to a suitable dataset (public or synthetic)",
            "Consider the complexity appropriate for a capstone project",
            "Think about unique evaluation challenges in your domain",
            "Consider production deployment requirements"
        ],
        
        "dataset_sources": [
            "Public datasets: Hugging Face, Kaggle, UCI ML Repository",
            "Academic datasets: ArXiv, PubMed, legal databases",
            "Synthetic datasets: Generate using LLMs",
            "Company datasets: Use with appropriate permissions",
            "Web scraping: Ensure compliance with terms of service"
        ]
    },
    
    "evaluation_rubric": {
        "rubric_id": "capstone_rubric",
        "criteria": {
            "domain_selection": {
                "points": 10,
                "description": "Clear domain definition with well-defined problem and success criteria",
                "excellent": "Comprehensive domain analysis with clear problem definition, user personas, and success criteria",
                "good": "Clear domain and problem definition with basic success criteria",
                "needs_improvement": "Vague domain or problem definition lacking clear success criteria"
            },
            "rag_implementation": {
                "points": 25,
                "description": "Complete RAG pipeline with appropriate component selection and configuration",
                "excellent": "Well-architected RAG system with justified design decisions and optimized configuration",
                "good": "Functional RAG system with reasonable component choices",
                "needs_improvement": "Incomplete RAG implementation or poor component choices"
            },
            "synthetic_data": {
                "points": 20,
                "description": "High-quality domain-specific synthetic test data with effective prompt engineering",
                "excellent": "Excellent prompt engineering with clear before/after improvements and realistic queries",
                "good": "Functional synthetic data generation with some customization",
                "needs_improvement": "Generic synthetic data lacking domain specificity"
            },
            "evaluation": {
                "points": 25,
                "description": "Comprehensive evaluation with standard and custom metrics, detailed analysis",
                "excellent": "Thorough evaluation with custom metrics, component-level analysis, and actionable insights",
                "good": "Solid evaluation with standard metrics and basic analysis",
                "needs_improvement": "Incomplete evaluation or lack of meaningful analysis"
            },
            "production_considerations": {
                "points": 10,
                "description": "Thoughtful consideration of production deployment requirements",
                "excellent": "Comprehensive production plan with error handling, monitoring, and compliance",
                "good": "Basic production considerations addressed",
                "needs_improvement": "Minimal or no production considerations"
            },
            "documentation": {
                "points": 10,
                "description": "Clear documentation and effective presentation of work",
                "excellent": "Excellent documentation with clear diagrams, visualizations, and insights",
                "good": "Adequate documentation covering key aspects",
                "needs_improvement": "Incomplete or unclear documentation"
            }
        },
        "total_points": 100,
        "passing_score": 70,
        "grading_scale": {
            "A (90-100)": "Exceptional work demonstrating mastery of all concepts",
            "B (80-89)": "Strong work with good understanding of core concepts",
            "C (70-79)": "Satisfactory work meeting basic requirements",
            "F (0-69)": "Incomplete or inadequate work not meeting requirements"
        }
    },
    
    "timeline": {
        "recommended_duration": "2-3 weeks",
        "milestones": [
            {
                "week": 1,
                "tasks": [
                    "Select domain and define problem",
                    "Gather or create dataset",
                    "Implement basic RAG pipeline",
                    "Generate baseline synthetic data"
                ]
            },
            {
                "week": 2,
                "tasks": [
                    "Optimize RAG components",
                    "Implement custom prompt engineering",
                    "Set up evaluation pipeline",
                    "Run initial evaluations"
                ]
            },
            {
                "week": 3,
                "tasks": [
                    "Implement custom metrics",
                    "Analyze results and optimize",
                    "Address production considerations",
                    "Complete documentation and presentation"
                ]
            }
        ]
    },
    
    "submission_requirements": {
        "format": "GitHub repository or ZIP file",
        "required_files": [
            "README.md with project overview",
            "requirements.txt with dependencies",
            "Source code for RAG pipeline",
            "Jupyter notebook with evaluation results",
            "Synthetic test data (JSON or CSV)",
            "Evaluation results and analysis",
            "Architecture diagram (Mermaid or image)",
            "Presentation slides or video link"
        ],
        "code_requirements": [
            "Python 3.10+ code with type hints",
            "Clear function and class documentation",
            "Modular code structure",
            "Error handling implemented",
            "Requirements.txt with all dependencies"
        ]
    },
    
    "example_projects": [
        {
            "title": "Medical Literature Q&A System",
            "domain": "Healthcare",
            "description": "RAG system for answering clinical questions using PubMed abstracts",
            "highlights": [
                "Domain-specific medical embeddings (BioBERT)",
                "HIPAA compliance considerations",
                "Custom faithfulness metric for medical accuracy",
                "Temporal handling for recent research"
            ]
        },
        {
            "title": "Financial Earnings Analysis Assistant",
            "domain": "Finance",
            "description": "RAG system for analyzing company earnings reports",
            "highlights": [
                "Temporal data handling for quarterly reports",
                "Custom metric for numerical accuracy",
                "Regulatory compliance (SEC)",
                "Performance optimization for real-time queries"
            ]
        },
        {
            "title": "Legal Case Law Research Tool",
            "domain": "Legal",
            "description": "RAG system for finding relevant case law precedents",
            "highlights": [
                "Citation accuracy validation",
                "Temporal precedent relationships",
                "Custom metric for legal relevance",
                "Complex document structure handling"
            ]
        }
    ],
    
    "resources": [
        "Course modules 1-7 for reference",
        "NVIDIA NIM documentation",
        "Ragas framework documentation",
        "Hugging Face datasets",
        "Course Slack channel for questions",
        "Office hours with instructors"
    ],
    
    "faq": [
        {
            "question": "Can I work in a team?",
            "answer": "Yes, teams of 2-3 students are allowed. All team members must contribute equally and this should be documented."
        },
        {
            "question": "What if I don't have access to a domain-specific dataset?",
            "answer": "You can use public datasets, create synthetic data, or use general datasets like Wikipedia. The key is demonstrating evaluation skills."
        },
        {
            "question": "Do I need to deploy the system to production?",
            "answer": "No, but you should document production considerations and have a deployment plan."
        },
        {
            "question": "Can I use frameworks other than Ragas?",
            "answer": "Yes, but you must justify your choice and demonstrate equivalent evaluation capabilities."
        },
        {
            "question": "How detailed should the custom metrics be?",
            "answer": "Custom metrics should be well-documented with clear prompts, scoring rubrics, and validation on test cases."
        }
    ]
}


def create_capstone_assessment() -> Assessment:
    """
    Create the capstone project assessment object.
    
    Returns:
        Assessment object for the capstone project
    """
    # Create evaluation rubric
    rubric = EvaluationRubric(
        rubric_id=CAPSTONE_PROJECT_SPEC["evaluation_rubric"]["rubric_id"],
        criteria=CAPSTONE_PROJECT_SPEC["evaluation_rubric"]["criteria"],
        total_points=CAPSTONE_PROJECT_SPEC["evaluation_rubric"]["total_points"],
        passing_score=CAPSTONE_PROJECT_SPEC["evaluation_rubric"]["passing_score"]
    )
    
    # Create project requirements as "questions" (deliverables)
    questions = []
    for req_id, req_data in CAPSTONE_PROJECT_SPEC["requirements"].items():
        question = Question(
            question_id=req_id,
            question_text=f"{req_data['title']}: {req_data['description']}",
            question_type=QuestionType.OPEN_ENDED,
            options=None,
            correct_answer=None,
            explanation="\n".join([f"- {d}" for d in req_data["deliverables"]]),
            exam_domain="Evaluation and Tuning",
            difficulty=Difficulty.ADVANCED,
            points=req_data["points"]
        )
        questions.append(question)
    
    # Create assessment
    assessment = Assessment(
        assessment_id="capstone_project",
        assessment_type=AssessmentType.CAPSTONE,
        module_number=None,  # Capstone spans all modules
        title=CAPSTONE_PROJECT_SPEC["title"],
        description=CAPSTONE_PROJECT_SPEC["description"],
        questions=questions,
        rubric=rubric,
        time_limit_minutes=None  # No time limit for capstone
    )
    
    return assessment


def get_domain_guidance(domain: str) -> Dict:
    """
    Get domain-specific guidance for capstone project.
    
    Args:
        domain: Domain name (e.g., "Healthcare", "Finance")
        
    Returns:
        Dictionary with domain-specific guidance
    """
    for domain_info in CAPSTONE_PROJECT_SPEC["domain_selection_guidance"]["recommended_domains"]:
        if domain_info["domain"].lower() == domain.lower():
            return domain_info
    
    return {
        "domain": domain,
        "description": "Custom domain",
        "considerations": ["Define domain-specific requirements"],
        "example_use_cases": ["Define your use case"]
    }


def get_evaluation_criteria() -> Dict:
    """
    Get detailed evaluation criteria for capstone project.
    
    Returns:
        Dictionary with evaluation criteria and rubric
    """
    return CAPSTONE_PROJECT_SPEC["evaluation_rubric"]


def get_submission_requirements() -> Dict:
    """
    Get submission requirements for capstone project.
    
    Returns:
        Dictionary with submission format and requirements
    """
    return CAPSTONE_PROJECT_SPEC["submission_requirements"]


# Example usage
if __name__ == "__main__":
    # Create capstone assessment
    capstone = create_capstone_assessment()
    
    print(f"Capstone Project: {capstone.title}")
    print(f"Total Points: {capstone.rubric.total_points}")
    print(f"Passing Score: {capstone.rubric.passing_score}")
    print(f"\nRequirements ({len(capstone.questions)} deliverables):")
    
    for question in capstone.questions:
        print(f"\n{question.question_text}")
        print(f"Points: {question.points}")
        print(f"Deliverables:\n{question.explanation}")
    
    print("\n" + "="*80)
    print("Domain Selection Guidance:")
    print("="*80)
    
    for domain_info in CAPSTONE_PROJECT_SPEC["domain_selection_guidance"]["recommended_domains"]:
        print(f"\n{domain_info['domain']}: {domain_info['description']}")
        print(f"Example use cases: {', '.join(domain_info['example_use_cases'][:2])}")
