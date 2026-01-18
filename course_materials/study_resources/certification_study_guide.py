"""
Certification Study Guide: NCP-AAI Exam Topic Mapping

This module provides comprehensive mapping of course modules to NVIDIA-Certified Professional:
Agentic AI (NCP-AAI) certification exam domains with coverage level indicators.

Requirements: 17.1, 2.1
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class CoverageLevel(Enum):
    """Coverage level indicators for exam domain mapping."""
    PRIMARY = "⭐⭐⭐"  # Primary focus - deep coverage
    CORE = "⭐⭐⭐"     # Core content - comprehensive coverage
    SUPPORTING = "⭐⭐"  # Supporting content - significant coverage
    CONTEXTUAL = "⭐"   # Contextual - referenced/supporting


@dataclass
class ExamDomain:
    """Represents an NCP-AAI certification exam domain."""
    name: str
    weight_percentage: float
    topics_covered: List[str]
    coverage_level: CoverageLevel
    
    def __str__(self) -> str:
        return f"{self.name} ({self.weight_percentage}%) - {self.coverage_level.value}"


@dataclass
class ModuleExamMapping:
    """Maps a course module to exam domains."""
    module_number: int
    module_title: str
    primary_domains: List[ExamDomain]
    secondary_domains: List[ExamDomain]
    key_exam_topics: List[str]
    
    def get_all_domains(self) -> List[ExamDomain]:
        """Get all domains (primary and secondary) for this module."""
        return self.primary_domains + self.secondary_domains


class CertificationStudyGuide:
    """
    Comprehensive study guide mapping course content to NCP-AAI exam domains.
    
    This guide helps students understand how each module contributes to certification
    preparation and which exam topics are covered at what depth.
    """
    
    def __init__(self):
        """Initialize the study guide with exam domain definitions and module mappings."""
        self.exam_domains = self._define_exam_domains()
        self.module_mappings = self._create_module_mappings()
    
    def _define_exam_domains(self) -> Dict[str, ExamDomain]:
        """Define all NCP-AAI certification exam domains."""
        return {
            "evaluation_tuning": ExamDomain(
                name="Evaluation and Tuning",
                weight_percentage=13.0,
                topics_covered=[
                    "Measuring agent performance",
                    "Comparing agents across tasks and datasets",
                    "Optimizing agent behavior",
                    "Metric selection and interpretation",
                    "LLM-as-a-Judge methodologies",
                    "Custom metric development",
                    "Performance benchmarking"
                ],
                coverage_level=CoverageLevel.PRIMARY
            ),
            "knowledge_integration": ExamDomain(
                name="Knowledge Integration and Data Handling",
                weight_percentage=10.0,
                topics_covered=[
                    "External knowledge integration",
                    "Diverse data type management",
                    "Vector stores and embeddings",
                    "Data preprocessing and augmentation",
                    "Retrieval pipeline implementation",
                    "Chunking strategies",
                    "Domain-specific data handling"
                ],
                coverage_level=CoverageLevel.CORE
            ),
            "agent_development": ExamDomain(
                name="Agent Development",
                weight_percentage=15.0,
                topics_covered=[
                    "Building and integrating agents",
                    "Agent enhancement techniques",
                    "RAG pipeline implementation",
                    "Component-level debugging",
                    "Synthetic data generation",
                    "Test-driven development for agents"
                ],
                coverage_level=CoverageLevel.SUPPORTING
            ),
            "agent_architecture": ExamDomain(
                name="Agent Architecture and Design",
                weight_percentage=15.0,
                topics_covered=[
                    "Foundational agent structuring",
                    "Agent interaction patterns",
                    "Reasoning and communication design",
                    "Search system evolution",
                    "RAG architecture patterns",
                    "Hybrid system design"
                ],
                coverage_level=CoverageLevel.SUPPORTING
            ),
            "deployment_scaling": ExamDomain(
                name="Deployment and Scaling",
                weight_percentage=13.0,
                topics_covered=[
                    "Operationalizing agentic systems",
                    "Scaling strategies",
                    "Production deployment",
                    "Performance optimization",
                    "Cost-efficiency trade-offs"
                ],
                coverage_level=CoverageLevel.CONTEXTUAL
            ),
            "run_monitor_maintain": ExamDomain(
                name="Run, Monitor, and Maintain",
                weight_percentage=5.0,
                topics_covered=[
                    "Ongoing operation",
                    "System monitoring",
                    "Maintenance procedures",
                    "Continuous evaluation",
                    "Feedback loops"
                ],
                coverage_level=CoverageLevel.CONTEXTUAL
            ),
            "nvidia_platform": ExamDomain(
                name="NVIDIA Platform Implementation",
                weight_percentage=7.0,
                topics_covered=[
                    "NVIDIA NIM for embeddings and inference",
                    "NVIDIA Nemotron for synthetic data",
                    "NVIDIA Triton for deployment",
                    "NVIDIA NeMo for agent development",
                    "NVIDIA Agent Intelligence Toolkit"
                ],
                coverage_level=CoverageLevel.SUPPORTING
            ),
            "cognition_planning_memory": ExamDomain(
                name="Cognition, Planning, and Memory",
                weight_percentage=10.0,
                topics_covered=[
                    "Reasoning strategies",
                    "Decision-making processes",
                    "Memory management",
                    "Multi-step reasoning",
                    "Orchestration patterns"
                ],
                coverage_level=CoverageLevel.CONTEXTUAL
            ),
            "safety_ethics_compliance": ExamDomain(
                name="Safety, Ethics, and Compliance",
                weight_percentage=5.0,
                topics_covered=[
                    "Responsible AI principles",
                    "Ethical standards",
                    "Legal compliance (GDPR, HIPAA)",
                    "Regulatory frameworks",
                    "Data privacy and security"
                ],
                coverage_level=CoverageLevel.CONTEXTUAL
            ),
            "human_ai_interaction": ExamDomain(
                name="Human-AI Interaction and Oversight",
                weight_percentage=5.0,
                topics_covered=[
                    "Human oversight design",
                    "Effective interaction patterns",
                    "User feedback collection",
                    "Human-in-the-loop systems"
                ],
                coverage_level=CoverageLevel.CONTEXTUAL
            )
        }
    
    def _create_module_mappings(self) -> List[ModuleExamMapping]:
        """Create detailed mappings for each course module to exam domains."""
        return [
            self._create_module_1_mapping(),
            self._create_module_2_mapping(),
            self._create_module_3_mapping(),
            self._create_module_4_mapping(),
            self._create_module_5_mapping(),
            self._create_module_6_mapping(),
            self._create_module_7_mapping()
        ]
    
    def _create_module_1_mapping(self) -> ModuleExamMapping:
        """Module 1: Evolution of Search to RAG Systems."""
        return ModuleExamMapping(
            module_number=1,
            module_title="Evolution of Search to RAG Systems",
            primary_domains=[
                self.exam_domains["agent_architecture"]
            ],
            secondary_domains=[
                self.exam_domains["knowledge_integration"]
            ],
            key_exam_topics=[
                "Classic search architecture (crawling, indexing, ranking)",
                "BM25 vs semantic search comparison",
                "Hybrid system design (BM25 + Vector + Re-ranking)",
                "Search approach selection frameworks",
                "RAG architecture foundations"
            ]
        )
    
    def _create_module_2_mapping(self) -> ModuleExamMapping:
        """Module 2: Embeddings and Vector Stores."""
        return ModuleExamMapping(
            module_number=2,
            module_title="Embeddings and Vector Stores",
            primary_domains=[
                self.exam_domains["knowledge_integration"]
            ],
            secondary_domains=[
                self.exam_domains["nvidia_platform"]
            ],
            key_exam_topics=[
                "Multi-dimensional similarity and embedding fundamentals",
                "Domain-specific embedding models (code, finance, healthcare, multilingual)",
                "NVIDIA NIM embedding models (NV-Embed-v2)",
                "Vector store configuration and optimization",
                "Chunking strategies for different data types",
                "Tabular data transformation for embeddings"
            ]
        )
    
    def _create_module_3_mapping(self) -> ModuleExamMapping:
        """Module 3: RAG Architecture and Component Analysis."""
        return ModuleExamMapping(
            module_number=3,
            module_title="RAG Architecture and Component Analysis",
            primary_domains=[
                self.exam_domains["agent_architecture"],
                self.exam_domains["agent_development"]
            ],
            secondary_domains=[
                self.exam_domains["cognition_planning_memory"]
            ],
            key_exam_topics=[
                "Three-stage RAG pipeline (Retrieval → Augmentation → Generation)",
                "Component-level failure diagnosis",
                "Retrieval vs generation failure differentiation",
                "Context relevance assessment",
                "Response accuracy and faithfulness evaluation",
                "Orchestration patterns and multi-step reasoning"
            ]
        )
    
    def _create_module_4_mapping(self) -> ModuleExamMapping:
        """Module 4: Synthetic Test Data Generation."""
        return ModuleExamMapping(
            module_number=4,
            module_title="Synthetic Test Data Generation",
            primary_domains=[
                self.exam_domains["evaluation_tuning"],
                self.exam_domains["agent_development"]
            ],
            secondary_domains=[
                self.exam_domains["nvidia_platform"]
            ],
            key_exam_topics=[
                "Test-driven development for LLMs and RAG",
                "LLM-based synthetic data generation",
                "Prompt engineering for data steering (3-5 example pattern)",
                "Domain-specific query generation",
                "NVIDIA Nemotron-4-340B for synthetic data",
                "Quality validation and filtering strategies",
                "Synthesizer mixing and customization"
            ]
        )
    
    def _create_module_5_mapping(self) -> ModuleExamMapping:
        """Module 5: RAG Evaluation Metrics and Frameworks."""
        return ModuleExamMapping(
            module_number=5,
            module_title="RAG Evaluation Metrics and Frameworks",
            primary_domains=[
                self.exam_domains["evaluation_tuning"]
            ],
            secondary_domains=[
                self.exam_domains["agent_development"]
            ],
            key_exam_topics=[
                "LLM-as-a-Judge methodology and limitations",
                "Ragas framework architecture and metrics",
                "Generation metrics (Faithfulness, Answer Relevancy, Context Utilization)",
                "Retrieval metrics (Context Precision, Context Recall, Context Relevance)",
                "Custom metric development from scratch",
                "Metric interpretation and actionable insights",
                "Multi-stage evaluation pipelines"
            ]
        )
    
    def _create_module_6_mapping(self) -> ModuleExamMapping:
        """Module 6: Semantic Search System Evaluation."""
        return ModuleExamMapping(
            module_number=6,
            module_title="Semantic Search System Evaluation",
            primary_domains=[
                self.exam_domains["evaluation_tuning"],
                self.exam_domains["knowledge_integration"]
            ],
            secondary_domains=[],
            key_exam_topics=[
                "Legacy BM25 system evaluation with modern techniques",
                "Applying Ragas to non-RAG search systems",
                "Hybrid evaluation strategies (RAG + Semantic Search)",
                "Ranking algorithm assessment and optimization",
                "Enterprise system integration considerations"
            ]
        )
    
    def _create_module_7_mapping(self) -> ModuleExamMapping:
        """Module 7: Production Deployment and Advanced Topics."""
        return ModuleExamMapping(
            module_number=7,
            module_title="Production Deployment and Advanced Topics",
            primary_domains=[
                self.exam_domains["deployment_scaling"],
                self.exam_domains["run_monitor_maintain"]
            ],
            secondary_domains=[
                self.exam_domains["safety_ethics_compliance"],
                self.exam_domains["human_ai_interaction"]
            ],
            key_exam_topics=[
                "Temporal data handling and time-weighted retrieval",
                "Regulatory compliance (GDPR, HIPAA)",
                "Continuous evaluation in production",
                "Performance profiling and cost-efficiency trade-offs",
                "A/B testing frameworks for RAG systems",
                "Monitoring, observability, and feedback loops",
                "Multi-language and low-resource language challenges"
            ]
        )
    
    def get_module_mapping(self, module_number: int) -> ModuleExamMapping:
        """Get exam domain mapping for a specific module."""
        for mapping in self.module_mappings:
            if mapping.module_number == module_number:
                return mapping
        raise ValueError(f"Module {module_number} not found")
    
    def get_coverage_by_domain(self, domain_name: str) -> List[ModuleExamMapping]:
        """Get all modules that cover a specific exam domain."""
        modules = []
        for mapping in self.module_mappings:
            all_domains = mapping.get_all_domains()
            if any(d.name == domain_name for d in all_domains):
                modules.append(mapping)
        return modules
    
    def generate_study_plan(self) -> str:
        """
        Generate a comprehensive study plan showing how the course prepares
        students for the NCP-AAI certification exam.
        """
        plan = []
        plan.append("=" * 80)
        plan.append("NCP-AAI CERTIFICATION STUDY GUIDE")
        plan.append("Course: Evaluating RAG and Semantic Search Systems")
        plan.append("=" * 80)
        plan.append("")
        
        # Exam overview
        plan.append("EXAM OVERVIEW")
        plan.append("-" * 80)
        plan.append("Certification: NVIDIA-Certified Professional: Agentic AI (NCP-AAI)")
        plan.append("Level: Professional (Intermediate)")
        plan.append("Duration: 120 minutes")
        plan.append("Questions: 60-70 questions")
        plan.append("Prerequisites: 1-2 years AI/ML experience with production agentic AI")
        plan.append("")
        
        # Coverage legend
        plan.append("COVERAGE LEVEL LEGEND")
        plan.append("-" * 80)
        plan.append(f"{CoverageLevel.PRIMARY.value} PRIMARY - Deep, comprehensive coverage (primary focus)")
        plan.append(f"{CoverageLevel.CORE.value} CORE - Comprehensive coverage (essential content)")
        plan.append(f"{CoverageLevel.SUPPORTING.value} SUPPORTING - Significant coverage (important context)")
        plan.append(f"{CoverageLevel.CONTEXTUAL.value} CONTEXTUAL - Referenced/supporting (background)")
        plan.append("")
        
        # Exam domains summary
        plan.append("EXAM DOMAINS AND COURSE COVERAGE")
        plan.append("-" * 80)
        for domain_key, domain in self.exam_domains.items():
            modules = self.get_coverage_by_domain(domain.name)
            module_nums = ", ".join([f"Module {m.module_number}" for m in modules])
            plan.append(f"{domain.coverage_level.value} {domain.name}")
            plan.append(f"   Weight: {domain.weight_percentage}% of exam")
            plan.append(f"   Covered in: {module_nums}")
            plan.append("")
        
        # Module-by-module breakdown
        plan.append("MODULE-BY-MODULE EXAM TOPIC MAPPING")
        plan.append("=" * 80)
        plan.append("")
        
        for mapping in self.module_mappings:
            plan.append(f"MODULE {mapping.module_number}: {mapping.module_title}")
            plan.append("-" * 80)
            
            plan.append("Primary Exam Domains:")
            for domain in mapping.primary_domains:
                plan.append(f"  {domain.coverage_level.value} {domain}")
            
            if mapping.secondary_domains:
                plan.append("Secondary Exam Domains:")
                for domain in mapping.secondary_domains:
                    plan.append(f"  {domain.coverage_level.value} {domain}")
            
            plan.append("Key Exam Topics Covered:")
            for topic in mapping.key_exam_topics:
                plan.append(f"  • {topic}")
            
            plan.append("")
        
        # Study recommendations
        plan.append("STUDY RECOMMENDATIONS")
        plan.append("=" * 80)
        plan.append("")
        plan.append("1. PRIMARY FOCUS AREAS (⭐⭐⭐)")
        plan.append("   - Modules 4, 5, 6: Evaluation and Tuning (13% of exam)")
        plan.append("   - Deep dive into Ragas, LLM-as-a-Judge, custom metrics")
        plan.append("   - Practice synthetic data generation and prompt engineering")
        plan.append("")
        plan.append("2. CORE CONTENT AREAS (⭐⭐⭐)")
        plan.append("   - Modules 2, 3: Knowledge Integration and Agent Development")
        plan.append("   - Master embeddings, vector stores, RAG architecture")
        plan.append("   - Practice component-level debugging")
        plan.append("")
        plan.append("3. SUPPORTING CONTENT (⭐⭐)")
        plan.append("   - Module 1: Agent Architecture foundations")
        plan.append("   - Module 7: Production deployment context")
        plan.append("   - Understand NVIDIA platform integration throughout")
        plan.append("")
        plan.append("4. ADDITIONAL PREPARATION")
        plan.append("   - Complete all hands-on notebooks and debugging exercises")
        plan.append("   - Take mock certification exam (60-70 questions)")
        plan.append("   - Review one-page concept summaries for each module")
        plan.append("   - Study recommended reading materials")
        plan.append("   - Consider additional NVIDIA courses for comprehensive coverage")
        plan.append("")
        
        # Exam success tips
        plan.append("EXAM SUCCESS TIPS")
        plan.append("-" * 80)
        plan.append("• Focus on practical application, not just theory")
        plan.append("• Understand component interactions in RAG pipelines")
        plan.append("• Master metric selection and interpretation")
        plan.append("• Know NVIDIA tools: NIM, NeMo, Triton, Nemotron")
        plan.append("• Practice debugging retrieval vs generation failures")
        plan.append("• Understand enterprise considerations (compliance, scaling)")
        plan.append("• Review RAG evaluation in context of agentic AI systems")
        plan.append("")
        
        return "\n".join(plan)
    
    def export_to_markdown(self, filepath: str) -> None:
        """Export the study guide to a markdown file."""
        content = self.generate_study_plan()
        with open(filepath, 'w') as f:
            f.write(content)


# Example usage
if __name__ == "__main__":
    guide = CertificationStudyGuide()
    
    # Print study plan
    print(guide.generate_study_plan())
    
    # Example: Get mapping for Module 5
    module_5 = guide.get_module_mapping(5)
    print(f"\nModule 5 covers {len(module_5.primary_domains)} primary domains")
    
    # Example: Find all modules covering Evaluation and Tuning
    eval_modules = guide.get_coverage_by_domain("Evaluation and Tuning")
    print(f"\nEvaluation and Tuning is covered in {len(eval_modules)} modules")
