"""
Post-Course Resource Package

Comprehensive package of resources for continued learning after course completion,
including downloadable materials, NVIDIA platform links, community resources,
and additional course recommendations.

Requirements: 17.5, 17.6, 17.7, 17.8, 17.9
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class ResourceCategory(Enum):
    """Categories for post-course resources."""
    DOWNLOADABLE = "Downloadable Materials"
    NVIDIA_PLATFORM = "NVIDIA Platform Access"
    COMMUNITY = "Community and Support"
    ADDITIONAL_COURSES = "Additional NVIDIA Courses"
    TOOLS = "Development Tools"
    GITHUB = "GitHub Repositories"


@dataclass
class PostCourseResource:
    """Represents a post-course resource."""
    title: str
    description: str
    category: ResourceCategory
    url: str
    access_requirements: str
    estimated_value: str  # Time saved, learning value, etc.
    
    def __str__(self) -> str:
        return f"{self.title} - {self.category.value}"


class PostCourseResourcePackage:
    """
    Comprehensive package of post-course resources for continued learning.
    
    Provides students with:
    - Downloadable notebooks and datasets
    - NVIDIA AI Catalog access information
    - Community forum and study group links
    - Additional NVIDIA course recommendations
    - GitHub repository structure for course materials
    """
    
    def __init__(self):
        """Initialize the resource package with all resources."""
        self.resources = self._compile_resources()
        self.github_structure = self._define_github_structure()
    
    def _compile_resources(self) -> List[PostCourseResource]:
        """Compile all post-course resources."""
        resources = []
        
        # Downloadable Materials
        resources.extend([
            PostCourseResource(
                title="Complete Jupyter Notebook Collection",
                description="All 8 hands-on Jupyter notebooks from the course (Notebooks 0-7) with solutions and intentional bugs for debugging practice.",
                category=ResourceCategory.DOWNLOADABLE,
                url="https://github.com/nvidia/rag-evaluation-course/tree/main/notebooks",
                access_requirements="Free - GitHub account recommended for cloning",
                estimated_value="20+ hours of hands-on practice"
            ),
            PostCourseResource(
                title="USC Course Catalog Dataset",
                description="Complete USC course catalog dataset in CSV format for tabular data practice and chunking experiments.",
                category=ResourceCategory.DOWNLOADABLE,
                url="https://github.com/nvidia/rag-evaluation-course/tree/main/datasets/usc_catalog",
                access_requirements="Free - No authentication required",
                estimated_value="Primary dataset for synthetic data generation"
            ),
            PostCourseResource(
                title="Amnesty Q&A Dataset",
                description="Pre-formatted Amnesty Q&A dataset for Ragas evaluation with user_input, retrieved_context, response, and ground_truth fields.",
                category=ResourceCategory.DOWNLOADABLE,
                url="https://github.com/nvidia/rag-evaluation-course/tree/main/datasets/amnesty_qa",
                access_requirements="Free - No authentication required",
                estimated_value="Ready-to-use evaluation dataset"
            ),
            PostCourseResource(
                title="Module Concept Summaries (PDF)",
                description="One-page concept summaries for all 7 modules in PDF format for quick review and exam preparation.",
                category=ResourceCategory.DOWNLOADABLE,
                url="https://github.com/nvidia/rag-evaluation-course/tree/main/study_resources",
                access_requirements="Free - No authentication required",
                estimated_value="Essential study materials for certification"
            ),
            PostCourseResource(
                title="Mock Certification Exam",
                description="60-70 question mock exam with detailed explanations, mirroring NCP-AAI exam format and difficulty.",
                category=ResourceCategory.DOWNLOADABLE,
                url="https://github.com/nvidia/rag-evaluation-course/tree/main/assessments",
                access_requirements="Free - No authentication required",
                estimated_value="2-hour exam simulation"
            ),
            PostCourseResource(
                title="Course Slide Decks",
                description="Complete slide decks for all 7 modules with Mermaid diagrams, speaker notes, and visual aids.",
                category=ResourceCategory.DOWNLOADABLE,
                url="https://github.com/nvidia/rag-evaluation-course/tree/main/slides",
                access_requirements="Free - No authentication required",
                estimated_value="Reference materials for review"
            )
        ])

        
        # NVIDIA Platform Access
        resources.extend([
            PostCourseResource(
                title="NVIDIA AI Catalog",
                description="Access to NVIDIA NIM microservices, embedding models, and LLM endpoints for continued experimentation.",
                category=ResourceCategory.NVIDIA_PLATFORM,
                url="https://build.nvidia.com/",
                access_requirements="Free tier available - NVIDIA account required",
                estimated_value="Production-ready AI models and APIs"
            ),
            PostCourseResource(
                title="NVIDIA NIM for Embeddings",
                description="Access to NV-Embed-v2 and domain-specific embedding models (code, finance, healthcare, multilingual).",
                category=ResourceCategory.NVIDIA_PLATFORM,
                url="https://build.nvidia.com/explore/retrieval",
                access_requirements="Free tier available - API key required",
                estimated_value="State-of-the-art embedding models"
            ),
            PostCourseResource(
                title="NVIDIA Nemotron Models",
                description="Access to Nemotron-4-340B and other models for synthetic data generation and reasoning.",
                category=ResourceCategory.NVIDIA_PLATFORM,
                url="https://build.nvidia.com/nvidia/nemotron-4-340b-instruct",
                access_requirements="Free tier available - API key required",
                estimated_value="Advanced synthetic data generation"
            ),
            PostCourseResource(
                title="NVIDIA Agent Intelligence Toolkit",
                description="Complete toolkit for building, optimizing, and deploying AI agents with NeMo integration.",
                category=ResourceCategory.NVIDIA_PLATFORM,
                url="https://docs.nvidia.com/aiqtoolkit/latest/index.html",
                access_requirements="Free - Open source",
                estimated_value="Comprehensive agent development platform"
            ),
            PostCourseResource(
                title="NVIDIA Triton Inference Server",
                description="Production-grade inference server for deploying RAG systems at scale with GPU acceleration.",
                category=ResourceCategory.NVIDIA_PLATFORM,
                url="https://developer.nvidia.com/triton-inference-server",
                access_requirements="Free - Open source",
                estimated_value="Enterprise deployment infrastructure"
            ),
            PostCourseResource(
                title="NVIDIA NeMo Framework",
                description="End-to-end framework for building, customizing, and deploying generative AI models.",
                category=ResourceCategory.NVIDIA_PLATFORM,
                url="https://www.nvidia.com/en-us/ai-data-science/products/nemo/",
                access_requirements="Free - Open source",
                estimated_value="Advanced model development"
            )
        ])
        
        # Community and Support
        resources.extend([
            PostCourseResource(
                title="NVIDIA Developer Forums - AI and Deep Learning",
                description="Active community forum for asking questions, sharing projects, and getting help from NVIDIA experts and peers.",
                category=ResourceCategory.COMMUNITY,
                url="https://forums.developer.nvidia.com/c/ai/",
                access_requirements="Free - NVIDIA Developer account",
                estimated_value="Community support and networking"
            ),
            PostCourseResource(
                title="NVIDIA Technical Blog",
                description="Regular updates on RAG, evaluation techniques, and agentic AI with tutorials and best practices.",
                category=ResourceCategory.COMMUNITY,
                url="https://developer.nvidia.com/blog/tag/retrieval-augmented-generation-rag/",
                access_requirements="Free - No account required",
                estimated_value="Latest techniques and updates"
            ),
            PostCourseResource(
                title="NCP-AAI Certification Study Group",
                description="Dedicated study group for NCP-AAI certification candidates with weekly discussions and practice sessions.",
                category=ResourceCategory.COMMUNITY,
                url="https://forums.developer.nvidia.com/c/certifications/ncp-aai/",
                access_requirements="Free - NVIDIA Developer account",
                estimated_value="Peer learning and exam preparation"
            ),
            PostCourseResource(
                title="Ragas Community Discord",
                description="Official Ragas framework Discord server for evaluation framework questions and community support.",
                category=ResourceCategory.COMMUNITY,
                url="https://discord.gg/ragas",
                access_requirements="Free - Discord account",
                estimated_value="Framework-specific support"
            ),
            PostCourseResource(
                title="NVIDIA AI Podcast",
                description="Weekly podcast featuring AI researchers, practitioners, and thought leaders discussing latest developments.",
                category=ResourceCategory.COMMUNITY,
                url="https://blogs.nvidia.com/ai-podcast/",
                access_requirements="Free - No account required",
                estimated_value="Industry insights and trends"
            )
        ])
        
        # Additional NVIDIA Courses
        resources.extend([
            PostCourseResource(
                title="Building RAG Agents With LLMs",
                description="Comprehensive course on building RAG agents from scratch with hands-on projects and real-world examples.",
                category=ResourceCategory.ADDITIONAL_COURSES,
                url="https://www.nvidia.com/en-us/training/instructor-led-workshops/building-rag-agents/",
                access_requirements="Paid - NVIDIA DLI enrollment",
                estimated_value="8 hours - Complements evaluation focus"
            ),
            PostCourseResource(
                title="Building Agentic AI Applications with LLMs",
                description="Advanced course on multi-agent systems, orchestration, and complex agentic workflows.",
                category=ResourceCategory.ADDITIONAL_COURSES,
                url="https://www.nvidia.com/en-us/training/instructor-led-workshops/agentic-ai/",
                access_requirements="Paid - NVIDIA DLI enrollment",
                estimated_value="8 hours - Agent Development (15% of exam)"
            ),
            PostCourseResource(
                title="Deploying RAG Pipelines for Production at Scale",
                description="Production deployment course covering Triton, Kubernetes, monitoring, and enterprise considerations.",
                category=ResourceCategory.ADDITIONAL_COURSES,
                url="https://www.nvidia.com/en-us/training/instructor-led-workshops/rag-production/",
                access_requirements="Paid - NVIDIA DLI enrollment",
                estimated_value="6 hours - Deployment & Scaling (13% of exam)"
            ),
            PostCourseResource(
                title="Generative AI Explained",
                description="Foundational course on generative AI concepts, LLMs, and practical applications.",
                category=ResourceCategory.ADDITIONAL_COURSES,
                url="https://www.nvidia.com/en-us/training/online/generative-ai-explained/",
                access_requirements="Free - NVIDIA DLI account",
                estimated_value="2 hours - Foundational knowledge"
            ),
            PostCourseResource(
                title="Building Transformer-Based Natural Language Processing Applications",
                description="Deep dive into transformer architectures and NLP applications with hands-on coding.",
                category=ResourceCategory.ADDITIONAL_COURSES,
                url="https://www.nvidia.com/en-us/training/instructor-led-workshops/transformer-nlp/",
                access_requirements="Paid - NVIDIA DLI enrollment",
                estimated_value="8 hours - Technical depth"
            )
        ])

        
        # Development Tools
        resources.extend([
            PostCourseResource(
                title="Ragas Framework",
                description="Open-source evaluation framework for RAG systems with comprehensive metrics and customization.",
                category=ResourceCategory.TOOLS,
                url="https://docs.ragas.io/",
                access_requirements="Free - pip install ragas",
                estimated_value="Core evaluation tool"
            ),
            PostCourseResource(
                title="LangChain",
                description="Framework for developing applications with LLMs, including RAG orchestration and agent tools.",
                category=ResourceCategory.TOOLS,
                url="https://python.langchain.com/",
                access_requirements="Free - pip install langchain",
                estimated_value="RAG orchestration framework"
            ),
            PostCourseResource(
                title="Hypothesis (Property-Based Testing)",
                description="Python library for property-based testing, useful for comprehensive RAG system testing.",
                category=ResourceCategory.TOOLS,
                url="https://hypothesis.readthedocs.io/",
                access_requirements="Free - pip install hypothesis",
                estimated_value="Advanced testing capabilities"
            ),
            PostCourseResource(
                title="Milvus Vector Database",
                description="Open-source vector database for billion-scale similarity search with GPU acceleration.",
                category=ResourceCategory.TOOLS,
                url="https://milvus.io/",
                access_requirements="Free - Open source",
                estimated_value="Production vector store"
            ),
            PostCourseResource(
                title="Weights & Biases",
                description="MLOps platform for experiment tracking, model monitoring, and evaluation visualization.",
                category=ResourceCategory.TOOLS,
                url="https://wandb.ai/",
                access_requirements="Free tier available",
                estimated_value="Experiment tracking and monitoring"
            )
        ])
        
        # GitHub Repositories
        resources.extend([
            PostCourseResource(
                title="Course Materials Repository",
                description="Complete course repository with all notebooks, datasets, assessments, and study materials.",
                category=ResourceCategory.GITHUB,
                url="https://github.com/nvidia/rag-evaluation-course",
                access_requirements="Free - Public repository",
                estimated_value="All course materials in one place"
            ),
            PostCourseResource(
                title="NVIDIA NeMo Agent Toolkit Repository",
                description="Official repository for NeMo Agent Toolkit with examples, documentation, and community contributions.",
                category=ResourceCategory.GITHUB,
                url="https://github.com/NVIDIA/NeMo-Agent-Toolkit",
                access_requirements="Free - Open source",
                estimated_value="Agent development examples"
            ),
            PostCourseResource(
                title="NVIDIA NeMo Guardrails Repository",
                description="Toolkit for adding programmable guardrails to LLM applications with safety examples.",
                category=ResourceCategory.GITHUB,
                url="https://github.com/NVIDIA-NeMo/Guardrails",
                access_requirements="Free - Open source",
                estimated_value="Safety and compliance tools"
            ),
            PostCourseResource(
                title="Ragas Examples Repository",
                description="Community-contributed examples of Ragas evaluation in various domains and use cases.",
                category=ResourceCategory.GITHUB,
                url="https://github.com/explodinggradients/ragas-examples",
                access_requirements="Free - Public repository",
                estimated_value="Real-world evaluation examples"
            ),
            PostCourseResource(
                title="NVIDIA TensorRT-LLM Repository",
                description="Optimized inference library for LLMs with state-of-the-art performance on NVIDIA GPUs.",
                category=ResourceCategory.GITHUB,
                url="https://github.com/NVIDIA/TensorRT-LLM",
                access_requirements="Free - Open source",
                estimated_value="Production inference optimization"
            )
        ])
        
        return resources
    
    def _define_github_structure(self) -> Dict[str, List[str]]:
        """Define the GitHub repository structure for course materials."""
        return {
            "root": [
                "README.md - Course overview and setup instructions",
                "requirements.txt - Python dependencies",
                "setup.py - Package installation",
                ".env.example - Environment variable template",
                "LICENSE - Course materials license"
            ],
            "notebooks/": [
                "notebook_0_search_paradigm_comparison.ipynb",
                "notebook_1_embeddings_chunking.ipynb",
                "notebook_2_rag_debugging.ipynb",
                "notebook_3_baseline_synthetic_data.ipynb",
                "notebook_4_customized_synthetic_data.ipynb",
                "notebook_5_ragas_evaluation.ipynb",
                "notebook_6_semantic_search_evaluation.ipynb",
                "notebook_7_production_monitoring.ipynb",
                "solutions/ - Solution notebooks for all exercises"
            ],
            "datasets/": [
                "usc_catalog/ - USC course catalog CSV files",
                "amnesty_qa/ - Amnesty Q&A JSON files",
                "README.md - Dataset documentation and usage"
            ],
            "course_materials/": [
                "modules/ - Lecture content for all 7 modules",
                "assessments/ - Quizzes, challenges, and mock exam",
                "study_resources/ - Concept summaries and study guides",
                "slides/ - Presentation slides with diagrams"
            ],
            "src/": [
                "models/ - Data models (Pydantic)",
                "evaluation/ - Evaluation framework code",
                "synthetic_data/ - Synthetic data generation",
                "platform_integration/ - NVIDIA platform clients",
                "utils/ - Utility functions"
            ],
            "tests/": [
                "unit/ - Unit tests",
                "property/ - Property-based tests",
                "integration/ - Integration tests",
                "conftest.py - Pytest configuration"
            ],
            "docs/": [
                "ARCHITECTURE.md - System architecture",
                "QUICKSTART.md - Quick start guide",
                "API.md - API documentation",
                "CONTRIBUTING.md - Contribution guidelines"
            ]
        }

    
    def get_resources_by_category(self, category: ResourceCategory) -> List[PostCourseResource]:
        """Get all resources in a specific category."""
        return [r for r in self.resources if r.category == category]
    
    def get_free_resources(self) -> List[PostCourseResource]:
        """Get all free resources."""
        return [r for r in self.resources if "Free" in r.access_requirements]
    
    def get_nvidia_platform_resources(self) -> List[PostCourseResource]:
        """Get all NVIDIA platform resources."""
        return self.get_resources_by_category(ResourceCategory.NVIDIA_PLATFORM)
    
    def generate_resource_package(self) -> str:
        """Generate formatted post-course resource package."""
        output = []
        output.append("=" * 80)
        output.append("POST-COURSE RESOURCE PACKAGE")
        output.append("Evaluating RAG and Semantic Search Systems")
        output.append("=" * 80)
        output.append("")
        
        output.append("WELCOME TO YOUR CONTINUED LEARNING JOURNEY!")
        output.append("-" * 80)
        output.append("Congratulations on completing the course! This resource package provides")
        output.append("everything you need to continue learning, practice your skills, and prepare")
        output.append("for the NVIDIA-Certified Professional: Agentic AI (NCP-AAI) certification.")
        output.append("")
        
        # Summary statistics
        total_resources = len(self.resources)
        free_resources = len(self.get_free_resources())
        output.append(f"Total Resources: {total_resources}")
        output.append(f"Free Resources: {free_resources}")
        output.append("")
        
        # Resources by category
        output.append("RESOURCES BY CATEGORY")
        output.append("=" * 80)
        output.append("")
        
        for category in ResourceCategory:
            category_resources = self.get_resources_by_category(category)
            if not category_resources:
                continue
            
            output.append(f"{category.value} ({len(category_resources)} resources)")
            output.append("-" * 80)
            
            for resource in category_resources:
                output.append(f"\n{resource.title}")
                output.append(f"  {resource.description}")
                output.append(f"  URL: {resource.url}")
                output.append(f"  Access: {resource.access_requirements}")
                output.append(f"  Value: {resource.estimated_value}")
            
            output.append("")
        
        # GitHub Repository Structure
        output.append("GITHUB REPOSITORY STRUCTURE")
        output.append("=" * 80)
        output.append("")
        output.append("Clone the complete course repository:")
        output.append("  git clone https://github.com/nvidia/rag-evaluation-course.git")
        output.append("")
        
        for directory, files in self.github_structure.items():
            output.append(f"{directory}")
            for file in files:
                output.append(f"  {file}")
            output.append("")
        
        # Quick Start Guide
        output.append("QUICK START GUIDE")
        output.append("=" * 80)
        output.append("")
        output.append("1. IMMEDIATE NEXT STEPS (First Week)")
        output.append("   • Clone the GitHub repository")
        output.append("   • Set up NVIDIA AI Catalog account and get API key")
        output.append("   • Re-run all notebooks with your own modifications")
        output.append("   • Take the mock certification exam")
        output.append("   • Join NVIDIA Developer Forums and NCP-AAI study group")
        output.append("")
        output.append("2. SHORT-TERM GOALS (First Month)")
        output.append("   • Complete additional NVIDIA courses (Building RAG Agents, Agentic AI)")
        output.append("   • Read recommended blog posts and tutorials")
        output.append("   • Build a personal RAG project in your domain of interest")
        output.append("   • Experiment with different evaluation metrics and synthesizers")
        output.append("   • Contribute to open-source projects (Ragas, NeMo)")
        output.append("")
        output.append("3. CERTIFICATION PREPARATION (2-3 Months)")
        output.append("   • Review all module concept summaries")
        output.append("   • Study certification exam blueprint thoroughly")
        output.append("   • Take multiple practice exams")
        output.append("   • Focus on weak areas identified in practice exams")
        output.append("   • Join study group sessions and discuss challenging topics")
        output.append("   • Schedule and take NCP-AAI certification exam")
        output.append("")
        output.append("4. LONG-TERM DEVELOPMENT (Ongoing)")
        output.append("   • Stay updated with NVIDIA Technical Blog")
        output.append("   • Attend NVIDIA GTC sessions on RAG and agentic AI")
        output.append("   • Build portfolio projects demonstrating RAG evaluation skills")
        output.append("   • Contribute to community forums and help other learners")
        output.append("   • Explore advanced topics (GraphRAG, multimodal RAG, reasoning)")
        output.append("")
        
        # Certification Preparation Checklist
        output.append("NCP-AAI CERTIFICATION PREPARATION CHECKLIST")
        output.append("=" * 80)
        output.append("")
        output.append("□ Completed all 7 course modules")
        output.append("□ Finished all hands-on notebooks (0-7)")
        output.append("□ Passed all module quizzes with 80%+ score")
        output.append("□ Completed capstone project")
        output.append("□ Scored 70%+ on mock certification exam")
        output.append("□ Reviewed all one-page concept summaries")
        output.append("□ Read recommended blog posts (at least 15)")
        output.append("□ Completed additional NVIDIA courses (at least 2)")
        output.append("□ Built personal RAG project with evaluation")
        output.append("□ Familiar with NVIDIA platform tools (NIM, NeMo, Triton)")
        output.append("□ Studied exam blueprint and understand all domains")
        output.append("□ Joined study group and participated in discussions")
        output.append("□ Practiced with NVIDIA AI Catalog APIs")
        output.append("□ Reviewed common pitfalls and best practices")
        output.append("□ Confident in evaluation metrics and LLM-as-a-Judge")
        output.append("")
        
        # Support and Contact
        output.append("SUPPORT AND CONTACT")
        output.append("-" * 80)
        output.append("• Technical Questions: NVIDIA Developer Forums")
        output.append("• Course Feedback: rag-evaluation-course@nvidia.com")
        output.append("• Certification Questions: certification@nvidia.com")
        output.append("• Community: Join NCP-AAI Study Group on forums")
        output.append("")
        
        output.append("GOOD LUCK WITH YOUR CERTIFICATION JOURNEY!")
        output.append("=" * 80)
        output.append("")
        
        return "\n".join(output)
    
    def export_to_markdown(self, filepath: str) -> None:
        """Export the resource package to a markdown file."""
        content = self.generate_resource_package()
        with open(filepath, 'w') as f:
            f.write(content)
    
    def generate_downloadable_package_manifest(self) -> Dict[str, List[str]]:
        """
        Generate a manifest of all downloadable materials for packaging.
        
        Returns a dictionary mapping categories to file paths for packaging.
        """
        manifest = {
            "notebooks": [
                "notebooks/notebook_0_search_paradigm_comparison.ipynb",
                "notebooks/notebook_1_embeddings_chunking.ipynb",
                "notebooks/notebook_2_rag_debugging.ipynb",
                "notebooks/notebook_3_baseline_synthetic_data.ipynb",
                "notebooks/notebook_4_customized_synthetic_data.ipynb",
                "notebooks/notebook_5_ragas_evaluation.ipynb",
                "notebooks/notebook_6_semantic_search_evaluation.ipynb",
                "notebooks/notebook_7_production_monitoring.ipynb",
                "notebooks/solutions/"
            ],
            "datasets": [
                "datasets/usc_catalog/",
                "datasets/amnesty_qa/",
                "datasets/README.md"
            ],
            "study_materials": [
                "course_materials/study_resources/module_1_concept_summary.py",
                "course_materials/study_resources/module_2_concept_summary.py",
                "course_materials/study_resources/module_3_concept_summary.py",
                "course_materials/study_resources/module_4_concept_summary.py",
                "course_materials/study_resources/module_5_concept_summary.py",
                "course_materials/study_resources/module_6_concept_summary.py",
                "course_materials/study_resources/module_7_concept_summary.py",
                "course_materials/study_resources/certification_study_guide.py",
                "course_materials/study_resources/recommended_reading_list.py"
            ],
            "assessments": [
                "course_materials/assessments/mock_certification_exam.py",
                "course_materials/assessments/capstone_project.py",
                "course_materials/assessments/module_*_quiz.py"
            ],
            "documentation": [
                "README.md",
                "ARCHITECTURE.md",
                "QUICKSTART.md",
                "requirements.txt",
                ".env.example"
            ]
        }
        return manifest


# Example usage
if __name__ == "__main__":
    package = PostCourseResourcePackage()
    
    # Print full resource package
    print(package.generate_resource_package())
    
    # Example: Get free resources
    free_resources = package.get_free_resources()
    print(f"\n{len(free_resources)} free resources available")
    
    # Example: Get NVIDIA platform resources
    nvidia_resources = package.get_nvidia_platform_resources()
    print(f"{len(nvidia_resources)} NVIDIA platform resources")
    
    # Example: Generate downloadable package manifest
    manifest = package.generate_downloadable_package_manifest()
    print(f"\nDownloadable package includes {len(manifest)} categories")
