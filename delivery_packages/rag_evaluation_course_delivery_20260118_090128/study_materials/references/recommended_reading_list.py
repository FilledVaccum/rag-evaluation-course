"""
Recommended Reading List for RAG Evaluation Course

Comprehensive list of external resources organized by topic and difficulty level
with annotations for each resource.

Requirements: 17.4
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty level for reading materials."""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class ResourceType(Enum):
    """Type of resource."""
    BLOG_POST = "Blog Post"
    DOCUMENTATION = "Documentation"
    TUTORIAL = "Tutorial"
    RESEARCH_PAPER = "Research Paper"
    VIDEO = "Video"
    GITHUB_REPO = "GitHub Repository"
    COURSE = "Course"


@dataclass
class ReadingResource:
    """Represents a reading resource with metadata."""
    title: str
    url: str
    resource_type: ResourceType
    difficulty: DifficultyLevel
    topics: List[str]
    annotation: str
    estimated_time_minutes: int
    
    def __str__(self) -> str:
        return f"{self.title} ({self.difficulty.value}) - {self.estimated_time_minutes} min"


class RecommendedReadingList:
    """
    Curated reading list organized by topic and difficulty level.
    
    Resources are extracted from external reference materials and organized
    to support comprehensive NCP-AAI certification preparation.
    """
    
    def __init__(self):
        """Initialize the reading list with all resources."""
        self.resources = self._compile_resources()
        self.topics = self._define_topics()
    
    def _define_topics(self) -> List[str]:
        """Define all topic categories."""
        return [
            "RAG Fundamentals",
            "Evaluation and Metrics",
            "Synthetic Data Generation",
            "Embeddings and Vector Stores",
            "NVIDIA Platform",
            "Production Deployment",
            "Agent Development",
            "Safety and Compliance",
            "Advanced Topics"
        ]

    
    def _compile_resources(self) -> List[ReadingResource]:
        """Compile all reading resources organized by topic."""
        resources = []
        
        # RAG Fundamentals
        resources.extend([
            ReadingResource(
                title="What is Retrieval-Augmented Generation?",
                url="https://www.nvidia.com/en-in/glossary/retrieval-augmented-generation/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.BEGINNER,
                topics=["RAG Fundamentals"],
                annotation="Essential introduction to RAG concepts from NVIDIA. Start here for foundational understanding.",
                estimated_time_minutes=10
            ),
            ReadingResource(
                title="Explainer: What Is Retrieval-Augmented Generation?",
                url="https://developer.nvidia.com/blog/explainer-what-is-retrieval-augmented-generation/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.BEGINNER,
                topics=["RAG Fundamentals"],
                annotation="Detailed explainer with examples and use cases. Great for understanding RAG architecture.",
                estimated_time_minutes=15
            ),
            ReadingResource(
                title="An Easy Introduction to Multimodal RAG",
                url="https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["RAG Fundamentals", "Advanced Topics"],
                annotation="Extends RAG concepts to multimodal data (text, images, video). Important for advanced applications.",
                estimated_time_minutes=20
            ),
            ReadingResource(
                title="Traditional RAG vs. Agentic RAG",
                url="https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["RAG Fundamentals", "Agent Development"],
                annotation="Compares traditional RAG with agentic approaches. Critical for understanding modern agent architectures.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="How to Take a RAG Application from Pilot to Production",
                url="https://developer.nvidia.com/blog/how-to-take-a-rag-application-from-pilot-to-production-in-four-steps/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["RAG Fundamentals", "Production Deployment"],
                annotation="Four-step guide for production deployment. Essential reading for Module 7.",
                estimated_time_minutes=20
            )
        ])
        
        # Evaluation and Metrics
        resources.extend([
            ReadingResource(
                title="Evaluating Retriever for Enterprise-Grade RAG",
                url="https://developer.nvidia.com/blog/evaluating-retriever-for-enterprise-grade-rag/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Evaluation and Metrics"],
                annotation="Deep dive into retrieval evaluation metrics. Core reading for Module 5.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="Evaluating Medical RAG with NVIDIA AI Endpoints and Ragas",
                url="https://developer.nvidia.com/blog/evaluating-medical-rag-with-nvidia-ai-endpoints-and-ragas/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Evaluation and Metrics", "NVIDIA Platform"],
                annotation="Practical example of Ragas evaluation in healthcare domain. Shows real-world application.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Mastering LLM Techniques: Evaluation",
                url="https://developer.nvidia.com/blog/mastering-llm-techniques-evaluation/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Evaluation and Metrics"],
                annotation="Comprehensive guide to LLM evaluation techniques. Must-read for certification preparation.",
                estimated_time_minutes=35
            ),
            ReadingResource(
                title="Evaluating and Enhancing RAG Pipeline Performance Using Synthetic Data",
                url="https://developer.nvidia.com/blog/evaluating-and-enhancing-rag-pipeline-performance-using-synthetic-data/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Evaluation and Metrics", "Synthetic Data Generation"],
                annotation="Combines evaluation with synthetic data generation. Bridges Modules 4 and 5.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Ragas Documentation: Agentic or Tool Use Metrics",
                url="https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/",
                resource_type=ResourceType.DOCUMENTATION,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Evaluation and Metrics", "Agent Development"],
                annotation="Official Ragas documentation for agent evaluation. Reference material for custom metrics.",
                estimated_time_minutes=20
            ),
            ReadingResource(
                title="What is AI Agent Evaluation?",
                url="https://www.ibm.com/think/topics/ai-agent-evaluation",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.BEGINNER,
                topics=["Evaluation and Metrics", "Agent Development"],
                annotation="IBM's overview of agent evaluation concepts. Good foundational reading.",
                estimated_time_minutes=15
            )
        ])

        
        # Synthetic Data Generation
        resources.extend([
            ReadingResource(
                title="Creating Synthetic Data Using Llama 3.1 405B",
                url="https://developer.nvidia.com/blog/creating-synthetic-data-using-llama-3-1-405b/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Synthetic Data Generation"],
                annotation="Practical guide to synthetic data generation with large models. Core reading for Module 4.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="An Introduction to Large Language Models: Prompt Engineering and P-Tuning",
                url="https://developer.nvidia.com/blog/an-introduction-to-large-language-models-prompt-engineering-and-p-tuning/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Synthetic Data Generation"],
                annotation="Foundational prompt engineering techniques. Essential for understanding data generation steering.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Chain of Thought Prompting Explained",
                url="https://www.codecademy.com/article/chain-of-thought-cot-prompting",
                resource_type=ResourceType.TUTORIAL,
                difficulty=DifficultyLevel.BEGINNER,
                topics=["Synthetic Data Generation"],
                annotation="Explains CoT prompting with examples. Useful for advanced prompt engineering.",
                estimated_time_minutes=15
            )
        ])
        
        # Embeddings and Vector Stores
        resources.extend([
            ReadingResource(
                title="NVIDIA Text Embedding Model Tops MTEB Leaderboard",
                url="https://developer.nvidia.com/blog/nvidia-text-embedding-model-tops-mteb-leaderboard/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Embeddings and Vector Stores", "NVIDIA Platform"],
                annotation="Introduces NV-Embed-v2 and its performance. Important for understanding NVIDIA embedding models.",
                estimated_time_minutes=20
            ),
            ReadingResource(
                title="Boost Embedding Model Accuracy for Custom Information Retrieval",
                url="https://developer.nvidia.com/blog/boost-embedding-model-accuracy-for-custom-information-retrieval/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Embeddings and Vector Stores"],
                annotation="Advanced techniques for improving embedding accuracy. Useful for domain-specific applications.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Finding the Best Chunking Strategy for Accurate AI Responses",
                url="https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Embeddings and Vector Stores"],
                annotation="Comprehensive guide to chunking strategies. Critical for Module 2 hands-on exercises.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="Enhancing GPU-Accelerated Vector Search in Faiss with NVIDIA cuVS",
                url="https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Embeddings and Vector Stores", "NVIDIA Platform"],
                annotation="GPU acceleration for vector search. Advanced topic for performance optimization.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Optimizing Vector Search for Indexing and Real-Time Retrieval",
                url="https://developer.nvidia.com/blog/optimizing-vector-search-for-indexing-and-real-time-retrieval-with-nvidia-cuvs/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Embeddings and Vector Stores", "Production Deployment"],
                annotation="Production-scale vector search optimization. Important for Module 7.",
                estimated_time_minutes=35
            ),
            ReadingResource(
                title="Develop Multilingual and Cross-Lingual Information Retrieval Systems",
                url="https://developer.nvidia.com/blog/develop-multilingual-and-cross-lingual-information-retrieval-systems-with-efficient-data-storage/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Embeddings and Vector Stores", "Advanced Topics"],
                annotation="Multilingual embedding strategies. Relevant for global enterprise applications.",
                estimated_time_minutes=30
            )
        ])

        
        # NVIDIA Platform
        resources.extend([
            ReadingResource(
                title="NVIDIA Agent Intelligence Toolkit Overview",
                url="https://docs.nvidia.com/aiqtoolkit/latest/index.html",
                resource_type=ResourceType.DOCUMENTATION,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["NVIDIA Platform", "Agent Development"],
                annotation="Official documentation for NVIDIA Agent Intelligence Toolkit. Essential reference material.",
                estimated_time_minutes=40
            ),
            ReadingResource(
                title="NVIDIA Agent Intelligence Toolkit Tutorials",
                url="https://docs.nvidia.com/aiqtoolkit/latest/tutorials/index.html",
                resource_type=ResourceType.TUTORIAL,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["NVIDIA Platform", "Agent Development"],
                annotation="Hands-on tutorials for the toolkit. Practical exercises beyond course content.",
                estimated_time_minutes=60
            ),
            ReadingResource(
                title="NeMo Agent Toolkit GitHub Repository",
                url="https://github.com/NVIDIA/NeMo-Agent-Toolkit",
                resource_type=ResourceType.GITHUB_REPO,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["NVIDIA Platform", "Agent Development"],
                annotation="Open-source repository for NeMo Agent Toolkit. For developers wanting to dive deep.",
                estimated_time_minutes=120
            ),
            ReadingResource(
                title="Build a RAG Agent with NVIDIA Nemotron",
                url="https://developer.nvidia.com/blog/build-a-rag-agent-with-nvidia-nemotron/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["NVIDIA Platform", "RAG Fundamentals"],
                annotation="Step-by-step guide using Nemotron models. Practical application of NVIDIA tools.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Enhancing RAG Applications with NVIDIA NIM",
                url="https://developer.nvidia.com/blog/enhancing-rag-applications-with-nvidia-nim/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["NVIDIA Platform", "RAG Fundamentals"],
                annotation="Introduction to NVIDIA NIM for RAG. Core platform knowledge for certification.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="Building AI Agents with NVIDIA NIM Microservices and LangChain",
                url="https://developer.nvidia.com/blog/building-ai-agents-with-nvidia-nim-microservices-and-langchain/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["NVIDIA Platform", "Agent Development"],
                annotation="Integration of NIM with LangChain. Shows ecosystem integration patterns.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="NVIDIA Triton Inference Server Optimization",
                url="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html",
                resource_type=ResourceType.DOCUMENTATION,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["NVIDIA Platform", "Production Deployment"],
                annotation="Triton optimization guide. Advanced production deployment knowledge.",
                estimated_time_minutes=45
            ),
            ReadingResource(
                title="NeMo Guardrails Toolkit",
                url="https://github.com/NVIDIA-NeMo/Guardrails",
                resource_type=ResourceType.GITHUB_REPO,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["NVIDIA Platform", "Safety and Compliance"],
                annotation="Open-source guardrails for LLM applications. Important for safety considerations.",
                estimated_time_minutes=60
            )
        ])
        
        # Production Deployment
        resources.extend([
            ReadingResource(
                title="Scaling Enterprise RAG with Accelerated Ethernet Networking",
                url="https://developer.nvidia.com/blog/scaling-enterprise-rag-with-accelerated-ethernet-networking-and-networked-storage/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Production Deployment"],
                annotation="Infrastructure considerations for scaling RAG. Enterprise deployment focus.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Enabling Horizontal Autoscaling of Enterprise RAG Components on Kubernetes",
                url="https://developer.nvidia.com/blog/enabling-horizontal-autoscaling-of-enterprise-rag-components-on-kubernetes/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Production Deployment"],
                annotation="Kubernetes-based scaling strategies. Critical for production systems.",
                estimated_time_minutes=35
            ),
            ReadingResource(
                title="A Guide to Monitoring Machine Learning Models in Production",
                url="https://developer.nvidia.com/blog/a-guide-to-monitoring-machine-learning-models-in-production/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Production Deployment"],
                annotation="Monitoring strategies for ML models. Essential for Module 7.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="Mastering LLM Techniques: LLMOps",
                url="https://developer.nvidia.com/blog/mastering-llm-techniques-llmops/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Production Deployment"],
                annotation="LLMOps best practices. Comprehensive guide to operational excellence.",
                estimated_time_minutes=40
            ),
            ReadingResource(
                title="Mastering LLM Techniques: Inference Optimization",
                url="https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Production Deployment", "NVIDIA Platform"],
                annotation="Inference optimization techniques. Performance tuning for production.",
                estimated_time_minutes=35
            )
        ])

        
        # Agent Development
        resources.extend([
            ReadingResource(
                title="What are Multi-Agent Systems?",
                url="https://www.nvidia.com/en-us/glossary/multi-agent-systems/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.BEGINNER,
                topics=["Agent Development"],
                annotation="Introduction to multi-agent systems. Foundational concept for agentic AI.",
                estimated_time_minutes=10
            ),
            ReadingResource(
                title="Building Autonomous AI with NVIDIA Agentic NeMo",
                url="https://medium.com/@zbabar/building-autonomous-ai-with-nvidia-agentic-nemo-e2992ae58cea",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Agent Development", "NVIDIA Platform"],
                annotation="Practical guide to building autonomous agents. Real-world implementation patterns.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="An Easy Introduction to LLM Reasoning, AI Agents, and Test Time Scaling",
                url="https://developer.nvidia.com/blog/an-easy-introduction-to-llm-reasoning-ai-agents-and-test-time-scaling/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Agent Development"],
                annotation="Explains reasoning in AI agents. Important for understanding agent cognition.",
                estimated_time_minutes=20
            ),
            ReadingResource(
                title="Understanding the Planning of LLM Agents: A Survey",
                url="https://arxiv.org/abs/2402.02716",
                resource_type=ResourceType.RESEARCH_PAPER,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Agent Development"],
                annotation="Academic survey of agent planning. Deep dive into planning mechanisms.",
                estimated_time_minutes=90
            ),
            ReadingResource(
                title="Improve AI Code Generation Using NVIDIA NeMo Agent Toolkit",
                url="https://developer.nvidia.com/blog/improve-ai-code-generation-using-nvidia-nemo-agent-toolkit/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Agent Development", "NVIDIA Platform"],
                annotation="Code generation use case with NeMo. Practical agent application example.",
                estimated_time_minutes=25
            )
        ])
        
        # Safety and Compliance
        resources.extend([
            ReadingResource(
                title="Securing Generative AI Deployments with NVIDIA NIM and NeMo Guardrails",
                url="https://developer.nvidia.com/blog/securing-generative-ai-deployments-with-nvidia-nim-and-nvidia-nemo-guardrails/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Safety and Compliance", "NVIDIA Platform"],
                annotation="Security best practices for AI deployments. Essential for production systems.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="Building Safer LLM Apps with LangChain Templates and NVIDIA NeMo Guardrails",
                url="https://developer.nvidia.com/blog/building-safer-llm-apps-with-langchain-templates-and-nvidia-nemo-guardrails/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Safety and Compliance"],
                annotation="Practical guardrail implementation. Shows how to add safety layers.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications",
                url="https://arxiv.org/abs/2310.10501",
                resource_type=ResourceType.RESEARCH_PAPER,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Safety and Compliance"],
                annotation="Research paper on guardrails. Academic foundation for safety mechanisms.",
                estimated_time_minutes=60
            ),
            ReadingResource(
                title="What are AI Guardrails: Definition, Types & Ethical Usage",
                url="https://coralogix.com/ai-blog/understanding-why-ai-guardrails-are-necessary-ensuring-ethical-and-responsible-ai-use/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.BEGINNER,
                topics=["Safety and Compliance"],
                annotation="Introduction to AI guardrails. Good starting point for safety concepts.",
                estimated_time_minutes=15
            ),
            ReadingResource(
                title="AI for Regulatory Compliance: Use Cases, Technologies, Benefits",
                url="https://www.leewayhertz.com/ai-for-regulatory-compliance/#core-technologies",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Safety and Compliance"],
                annotation="Regulatory compliance in AI systems. Important for enterprise deployments.",
                estimated_time_minutes=30
            )
        ])
        
        # Advanced Topics
        resources.extend([
            ReadingResource(
                title="Boosting Q&A Accuracy with GraphRAG Using PyG and Graph Databases",
                url="https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Advanced Topics", "RAG Fundamentals"],
                annotation="GraphRAG techniques for complex queries. Advanced RAG architecture pattern.",
                estimated_time_minutes=35
            ),
            ReadingResource(
                title="How to Enhance RAG Pipelines with Reasoning Using NVIDIA Llama Nemotron Models",
                url="https://developer.nvidia.com/blog/how-to-enhance-rag-pipelines-with-reasoning-using-nvidia-llama-nemotron-models/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Advanced Topics", "NVIDIA Platform"],
                annotation="Reasoning-enhanced RAG. Cutting-edge techniques for improved accuracy.",
                estimated_time_minutes=30
            ),
            ReadingResource(
                title="From Human Memory to AI Memory: A Survey on Memory Mechanisms in LLMs",
                url="https://arxiv.org/html/2504.15965v1",
                resource_type=ResourceType.RESEARCH_PAPER,
                difficulty=DifficultyLevel.ADVANCED,
                topics=["Advanced Topics", "Agent Development"],
                annotation="Academic survey on memory in LLMs. Deep dive into memory architectures.",
                estimated_time_minutes=120
            ),
            ReadingResource(
                title="What Is AI Agent Memory?",
                url="https://www.ibm.com/think/topics/ai-agent-memory",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Advanced Topics", "Agent Development"],
                annotation="IBM's overview of agent memory. Accessible introduction to memory concepts.",
                estimated_time_minutes=15
            ),
            ReadingResource(
                title="Approaches to PDF Data Extraction for Information Retrieval",
                url="https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Advanced Topics", "Embeddings and Vector Stores"],
                annotation="PDF processing for RAG. Practical techniques for document extraction.",
                estimated_time_minutes=25
            ),
            ReadingResource(
                title="Enhancing RAG Pipelines with Re-Ranking",
                url="https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/",
                resource_type=ResourceType.BLOG_POST,
                difficulty=DifficultyLevel.INTERMEDIATE,
                topics=["Advanced Topics", "RAG Fundamentals"],
                annotation="Re-ranking strategies for improved retrieval. Advanced optimization technique.",
                estimated_time_minutes=25
            )
        ])
        
        return resources

    
    def get_resources_by_topic(self, topic: str) -> List[ReadingResource]:
        """Get all resources for a specific topic."""
        return [r for r in self.resources if topic in r.topics]
    
    def get_resources_by_difficulty(self, difficulty: DifficultyLevel) -> List[ReadingResource]:
        """Get all resources at a specific difficulty level."""
        return [r for r in self.resources if r.difficulty == difficulty]
    
    def get_resources_by_type(self, resource_type: ResourceType) -> List[ReadingResource]:
        """Get all resources of a specific type."""
        return [r for r in self.resources if r.resource_type == resource_type]
    
    def get_beginner_path(self) -> List[ReadingResource]:
        """Get recommended reading path for beginners."""
        beginner_resources = self.get_resources_by_difficulty(DifficultyLevel.BEGINNER)
        # Sort by estimated time (shorter first)
        return sorted(beginner_resources, key=lambda r: r.estimated_time_minutes)
    
    def get_certification_prep_path(self) -> List[ReadingResource]:
        """Get recommended reading path for certification preparation."""
        # Focus on evaluation, NVIDIA platform, and agent development
        priority_topics = ["Evaluation and Metrics", "NVIDIA Platform", "Agent Development"]
        cert_resources = []
        for topic in priority_topics:
            cert_resources.extend(self.get_resources_by_topic(topic))
        # Remove duplicates and sort by difficulty
        unique_resources = list({r.url: r for r in cert_resources}.values())
        return sorted(unique_resources, key=lambda r: (r.difficulty.value, r.estimated_time_minutes))
    
    def estimate_total_reading_time(self, resources: List[ReadingResource] = None) -> int:
        """Estimate total reading time in minutes."""
        if resources is None:
            resources = self.resources
        return sum(r.estimated_time_minutes for r in resources)
    
    def generate_reading_list(self) -> str:
        """Generate formatted reading list organized by topic and difficulty."""
        output = []
        output.append("=" * 80)
        output.append("RECOMMENDED READING LIST")
        output.append("RAG Evaluation Course - NCP-AAI Certification Preparation")
        output.append("=" * 80)
        output.append("")
        
        # Summary statistics
        total_resources = len(self.resources)
        total_time = self.estimate_total_reading_time()
        output.append(f"Total Resources: {total_resources}")
        output.append(f"Total Estimated Reading Time: {total_time} minutes ({total_time // 60} hours)")
        output.append("")
        
        # Difficulty distribution
        beginner_count = len(self.get_resources_by_difficulty(DifficultyLevel.BEGINNER))
        intermediate_count = len(self.get_resources_by_difficulty(DifficultyLevel.INTERMEDIATE))
        advanced_count = len(self.get_resources_by_difficulty(DifficultyLevel.ADVANCED))
        output.append(f"Difficulty Distribution:")
        output.append(f"  Beginner: {beginner_count} resources")
        output.append(f"  Intermediate: {intermediate_count} resources")
        output.append(f"  Advanced: {advanced_count} resources")
        output.append("")
        
        # Resources by topic
        output.append("RESOURCES BY TOPIC")
        output.append("=" * 80)
        output.append("")
        
        for topic in self.topics:
            topic_resources = self.get_resources_by_topic(topic)
            if not topic_resources:
                continue
            
            output.append(f"{topic} ({len(topic_resources)} resources)")
            output.append("-" * 80)
            
            # Sort by difficulty
            sorted_resources = sorted(topic_resources, key=lambda r: r.difficulty.value)
            
            for resource in sorted_resources:
                output.append(f"\n[{resource.difficulty.value}] {resource.title}")
                output.append(f"  Type: {resource.resource_type.value}")
                output.append(f"  Time: {resource.estimated_time_minutes} minutes")
                output.append(f"  URL: {resource.url}")
                output.append(f"  {resource.annotation}")
            
            output.append("")
        
        # Recommended learning paths
        output.append("RECOMMENDED LEARNING PATHS")
        output.append("=" * 80)
        output.append("")
        
        output.append("1. BEGINNER PATH (Start Here)")
        output.append("-" * 80)
        beginner_path = self.get_beginner_path()
        for i, resource in enumerate(beginner_path, 1):
            output.append(f"{i}. {resource.title} ({resource.estimated_time_minutes} min)")
            output.append(f"   {resource.url}")
        output.append(f"Total time: {sum(r.estimated_time_minutes for r in beginner_path)} minutes")
        output.append("")
        
        output.append("2. CERTIFICATION PREPARATION PATH")
        output.append("-" * 80)
        cert_path = self.get_certification_prep_path()
        for i, resource in enumerate(cert_path[:15], 1):  # Top 15 for cert prep
            output.append(f"{i}. {resource.title} ({resource.estimated_time_minutes} min)")
            output.append(f"   Topics: {', '.join(resource.topics)}")
        output.append(f"Total time (top 15): {sum(r.estimated_time_minutes for r in cert_path[:15])} minutes")
        output.append("")
        
        # Study tips
        output.append("STUDY TIPS")
        output.append("-" * 80)
        output.append("• Start with Beginner resources to build foundational understanding")
        output.append("• Focus on Evaluation and Metrics resources (13% of exam)")
        output.append("• Read NVIDIA platform documentation thoroughly")
        output.append("• Prioritize blog posts and tutorials over research papers initially")
        output.append("• Revisit Advanced resources after completing course modules")
        output.append("• Use annotations to understand relevance to course content")
        output.append("• Allocate 10-15 hours for comprehensive reading beyond course time")
        output.append("")
        
        return "\n".join(output)
    
    def export_to_markdown(self, filepath: str) -> None:
        """Export the reading list to a markdown file."""
        content = self.generate_reading_list()
        with open(filepath, 'w') as f:
            f.write(content)


# Example usage
if __name__ == "__main__":
    reading_list = RecommendedReadingList()
    
    # Print full reading list
    print(reading_list.generate_reading_list())
    
    # Example: Get resources for specific topic
    eval_resources = reading_list.get_resources_by_topic("Evaluation and Metrics")
    print(f"\nFound {len(eval_resources)} resources on Evaluation and Metrics")
    
    # Example: Get beginner path
    beginner_path = reading_list.get_beginner_path()
    print(f"\nBeginner path has {len(beginner_path)} resources")
