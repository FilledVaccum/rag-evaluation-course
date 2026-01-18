"""
Synthetic Data Generator for RAG Evaluation
Implements LLM-based test data generation using NVIDIA Nemotron-4-340B

Requirements Coverage: 6.2, 6.3, 6.5
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import json


class SynthesizerType(Enum):
    """Types of query synthesizers available."""
    SPECIFIC = "specific"  # Fact-seeking, detailed questions
    ABSTRACT = "abstract"  # High-level, conceptual questions
    REASONING = "reasoning"  # Multi-hop reasoning questions


@dataclass
class SyntheticQuery:
    """Represents a generated synthetic query."""
    query_text: str
    synthesizer_type: SynthesizerType
    quality_score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """Template for synthetic data generation prompts."""
    system_instruction: str
    user_persona: str
    constraints: List[str]
    examples: List[str]  # Should be 3-5 examples
    negative_examples: List[str]
    
    def validate(self) -> bool:
        """Validate that prompt follows best practices."""
        if len(self.examples) < 3 or len(self.examples) > 5:
            return False
        return True
    
    def to_prompt(self) -> str:
        """Convert template to actual prompt string."""
        prompt_parts = [
            f"System: {self.system_instruction}",
            f"\nUser Persona: {self.user_persona}",
            "\nConstraints:"
        ]
        
        for constraint in self.constraints:
            prompt_parts.append(f"  - {constraint}")
        
        prompt_parts.append("\nDO NOT generate:")
        for neg_example in self.negative_examples:
            prompt_parts.append(f"  - {neg_example}")
        
        prompt_parts.append("\nExamples of good queries:")
        for i, example in enumerate(self.examples, 1):
            prompt_parts.append(f"{i}. \"{example}\"")
        
        prompt_parts.append("\nGenerate similar queries following these patterns.")
        
        return "\n".join(prompt_parts)


class QualityValidator:
    """Validates quality of generated synthetic queries."""
    
    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 50,
        banned_keywords: Optional[List[str]] = None,
        similarity_threshold: float = 0.9
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.banned_keywords = banned_keywords or []
        self.similarity_threshold = similarity_threshold
    
    def validate_length(self, query: str) -> bool:
        """Check if query length is within acceptable range."""
        word_count = len(query.split())
        return self.min_length <= word_count <= self.max_length
    
    def validate_keywords(self, query: str) -> bool:
        """Check if query contains banned keywords."""
        query_lower = query.lower()
        return not any(keyword.lower() in query_lower for keyword in self.banned_keywords)
    
    def validate_domain_relevance(self, query: str, domain_keywords: List[str]) -> bool:
        """Check if query is relevant to domain."""
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in domain_keywords)
    
    def calculate_quality_score(
        self,
        query: str,
        domain_keywords: Optional[List[str]] = None
    ) -> float:
        """Calculate overall quality score for a query."""
        score = 0.0
        checks = 0
        
        # Length check
        if self.validate_length(query):
            score += 1.0
        checks += 1
        
        # Keyword check
        if self.validate_keywords(query):
            score += 1.0
        checks += 1
        
        # Domain relevance check (if domain keywords provided)
        if domain_keywords:
            if self.validate_domain_relevance(query, domain_keywords):
                score += 1.0
            checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    def filter_duplicates(
        self,
        queries: List[str],
        similarity_func: Optional[Callable[[str, str], float]] = None
    ) -> List[str]:
        """Remove near-duplicate queries based on similarity."""
        if not similarity_func:
            # Simple word-based similarity if no function provided
            def simple_similarity(q1: str, q2: str) -> float:
                words1 = set(q1.lower().split())
                words2 = set(q2.lower().split())
                if not words1 or not words2:
                    return 0.0
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union)
            
            similarity_func = simple_similarity
        
        filtered = []
        for query in queries:
            is_duplicate = False
            for existing in filtered:
                if similarity_func(query, existing) > self.similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(query)
        
        return filtered


class SyntheticDataGenerator:
    """
    Generates synthetic test data for RAG evaluation using LLMs.
    
    Uses NVIDIA Nemotron-4-340B for high-quality synthetic query generation
    with customizable prompts and quality validation.
    
    Requirements: 6.2, 6.3, 6.5
    """
    
    def __init__(
        self,
        llm_endpoint: str,
        api_key: str,
        model_name: str = "nvidia/nemotron-4-340b-instruct",
        validator: Optional[QualityValidator] = None
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            llm_endpoint: NVIDIA NIM endpoint URL
            api_key: API key for authentication
            model_name: Model to use (default: Nemotron-4-340B)
            validator: Quality validator instance
        """
        self.llm_endpoint = llm_endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.validator = validator or QualityValidator()
        self.prompt_template: Optional[PromptTemplate] = None
    
    def customize_prompt(
        self,
        system_instruction: str,
        user_persona: str,
        constraints: List[str],
        examples: List[str],
        negative_examples: List[str]
    ) -> None:
        """
        Customize the prompt template for data generation.
        
        Best practices:
        - Use exactly 3-5 examples (optimal pattern)
        - Be extremely specific in instructions
        - Include explicit negative examples
        - Define clear user persona
        
        Args:
            system_instruction: High-level instruction for the LLM
            user_persona: Description of user being simulated
            constraints: List of constraints for generated queries
            examples: 3-5 example queries (MUST be 3-5)
            negative_examples: Examples of what NOT to generate
        
        Raises:
            ValueError: If examples count is not 3-5
        """
        if len(examples) < 3 or len(examples) > 5:
            raise ValueError(
                f"Examples must be 3-5 for optimal steering. Got {len(examples)}. "
                "Too few leads to generic outputs, too many leads to overfitting."
            )
        
        self.prompt_template = PromptTemplate(
            system_instruction=system_instruction,
            user_persona=user_persona,
            constraints=constraints,
            examples=examples,
            negative_examples=negative_examples
        )
    
    def generate_questions(
        self,
        dataset_context: str,
        num_samples: int = 10,
        synthesizer_type: SynthesizerType = SynthesizerType.SPECIFIC,
        temperature: float = 0.7
    ) -> List[SyntheticQuery]:
        """
        Generate synthetic questions using the configured LLM.
        
        Args:
            dataset_context: Context from the knowledge base (e.g., course catalog)
            num_samples: Number of queries to generate
            synthesizer_type: Type of synthesizer to use
            temperature: Sampling temperature (higher = more diverse)
        
        Returns:
            List of generated synthetic queries
        
        Raises:
            ValueError: If prompt template not configured
        """
        if not self.prompt_template:
            raise ValueError(
                "Prompt template not configured. Call customize_prompt() first."
            )
        
        # Build the full prompt
        base_prompt = self.prompt_template.to_prompt()
        
        # Add synthesizer-specific instructions
        synthesizer_instructions = self._get_synthesizer_instructions(synthesizer_type)
        
        full_prompt = f"{base_prompt}\n\n{synthesizer_instructions}\n\nDataset Context:\n{dataset_context}\n\nGenerate {num_samples} queries:"
        
        # Call LLM (placeholder - actual implementation would call NVIDIA NIM)
        generated_texts = self._call_llm(full_prompt, num_samples, temperature)
        
        # Parse and create SyntheticQuery objects
        queries = []
        for text in generated_texts:
            query = SyntheticQuery(
                query_text=text.strip(),
                synthesizer_type=synthesizer_type,
                metadata={"temperature": temperature}
            )
            queries.append(query)
        
        return queries
    
    def _get_synthesizer_instructions(self, synthesizer_type: SynthesizerType) -> str:
        """Get specific instructions for each synthesizer type."""
        instructions = {
            SynthesizerType.SPECIFIC: """
Generate specific, fact-seeking questions that:
- Ask about concrete details (dates, names, numbers, requirements)
- Can be answered with direct information from the dataset
- Are precise and unambiguous
Example: "What is the prerequisite for CSCI 567?"
            """,
            SynthesizerType.ABSTRACT: """
Generate abstract, conceptual questions that:
- Ask about high-level concepts and relationships
- Require understanding of broader patterns
- May need synthesis of multiple pieces of information
Example: "What are the main areas of focus in the CS curriculum?"
            """,
            SynthesizerType.REASONING: """
Generate reasoning questions that:
- Require multi-hop reasoning across multiple facts
- Involve comparison, planning, or decision-making
- Cannot be answered with a single fact lookup
Example: "If I want to specialize in AI but I'm weak at math, what's my course path?"
            """
        }
        return instructions.get(synthesizer_type, "")
    
    def _call_llm(
        self,
        prompt: str,
        num_samples: int,
        temperature: float
    ) -> List[str]:
        """
        Call the LLM endpoint to generate queries.
        
        This is a placeholder implementation. In production, this would:
        1. Make HTTP request to NVIDIA NIM endpoint
        2. Handle authentication with API key
        3. Parse response and extract generated queries
        4. Handle errors and retries
        
        Args:
            prompt: Full prompt to send to LLM
            num_samples: Number of samples to generate
            temperature: Sampling temperature
        
        Returns:
            List of generated query strings
        """
        # Placeholder implementation
        # In real implementation, would call NVIDIA NIM API
        return [
            f"Generated query {i+1} (placeholder)" 
            for i in range(num_samples)
        ]
    
    def validate_quality(
        self,
        queries: List[SyntheticQuery],
        domain_keywords: Optional[List[str]] = None,
        min_quality_score: float = 0.7
    ) -> List[SyntheticQuery]:
        """
        Validate and filter generated queries based on quality metrics.
        
        Applies multiple quality checks:
        - Length validation (5-50 words)
        - Banned keyword filtering
        - Domain relevance checking
        - Duplicate removal
        
        Args:
            queries: List of generated queries to validate
            domain_keywords: Keywords that indicate domain relevance
            min_quality_score: Minimum quality score to keep (0.0-1.0)
        
        Returns:
            Filtered list of high-quality queries
        """
        validated = []
        
        for query in queries:
            # Calculate quality score
            score = self.validator.calculate_quality_score(
                query.query_text,
                domain_keywords
            )
            query.quality_score = score
            
            # Keep if meets minimum threshold
            if score >= min_quality_score:
                validated.append(query)
        
        # Remove duplicates
        query_texts = [q.query_text for q in validated]
        unique_texts = self.validator.filter_duplicates(query_texts)
        
        # Rebuild list with only unique queries
        unique_queries = [
            q for q in validated 
            if q.query_text in unique_texts
        ]
        
        return unique_queries
    
    def mix_synthesizers(
        self,
        dataset_context: str,
        synthesizer_configs: List[Dict],
        total_samples: int = 100
    ) -> List[SyntheticQuery]:
        """
        Generate queries using multiple synthesizers with specified ratios.
        
        Example: 50% specific, 50% reasoning queries
        
        Args:
            dataset_context: Context from knowledge base
            synthesizer_configs: List of dicts with 'type' and 'ratio' keys
                Example: [
                    {'type': SynthesizerType.SPECIFIC, 'ratio': 0.5},
                    {'type': SynthesizerType.REASONING, 'ratio': 0.5}
                ]
            total_samples: Total number of queries to generate
        
        Returns:
            Combined list of queries from all synthesizers
        
        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        # Validate ratios
        total_ratio = sum(config['ratio'] for config in synthesizer_configs)
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(
                f"Synthesizer ratios must sum to 1.0, got {total_ratio}"
            )
        
        all_queries = []
        
        for config in synthesizer_configs:
            synthesizer_type = config['type']
            ratio = config['ratio']
            num_samples = int(total_samples * ratio)
            
            # Generate queries for this synthesizer
            queries = self.generate_questions(
                dataset_context=dataset_context,
                num_samples=num_samples,
                synthesizer_type=synthesizer_type
            )
            
            all_queries.extend(queries)
        
        return all_queries
    
    def get_default_prompt_template(self, domain: str = "courses") -> PromptTemplate:
        """
        Get a default prompt template for common domains.
        
        Args:
            domain: Domain type ('courses', 'qa', 'documentation')
        
        Returns:
            Default PromptTemplate for the domain
        """
        if domain == "courses":
            return PromptTemplate(
                system_instruction="You are simulating a student asking questions about courses.",
                user_persona="Undergraduate student planning their schedule",
                constraints=[
                    "Questions should be 10-25 words long",
                    "Focus on practical course selection concerns",
                    "Use casual, student-like language",
                    "Questions must be answerable with course catalog"
                ],
                examples=[
                    "I want to take machine learning but I only know Python, what should I do?",
                    "Which database course has the most hands-on projects?",
                    "Are there any 2-unit courses that count toward the CS major?",
                    "What's the typical workload for graduate-level systems courses?"
                ],
                negative_examples=[
                    "Questions about admissions or tuition",
                    "Philosophical questions about education",
                    "Questions answerable by course title alone",
                    "Questions about general university policies"
                ]
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")


def create_default_generator(
    llm_endpoint: str,
    api_key: str,
    domain: str = "courses"
) -> SyntheticDataGenerator:
    """
    Create a SyntheticDataGenerator with sensible defaults.
    
    Args:
        llm_endpoint: NVIDIA NIM endpoint URL
        api_key: API key for authentication
        domain: Domain type for default prompt template
    
    Returns:
        Configured SyntheticDataGenerator instance
    """
    generator = SyntheticDataGenerator(
        llm_endpoint=llm_endpoint,
        api_key=api_key,
        validator=QualityValidator(
            min_length=5,
            max_length=50,
            banned_keywords=["weather", "admission", "tuition", "apply"],
            similarity_threshold=0.9
        )
    )
    
    # Set default prompt template
    template = generator.get_default_prompt_template(domain)
    generator.customize_prompt(
        system_instruction=template.system_instruction,
        user_persona=template.user_persona,
        constraints=template.constraints,
        examples=template.examples,
        negative_examples=template.negative_examples
    )
    
    return generator


if __name__ == "__main__":
    # Example usage
    print("Synthetic Data Generator")
    print("=" * 50)
    
    # Create generator with default settings
    generator = create_default_generator(
        llm_endpoint="https://api.nvidia.com/nim",
        api_key="your-api-key",
        domain="courses"
    )
    
    print("\nPrompt Template Configured:")
    print(f"  Examples: {len(generator.prompt_template.examples)}")
    print(f"  Constraints: {len(generator.prompt_template.constraints)}")
    print(f"  Negative Examples: {len(generator.prompt_template.negative_examples)}")
    
    print("\nGenerator ready for synthetic data generation!")
    print("\nNext steps:")
    print("  1. Call generate_questions() with dataset context")
    print("  2. Validate quality with validate_quality()")
    print("  3. Mix synthesizers with mix_synthesizers()")
