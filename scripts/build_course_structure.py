"""
Build integrated course structure.

This script creates the complete course structure with all modules,
dependencies, cross-references, and navigation paths.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.course import Module
from models.course_integration import (
    CourseStructure,
    create_default_dependencies,
    create_default_cross_references,
    create_default_terminology,
    create_default_navigation_paths,
)


def create_module_definitions() -> list[Module]:
    """Create all 7 module definitions with proper time allocations."""
    
    modules = [
        Module(
            module_number=1,
            title="Evolution of Search to RAG Systems",
            duration_minutes=40,
            learning_objectives=[
                "Explain the progression from traditional search to RAG",
                "Compare BM25 keyword search with semantic search paradigms",
                "Understand enterprise hybrid systems combining multiple approaches",
                "Make informed decisions about when to use each search approach",
            ],
            exam_domain_mapping={
                "Agent Architecture and Design": 15.0,
                "Knowledge Integration and Data Handling": 10.0,
            },
            is_technical=True,
            lecture_time_minutes=16,  # 40%
            hands_on_time_minutes=20,  # 50%
            discussion_time_minutes=4,  # 10%
        ),
        Module(
            module_number=2,
            title="Embeddings and Vector Stores",
            duration_minutes=50,
            learning_objectives=[
                "Understand embedding fundamentals and multi-dimensional similarity",
                "Select appropriate domain-specific embedding models",
                "Configure and optimize vector stores for retrieval",
                "Handle diverse data types including tabular data in RAG systems",
                "Apply effective chunking strategies",
            ],
            exam_domain_mapping={
                "Knowledge Integration and Data Handling": 10.0,
                "NVIDIA Platform Implementation": 7.0,
            },
            is_technical=True,
            lecture_time_minutes=20,  # 40%
            hands_on_time_minutes=25,  # 50%
            discussion_time_minutes=5,  # 10%
        ),
        Module(
            module_number=3,
            title="RAG Architecture and Component Analysis",
            duration_minutes=70,
            learning_objectives=[
                "Architect end-to-end RAG systems with three-stage pipeline",
                "Diagnose failures at specific pipeline stages",
                "Evaluate retrieval and generation independently",
                "Implement error handling and graceful failure recovery",
                "Understand orchestration patterns and multi-step reasoning",
            ],
            exam_domain_mapping={
                "Agent Architecture and Design": 15.0,
                "Agent Development": 15.0,
            },
            is_technical=True,
            lecture_time_minutes=28,  # 40%
            hands_on_time_minutes=35,  # 50%
            discussion_time_minutes=7,  # 10%
        ),
        Module(
            module_number=4,
            title="Synthetic Test Data Generation",
            duration_minutes=80,
            learning_objectives=[
                "Generate robust test sets for RAG evaluation",
                "Apply prompt engineering techniques for data steering",
                "Customize prompts to reflect authentic user queries",
                "Validate synthetic data quality",
                "Support continuous evaluation workflows",
            ],
            exam_domain_mapping={
                "Evaluation and Tuning": 13.0,
                "Agent Development": 15.0,
            },
            is_technical=True,
            lecture_time_minutes=32,  # 40%
            hands_on_time_minutes=40,  # 50%
            discussion_time_minutes=8,  # 10%
        ),
        Module(
            module_number=5,
            title="RAG Evaluation Metrics and Frameworks",
            duration_minutes=100,
            learning_objectives=[
                "Implement comprehensive evaluation pipelines using Ragas",
                "Select appropriate metrics for different RAG components",
                "Customize metrics for domain-specific requirements",
                "Interpret evaluation results to guide optimization",
                "Create custom metrics from scratch",
            ],
            exam_domain_mapping={
                "Evaluation and Tuning": 13.0,
            },
            is_technical=True,
            lecture_time_minutes=40,  # 40%
            hands_on_time_minutes=50,  # 50%
            discussion_time_minutes=10,  # 10%
        ),
        Module(
            module_number=6,
            title="Semantic Search System Evaluation",
            duration_minutes=70,
            learning_objectives=[
                "Evaluate legacy search systems with modern LLM techniques",
                "Apply Ragas to non-RAG search systems",
                "Bridge traditional and modern evaluation approaches",
                "Support enterprise hybrid system requirements",
                "Optimize ranking algorithms",
            ],
            exam_domain_mapping={
                "Evaluation and Tuning": 13.0,
                "Knowledge Integration and Data Handling": 10.0,
            },
            is_technical=True,
            lecture_time_minutes=28,  # 40%
            hands_on_time_minutes=35,  # 50%
            discussion_time_minutes=7,  # 10%
        ),
        Module(
            module_number=7,
            title="Production Deployment and Advanced Topics",
            duration_minutes=60,
            learning_objectives=[
                "Deploy evaluation systems at production scale",
                "Monitor RAG performance continuously",
                "Handle enterprise-specific requirements",
                "Implement feedback loops for improvement",
                "Balance cost-efficiency with accuracy",
            ],
            exam_domain_mapping={
                "Deployment and Scaling": 13.0,
                "Run, Monitor, and Maintain": 5.0,
            },
            is_technical=True,
            lecture_time_minutes=24,  # 40%
            hands_on_time_minutes=30,  # 50%
            discussion_time_minutes=6,  # 10%
        ),
    ]
    
    return modules


def build_course_structure() -> CourseStructure:
    """Build complete integrated course structure."""
    
    modules = create_module_definitions()
    dependencies = create_default_dependencies()
    cross_references = create_default_cross_references()
    terminology = create_default_terminology()
    navigation_paths = create_default_navigation_paths()
    
    course = CourseStructure(
        course_title="Evaluating RAG and Semantic Search Systems",
        course_version="1.0.0",
        modules=modules,
        dependencies=dependencies,
        cross_references=cross_references,
        terminology=terminology,
        navigation_paths=navigation_paths,
    )
    
    return course


def print_course_summary(course: CourseStructure):
    """Print summary of course structure."""
    
    print(f"\n{'='*80}")
    print(f"Course: {course.course_title}")
    print(f"Version: {course.course_version}")
    print(f"{'='*80}\n")
    
    print(f"Total Modules: {len(course.modules)}")
    print(f"Total Duration: {course.calculate_total_duration()} minutes ({course.calculate_total_duration() / 60:.1f} hours)")
    print(f"Dependencies: {len(course.dependencies)}")
    print(f"Cross-References: {len(course.cross_references)}")
    print(f"Terminology Entries: {len(course.terminology)}")
    print(f"Navigation Paths: {len(course.navigation_paths)}\n")
    
    print("Module Sequence:")
    print("-" * 80)
    for module in course.modules:
        prereqs = course.get_prerequisites(module.module_number)
        prereq_str = f" (requires: {', '.join([str(p.module_number) for p in prereqs])})" if prereqs else ""
        print(f"  {module.module_number}. {module.title} - {module.duration_minutes}min{prereq_str}")
    
    print("\nRecommended Learning Sequence:")
    print("-" * 80)
    sequence = course.get_learning_sequence()
    print(f"  {' → '.join([str(n) for n in sequence])}")
    
    print("\nExam Domain Coverage:")
    print("-" * 80)
    coverage = course.get_exam_domain_coverage()
    for domain, modules in sorted(coverage.items()):
        module_str = ', '.join([str(m) for m in sorted(modules)])
        print(f"  {domain}: Modules {module_str}")
    
    print("\nNavigation Paths:")
    print("-" * 80)
    for path in course.navigation_paths:
        sequence_str = ' → '.join([str(n) for n in path.module_sequence])
        print(f"  {path.path_name}: {sequence_str} ({path.estimated_hours}h)")
    
    print("\nKey Terminology (by module):")
    print("-" * 80)
    for i in range(1, 8):
        terms = course.get_terminology_by_module(i)
        if terms:
            term_names = ', '.join([t.term for t in terms])
            print(f"  Module {i}: {term_names}")
    
    print(f"\n{'='*80}\n")


def validate_course_structure(course: CourseStructure) -> bool:
    """Validate course structure integrity."""
    
    print("Validating course structure...")
    
    # Check module count
    if len(course.modules) != 7:
        print(f"  ❌ Expected 7 modules, found {len(course.modules)}")
        return False
    print(f"  ✓ Module count: 7")
    
    # Check module numbers
    module_numbers = sorted([m.module_number for m in course.modules])
    if module_numbers != list(range(1, 8)):
        print(f"  ❌ Module numbers not sequential: {module_numbers}")
        return False
    print(f"  ✓ Module numbers: 1-7")
    
    # Validate dependencies
    if not course.validate_dependencies():
        print(f"  ❌ Invalid dependencies detected")
        return False
    print(f"  ✓ Dependencies valid")
    
    # Check time allocations
    all_valid = True
    for module in course.modules:
        if not module.validate_time_allocation(tolerance=5.0):
            allocation = module.calculate_time_allocation()
            print(f"  ⚠️  Module {module.module_number} time allocation: "
                  f"{allocation['lecture']:.1f}% / {allocation['hands_on']:.1f}% / {allocation['discussion']:.1f}%")
            all_valid = False
    
    if all_valid:
        print(f"  ✓ All modules have valid time allocation (40/50/10 ±5%)")
    
    # Check exam domain mappings
    for module in course.modules:
        if not module.exam_domain_mapping:
            print(f"  ❌ Module {module.module_number} has no exam domain mapping")
            return False
    print(f"  ✓ All modules have exam domain mappings")
    
    # Check technical modules have notebooks
    for module in course.modules:
        if module.is_technical and len(module.notebooks) == 0:
            print(f"  ⚠️  Module {module.module_number} is technical but has no notebooks (will be added during implementation)")
    
    print("\n✓ Course structure validation complete\n")
    return True


def main():
    """Main entry point."""
    
    print("Building integrated course structure...")
    
    course = build_course_structure()
    
    # Validate structure
    if not validate_course_structure(course):
        print("❌ Course structure validation failed")
        return 1
    
    # Print summary
    print_course_summary(course)
    
    # Save to JSON
    output_path = Path(__file__).parent.parent / "course_materials" / "course_structure.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(course.model_dump_json(indent=2))
    
    print(f"✓ Course structure saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
