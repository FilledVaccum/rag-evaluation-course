"""
Create course delivery package.

This script packages all course materials for delivery:
- Lecture materials (slides, speaker notes, diagrams)
- Jupyter notebooks
- Datasets
- Assessments (quizzes, exams, challenges)
- Study materials (guides, summaries, references)
- Instructor guide
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.course_integration import CourseStructure


def load_course_structure() -> CourseStructure:
    """Load course structure from JSON file."""
    course_file = Path(__file__).parent.parent / "course_materials" / "course_structure.json"
    
    if not course_file.exists():
        print(f"❌ Course structure file not found: {course_file}")
        print("   Run 'python scripts/build_course_structure.py' first")
        sys.exit(1)
    
    with open(course_file, 'r') as f:
        data = json.load(f)
    
    return CourseStructure(**data)


def create_package_structure(package_dir: Path):
    """Create directory structure for delivery package."""
    
    directories = [
        package_dir / "lecture_materials",
        package_dir / "lecture_materials" / "slides",
        package_dir / "lecture_materials" / "speaker_notes",
        package_dir / "lecture_materials" / "diagrams",
        package_dir / "lecture_materials" / "case_studies",
        package_dir / "notebooks",
        package_dir / "datasets",
        package_dir / "assessments",
        package_dir / "assessments" / "quizzes",
        package_dir / "assessments" / "challenges",
        package_dir / "assessments" / "mock_exam",
        package_dir / "study_materials",
        package_dir / "study_materials" / "concept_summaries",
        package_dir / "study_materials" / "guides",
        package_dir / "study_materials" / "references",
        package_dir / "instructor_guide",
        package_dir / "scripts",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories


def copy_lecture_materials(source_dir: Path, package_dir: Path) -> Dict[str, int]:
    """Copy lecture materials to package."""
    stats = {"slides": 0, "speaker_notes": 0, "diagrams": 0, "case_studies": 0}
    
    slides_dir = source_dir / "course_materials" / "slides"
    if slides_dir.exists():
        for file in slides_dir.glob("*.py"):
            if file.name != "__init__.py":
                shutil.copy2(file, package_dir / "lecture_materials" / "slides" / file.name)
                stats["slides"] += 1
    
    # Copy README
    readme = slides_dir / "README.md"
    if readme.exists():
        shutil.copy2(readme, package_dir / "lecture_materials" / "README.md")
    
    return stats


def copy_notebooks(source_dir: Path, package_dir: Path) -> int:
    """Copy Jupyter notebooks to package."""
    count = 0
    
    notebooks_dir = source_dir / "course_materials" / "notebooks"
    if notebooks_dir.exists():
        for file in notebooks_dir.glob("*.py"):
            if file.name != "__init__.py" and file.name != ".gitkeep":
                shutil.copy2(file, package_dir / "notebooks" / file.name)
                count += 1
    
    return count


def copy_datasets(source_dir: Path, package_dir: Path) -> int:
    """Copy datasets to package."""
    count = 0
    
    datasets_dir = source_dir / "course_materials" / "datasets"
    if datasets_dir.exists():
        for file in datasets_dir.iterdir():
            if file.is_file() and file.name != ".gitkeep":
                shutil.copy2(file, package_dir / "datasets" / file.name)
                count += 1
    
    return count


def copy_assessments(source_dir: Path, package_dir: Path) -> Dict[str, int]:
    """Copy assessments to package."""
    stats = {"quizzes": 0, "challenges": 0, "mock_exam": 0}
    
    assessments_dir = source_dir / "course_materials" / "assessments"
    if assessments_dir.exists():
        for file in assessments_dir.glob("*.py"):
            if file.name != "__init__.py" and file.name != ".gitkeep":
                if "quiz" in file.name:
                    shutil.copy2(file, package_dir / "assessments" / "quizzes" / file.name)
                    stats["quizzes"] += 1
                elif "challenge" in file.name or "exercise" in file.name:
                    shutil.copy2(file, package_dir / "assessments" / "challenges" / file.name)
                    stats["challenges"] += 1
                elif "mock" in file.name or "exam" in file.name:
                    shutil.copy2(file, package_dir / "assessments" / "mock_exam" / file.name)
                    stats["mock_exam"] += 1
    
    return stats


def copy_study_materials(source_dir: Path, package_dir: Path) -> Dict[str, int]:
    """Copy study materials to package."""
    stats = {"summaries": 0, "guides": 0, "references": 0}
    
    study_dir = source_dir / "course_materials" / "study_resources"
    if study_dir.exists():
        for file in study_dir.glob("*.py"):
            if file.name != "__init__.py" and file.name != ".gitkeep":
                if "summary" in file.name:
                    shutil.copy2(file, package_dir / "study_materials" / "concept_summaries" / file.name)
                    stats["summaries"] += 1
                elif "guide" in file.name or "pitfalls" in file.name or "prompt" in file.name:
                    shutil.copy2(file, package_dir / "study_materials" / "guides" / file.name)
                    stats["guides"] += 1
                elif "reading" in file.name or "resources" in file.name:
                    shutil.copy2(file, package_dir / "study_materials" / "references" / file.name)
                    stats["references"] += 1
    
    return stats


def create_instructor_guide(course: CourseStructure, package_dir: Path):
    """Create comprehensive instructor guide."""
    
    guide_content = f"""# Instructor Guide: {course.course_title}

## Course Overview

**Version:** {course.course_version}
**Total Duration:** {course.calculate_total_duration() / 60:.1f} hours ({course.calculate_total_duration()} minutes)
**Target Certification:** NVIDIA-Certified Professional: Agentic AI (NCP-AAI)
**Target Audience:** AI/ML professionals with 1-2 years of experience

## Course Objectives

This course prepares students for the NCP-AAI certification exam with primary focus on:
- Evaluation and Tuning (13% of exam)
- Knowledge Integration and Data Handling (10% of exam)
- Supporting coverage of Agent Development and Architecture

## Module Structure

"""
    
    for module in course.modules:
        allocation = module.calculate_time_allocation()
        prereqs = course.get_prerequisites(module.module_number)
        prereq_str = f" (Prerequisites: Modules {', '.join([str(p.module_number) for p in prereqs])})" if prereqs else ""
        
        guide_content += f"""
### Module {module.module_number}: {module.title}

**Duration:** {module.duration_minutes} minutes ({module.duration_minutes / 60:.1f} hours)
**Time Allocation:** {allocation['lecture']:.0f}% Lecture / {allocation['hands_on']:.0f}% Hands-On / {allocation['discussion']:.0f}% Discussion{prereq_str}

**Learning Objectives:**
"""
        for obj in module.learning_objectives:
            guide_content += f"- {obj}\n"
        
        guide_content += "\n**Exam Domain Mapping:**\n"
        for domain, weight in module.exam_domain_mapping.items():
            guide_content += f"- {domain} ({weight}%)\n"
        
        guide_content += "\n**Delivery Tips:**\n"
        guide_content += f"- Lecture/Demo: {module.lecture_time_minutes} minutes\n"
        guide_content += f"- Hands-On Practice: {module.hands_on_time_minutes} minutes\n"
        guide_content += f"- Discussion/Q&A: {module.discussion_time_minutes} minutes\n"
        
        # Add cross-references
        refs_from = course.get_cross_references_from(module.module_number)
        refs_to = course.get_cross_references_to(module.module_number)
        
        if refs_from or refs_to:
            guide_content += "\n**Cross-References:**\n"
            if refs_from:
                guide_content += "- References to other modules:\n"
                for ref in refs_from:
                    guide_content += f"  - Module {ref.target_module}: {ref.description}\n"
            if refs_to:
                guide_content += "- Referenced by other modules:\n"
                for ref in refs_to:
                    guide_content += f"  - Module {ref.source_module}: {ref.description}\n"
        
        guide_content += "\n---\n"
    
    # Add navigation paths
    guide_content += """
## Learning Paths

The course supports multiple learning paths for different student needs:

"""
    
    for path in course.navigation_paths:
        sequence_str = ' → '.join([str(n) for n in path.module_sequence])
        guide_content += f"""
### {path.path_name}

**Target Audience:** {path.target_audience}
**Estimated Duration:** {path.estimated_hours} hours
**Module Sequence:** {sequence_str}

{path.description}

"""
    
    # Add terminology reference
    guide_content += """
## Key Terminology

Students should master these terms throughout the course:

"""
    
    for i in range(1, 8):
        terms = course.get_terminology_by_module(i)
        if terms:
            guide_content += f"\n**Module {i}:**\n"
            for term in terms:
                guide_content += f"- **{term.term}**: {term.definition}\n"
                if term.aliases:
                    guide_content += f"  - Also known as: {', '.join(term.aliases)}\n"
    
    # Add delivery recommendations
    guide_content += """
## Delivery Recommendations

### Time Management

- **Lecture/Demo (40%):** Focus on concepts, demonstrations, and examples
- **Hands-On Practice (50%):** Students work through notebooks and exercises
- **Discussion/Q&A (10%):** Address questions, clarify concepts, share insights

### Teaching Strategies

1. **Start with Motivation:** Begin each module with real-world problems
2. **Hands-On First:** Let students experiment before explaining theory
3. **Intentional Bugs:** Notebooks include bugs for debugging practice
4. **Open-Ended Exercises:** Multiple valid solutions encourage exploration
5. **Certification Focus:** Regularly reference exam domains and requirements

### Common Student Questions

**Module 1-2:**
- "When should I use BM25 vs. semantic search?"
- "How do I choose the right embedding model?"
- "What chunk size should I use?"

**Module 3-4:**
- "How do I know if retrieval or generation is failing?"
- "How many examples should I include in prompts?"
- "How do I validate synthetic data quality?"

**Module 5-6:**
- "Why can't I use BLEU or F1 scores for RAG?"
- "How do I interpret faithfulness scores?"
- "Can I apply Ragas to non-RAG systems?"

**Module 7:**
- "How do I monitor RAG systems in production?"
- "What's the cost-accuracy trade-off?"
- "How do I handle regulatory compliance?"

### Technical Setup

Ensure students have:
- Python 3.10+ installed
- JupyterLab environment configured
- NVIDIA API keys (provide instructions)
- All dependencies installed (requirements.txt)
- Datasets pre-loaded

### Troubleshooting

Common issues and solutions:
- **API Rate Limits:** Use retry logic with exponential backoff
- **Memory Issues:** Reduce batch sizes, use streaming
- **Slow Embeddings:** Cache embeddings, use smaller models for testing
- **Missing Dependencies:** Provide requirements.txt and setup script

## Assessment Guidelines

### Module Quizzes
- 5-10 questions per module
- Mix of conceptual and applied questions
- Immediate feedback with explanations

### Hands-On Challenges
- Open-ended exercises
- Evaluation rubrics provided
- Multiple valid solutions

### Mock Certification Exam
- 60-70 questions
- 120-minute time limit
- Mirrors actual NCP-AAI exam format
- Detailed explanations for all answers

## Resources

### For Instructors
- Slide decks with speaker notes
- Case studies and examples
- Timing guidance for each section
- Common student questions and answers

### For Students
- Jupyter notebooks with exercises
- Datasets (USC Course Catalog, Amnesty Q&A)
- Study guides and concept summaries
- Practice questions and mock exam
- Recommended reading list

## Support

For questions or issues:
- Technical support: [support email]
- Course updates: [GitHub repository]
- Community forum: [forum link]
- NVIDIA certification: https://www.nvidia.com/en-us/training/certification/

---

**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}
**Course Version:** {course.course_version}
"""
    
    # Write instructor guide
    guide_file = package_dir / "instructor_guide" / "INSTRUCTOR_GUIDE.md"
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    return guide_file


def create_package_manifest(course: CourseStructure, package_dir: Path, stats: Dict):
    """Create manifest file listing all package contents."""
    
    manifest = {
        "course_title": course.course_title,
        "course_version": course.course_version,
        "package_date": datetime.now().isoformat(),
        "total_duration_hours": course.calculate_total_duration() / 60,
        "module_count": len(course.modules),
        "contents": stats,
        "modules": [
            {
                "number": m.module_number,
                "title": m.title,
                "duration_minutes": m.duration_minutes,
                "exam_domains": list(m.exam_domain_mapping.keys())
            }
            for m in course.modules
        ],
        "navigation_paths": [
            {
                "name": p.path_name,
                "sequence": p.module_sequence,
                "estimated_hours": p.estimated_hours
            }
            for p in course.navigation_paths
        ]
    }
    
    manifest_file = package_dir / "MANIFEST.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_file


def create_readme(course: CourseStructure, package_dir: Path):
    """Create README for the delivery package."""
    
    readme_content = f"""# {course.course_title} - Delivery Package

**Version:** {course.course_version}
**Package Date:** {datetime.now().strftime("%Y-%m-%d")}

## Contents

This delivery package contains all materials needed to deliver the "{course.course_title}" course.

### Directory Structure

```
delivery_package/
├── lecture_materials/       # Lecture slides, speaker notes, diagrams
│   ├── slides/             # Module slide decks
│   ├── speaker_notes/      # Instructor notes
│   ├── diagrams/           # Architecture diagrams
│   └── case_studies/       # Real-world examples
├── notebooks/              # Jupyter notebooks for hands-on practice
├── datasets/               # Course datasets (USC Catalog, Amnesty Q&A)
├── assessments/            # Quizzes, challenges, mock exam
│   ├── quizzes/           # Module-end quizzes
│   ├── challenges/        # Hands-on challenges
│   └── mock_exam/         # Mock certification exam
├── study_materials/        # Study guides and references
│   ├── concept_summaries/ # One-page module summaries
│   ├── guides/            # Study guides and best practices
│   └── references/        # Recommended reading
├── instructor_guide/       # Comprehensive instructor guide
├── scripts/               # Setup and validation scripts
├── MANIFEST.json          # Package contents manifest
└── README.md              # This file
```

## Course Overview

**Total Duration:** {course.calculate_total_duration() / 60:.1f} hours
**Modules:** {len(course.modules)}
**Target Certification:** NVIDIA-Certified Professional: Agentic AI (NCP-AAI)

### Modules

"""
    
    for module in course.modules:
        readme_content += f"{module.module_number}. {module.title} ({module.duration_minutes}min)\n"
    
    readme_content += f"""
## Quick Start

### For Instructors

1. Review the **Instructor Guide** in `instructor_guide/INSTRUCTOR_GUIDE.md`
2. Familiarize yourself with lecture materials in `lecture_materials/`
3. Test all notebooks in `notebooks/` before delivery
4. Review assessment rubrics in `assessments/`

### For Students

1. Install Python 3.10+ and JupyterLab
2. Install dependencies: `pip install -r requirements.txt`
3. Set up NVIDIA API keys (see technical setup guide)
4. Start with Module 1 notebooks

## Technical Requirements

- Python 3.10 or higher
- JupyterLab
- NVIDIA API key (for NIM access)
- 8GB RAM minimum
- Internet connection for API access

## Support

- Course documentation: See `instructor_guide/`
- Technical issues: [support contact]
- Course updates: [repository link]

## License

[License information]

---

**Package Version:** {course.course_version}
**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}
"""
    
    readme_file = package_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    return readme_file


def create_delivery_package() -> Path:
    """Create complete delivery package."""
    
    print("Creating course delivery package...\n")
    
    # Load course structure
    course = load_course_structure()
    print(f"✓ Loaded course structure: {course.course_title}")
    print(f"  Version: {course.course_version}")
    print(f"  Modules: {len(course.modules)}")
    print(f"  Duration: {course.calculate_total_duration() / 60:.1f} hours\n")
    
    # Create package directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"rag_evaluation_course_delivery_{timestamp}"
    package_dir = Path(__file__).parent.parent / "delivery_packages" / package_name
    
    print(f"Creating package directory: {package_dir}\n")
    create_package_structure(package_dir)
    
    # Copy materials
    source_dir = Path(__file__).parent.parent
    
    print("Copying course materials...")
    stats = {}
    
    lecture_stats = copy_lecture_materials(source_dir, package_dir)
    print(f"  ✓ Lecture materials: {lecture_stats['slides']} slide files")
    stats["lecture_materials"] = lecture_stats
    
    notebook_count = copy_notebooks(source_dir, package_dir)
    print(f"  ✓ Notebooks: {notebook_count} files")
    stats["notebooks"] = notebook_count
    
    dataset_count = copy_datasets(source_dir, package_dir)
    print(f"  ✓ Datasets: {dataset_count} files")
    stats["datasets"] = dataset_count
    
    assessment_stats = copy_assessments(source_dir, package_dir)
    print(f"  ✓ Assessments: {assessment_stats['quizzes']} quizzes, {assessment_stats['challenges']} challenges, {assessment_stats['mock_exam']} mock exam")
    stats["assessments"] = assessment_stats
    
    study_stats = copy_study_materials(source_dir, package_dir)
    print(f"  ✓ Study materials: {study_stats['summaries']} summaries, {study_stats['guides']} guides, {study_stats['references']} references")
    stats["study_materials"] = study_stats
    
    print("\nCreating package documentation...")
    
    # Create instructor guide
    guide_file = create_instructor_guide(course, package_dir)
    print(f"  ✓ Instructor guide: {guide_file.name}")
    
    # Create manifest
    manifest_file = create_package_manifest(course, package_dir, stats)
    print(f"  ✓ Package manifest: {manifest_file.name}")
    
    # Create README
    readme_file = create_readme(course, package_dir)
    print(f"  ✓ Package README: {readme_file.name}")
    
    # Copy course structure
    shutil.copy2(
        source_dir / "course_materials" / "course_structure.json",
        package_dir / "course_structure.json"
    )
    print(f"  ✓ Course structure: course_structure.json")
    
    # Copy setup scripts
    scripts_to_copy = [
        "setup_environment.py",
        "validate_environment.py",
        "preload_datasets.py",
    ]
    
    for script in scripts_to_copy:
        script_path = source_dir / "scripts" / script
        if script_path.exists():
            shutil.copy2(script_path, package_dir / "scripts" / script)
    
    print(f"  ✓ Setup scripts: {len(scripts_to_copy)} files")
    
    print(f"\n{'='*80}")
    print(f"✓ DELIVERY PACKAGE CREATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nPackage location: {package_dir}")
    print(f"Package size: {sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB")
    print(f"\nTo use this package:")
    print(f"  1. Review {package_dir / 'README.md'}")
    print(f"  2. Read {package_dir / 'instructor_guide' / 'INSTRUCTOR_GUIDE.md'}")
    print(f"  3. Test notebooks and materials before delivery")
    print(f"\n")
    
    return package_dir


def main():
    """Main entry point."""
    
    try:
        package_dir = create_delivery_package()
        return 0
    except Exception as e:
        print(f"\n❌ Error creating delivery package: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
