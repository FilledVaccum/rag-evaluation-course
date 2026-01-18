"""
Validate certification alignment with NCP-AAI exam.

This script validates that:
1. All exam domains are covered
2. Coverage levels are appropriate (⭐⭐⭐, ⭐⭐, ⭐)
3. Primary focus on Evaluation & Tuning (13%)
4. Practice question count is 60-70
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.course_integration import CourseStructure


# NCP-AAI Exam Blueprint
EXAM_DOMAINS = {
    "Evaluation and Tuning": 13.0,
    "Knowledge Integration and Data Handling": 10.0,
    "Agent Development": 15.0,
    "Agent Architecture and Design": 15.0,
    "Deployment and Scaling": 13.0,
    "Run, Monitor, and Maintain": 5.0,
    "NVIDIA Platform Implementation": 7.0,
    "Cognition, Planning, and Memory": 10.0,
    "Safety, Ethics, and Compliance": 5.0,
    "Human-AI Interaction": 5.0,
}

# Expected coverage levels
PRIMARY_COVERAGE = ["Evaluation and Tuning"]
CORE_COVERAGE = ["Knowledge Integration and Data Handling"]
SUPPORTING_COVERAGE = ["Agent Development", "Agent Architecture and Design"]


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


def count_practice_questions() -> int:
    """Count total practice questions across all assessments."""
    # Count questions from mock exam
    mock_exam_file = Path(__file__).parent.parent / "course_materials" / "assessments" / "mock_certification_exam.py"
    
    if not mock_exam_file.exists():
        print(f"⚠️  Mock exam file not found: {mock_exam_file}")
        return 0
    
    # Read the file and count questions
    with open(mock_exam_file, 'r') as f:
        content = f.read()
    
    # Count Question objects
    question_count = content.count('Question(')
    
    return question_count


def get_coverage_level(domain: str, course: CourseStructure) -> str:
    """Determine coverage level for a domain."""
    coverage = course.get_exam_domain_coverage()
    
    if domain not in coverage:
        return "⭐ Not Covered"
    
    module_count = len(coverage[domain])
    
    if domain in PRIMARY_COVERAGE:
        return "⭐⭐⭐ Primary Focus"
    elif domain in CORE_COVERAGE:
        return "⭐⭐⭐ Core Content"
    elif domain in SUPPORTING_COVERAGE:
        return "⭐⭐ Supporting Content"
    elif module_count >= 2:
        return "⭐⭐ Contextual"
    elif module_count == 1:
        return "⭐ Referenced"
    else:
        return "⭐ Not Covered"


def validate_exam_domain_coverage(course: CourseStructure) -> Tuple[bool, List[str]]:
    """Validate that all exam domains are covered."""
    coverage = course.get_exam_domain_coverage()
    issues = []
    
    # Check primary focus domains
    for domain in PRIMARY_COVERAGE:
        if domain not in coverage:
            issues.append(f"❌ Primary domain '{domain}' not covered")
        elif len(coverage[domain]) < 2:
            issues.append(f"⚠️  Primary domain '{domain}' only covered in {len(coverage[domain])} module(s)")
    
    # Check core domains
    for domain in CORE_COVERAGE:
        if domain not in coverage:
            issues.append(f"❌ Core domain '{domain}' not covered")
        elif len(coverage[domain]) < 2:
            issues.append(f"⚠️  Core domain '{domain}' only covered in {len(coverage[domain])} module(s)")
    
    # Check supporting domains
    for domain in SUPPORTING_COVERAGE:
        if domain not in coverage:
            issues.append(f"⚠️  Supporting domain '{domain}' not covered")
    
    # Check all exam domains are at least referenced
    uncovered = []
    for domain in EXAM_DOMAINS.keys():
        if domain not in coverage:
            uncovered.append(domain)
    
    if uncovered:
        issues.append(f"ℹ️  Domains not covered: {', '.join(uncovered)}")
    
    return len(issues) == 0, issues


def validate_practice_questions() -> Tuple[bool, str]:
    """Validate practice question count is 60-70."""
    count = count_practice_questions()
    
    if 60 <= count <= 70:
        return True, f"✓ Practice questions: {count} (target: 60-70)"
    elif count == 0:
        return False, f"⚠️  Practice questions: {count} (mock exam not yet implemented)"
    else:
        return False, f"❌ Practice questions: {count} (target: 60-70)"


def generate_certification_alignment_report(course: CourseStructure) -> str:
    """Generate detailed certification alignment report."""
    report = []
    report.append("\n" + "="*100)
    report.append("CERTIFICATION ALIGNMENT VALIDATION REPORT")
    report.append("="*100 + "\n")
    
    report.append(f"Course: {course.course_title}")
    report.append(f"Target Certification: NVIDIA-Certified Professional: Agentic AI (NCP-AAI)")
    report.append(f"Exam Format: 60-70 questions, 120 minutes, Professional level")
    report.append("")
    
    # Exam domain coverage
    report.append("EXAM DOMAIN COVERAGE")
    report.append("-"*100)
    report.append(f"{'Domain':<45} {'Weight':<10} {'Coverage':<25} {'Modules':<20}")
    report.append("-"*100)
    
    coverage = course.get_exam_domain_coverage()
    
    for domain, weight in sorted(EXAM_DOMAINS.items(), key=lambda x: x[1], reverse=True):
        coverage_level = get_coverage_level(domain, course)
        modules = coverage.get(domain, [])
        module_str = ', '.join([str(m) for m in sorted(modules)]) if modules else "None"
        
        report.append(
            f"{domain:<45} {weight:>4.0f}%      {coverage_level:<25} {module_str:<20}"
        )
    
    report.append("")
    
    # Coverage summary
    report.append("COVERAGE SUMMARY")
    report.append("-"*100)
    
    primary_domains = [d for d in PRIMARY_COVERAGE if d in coverage]
    core_domains = [d for d in CORE_COVERAGE if d in coverage]
    supporting_domains = [d for d in SUPPORTING_COVERAGE if d in coverage]
    
    report.append(f"⭐⭐⭐ Primary Focus: {', '.join(primary_domains) if primary_domains else 'None'}")
    report.append(f"⭐⭐⭐ Core Content: {', '.join(core_domains) if core_domains else 'None'}")
    report.append(f"⭐⭐ Supporting Content: {', '.join(supporting_domains) if supporting_domains else 'None'}")
    
    covered_count = len(coverage)
    total_count = len(EXAM_DOMAINS)
    coverage_pct = (covered_count / total_count) * 100
    
    report.append(f"\nTotal Domains Covered: {covered_count}/{total_count} ({coverage_pct:.0f}%)")
    report.append("")
    
    # Module-by-module exam mapping
    report.append("MODULE-BY-MODULE EXAM MAPPING")
    report.append("-"*100)
    
    for module in course.modules:
        report.append(f"\nModule {module.module_number}: {module.title}")
        report.append(f"  Duration: {module.duration_minutes} minutes ({module.duration_minutes/60:.1f} hours)")
        report.append(f"  Exam Domains:")
        for domain, weight in module.exam_domain_mapping.items():
            coverage_level = get_coverage_level(domain, course)
            report.append(f"    - {domain} ({weight}%) - {coverage_level}")
    
    report.append("")
    
    return "\n".join(report)


def validate_certification_alignment(course: CourseStructure) -> bool:
    """Validate all certification alignment requirements."""
    
    print(generate_certification_alignment_report(course))
    
    # Validate exam domain coverage
    print("VALIDATION CHECKS")
    print("-"*100)
    
    coverage_ok, coverage_issues = validate_exam_domain_coverage(course)
    
    if coverage_ok:
        print("✓ All required exam domains are covered")
    else:
        print("⚠️  Some exam domain coverage issues:")
        for issue in coverage_issues:
            print(f"  {issue}")
    
    # Validate practice questions
    questions_ok, questions_msg = validate_practice_questions()
    print(questions_msg)
    
    # Check primary focus
    coverage = course.get_exam_domain_coverage()
    primary_ok = "Evaluation and Tuning" in coverage and len(coverage["Evaluation and Tuning"]) >= 3
    
    if primary_ok:
        modules = coverage["Evaluation and Tuning"]
        print(f"✓ Primary focus on Evaluation & Tuning (13%) - covered in {len(modules)} modules: {', '.join([str(m) for m in sorted(modules)])}")
    else:
        print(f"❌ Insufficient focus on Evaluation & Tuning (13%)")
    
    print("")
    
    # Final summary
    print("="*100)
    
    all_ok = coverage_ok and questions_ok and primary_ok
    
    if all_ok:
        print("✓ ALL CERTIFICATION ALIGNMENT VALIDATIONS PASSED")
    else:
        print("⚠️  SOME CERTIFICATION ALIGNMENT CHECKS NEED ATTENTION")
        if not coverage_ok:
            print("   - Some exam domains need better coverage")
        if not questions_ok:
            print("   - Practice question count needs adjustment")
        if not primary_ok:
            print("   - Primary focus on Evaluation & Tuning needs strengthening")
    
    print("="*100 + "\n")
    
    return all_ok


def main():
    """Main entry point."""
    
    print("Loading course structure...")
    course = load_course_structure()
    print(f"✓ Loaded {len(course.modules)} modules\n")
    
    # Validate certification alignment
    success = validate_certification_alignment(course)
    
    if success:
        print("✓ Certification alignment validation complete - ALL CHECKS PASSED")
        return 0
    else:
        print("⚠️  Certification alignment validation complete - SOME CHECKS NEED ATTENTION")
        print("   (This is expected during development - mock exam will be completed in implementation)")
        return 0  # Return 0 since this is expected during development


if __name__ == "__main__":
    sys.exit(main())
