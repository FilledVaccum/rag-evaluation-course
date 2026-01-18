"""
Validate time allocations across all course modules.

This script validates that:
1. Total course duration is 6-8 hours
2. Each module follows 40/50/10 split (±5% tolerance)
3. Time allocations are consistent and properly documented
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.course import Module
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


def validate_total_duration(course: CourseStructure) -> Tuple[bool, str]:
    """Validate total course duration is 6-8 hours."""
    total_minutes = course.calculate_total_duration()
    total_hours = total_minutes / 60
    
    if 6.0 <= total_hours <= 8.0:
        return True, f"✓ Total duration: {total_hours:.1f} hours ({total_minutes} minutes) - WITHIN TARGET"
    else:
        return False, f"❌ Total duration: {total_hours:.1f} hours ({total_minutes} minutes) - OUTSIDE 6-8 hour target"


def validate_module_time_split(module: Module, tolerance: float = 5.0) -> Tuple[bool, Dict[str, float], str]:
    """Validate module follows 40/50/10 time split."""
    allocation = module.calculate_time_allocation()
    
    lecture_ok = abs(allocation["lecture"] - 40.0) <= tolerance
    hands_on_ok = abs(allocation["hands_on"] - 50.0) <= tolerance
    discussion_ok = abs(allocation["discussion"] - 10.0) <= tolerance
    
    all_ok = lecture_ok and hands_on_ok and discussion_ok
    
    status = "✓" if all_ok else "❌"
    message = (
        f"{status} Module {module.module_number}: "
        f"Lecture {allocation['lecture']:.1f}% "
        f"(target 40% ±{tolerance}%), "
        f"Hands-on {allocation['hands_on']:.1f}% "
        f"(target 50% ±{tolerance}%), "
        f"Discussion {allocation['discussion']:.1f}% "
        f"(target 10% ±{tolerance}%)"
    )
    
    return all_ok, allocation, message


def generate_time_allocation_report(course: CourseStructure) -> str:
    """Generate detailed time allocation report."""
    report = []
    report.append("\n" + "="*100)
    report.append("TIME ALLOCATION VALIDATION REPORT")
    report.append("="*100 + "\n")
    
    # Overall duration
    total_minutes = course.calculate_total_duration()
    total_hours = total_minutes / 60
    report.append(f"Course: {course.course_title}")
    report.append(f"Version: {course.course_version}")
    report.append(f"Total Duration: {total_hours:.1f} hours ({total_minutes} minutes)")
    report.append(f"Target Range: 6-8 hours")
    report.append("")
    
    # Module-by-module breakdown
    report.append("MODULE TIME ALLOCATIONS")
    report.append("-"*100)
    report.append(f"{'Module':<50} {'Duration':<12} {'Lecture':<12} {'Hands-On':<12} {'Discussion':<12}")
    report.append("-"*100)
    
    total_lecture = 0
    total_hands_on = 0
    total_discussion = 0
    
    for module in course.modules:
        allocation = module.calculate_time_allocation()
        total_lecture += module.lecture_time_minutes
        total_hands_on += module.hands_on_time_minutes
        total_discussion += module.discussion_time_minutes
        
        report.append(
            f"{module.module_number}. {module.title:<46} "
            f"{module.duration_minutes:>3}min "
            f"({module.duration_minutes/60:.1f}h)  "
            f"{allocation['lecture']:>5.1f}% "
            f"({module.lecture_time_minutes:>2}min)  "
            f"{allocation['hands_on']:>5.1f}% "
            f"({module.hands_on_time_minutes:>2}min)  "
            f"{allocation['discussion']:>5.1f}% "
            f"({module.discussion_time_minutes:>2}min)"
        )
    
    report.append("-"*100)
    
    # Overall percentages
    overall_lecture_pct = (total_lecture / total_minutes) * 100
    overall_hands_on_pct = (total_hands_on / total_minutes) * 100
    overall_discussion_pct = (total_discussion / total_minutes) * 100
    
    report.append(
        f"{'OVERALL':<50} "
        f"{total_minutes:>3}min "
        f"({total_hours:.1f}h)  "
        f"{overall_lecture_pct:>5.1f}% "
        f"({total_lecture:>3}min)  "
        f"{overall_hands_on_pct:>5.1f}% "
        f"({total_hands_on:>3}min)  "
        f"{overall_discussion_pct:>5.1f}% "
        f"({total_discussion:>2}min)"
    )
    
    report.append("")
    report.append("TARGET ALLOCATION: 40% Lecture / 50% Hands-On / 10% Discussion (±5% tolerance)")
    report.append("")
    
    return "\n".join(report)


def validate_all_time_allocations(course: CourseStructure, tolerance: float = 5.0) -> bool:
    """Validate all time allocations and generate report."""
    
    print(generate_time_allocation_report(course))
    
    # Validate total duration
    duration_ok, duration_msg = validate_total_duration(course)
    print(duration_msg)
    print("")
    
    # Validate each module
    print("MODULE-BY-MODULE VALIDATION")
    print("-"*100)
    
    all_modules_ok = True
    for module in course.modules:
        module_ok, allocation, message = validate_module_time_split(module, tolerance)
        print(message)
        if not module_ok:
            all_modules_ok = False
    
    print("")
    
    # Overall validation
    total_minutes = course.calculate_total_duration()
    overall_lecture_pct = sum(m.lecture_time_minutes for m in course.modules) / total_minutes * 100
    overall_hands_on_pct = sum(m.hands_on_time_minutes for m in course.modules) / total_minutes * 100
    overall_discussion_pct = sum(m.discussion_time_minutes for m in course.modules) / total_minutes * 100
    
    overall_lecture_ok = abs(overall_lecture_pct - 40.0) <= tolerance
    overall_hands_on_ok = abs(overall_hands_on_pct - 50.0) <= tolerance
    overall_discussion_ok = abs(overall_discussion_pct - 10.0) <= tolerance
    overall_ok = overall_lecture_ok and overall_hands_on_ok and overall_discussion_ok
    
    print("OVERALL COURSE ALLOCATION")
    print("-"*100)
    status = "✓" if overall_ok else "❌"
    print(
        f"{status} Overall: "
        f"Lecture {overall_lecture_pct:.1f}% (target 40% ±{tolerance}%), "
        f"Hands-on {overall_hands_on_pct:.1f}% (target 50% ±{tolerance}%), "
        f"Discussion {overall_discussion_pct:.1f}% (target 10% ±{tolerance}%)"
    )
    print("")
    
    # Final summary
    print("="*100)
    if duration_ok and all_modules_ok and overall_ok:
        print("✓ ALL TIME ALLOCATION VALIDATIONS PASSED")
        print("="*100 + "\n")
        return True
    else:
        print("❌ SOME TIME ALLOCATION VALIDATIONS FAILED")
        if not duration_ok:
            print("   - Total duration outside 6-8 hour target")
        if not all_modules_ok:
            print("   - Some modules outside 40/50/10 split tolerance")
        if not overall_ok:
            print("   - Overall course allocation outside 40/50/10 split tolerance")
        print("="*100 + "\n")
        return False


def main():
    """Main entry point."""
    
    print("Loading course structure...")
    course = load_course_structure()
    print(f"✓ Loaded {len(course.modules)} modules\n")
    
    # Validate time allocations
    success = validate_all_time_allocations(course, tolerance=5.0)
    
    if success:
        print("✓ Time allocation validation complete - ALL CHECKS PASSED")
        return 0
    else:
        print("❌ Time allocation validation complete - SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
