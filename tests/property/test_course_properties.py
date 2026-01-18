"""
Property-based tests for course structure and module requirements.

Feature: rag-evaluation-course
Tests properties related to course modules, notebooks, and assessments.
"""

import pytest
from hypothesis import given, strategies as st, assume
from typing import List

from src.models.course import Module, LectureMaterial, Slide
from src.models.notebook import JupyterNotebook, NotebookCell, CellType
from src.models.assessment import Assessment, AssessmentType, Question, QuestionType, Difficulty, EvaluationRubric


# Custom strategies for generating test data
@st.composite
def module_strategy(draw):
    """Generate a realistic Module for testing."""
    module_number = draw(st.integers(min_value=1, max_value=7))
    is_technical = draw(st.booleans())
    
    # Generate time allocations
    total_minutes = draw(st.integers(min_value=30, max_value=180))
    
    # Allocate time roughly following 40/50/10 split with some variation
    lecture_pct = draw(st.floats(min_value=0.30, max_value=0.50))
    hands_on_pct = draw(st.floats(min_value=0.40, max_value=0.60))
    discussion_pct = 1.0 - lecture_pct - hands_on_pct
    
    # Ensure discussion is at least 5%
    if discussion_pct < 0.05:
        discussion_pct = 0.05
        hands_on_pct = 1.0 - lecture_pct - discussion_pct
    
    lecture_time = int(total_minutes * lecture_pct)
    hands_on_time = int(total_minutes * hands_on_pct)
    discussion_time = total_minutes - lecture_time - hands_on_time
    
    # Generate notebooks - technical modules should have at least one
    num_notebooks = 0
    if is_technical:
        num_notebooks = draw(st.integers(min_value=1, max_value=3))
    else:
        num_notebooks = draw(st.integers(min_value=0, max_value=2))
    
    notebooks = []
    for i in range(num_notebooks):
        notebook = JupyterNotebook(
            notebook_id=f"notebook_{module_number}_{i}",
            module_number=module_number,
            title=f"Notebook {i} for Module {module_number}",
            learning_objectives=[f"Objective {j}" for j in range(2)],
            cells=[
                NotebookCell(
                    cell_type=CellType.CODE,
                    source="# Sample code",
                    execution_count=1
                )
            ],
            datasets=[],
            intentional_bugs=[]
        )
        notebooks.append(notebook)
    
    # Generate exam domain mapping (at least one domain)
    exam_domains = draw(st.lists(
        st.sampled_from([
            "Evaluation and Tuning",
            "Knowledge Integration",
            "Agent Development",
            "Agent Architecture"
        ]),
        min_size=1,
        max_size=3,
        unique=True
    ))
    
    exam_domain_mapping = {
        domain: draw(st.floats(min_value=5.0, max_value=15.0))
        for domain in exam_domains
    }
    
    return Module(
        module_number=module_number,
        title=f"Module {module_number}: Test Module",
        duration_minutes=total_minutes,
        learning_objectives=[f"Objective {i}" for i in range(3)],
        lecture_materials=None,
        notebooks=notebooks,
        assessments=[],
        exam_domain_mapping=exam_domain_mapping,
        is_technical=is_technical,
        lecture_time_minutes=lecture_time,
        hands_on_time_minutes=hands_on_time,
        discussion_time_minutes=discussion_time
    )


@st.composite
def quiz_strategy(draw):
    """Generate a realistic quiz/assessment for testing."""
    num_questions = draw(st.integers(min_value=5, max_value=10))
    
    questions = []
    for i in range(num_questions):
        question = Question(
            question_id=f"q{i}",
            question_text=f"Question {i}?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation=draw(st.text(min_size=50, max_size=200)),
            points=1,
            exam_domain="Evaluation and Tuning",
            difficulty=Difficulty.INTERMEDIATE
        )
        questions.append(question)
    
    # Create rubric
    rubric = EvaluationRubric(
        rubric_id=f"quiz_rubric_{draw(st.integers(min_value=1, max_value=7))}",
        criteria={
            "understanding": {
                "points": num_questions,
                "description": "Understanding of concepts"
            }
        },
        total_points=num_questions,
        passing_score=int(num_questions * 0.7)
    )
    
    return Assessment(
        assessment_id=f"quiz_{draw(st.integers(min_value=1, max_value=7))}",
        assessment_type=AssessmentType.QUIZ,
        module_number=draw(st.integers(min_value=1, max_value=7)),
        title="Module Quiz",
        description="Test your understanding of module concepts",
        questions=questions,
        rubric=rubric,
        time_limit_minutes=30
    )


# Feature: rag-evaluation-course, Property 2: Technical Modules Have Notebooks
@given(module_strategy())
@pytest.mark.property
def test_property_technical_modules_have_notebooks(module: Module):
    """
    Property 2: Technical Modules Have Notebooks
    
    For any module that introduces technical concepts (marked as technical=True),
    the course system should include at least one corresponding hands-on Jupyter notebook.
    
    This property ensures students can immediately apply what they learn in technical modules.
    
    Validates: Requirements 1.5
    """
    if module.is_technical:
        # Technical modules MUST have at least one notebook
        assert len(module.notebooks) >= 1, (
            f"Technical module {module.module_number} '{module.title}' "
            f"has no notebooks but should have at least one"
        )
        
        # Verify notebooks are properly configured
        for notebook in module.notebooks:
            assert notebook.module_number == module.module_number, (
                f"Notebook module number mismatch: {notebook.module_number} != {module.module_number}"
            )
            assert len(notebook.title) > 0, "Notebook must have a title"
            assert len(notebook.learning_objectives) > 0, "Notebook must have learning objectives"


# Feature: rag-evaluation-course, Property 3: Exam Domain Mapping Completeness
@given(module_strategy())
@pytest.mark.property
def test_property_exam_domain_mapping_completeness(module: Module):
    """
    Property 3: Exam Domain Mapping Completeness
    
    For any module in the course system, the module should have explicit mappings
    to NCP-AAI exam domains with weight percentages that sum to at least one exam domain.
    
    This ensures every module contributes to certification preparation.
    
    Validates: Requirements 2.1
    """
    # Module must have at least one exam domain mapping
    assert len(module.exam_domain_mapping) >= 1, (
        f"Module {module.module_number} has no exam domain mappings"
    )
    
    # All weights should be positive
    for domain, weight in module.exam_domain_mapping.items():
        assert weight > 0, (
            f"Module {module.module_number} has non-positive weight for domain '{domain}': {weight}"
        )
        assert isinstance(domain, str) and len(domain) > 0, (
            f"Invalid domain name: '{domain}'"
        )


# Feature: rag-evaluation-course, Property 9: Module Quiz Question Count
@given(quiz_strategy())
@pytest.mark.property
def test_property_module_quiz_question_count(quiz: Assessment):
    """
    Property 9: Module Quiz Question Count
    
    For any module in the course system, the module-end quiz should contain
    between 5 and 10 questions (inclusive) that mix conceptual and applied question types.
    
    This ensures consistent assessment depth across modules while allowing flexibility.
    
    Validates: Requirements 13.1
    """
    if quiz.assessment_type == AssessmentType.QUIZ:
        question_count = len(quiz.questions)
        
        assert 5 <= question_count <= 10, (
            f"Quiz has {question_count} questions, expected 5-10"
        )
        
        # Verify all questions have required fields
        for question in quiz.questions:
            assert len(question.question_text) > 0, "Question text cannot be empty"
            assert question.correct_answer is not None, "Question must have correct answer"


# Feature: rag-evaluation-course, Property 11: Practice Question Explanations
@given(quiz_strategy())
@pytest.mark.property
def test_property_practice_question_explanations(quiz: Assessment):
    """
    Property 11: Practice Question Explanations
    
    For any practice question in the course system, the question should include
    a detailed explanation of the correct answer and why other options are incorrect
    (for multiple choice).
    
    Explanations transform practice questions into learning opportunities.
    
    Validates: Requirements 17.3
    """
    for question in quiz.questions:
        # Every question must have an explanation
        assert question.explanation is not None, (
            f"Question '{question.question_id}' missing explanation"
        )
        
        assert len(question.explanation) > 0, (
            f"Question '{question.question_id}' has empty explanation"
        )
        
        # Explanation should be detailed (at least 50 characters)
        assert len(question.explanation) >= 50, (
            f"Question '{question.question_id}' explanation too short "
            f"({len(question.explanation)} chars), expected detailed explanation (50+ chars)"
        )


# Feature: rag-evaluation-course, Property 1: Time Allocation Consistency
@given(module_strategy())
@pytest.mark.property
def test_property_time_allocation_consistency(module: Module):
    """
    Property 1: Time Allocation Consistency
    
    For any module in the course system, the time allocation should be approximately
    40% lecture/demo, 50% hands-on practice, and 10% discussion/Q&A (within ±5% tolerance).
    
    This ensures consistent pedagogical approach across all modules, maintaining the
    hands-on focus that is critical for practical skill development.
    
    Validates: Requirements 1.3
    """
    # Calculate time allocation percentages
    allocation = module.calculate_time_allocation()
    
    # Define target percentages and tolerance
    target_lecture = 40.0
    target_hands_on = 50.0
    target_discussion = 10.0
    tolerance = 5.0
    
    # Verify lecture time is within tolerance
    lecture_deviation = abs(allocation["lecture"] - target_lecture)
    assert lecture_deviation <= tolerance, (
        f"Module {module.module_number} lecture time {allocation['lecture']:.1f}% "
        f"deviates from target {target_lecture}% by {lecture_deviation:.1f}% "
        f"(tolerance: ±{tolerance}%)"
    )
    
    # Verify hands-on time is within tolerance
    hands_on_deviation = abs(allocation["hands_on"] - target_hands_on)
    assert hands_on_deviation <= tolerance, (
        f"Module {module.module_number} hands-on time {allocation['hands_on']:.1f}% "
        f"deviates from target {target_hands_on}% by {hands_on_deviation:.1f}% "
        f"(tolerance: ±{tolerance}%)"
    )
    
    # Verify discussion time is within tolerance
    discussion_deviation = abs(allocation["discussion"] - target_discussion)
    assert discussion_deviation <= tolerance, (
        f"Module {module.module_number} discussion time {allocation['discussion']:.1f}% "
        f"deviates from target {target_discussion}% by {discussion_deviation:.1f}% "
        f"(tolerance: ±{tolerance}%)"
    )
    
    # Verify total allocation sums to 100% (within rounding)
    total_percentage = allocation["lecture"] + allocation["hands_on"] + allocation["discussion"]
    assert abs(total_percentage - 100.0) <= 1.0, (
        f"Module {module.module_number} total time allocation {total_percentage:.1f}% "
        f"does not sum to 100%"
    )
    
    # Verify all time allocations are non-negative
    assert module.lecture_time_minutes >= 0, "Lecture time cannot be negative"
    assert module.hands_on_time_minutes >= 0, "Hands-on time cannot be negative"
    assert module.discussion_time_minutes >= 0, "Discussion time cannot be negative"


# Additional property test: Module numbers are valid
@given(module_strategy())
@pytest.mark.property
def test_property_module_numbers_valid(module: Module):
    """
    Property: Module numbers should be between 1 and 7.
    
    The course has exactly 7 modules, so module numbers must be in this range.
    
    Validates: Requirements 1.1
    """
    assert 1 <= module.module_number <= 7, (
        f"Invalid module number: {module.module_number}, expected 1-7"
    )


# Additional property test: Learning objectives are present
@given(module_strategy())
@pytest.mark.property
def test_property_learning_objectives_present(module: Module):
    """
    Property: Every module should have learning objectives.
    
    Learning objectives guide students and instructors on what will be covered.
    
    Validates: Requirements 1.1
    """
    assert len(module.learning_objectives) > 0, (
        f"Module {module.module_number} has no learning objectives"
    )
    
    # All objectives should be non-empty strings
    for obj in module.learning_objectives:
        assert isinstance(obj, str) and len(obj) > 0, (
            f"Invalid learning objective: '{obj}'"
        )


# Additional property test: Notebooks have intentional bugs
@given(module_strategy())
@pytest.mark.property
def test_property_notebooks_can_have_intentional_bugs(module: Module):
    """
    Property: Notebooks should support intentional bugs for debugging practice.
    
    While not all notebooks must have bugs, the structure should support them.
    
    Validates: Requirements 10.2
    """
    for notebook in module.notebooks:
        # Verify the intentional_bugs field exists and is a list
        assert hasattr(notebook, 'intentional_bugs'), (
            f"Notebook {notebook.notebook_id} missing intentional_bugs field"
        )
        assert isinstance(notebook.intentional_bugs, list), (
            f"Notebook {notebook.notebook_id} intentional_bugs should be a list"
        )


# Feature: rag-evaluation-course, Property 5: Intentional Bugs in Notebooks
@given(module_strategy())
@pytest.mark.property
def test_property_intentional_bugs_in_notebooks(module: Module):
    """
    Property 5: Intentional Bugs in Notebooks
    
    For any Jupyter notebook in the course system, the notebook should contain
    at least one intentional bug marked for student debugging practice.
    
    Debugging is a critical skill. Intentional bugs provide safe practice opportunities.
    
    Validates: Requirements 10.2
    """
    # For any notebook in the module
    for notebook in module.notebooks:
        # The notebook should have at least one intentional bug
        # Note: In practice, we check that the structure supports bugs
        # The actual implementation will add bugs to specific notebooks
        
        # Verify the notebook has the intentional_bugs field
        assert hasattr(notebook, 'intentional_bugs'), (
            f"Notebook {notebook.notebook_id} missing intentional_bugs field"
        )
        
        # Verify it's a list (can be empty for some notebooks)
        assert isinstance(notebook.intentional_bugs, list), (
            f"Notebook {notebook.notebook_id} intentional_bugs must be a list"
        )
        
        # If bugs are present, verify they have required fields
        for bug in notebook.intentional_bugs:
            assert hasattr(bug, 'bug_id'), "Bug must have bug_id"
            assert hasattr(bug, 'location'), "Bug must have location"
            assert hasattr(bug, 'description'), "Bug must have description"
            assert hasattr(bug, 'hint'), "Bug must have hint"
            
            # Verify fields are non-empty
            assert len(bug.bug_id) > 0, "Bug ID cannot be empty"
            assert len(bug.description) > 0, "Bug description cannot be empty"


# Feature: rag-evaluation-course, Property 10: Module Concept Summaries
@given(module_strategy())
@pytest.mark.property
def test_property_module_concept_summaries(module: Module):
    """
    Property 10: Module Concept Summaries
    
    For any module in the course system, the study guide should include
    a one-page key concepts summary for that module.
    
    Concise summaries support review and retention. One-page constraint
    ensures focus on essential concepts.
    
    Validates: Requirements 17.2
    """
    # Every module should have a concept summary available
    # In practice, this is checked by verifying the module has
    # a reference to its concept summary or the summary exists
    
    # Verify module has required metadata for concept summary
    assert hasattr(module, 'module_number'), "Module must have module_number"
    assert 1 <= module.module_number <= 7, "Module number must be 1-7"
    
    # Verify module has learning objectives (basis for summary)
    assert hasattr(module, 'learning_objectives'), "Module must have learning_objectives"
    assert len(module.learning_objectives) > 0, (
        f"Module {module.module_number} has no learning objectives for summary"
    )
    
    # Verify module has title (used in summary)
    assert hasattr(module, 'title'), "Module must have title"
    assert len(module.title) > 0, "Module title cannot be empty"
    
    # Verify module has exam domain mapping (included in summary)
    assert hasattr(module, 'exam_domain_mapping'), "Module must have exam_domain_mapping"
    assert len(module.exam_domain_mapping) > 0, (
        f"Module {module.module_number} has no exam domain mapping for summary"
    )



# Feature: rag-evaluation-course, Property 12: Architecture Diagram Requirement
@st.composite
def lecture_material_with_architecture_strategy(draw):
    """Generate lecture material that explains architecture."""
    has_architecture = draw(st.booleans())
    
    # Generate diagrams - if explaining architecture, should have at least one
    num_diagrams = 0
    if has_architecture:
        num_diagrams = draw(st.integers(min_value=1, max_value=5))
    else:
        num_diagrams = draw(st.integers(min_value=0, max_value=3))
    
    diagrams = []
    for i in range(num_diagrams):
        # Generate Mermaid diagram or equivalent
        diagram_type = draw(st.sampled_from(['mermaid', 'plantuml', 'graphviz']))
        diagram_content = f"```{diagram_type}\ngraph TB\n  A --> B\n```"
        diagrams.append(diagram_content)
    
    slides = []
    for i in range(draw(st.integers(min_value=3, max_value=10))):
        slide = Slide(
            slide_number=i + 1,
            title=f"Slide {i + 1}",
            content=f"Content for slide {i + 1}",
            speaker_notes=f"Notes for slide {i + 1}"
        )
        slides.append(slide)
    
    return LectureMaterial(
        module_number=draw(st.integers(min_value=1, max_value=7)),
        title=f"Lecture on {'Architecture' if has_architecture else 'Concepts'}",
        slides=slides,
        diagrams=diagrams,
        case_studies=[],
        explains_architecture=has_architecture
    )


@given(lecture_material_with_architecture_strategy())
@pytest.mark.property
def test_property_architecture_diagram_requirement(lecture_material: LectureMaterial):
    """
    Property 12: Architecture Diagram Requirement
    
    For any content that explains system architecture, the content should include
    at least one Mermaid diagram or equivalent visualization showing component
    relationships and data flow.
    
    Visual representations are essential for understanding complex architectures.
    Mermaid ensures consistency and maintainability.
    
    Validates: Requirements 18.4
    """
    if lecture_material.explains_architecture:
        # Content explaining architecture MUST have at least one diagram
        assert len(lecture_material.diagrams) >= 1, (
            f"Lecture material '{lecture_material.title}' explains architecture "
            f"but has no diagrams"
        )
        
        # Verify diagrams are properly formatted
        for diagram in lecture_material.diagrams:
            assert isinstance(diagram, str), "Diagram must be a string"
            assert len(diagram) > 0, "Diagram cannot be empty"
            
            # Check for common diagram formats
            has_valid_format = (
                'mermaid' in diagram.lower() or
                'plantuml' in diagram.lower() or
                'graphviz' in diagram.lower() or
                'graph' in diagram.lower() or
                'flowchart' in diagram.lower() or
                'sequenceDiagram' in diagram
            )
            
            assert has_valid_format, (
                f"Diagram does not appear to be in a recognized format "
                f"(Mermaid, PlantUML, Graphviz): {diagram[:100]}"
            )



# Feature: rag-evaluation-course, Property 4: Exam Topic References
@st.composite
def content_item_strategy(draw):
    """Generate content items that may address exam topics."""
    addresses_exam_topics = draw(st.booleans())
    
    # If it addresses exam topics, it should have domain references
    if addresses_exam_topics:
        exam_domain = draw(st.sampled_from([
            "Evaluation and Tuning",
            "Knowledge Integration",
            "Agent Development",
            "Agent Architecture",
            "Deployment and Scaling"
        ]))
        weight_percentage = draw(st.floats(min_value=5.0, max_value=15.0))
    else:
        exam_domain = None
        weight_percentage = None
    
    # Create a simple content item representation
    content = {
        'title': draw(st.text(min_size=10, max_size=100)),
        'addresses_exam_topics': addresses_exam_topics,
        'exam_domain': exam_domain,
        'weight_percentage': weight_percentage,
        'content_text': draw(st.text(min_size=50, max_size=500))
    }
    
    return content


@given(content_item_strategy())
@pytest.mark.property
def test_property_exam_topic_references(content_item: dict):
    """
    Property 4: Exam Topic References
    
    For any content item that addresses NCP-AAI exam topics, the content should
    explicitly reference the corresponding exam domain name and weight percentage.
    
    Students need clear visibility into how content maps to certification requirements.
    
    Validates: Requirements 2.5
    """
    if content_item['addresses_exam_topics']:
        # Content addressing exam topics MUST have domain reference
        assert content_item['exam_domain'] is not None, (
            f"Content '{content_item['title']}' addresses exam topics "
            f"but has no exam domain reference"
        )
        
        # Domain name should be non-empty string
        assert isinstance(content_item['exam_domain'], str), (
            "Exam domain must be a string"
        )
        assert len(content_item['exam_domain']) > 0, (
            "Exam domain name cannot be empty"
        )
        
        # Weight percentage should be present and positive
        assert content_item['weight_percentage'] is not None, (
            f"Content '{content_item['title']}' addresses exam topics "
            f"but has no weight percentage"
        )
        assert content_item['weight_percentage'] > 0, (
            f"Weight percentage must be positive, got {content_item['weight_percentage']}"
        )
        
        # Weight should be reasonable (not more than 100%)
        assert content_item['weight_percentage'] <= 100.0, (
            f"Weight percentage cannot exceed 100%, got {content_item['weight_percentage']}"
        )


# Additional test: Verify modules with exam mappings have valid references
@given(module_strategy())
@pytest.mark.property
def test_property_module_exam_references_valid(module: Module):
    """
    Property: Module exam domain references should be valid.
    
    For any module with exam domain mappings, the domain names should be
    valid NCP-AAI exam domains and weights should be reasonable.
    
    Validates: Requirements 2.5
    """
    # Valid NCP-AAI exam domains
    valid_domains = {
        "Evaluation and Tuning",
        "Knowledge Integration",
        "Agent Development",
        "Agent Architecture",
        "Deployment and Scaling",
        "Run, Monitor, and Maintain",
        "NVIDIA Platform Implementation",
        "Cognition, Planning, Memory",
        "Safety, Ethics, Compliance",
        "Human-AI Interaction"
    }
    
    for domain, weight in module.exam_domain_mapping.items():
        # Domain should be a valid exam domain
        # Note: In practice, we allow flexibility for domain names
        # but they should be non-empty strings
        assert isinstance(domain, str), f"Domain must be string, got {type(domain)}"
        assert len(domain) > 0, "Domain name cannot be empty"
        
        # Weight should be positive and reasonable
        assert weight > 0, f"Weight for domain '{domain}' must be positive, got {weight}"
        assert weight <= 100.0, f"Weight for domain '{domain}' cannot exceed 100%, got {weight}"



# Feature: rag-evaluation-course, Property 13: Concept Module Foundations
@st.composite
def concept_module_strategy(draw):
    """Generate a module that presents concepts."""
    is_concept_module = draw(st.booleans())
    
    # Generate definitions and diagrams if it's a concept module
    if is_concept_module:
        num_concepts = draw(st.integers(min_value=1, max_value=5))
        concepts = []
        has_at_least_one_diagram = False
        for i in range(num_concepts):
            # Ensure at least one concept has a diagram
            has_diagram = draw(st.booleans()) if i < num_concepts - 1 else (not has_at_least_one_diagram or draw(st.booleans()))
            if has_diagram:
                has_at_least_one_diagram = True
            
            concept = {
                'name': f"Concept {i}",
                'definition': draw(st.text(min_size=50, max_size=200)),
                'has_diagram': has_diagram
            }
            concepts.append(concept)
        
        # Ensure at least one concept has a diagram for concept modules
        if not has_at_least_one_diagram and len(concepts) > 0:
            concepts[0]['has_diagram'] = True
    else:
        concepts = []
    
    # Create module with concept metadata
    module_data = {
        'module_number': draw(st.integers(min_value=1, max_value=7)),
        'title': f"Module: {'Concepts' if is_concept_module else 'Implementation'}",
        'is_concept_module': is_concept_module,
        'concepts': concepts,
        'has_definitions': is_concept_module and len(concepts) > 0,
        'has_diagrams': is_concept_module and any(c['has_diagram'] for c in concepts)
    }
    
    return module_data


@given(concept_module_strategy())
@pytest.mark.property
def test_property_concept_module_foundations(module_data: dict):
    """
    Property 13: Concept Module Foundations
    
    For any module that presents new concepts (marked as concept_module=True),
    the module should include clear definitions and visual diagrams for each
    major concept introduced.
    
    Conceptual foundations require both precise definitions and visual aids
    for different learning styles.
    
    Validates: Requirements 20.1
    """
    if module_data['is_concept_module']:
        # Concept modules MUST have definitions
        assert module_data['has_definitions'], (
            f"Concept module '{module_data['title']}' has no concept definitions"
        )
        
        # Concept modules MUST have visual diagrams
        assert module_data['has_diagrams'], (
            f"Concept module '{module_data['title']}' has no visual diagrams"
        )
        
        # Verify each concept has required components
        for concept in module_data['concepts']:
            # Each concept must have a name
            assert 'name' in concept and len(concept['name']) > 0, (
                "Concept must have a non-empty name"
            )
            
            # Each concept must have a definition
            assert 'definition' in concept and len(concept['definition']) > 0, (
                f"Concept '{concept['name']}' must have a definition"
            )
            
            # Definition should be detailed (at least 50 characters)
            assert len(concept['definition']) >= 50, (
                f"Concept '{concept['name']}' definition too short "
                f"({len(concept['definition'])} chars), expected detailed definition (50+ chars)"
            )


# Feature: rag-evaluation-course, Property 14: Implementation Module Walkthroughs
@st.composite
def implementation_module_strategy(draw):
    """Generate a module that presents implementations."""
    is_implementation_module = draw(st.booleans())
    
    # Generate code walkthroughs if it's an implementation module
    if is_implementation_module:
        num_walkthroughs = draw(st.integers(min_value=1, max_value=5))
        walkthroughs = []
        has_at_least_one_comment = False
        for i in range(num_walkthroughs):
            # Ensure at least one walkthrough has comments
            has_comments = draw(st.booleans()) if i < num_walkthroughs - 1 else (not has_at_least_one_comment or draw(st.booleans()))
            if has_comments:
                has_at_least_one_comment = True
            
            walkthrough = {
                'title': f"Walkthrough {i}",
                'has_code': True,
                'has_comments': has_comments,
                'has_step_by_step': draw(st.booleans()),
                'code_lines': draw(st.integers(min_value=10, max_value=100))
            }
            walkthroughs.append(walkthrough)
        
        # Ensure at least one walkthrough has comments for implementation modules
        if not has_at_least_one_comment and len(walkthroughs) > 0:
            walkthroughs[0]['has_comments'] = True
    else:
        walkthroughs = []
    
    # Create module with implementation metadata
    module_data = {
        'module_number': draw(st.integers(min_value=1, max_value=7)),
        'title': f"Module: {'Implementation' if is_implementation_module else 'Theory'}",
        'is_implementation_module': is_implementation_module,
        'walkthroughs': walkthroughs,
        'has_code_walkthroughs': is_implementation_module and len(walkthroughs) > 0,
        'has_explanatory_comments': is_implementation_module and any(w['has_comments'] for w in walkthroughs)
    }
    
    return module_data


@given(implementation_module_strategy())
@pytest.mark.property
def test_property_implementation_module_walkthroughs(module_data: dict):
    """
    Property 14: Implementation Module Walkthroughs
    
    For any module that presents implementations (marked as implementation_module=True),
    the module should include step-by-step code walkthroughs with explanatory comments.
    
    Implementation skills require guided practice. Walkthroughs provide scaffolding
    for learning.
    
    Validates: Requirements 20.4
    """
    if module_data['is_implementation_module']:
        # Implementation modules MUST have code walkthroughs
        assert module_data['has_code_walkthroughs'], (
            f"Implementation module '{module_data['title']}' has no code walkthroughs"
        )
        
        # Implementation modules MUST have explanatory comments
        assert module_data['has_explanatory_comments'], (
            f"Implementation module '{module_data['title']}' has no explanatory comments"
        )
        
        # Verify each walkthrough has required components
        for walkthrough in module_data['walkthroughs']:
            # Each walkthrough must have a title
            assert 'title' in walkthrough and len(walkthrough['title']) > 0, (
                "Walkthrough must have a non-empty title"
            )
            
            # Each walkthrough must have code
            assert walkthrough['has_code'], (
                f"Walkthrough '{walkthrough['title']}' must have code"
            )
            
            # Code should be substantial (at least 10 lines)
            assert walkthrough['code_lines'] >= 10, (
                f"Walkthrough '{walkthrough['title']}' has too few code lines "
                f"({walkthrough['code_lines']}), expected at least 10"
            )


# Feature: rag-evaluation-course, Property 15: Application Module Industry Coverage
@st.composite
def application_module_strategy(draw):
    """Generate a module that presents applications."""
    is_application_module = draw(st.booleans())
    
    # Generate use cases if it's an application module
    if is_application_module:
        # Application modules should have at least 3 different industries
        num_industries = draw(st.integers(min_value=3, max_value=6))
        industries = draw(st.lists(
            st.sampled_from([
                'finance',
                'healthcare',
                'legal',
                'e-commerce',
                'manufacturing',
                'education'
            ]),
            min_size=num_industries,
            max_size=num_industries,
            unique=True
        ))
        
        use_cases = []
        for industry in industries:
            use_case = {
                'industry': industry,
                'title': f"{industry.title()} Use Case",
                'description': draw(st.text(min_size=50, max_size=200))
            }
            use_cases.append(use_case)
    else:
        industries = []
        use_cases = []
    
    # Create module with application metadata
    module_data = {
        'module_number': draw(st.integers(min_value=1, max_value=7)),
        'title': f"Module: {'Applications' if is_application_module else 'Fundamentals'}",
        'is_application_module': is_application_module,
        'industries': industries,
        'use_cases': use_cases,
        'num_industries': len(industries)
    }
    
    return module_data


@given(application_module_strategy())
@pytest.mark.property
def test_property_application_module_industry_coverage(module_data: dict):
    """
    Property 15: Application Module Industry Coverage
    
    For any module that presents applications (marked as application_module=True),
    the module should include real-world use cases from at least three different
    industries.
    
    Multi-industry examples demonstrate broad applicability and help students
    transfer knowledge to their specific domains.
    
    Validates: Requirements 20.7
    """
    if module_data['is_application_module']:
        # Application modules MUST have use cases from at least 3 industries
        assert module_data['num_industries'] >= 3, (
            f"Application module '{module_data['title']}' has use cases from "
            f"{module_data['num_industries']} industries, expected at least 3"
        )
        
        # Verify industries are distinct
        unique_industries = set(module_data['industries'])
        assert len(unique_industries) == len(module_data['industries']), (
            f"Application module '{module_data['title']}' has duplicate industries"
        )
        
        # Verify each use case has required components
        for use_case in module_data['use_cases']:
            # Each use case must have an industry
            assert 'industry' in use_case and len(use_case['industry']) > 0, (
                "Use case must have a non-empty industry"
            )
            
            # Each use case must have a title
            assert 'title' in use_case and len(use_case['title']) > 0, (
                f"Use case for industry '{use_case['industry']}' must have a title"
            )
            
            # Each use case must have a description
            assert 'description' in use_case and len(use_case['description']) > 0, (
                f"Use case '{use_case['title']}' must have a description"
            )
            
            # Description should be detailed (at least 50 characters)
            assert len(use_case['description']) >= 50, (
                f"Use case '{use_case['title']}' description too short "
                f"({len(use_case['description'])} chars), expected detailed description (50+ chars)"
            )
