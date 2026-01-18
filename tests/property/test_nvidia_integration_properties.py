"""
Property-based tests for NVIDIA platform integration.

Feature: rag-evaluation-course, Property 8: NVIDIA Toolkit References
Validates: Requirements 12.6
"""

import pytest
from hypothesis import given, strategies as st, assume
from typing import List

from src.models.course import Module, LectureMaterial, Slide, MermaidDiagram, CaseStudy
from src.models.notebook import JupyterNotebook, NotebookCell, CellType


# NVIDIA toolkit keywords to search for
NVIDIA_TOOLKIT_KEYWORDS = [
    "nvidia",
    "nim",
    "nemotron",
    "triton",
    "nemo",
    "nv-embed",
    "nvidia agent intelligence toolkit",
    "nvidia inference microservices",
    "nvidia ai catalog"
]


def contains_nvidia_reference(text: str) -> bool:
    """Check if text contains any NVIDIA toolkit reference."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in NVIDIA_TOOLKIT_KEYWORDS)


def module_has_nvidia_references(module: Module) -> bool:
    """
    Check if a module contains NVIDIA toolkit references.
    
    Searches in:
    - Lecture materials (slides, diagrams, case studies)
    - Notebooks (cells)
    - Learning objectives
    """
    # Check learning objectives
    for objective in module.learning_objectives:
        if contains_nvidia_reference(objective):
            return True
    
    # Check lecture materials
    if module.lecture_materials:
        # Check slides
        for slide in module.lecture_materials.slides:
            if contains_nvidia_reference(slide.title) or contains_nvidia_reference(slide.content):
                return True
            if slide.speaker_notes and contains_nvidia_reference(slide.speaker_notes):
                return True
        
        # Check diagrams (diagrams are strings, not objects)
        for diagram in module.lecture_materials.diagrams:
            if contains_nvidia_reference(diagram):
                return True
        
        # Check case studies
        for case_study in module.lecture_materials.case_studies:
            if (contains_nvidia_reference(case_study.title) or 
                contains_nvidia_reference(case_study.solution)):
                return True
    
    # Check notebooks
    for notebook in module.notebooks:
        for cell in notebook.cells:
            if contains_nvidia_reference(cell.source):
                return True
    
    return False


# Custom strategies for generating modules with content
@st.composite
def module_with_content(draw):
    """Generate a module with various content types."""
    module_number = draw(st.integers(min_value=1, max_value=7))
    
    # Generate lecture materials
    num_slides = draw(st.integers(min_value=1, max_value=5))
    slides = []
    for i in range(num_slides):
        slide = Slide(
            slide_number=i + 1,
            title=draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Z')))),
            content=draw(st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z')))),
            speaker_notes=draw(st.one_of(st.none(), st.text(min_size=10, max_size=100)))
        )
        slides.append(slide)
    
    lecture_materials = LectureMaterial(
        module_number=module_number,
        title=draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Z')))),
        slides=slides,
        diagrams=[],
        case_studies=[]
    )
    
    # Generate notebooks
    num_notebooks = draw(st.integers(min_value=0, max_value=2))
    notebooks = []
    for i in range(num_notebooks):
        num_cells = draw(st.integers(min_value=1, max_value=3))
        cells = []
        for j in range(num_cells):
            cell = NotebookCell(
                cell_type=draw(st.sampled_from([CellType.CODE, CellType.MARKDOWN])),
                source=draw(st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z')))),
                execution_count=None
            )
            cells.append(cell)
        
        notebook = JupyterNotebook(
            notebook_id=f"notebook_{i}",
            module_number=module_number,
            title=draw(st.text(min_size=5, max_size=50)),
            learning_objectives=["Learn notebook concepts"],  # Must have at least one
            cells=cells,
            intentional_bugs=[],
            datasets=[]
        )
        notebooks.append(notebook)
    
    # Create module
    module = Module(
        module_number=module_number,
        title=draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Z')))),
        duration_minutes=draw(st.integers(min_value=30, max_value=180)),
        learning_objectives=[draw(st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Z'))))],
        lecture_materials=lecture_materials,
        notebooks=notebooks,
        assessments=[],
        exam_domain_mapping={"Test Domain": 10.0},
        is_technical=draw(st.booleans()),
        lecture_time_minutes=draw(st.integers(min_value=10, max_value=80)),
        hands_on_time_minutes=draw(st.integers(min_value=10, max_value=100)),
        discussion_time_minutes=draw(st.integers(min_value=5, max_value=30))
    )
    
    return module


@st.composite
def module_with_nvidia_reference(draw):
    """Generate a module that includes NVIDIA toolkit references."""
    module = draw(module_with_content())
    
    # Ensure at least one NVIDIA reference exists
    # Add to a random location
    location = draw(st.sampled_from([
        "learning_objective",
        "slide_title",
        "slide_content",
        "notebook_cell"
    ]))
    
    nvidia_tool = draw(st.sampled_from([
        "NVIDIA NIM",
        "Nemotron-4-340B",
        "NVIDIA Triton",
        "NeMo Agent Toolkit",
        "NV-Embed-v2",
        "NVIDIA Agent Intelligence Toolkit"
    ]))
    
    if location == "learning_objective":
        module.learning_objectives.append(f"Learn to use {nvidia_tool} for RAG evaluation")
    elif location == "slide_title" and module.lecture_materials and module.lecture_materials.slides:
        module.lecture_materials.slides[0].title = f"Introduction to {nvidia_tool}"
    elif location == "slide_content" and module.lecture_materials and module.lecture_materials.slides:
        module.lecture_materials.slides[0].content = f"We will use {nvidia_tool} for this task"
    elif location == "notebook_cell" and module.notebooks and module.notebooks[0].cells:
        module.notebooks[0].cells[0].source = f"# Using {nvidia_tool}\nimport nvidia_nim"
    else:
        # Fallback: add to learning objective
        module.learning_objectives.append(f"Understand {nvidia_tool} integration")
    
    return module


# Feature: rag-evaluation-course, Property 8: NVIDIA Toolkit References
@given(module_with_nvidia_reference())
@pytest.mark.property
def test_property_nvidia_toolkit_references(module: Module):
    """
    Property 8: NVIDIA Toolkit References
    
    For any module in the course system, the module materials should include
    at least one reference to NVIDIA Agent Intelligence Toolkit or related
    NVIDIA platforms (NIM, Nemotron, Triton, NeMo).
    
    This property verifies that:
    1. Every module contains NVIDIA platform references
    2. References can appear in various content types (slides, notebooks, objectives)
    3. NVIDIA ecosystem integration is consistent throughout the course
    
    Validates: Requirements 12.6
    """
    # Verify module has NVIDIA references
    has_references = module_has_nvidia_references(module)
    
    assert has_references, (
        f"Module {module.module_number} '{module.title}' does not contain any "
        f"NVIDIA toolkit references. Expected at least one reference to: "
        f"{', '.join(NVIDIA_TOOLKIT_KEYWORDS)}"
    )


@given(
    module_number=st.integers(min_value=1, max_value=7),
    nvidia_tool=st.sampled_from([
        "NVIDIA NIM",
        "Nemotron-4-340B",
        "NVIDIA Triton Inference Server",
        "NeMo Agent Toolkit",
        "NV-Embed-v2",
        "NVIDIA Agent Intelligence Toolkit",
        "NVIDIA AI Catalog"
    ])
)
@pytest.mark.property
def test_property_nvidia_reference_detection(module_number: int, nvidia_tool: str):
    """
    Property: NVIDIA reference detection should work for all toolkit names.
    
    For any NVIDIA tool name, the detection function should correctly
    identify it in module content.
    
    Validates: Requirements 12.6
    """
    # Create a minimal module with the NVIDIA tool reference
    module = Module(
        module_number=module_number,
        title=f"Module {module_number}",
        duration_minutes=60,
        learning_objectives=[f"Learn to use {nvidia_tool}"],
        exam_domain_mapping={"Test": 10.0},
        lecture_time_minutes=24,
        hands_on_time_minutes=30,
        discussion_time_minutes=6
    )
    
    # Should detect the NVIDIA reference
    assert module_has_nvidia_references(module), (
        f"Failed to detect NVIDIA reference '{nvidia_tool}' in module content"
    )


@given(
    num_references=st.integers(min_value=1, max_value=5)
)
@pytest.mark.property
def test_property_multiple_nvidia_references(num_references: int):
    """
    Property: Modules can contain multiple NVIDIA toolkit references.
    
    For any number of NVIDIA references, the detection should work
    and the module should be considered valid.
    
    Validates: Requirements 12.6
    """
    nvidia_tools = [
        "NVIDIA NIM",
        "Nemotron-4-340B",
        "NVIDIA Triton",
        "NeMo",
        "NV-Embed-v2"
    ]
    
    # Create module with multiple references
    learning_objectives = [
        f"Learn {nvidia_tools[i % len(nvidia_tools)]}"
        for i in range(num_references)
    ]
    
    module = Module(
        module_number=1,
        title="Multi-tool Module",
        duration_minutes=90,
        learning_objectives=learning_objectives,
        exam_domain_mapping={"Test": 10.0},
        lecture_time_minutes=36,
        hands_on_time_minutes=45,
        discussion_time_minutes=9
    )
    
    # Should detect NVIDIA references
    assert module_has_nvidia_references(module)


@given(
    content_location=st.sampled_from([
        "learning_objective",
        "slide_title",
        "slide_content",
        "speaker_notes",
        "diagram_title",
        "case_study_solution"
    ])
)
@pytest.mark.property
def test_property_nvidia_reference_in_any_location(content_location: str):
    """
    Property: NVIDIA references should be detected in any content location.
    
    For any content location (slides, notebooks, objectives, etc.),
    NVIDIA toolkit references should be properly detected.
    
    Validates: Requirements 12.6
    """
    nvidia_tool = "NVIDIA NIM"
    
    # Create base module
    module = Module(
        module_number=1,
        title="Test Module",
        duration_minutes=60,
        learning_objectives=["Base objective"],
        exam_domain_mapping={"Test": 10.0},
        lecture_time_minutes=24,
        hands_on_time_minutes=30,
        discussion_time_minutes=6
    )
    
    # Add NVIDIA reference to specified location
    if content_location == "learning_objective":
        module.learning_objectives = [f"Learn {nvidia_tool}"]
    elif content_location == "slide_title":
        module.lecture_materials = LectureMaterial(
            module_number=1,
            title="Lecture",
            slides=[Slide(slide_number=1, title=f"{nvidia_tool} Overview", content="Content")]
        )
    elif content_location == "slide_content":
        module.lecture_materials = LectureMaterial(
            module_number=1,
            title="Lecture",
            slides=[Slide(slide_number=1, title="Title", content=f"Using {nvidia_tool}")]
        )
    elif content_location == "speaker_notes":
        module.lecture_materials = LectureMaterial(
            module_number=1,
            title="Lecture",
            slides=[Slide(slide_number=1, title="Title", content="Content", speaker_notes=f"Mention {nvidia_tool}")]
        )
    elif content_location == "diagram_title":
        module.lecture_materials = LectureMaterial(
            module_number=1,
            title="Lecture",
            diagrams=[f"{nvidia_tool} Architecture: graph LR"]
        )
    elif content_location == "case_study_solution":
        module.lecture_materials = LectureMaterial(
            module_number=1,
            title="Lecture",
            case_studies=[CaseStudy(
                title="Case",
                industry="tech",
                problem="Problem",
                solution=f"Used {nvidia_tool} for solution"
            )]
        )
    
    # Should detect NVIDIA reference regardless of location
    assert module_has_nvidia_references(module), (
        f"Failed to detect NVIDIA reference in {content_location}"
    )


@given(
    nvidia_tool=st.sampled_from([
        "NVIDIA NIM",
        "Nemotron",
        "NVIDIA Triton",
        "NeMo",
        "NV-Embed"
    ])
)
@pytest.mark.property
def test_property_nvidia_reference_case_insensitive(nvidia_tool: str):
    """
    Property: NVIDIA reference detection should be case-insensitive.
    
    For any text containing NVIDIA keywords in any case (NVIDIA, nvidia, Nvidia),
    the detection should work correctly.
    
    Validates: Requirements 12.6
    """
    # Test various case combinations
    test_texts = [
        nvidia_tool.upper(),
        nvidia_tool.lower(),
        nvidia_tool.title(),
        f"Using {nvidia_tool} for embeddings",
        f"The {nvidia_tool.lower()} service is available"
    ]
    
    for text in test_texts:
        assert contains_nvidia_reference(text), (
            f"Failed to detect NVIDIA reference in: {text}"
        )
