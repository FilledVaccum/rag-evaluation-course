"""
Unit tests for data models.

Tests basic functionality of Pydantic models.
"""

import pytest
from models.course import Module, LectureMaterial, Slide
from models.notebook import JupyterNotebook, NotebookCell, CellType
from models.assessment import Assessment, Question, QuestionType, Difficulty, AssessmentType
from models.dataset import Dataset, TestSet, CourseRecord, DatasetFormat
from models.evaluation import EvaluationResults, Metric, MetricType
from models.rag import RAGComponent, ComponentType
from models.certification import ExamDomain, CoverageLevel


@pytest.mark.unit
class TestCourseModels:
    """Test course-related models."""
    
    def test_module_creation(self, sample_module_data):
        """Test creating a Module instance."""
        module = Module(**sample_module_data)
        assert module.module_number == 1
        assert module.title == "Evolution of Search to RAG"
        assert module.duration_minutes == 45
    
    def test_module_time_allocation(self, sample_module_data):
        """Test time allocation calculation."""
        module = Module(**sample_module_data)
        allocation = module.calculate_time_allocation()
        
        assert "lecture" in allocation
        assert "hands_on" in allocation
        assert "discussion" in allocation
        
        # Check approximate 40/50/10 split
        assert 35 <= allocation["lecture"] <= 45
        assert 45 <= allocation["hands_on"] <= 55
        assert 5 <= allocation["discussion"] <= 15
    
    def test_module_validation(self, sample_module_data):
        """Test module time allocation validation."""
        module = Module(**sample_module_data)
        assert module.validate_time_allocation(tolerance=5.0)
    
    def test_slide_creation(self):
        """Test creating a Slide instance."""
        slide = Slide(
            slide_number=1,
            title="Introduction",
            content="Welcome to the course",
            speaker_notes="Start with enthusiasm"
        )
        assert slide.slide_number == 1
        assert slide.title == "Introduction"


@pytest.mark.unit
class TestNotebookModels:
    """Test notebook-related models."""
    
    def test_notebook_creation(self, sample_notebook_data):
        """Test creating a JupyterNotebook instance."""
        notebook = JupyterNotebook(**sample_notebook_data)
        assert notebook.notebook_id == "notebook_0"
        assert notebook.module_number == 1
        assert len(notebook.learning_objectives) == 2
    
    def test_notebook_cell_creation(self):
        """Test creating a NotebookCell instance."""
        cell = NotebookCell(
            cell_type=CellType.CODE,
            source="print('Hello, World!')",
            metadata={},
            execution_count=1
        )
        assert cell.cell_type == CellType.CODE
        assert "Hello" in cell.source


@pytest.mark.unit
class TestAssessmentModels:
    """Test assessment-related models."""
    
    def test_question_creation(self, sample_question_data):
        """Test creating a Question instance."""
        question = Question(**sample_question_data)
        assert question.question_id == "q1_1"
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert len(question.options) == 4
    
    def test_question_validation_requires_options(self):
        """Test that multiple choice questions require options."""
        with pytest.raises(ValueError, match="Multiple choice questions must have options"):
            Question(
                question_id="q1",
                question_text="Test question?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                options=None,  # Should fail
                correct_answer="A",
                explanation="Test",
                exam_domain="Test",
                difficulty=Difficulty.BEGINNER
            )


@pytest.mark.unit
class TestDatasetModels:
    """Test dataset-related models."""
    
    def test_test_set_creation(self, sample_test_set_data):
        """Test creating a TestSet instance."""
        test_set = TestSet(**sample_test_set_data)
        assert test_set.test_set_id == "test_001"
        assert len(test_set) == 2
        assert test_set.validate_lengths()
    
    def test_course_record_to_embedding_string(self, sample_course_record_data):
        """Test CourseRecord to embedding string conversion."""
        record = CourseRecord(**sample_course_record_data)
        
        # With labels
        embedding_str = record.to_embedding_string(include_labels=True)
        assert "Class name:" in embedding_str
        assert "CSCI 567" in embedding_str
        assert "Units:" in embedding_str
        
        # Without labels
        embedding_str_no_labels = record.to_embedding_string(include_labels=False)
        assert "Class name:" not in embedding_str_no_labels
        assert "CSCI 567" in embedding_str_no_labels


@pytest.mark.unit
class TestEvaluationModels:
    """Test evaluation-related models."""
    
    def test_evaluation_results_creation(self, sample_evaluation_results_data):
        """Test creating EvaluationResults instance."""
        results = EvaluationResults(**sample_evaluation_results_data)
        assert results.evaluation_id == "eval_001"
        assert "faithfulness" in results.metrics
        assert results.metrics["faithfulness"] == 0.92
    
    def test_metric_creation(self):
        """Test creating a Metric instance."""
        metric = Metric(
            metric_id="faithfulness",
            name="Faithfulness",
            metric_type=MetricType.GENERATION,
            description="Measures if response claims are supported by context",
            score_range=(0.0, 1.0),
            higher_is_better=True
        )
        assert metric.metric_id == "faithfulness"
        assert metric.metric_type == MetricType.GENERATION


@pytest.mark.unit
class TestRAGModels:
    """Test RAG-related models."""
    
    def test_rag_component_creation(self):
        """Test creating a RAGComponent instance."""
        component = RAGComponent(
            component_id="retriever_1",
            component_type=ComponentType.RETRIEVAL,
            implementation="VectorStoreRetriever",
            configuration={"top_k": 5}
        )
        assert component.component_id == "retriever_1"
        assert component.component_type == ComponentType.RETRIEVAL


@pytest.mark.unit
class TestCertificationModels:
    """Test certification-related models."""
    
    def test_exam_domain_creation(self):
        """Test creating an ExamDomain instance."""
        domain = ExamDomain(
            domain_id="eval_tuning",
            name="Evaluation and Tuning",
            weight_percentage=13.0,
            topics_covered=["RAG evaluation", "Metrics"],
            coverage_level=CoverageLevel.PRIMARY
        )
        assert domain.domain_id == "eval_tuning"
        assert domain.weight_percentage == 13.0
        assert domain.coverage_level == CoverageLevel.PRIMARY



@pytest.mark.unit
class TestMockExamFormat:
    """Test mock certification exam format and requirements."""
    
    def test_mock_exam_question_count(self):
        """Test that mock exam has 60-70 questions."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        
        mock_exam = create_mock_certification_exam()
        question_count = len(mock_exam.questions)
        
        assert 60 <= question_count <= 70, (
            f"Mock exam has {question_count} questions, expected 60-70"
        )
    
    def test_mock_exam_time_limit(self):
        """Test that mock exam has 120-minute time limit."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        
        mock_exam = create_mock_certification_exam()
        
        assert mock_exam.time_limit_minutes == 120, (
            f"Mock exam time limit is {mock_exam.time_limit_minutes} minutes, expected 120"
        )
    
    def test_mock_exam_question_type_distribution(self):
        """Test that mock exam has appropriate question type distribution."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        
        mock_exam = create_mock_certification_exam()
        
        # Count question types
        type_counts = {}
        for question in mock_exam.questions:
            qtype = question.question_type
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        # Should have multiple choice questions
        assert QuestionType.MULTIPLE_CHOICE in type_counts, (
            "Mock exam should have multiple choice questions"
        )
        
        # Multiple choice should be majority
        mc_count = type_counts.get(QuestionType.MULTIPLE_CHOICE, 0)
        assert mc_count >= len(mock_exam.questions) * 0.5, (
            f"Multiple choice questions ({mc_count}) should be at least 50% of exam"
        )
    
    def test_mock_exam_domain_coverage(self):
        """Test that mock exam covers all major exam domains."""
        from course_materials.assessments.mock_certification_exam import (
            create_mock_certification_exam,
            get_domain_distribution
        )
        
        mock_exam = create_mock_certification_exam()
        distribution = get_domain_distribution(mock_exam.questions)
        
        # Key domains that should be covered
        key_domains = [
            "Evaluation and Tuning",
            "Agent Development",
            "Agent Architecture"
        ]
        
        for domain in key_domains:
            assert domain in distribution, (
                f"Mock exam missing key domain: {domain}"
            )
            assert distribution[domain] > 0, (
                f"Domain '{domain}' has no questions"
            )
    
    def test_mock_exam_all_questions_have_explanations(self):
        """Test that all mock exam questions have explanations."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        
        mock_exam = create_mock_certification_exam()
        
        for question in mock_exam.questions:
            assert question.explanation is not None, (
                f"Question {question.question_id} missing explanation"
            )
            assert len(question.explanation) > 0, (
                f"Question {question.question_id} has empty explanation"
            )
    
    def test_mock_exam_passing_score(self):
        """Test that mock exam has appropriate passing score (70%)."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        
        mock_exam = create_mock_certification_exam()
        
        expected_passing = int(len(mock_exam.questions) * 0.70)
        
        assert mock_exam.rubric.passing_score == expected_passing, (
            f"Passing score {mock_exam.rubric.passing_score} != expected {expected_passing}"
        )
    
    def test_mock_exam_assessment_type(self):
        """Test that mock exam has correct assessment type."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        from src.models.assessment import AssessmentType
        
        mock_exam = create_mock_certification_exam()
        
        assert mock_exam.assessment_type == AssessmentType.MOCK_EXAM, (
            f"Mock exam type is {mock_exam.assessment_type}, expected MOCK_EXAM"
        )
    
    def test_mock_exam_validation(self):
        """Test mock exam validation function."""
        from course_materials.assessments.mock_certification_exam import (
            create_mock_certification_exam,
            validate_exam_structure
        )
        
        mock_exam = create_mock_certification_exam()
        validation = validate_exam_structure(mock_exam)
        
        # Should pass validation
        assert validation["valid"], (
            f"Mock exam failed validation: {validation['issues']}"
        )
    
    def test_mock_exam_domain_distribution_matches_weights(self):
        """Test that domain distribution roughly matches exam weights."""
        from course_materials.assessments.mock_certification_exam import (
            create_mock_certification_exam,
            get_domain_distribution,
            EXAM_DOMAINS
        )
        
        mock_exam = create_mock_certification_exam()
        distribution = get_domain_distribution(mock_exam.questions)
        total_questions = len(mock_exam.questions)
        
        # Check that distribution is within reasonable tolerance of weights
        for domain, expected_weight in EXAM_DOMAINS.items():
            if domain in distribution:
                actual_count = distribution[domain]
                actual_percentage = (actual_count / total_questions) * 100
                
                # Allow ±3% tolerance
                tolerance = 3.0
                assert abs(actual_percentage - expected_weight) <= tolerance + 1, (
                    f"Domain '{domain}' has {actual_percentage:.1f}% of questions, "
                    f"expected ~{expected_weight}% (±{tolerance}%)"
                )
    
    def test_mock_exam_questions_have_valid_domains(self):
        """Test that all questions reference valid exam domains."""
        from course_materials.assessments.mock_certification_exam import (
            create_mock_certification_exam,
            EXAM_DOMAINS
        )
        
        mock_exam = create_mock_certification_exam()
        valid_domains = set(EXAM_DOMAINS.keys())
        
        for question in mock_exam.questions:
            assert question.exam_domain in valid_domains, (
                f"Question {question.question_id} has invalid domain: {question.exam_domain}"
            )
    
    def test_mock_exam_questions_have_difficulty_levels(self):
        """Test that mock exam has questions at different difficulty levels."""
        from course_materials.assessments.mock_certification_exam import create_mock_certification_exam
        from src.models.assessment import Difficulty
        
        mock_exam = create_mock_certification_exam()
        
        # Count difficulty levels
        difficulty_counts = {}
        for question in mock_exam.questions:
            diff = question.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        # Should have at least intermediate questions
        assert Difficulty.INTERMEDIATE in difficulty_counts, (
            "Mock exam should have intermediate difficulty questions"
        )
        
        # Should have some variety (at least 2 difficulty levels)
        assert len(difficulty_counts) >= 2, (
            f"Mock exam should have multiple difficulty levels, found {len(difficulty_counts)}"
        )
