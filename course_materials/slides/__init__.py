"""
Slide Decks for RAG Evaluation Course
Comprehensive lecture materials with visual diagrams and speaker notes
"""

from .module_1_slides import get_module_1_slides, MODULE_1_SLIDES
from .module_2_slides import get_module_2_slides, MODULE_2_SLIDES
from .module_3_slides import get_module_3_slides, MODULE_3_SLIDES
from .module_4_slides import get_module_4_slides, MODULE_4_SLIDES
from .module_5_slides import get_module_5_slides, MODULE_5_SLIDES
from .module_6_slides import get_module_6_slides, MODULE_6_SLIDES
from .module_7_slides import get_module_7_slides, MODULE_7_SLIDES

__all__ = [
    'get_module_1_slides',
    'get_module_2_slides',
    'get_module_3_slides',
    'get_module_4_slides',
    'get_module_5_slides',
    'get_module_6_slides',
    'get_module_7_slides',
    'MODULE_1_SLIDES',
    'MODULE_2_SLIDES',
    'MODULE_3_SLIDES',
    'MODULE_4_SLIDES',
    'MODULE_5_SLIDES',
    'MODULE_6_SLIDES',
    'MODULE_7_SLIDES',
    'get_all_slides',
    'export_all_slides_to_markdown'
]


def get_all_slides():
    """Get all slides for all modules"""
    return {
        'module_1': get_module_1_slides(),
        'module_2': get_module_2_slides(),
        'module_3': get_module_3_slides(),
        'module_4': get_module_4_slides(),
        'module_5': get_module_5_slides(),
        'module_6': get_module_6_slides(),
        'module_7': get_module_7_slides(),
    }


def export_all_slides_to_markdown():
    """Export all slides to a single markdown document"""
    from .module_1_slides import export_slides_to_markdown as export_m1
    from .module_2_slides import export_slides_to_markdown as export_m2
    from .module_3_slides import export_slides_to_markdown as export_m3
    from .module_4_slides import export_slides_to_markdown as export_m4
    from .module_5_slides import export_slides_to_markdown as export_m5
    from .module_6_slides import export_slides_to_markdown as export_m6
    from .module_7_slides import export_slides_to_markdown as export_m7
    
    markdown = "# RAG Evaluation Course - Complete Slide Deck\n\n"
    markdown += "## Course: Evaluating RAG and Semantic Search Systems\n"
    markdown += "## Preparing for NVIDIA-Certified Professional: Agentic AI (NCP-AAI)\n\n"
    markdown += "---\n\n"
    
    markdown += export_m1(get_module_1_slides())
    markdown += "\n\n"
    markdown += export_m2(get_module_2_slides())
    markdown += "\n\n"
    markdown += export_m3(get_module_3_slides())
    markdown += "\n\n"
    markdown += export_m4(get_module_4_slides())
    markdown += "\n\n"
    markdown += export_m5(get_module_5_slides())
    markdown += "\n\n"
    markdown += export_m6(get_module_6_slides())
    markdown += "\n\n"
    markdown += export_m7(get_module_7_slides())
    
    return markdown
