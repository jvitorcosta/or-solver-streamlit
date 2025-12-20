from enum import Enum
from pathlib import Path

import pydantic
import yaml
from pydantic import BaseModel


class LanguageCode(str, Enum):
    """Supported language codes for application translations."""

    ENGLISH = "en"
    PORTUGUESE = "pt"


class AppTranslations(BaseModel):
    """Application-level translations."""

    title: str
    subtitle: str
    footer: str
    about: str


class SidebarTranslations(BaseModel):
    """Sidebar translations."""

    settings: str
    language: str


class InterfaceTranslations(BaseModel):
    """Interface translations."""

    problem_label: str
    solve_button: str
    loaded_example: str
    placeholder: str
    help_text: str
    clear_workspace: str
    clear_help: str
    solve_help: str
    processing_toast: str
    workspace_title: str
    using_example: str


class ResourcesTranslations(BaseModel):
    """Resources section translations."""

    title: str
    thesis: str
    readme: str
    contributing: str
    changelog: str
    structure: str
    view_pdf: str
    download_pdf: str


class MessagesTranslations(BaseModel):
    """Message translations."""

    empty_problem_solve: str
    solution_found: str
    problem_copied: str


class StatusTranslations(BaseModel):
    """Status message translations."""

    parsing: str
    setting_up: str
    solving: str
    solution_found: str
    complete: str
    syntax_error: str
    solving_failed: str


class ErrorsTranslations(BaseModel):
    """Error message translations."""

    syntax_error: str
    solving_error: str
    thesis_not_found: str
    file_not_found: str
    file_read_error: str
    pdf_load_error: str
    pdf_display_error: str
    visualization_error: str
    debug_info: str


class ResultTabsTranslations(BaseModel):
    """Result tabs translations."""

    summary: str


class ResultSummaryTranslations(BaseModel):
    """Result summary translations."""

    objective: str
    status: str
    optimal: str


class TabsTranslations(BaseModel):
    """Tab names translations."""

    workspace: str
    paper: str
    guide: str
    readme: str
    contributing: str


class GalleryTranslations(BaseModel):
    """Gallery section translations."""

    title: str
    choose_text: str
    select_help: str
    copy_button: str
    copy_help: str
    copied_toast: str


class VisualizationTranslations(BaseModel):
    """Visualization section translations."""

    title: str
    integer_info: str
    guide_title: str
    feasible_region: str
    constraint_lines: str
    optimal_point: str
    gradient_arrow: str


class LanguageLabelsTranslations(BaseModel):
    """Language label translations."""

    english: str
    portuguese: str


class ResultsTranslations(BaseModel):
    """Results section translations."""

    solution: str
    optimal_found: str
    objective_value: str
    solve_time: str
    variables: str
    variable_values: str
    constraints: str
    solution_variables: str
    no_variable_values: str
    problem_infeasible: str
    infeasible_help: str
    problem_unbounded: str
    unbounded_help: str
    solver_error: str
    error_details: str
    problem_type: str
    solver: str
    objective_value_help: str
    solve_time_help: str
    variable: str
    value: str
    type: str
    tabs: ResultTabsTranslations
    summary: ResultSummaryTranslations


class TranslationSchema(BaseModel):
    """Complete translation schema."""

    app: AppTranslations
    sidebar: SidebarTranslations
    interface: InterfaceTranslations
    resources: ResourcesTranslations
    messages: MessagesTranslations
    status: StatusTranslations
    errors: ErrorsTranslations
    results: ResultsTranslations
    tabs: TabsTranslations
    gallery: GalleryTranslations
    visualization: VisualizationTranslations
    language_labels: LanguageLabelsTranslations

    model_config = pydantic.ConfigDict(frozen=True)  # Immutable after creation


def _parse_yaml_translation_file_for_language(
    *, language_code: LanguageCode
) -> TranslationSchema | None:
    """Parse YAML translation file and return validated schema for specific language.

    Args:
        language_code: Language code to load translations for.

    Returns:
        TranslationSchema if file exists and is valid, None otherwise.
    """
    translations_directory = Path(__file__).parent.parent / "resources" / "translations"
    yaml_file_path = translations_directory / f"{language_code.value}.yaml"

    try:
        yaml_file_content = yaml_file_path.read_text(encoding="utf-8")
        parsed_translation_data = yaml.safe_load(yaml_file_content)
        return (
            TranslationSchema(**parsed_translation_data)
            if parsed_translation_data
            else None
        )
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return None


def load_language_translations(*, language_code: str) -> TranslationSchema:
    """Load localization strings for specified language.

    Args:
        language_code: Language code string (e.g., 'en', 'pt').

    Returns:
        TranslationSchema instance with validated translations for the language.
    """
    # Validate language code
    try:
        validated_language_code = LanguageCode(language_code)
    except ValueError:
        validated_language_code = LanguageCode.ENGLISH

    # Load the translation file
    translation_schema = _parse_yaml_translation_file_for_language(
        language_code=validated_language_code
    )

    if translation_schema is None:
        raise FileNotFoundError(
            f"Translation file not found: {validated_language_code.value}.yaml"
        )

    return translation_schema
