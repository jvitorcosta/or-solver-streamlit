from enum import StrEnum
from functools import cache
from pathlib import Path

import pydantic
import yaml
from pydantic import BaseModel


class LanguageCode(StrEnum):
    """Supported language codes for application translations."""

    ENGLISH = "en"
    PORTUGUESE = "pt"


class AppTranslations(BaseModel):
    """Application-level translations."""

    title: str
    subtitle: str
    footer: str


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
    region_label: str
    optimum_label: str
    direction_label: str


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
    feasible_found: str
    time_limit: str
    time_limit_help: str
    linear: str
    mixed_integer: str
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


_TRANSLATIONS_DIR = Path(__file__).parents[1] / "translations"


@cache
def load_language_translations(*, language_code: str) -> TranslationSchema:
    """The translations for a language code, English for an unknown code. Loaded once per code.

    A missing or broken translation file raises: it ships inside the package, so that is a bug.
    """
    try:
        code = LanguageCode(language_code)
    except ValueError:
        code = LanguageCode.ENGLISH
    data = yaml.safe_load((_TRANSLATIONS_DIR / f"{code.value}.yaml").read_text(encoding="utf-8"))
    return TranslationSchema(**data)
