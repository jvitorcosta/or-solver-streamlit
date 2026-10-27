"""UI text in English and Portuguese, read from translations/<code>.yaml.

English is the base: a key missing from another language falls back to its English text, so
a half-translated file never breaks the page.
"""

import logging
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

import pydantic
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_TRANSLATIONS_DIR = Path(__file__).parents[1] / "translations"


class LanguageCode(StrEnum):
    ENGLISH = "en"
    PORTUGUESE = "pt"


class AppTranslations(BaseModel):
    title: str
    subtitle: str
    about: str
    footer: str


class SidebarTranslations(BaseModel):
    settings: str
    language: str


class LanguageLabelsTranslations(BaseModel):
    english: str
    portuguese: str


class InterfaceTranslations(BaseModel):
    solve_button: str
    placeholder: str
    help_text: str
    clear_workspace: str
    clear_help: str
    solve_help: str
    workspace_title: str
    using_example: str


class TabsTranslations(BaseModel):
    workspace: str
    paper: str
    guide: str
    readme: str
    contributing: str


class GalleryTranslations(BaseModel):
    title: str
    choose_text: str
    select_help: str
    copy_button: str
    copy_help: str
    examples: dict[str, str]  # example key -> display name


class VisualizationTranslations(BaseModel):
    title: str
    integer_info: str
    region_label: str
    optimum_label: str
    direction_label: str


class ResourcesTranslations(BaseModel):
    thesis: str
    thesis_caption: str
    download_pdf: str


class MessagesTranslations(BaseModel):
    empty_problem_solve: str


class StatusTranslations(BaseModel):
    solving: str


class ErrorsTranslations(BaseModel):
    syntax_error: str
    solving_error: str
    thesis_not_found: str
    file_not_found: str
    file_read_error: str
    pdf_load_error: str
    pdf_display_error: str


class ResultsTranslations(BaseModel):
    solution: str
    optimal_found: str
    feasible_found: str
    objective_value: str
    objective_value_help: str
    solve_time: str
    solve_time_help: str
    variables: str
    variable_values: str
    variable: str
    value: str
    type: str
    problem_type: str
    linear: str
    mixed_integer: str
    solver: str
    problem_infeasible: str
    infeasible_help: str
    problem_unbounded: str
    unbounded_help: str
    time_limit: str
    time_limit_help: str
    solver_error: str
    error_details: str


class TranslationSchema(BaseModel):
    app: AppTranslations
    sidebar: SidebarTranslations
    language_labels: LanguageLabelsTranslations
    interface: InterfaceTranslations
    tabs: TabsTranslations
    gallery: GalleryTranslations
    visualization: VisualizationTranslations
    resources: ResourcesTranslations
    messages: MessagesTranslations
    status: StatusTranslations
    errors: ErrorsTranslations
    results: ResultsTranslations

    model_config = pydantic.ConfigDict(frozen=True)


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge(base[key], value)
        else:
            merged[key] = value
    return merged


def build_translations(english: dict[str, Any], other: dict[str, Any]) -> TranslationSchema:
    """Another language on top of English; all English if the other file is malformed."""
    try:
        return TranslationSchema(**_merge(english, other))
    except pydantic.ValidationError:
        logger.warning("broken translation file, using English", exc_info=True)
        return TranslationSchema(**english)


def _read(code: LanguageCode) -> dict[str, Any]:
    text = (_TRANSLATIONS_DIR / f"{code.value}.yaml").read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text) or {}
    return data


@cache
def load_language_translations(*, language_code: str) -> TranslationSchema:
    """The translations for a language code, English for an unknown code. Loaded once per code."""
    try:
        code = LanguageCode(language_code)
    except ValueError:
        code = LanguageCode.ENGLISH
    english = _read(LanguageCode.ENGLISH)
    if code is LanguageCode.ENGLISH:
        return TranslationSchema(**english)
    return build_translations(english, _read(code))
