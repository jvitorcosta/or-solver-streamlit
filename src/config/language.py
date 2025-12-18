from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LanguageCode(str, Enum):
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

    model_config = ConfigDict(frozen=True)  # Immutable after creation


class LanguageManager(BaseModel):
    """Manages language translations with Pydantic validation and immutable state."""

    translation_schemas: dict[LanguageCode, TranslationSchema] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(
        use_enum_values=True, frozen=True
    )  # Immutable after creation

    @staticmethod
    def load_translations() -> "LanguageManager":
        """Load all available translations from YAML files.

        Returns:
            LanguageManager instance with loaded translation schemas.
        """
        return LanguageManager(translation_schemas=_load_all_translation_files())

    @staticmethod
    def from_dict(
        *, language_code: LanguageCode, translations: dict[str, Any]
    ) -> "LanguageManager":
        """Create LanguageManager from translation dictionary.

        Args:
            language_code: Primary language code for the manager.
            translations: Translation data dictionary.

        Returns:
            LanguageManager instance with validated translations.
        """
        validated_schema = TranslationSchema(**translations)
        translation_schemas = {language_code: validated_schema}
        return LanguageManager(translation_schemas=translation_schemas)

    def get_translations(self, language_code: LanguageCode) -> TranslationSchema:
        """Get translation schema for specified language.

        Args:
            language_code: Language code for translations.

        Returns:
            TranslationSchema for the requested language.
        """
        if language_code not in self.translation_schemas:
            raise ValueError(
                f"Translations for language code '{language_code}' not found."
            )
        return self.translation_schemas[language_code]

    def get_translation(self, key_path: str, language_code: LanguageCode) -> str:
        """Get specific translation using dot notation key path.

        Args:
            key_path: Dot-separated path to translation (e.g., 'app.title').
            language_code: Language code for translation lookup.

        Returns:
            Translation string for the specified key path.
        """
        translation_schema = self.get_translations(language_code)
        return _navigate_translation_path(
            translation_object=translation_schema, key_path=key_path
        )


def _load_translation_file(*, language_code: LanguageCode) -> TranslationSchema | None:
    """Load translation file for specific language code.

    Args:
        language_code: Language code to load translations for.

    Returns:
        TranslationSchema if file exists and is valid, None otherwise.
    """
    translations_directory = Path(__file__).parents[2] / "translations"
    file_path = translations_directory / f"{language_code.value}.yaml"

    try:
        content = file_path.read_text(encoding="utf-8")
        translation_data = yaml.safe_load(content)
        return TranslationSchema(**translation_data) if translation_data else None
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return None


def _load_all_translation_files() -> dict[LanguageCode, TranslationSchema]:
    """Load all available translation files.

    Returns:
        Dictionary mapping language codes to their translation schemas.
    """
    return {
        lang_code: schema
        for lang_code in LanguageCode
        if (schema := _load_translation_file(language_code=lang_code)) is not None
    }


def _navigate_translation_path(*, translation_object: Any, key_path: str) -> str:
    """Navigate through nested translation attributes using dot notation.

    Args:
        translation_object: Starting translation object to navigate.
        key_path: Dot-separated path to desired translation.

    Returns:
        Translation string at the specified path.

    Raises:
        AttributeError: If the key path is invalid or not found.
    """
    from functools import reduce

    return reduce(getattr, key_path.split("."), translation_object)


def load_language_translations(*, language_code: str) -> TranslationSchema:
    """Load localization strings for specified language.

    Args:
        language_code: Language code string (e.g., 'en', 'pt').

    Returns:
        TranslationSchema instance with validated translations for the language.
        Falls back to English if the requested language is invalid.
    """
    # Validate and fallback to English if invalid
    try:
        lang_code = LanguageCode(language_code)
    except ValueError:
        lang_code = LanguageCode.ENGLISH

    # Load requested language or fallback to English
    return (
        _load_translation_file(language_code=lang_code)
        or _load_translation_file(language_code=LanguageCode.ENGLISH)
        or TranslationSchema(**{})  # Empty schema as last resort
    )
