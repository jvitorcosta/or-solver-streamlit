from enum import Enum
from pathlib import Path
from typing import Any

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


class LanguageManager(BaseModel):
    """Manages language translations with Pydantic validation and immutable state."""

    translation_schemas: dict[LanguageCode, TranslationSchema] = pydantic.Field(
        default_factory=dict
    )

    model_config = pydantic.ConfigDict(
        use_enum_values=True, frozen=True
    )  # Immutable after creation

    @staticmethod
    def load_translations() -> "LanguageManager":
        """Load all available translations from YAML files.

        Returns:
            LanguageManager instance with loaded translation schemas.
        """
        return LanguageManager(translation_schemas=build_language_schema_dictionary())

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
        validated_translation_schema = TranslationSchema(**translations)
        language_schema_mapping = {language_code: validated_translation_schema}
        return LanguageManager(translation_schemas=language_schema_mapping)

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
        language_translation_schema = self.get_translations(language_code)
        return traverse_nested_translation_attributes(
            translation_object=language_translation_schema, key_path=key_path
        )


def parse_yaml_translation_file_for_language(
    *, language_code: LanguageCode
) -> TranslationSchema | None:
    """Parse YAML translation file and return validated schema for specific language.

    Args:
        language_code: Language code to load translations for.

    Returns:
        TranslationSchema if file exists and is valid, None otherwise.
    """
    translations_directory = Path(__file__).parents[2] / "translations"
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


def build_language_schema_dictionary() -> dict[LanguageCode, TranslationSchema]:
    """Build dictionary of all supported languages with their translation schemas.

    Returns:
        Dictionary mapping language codes to their translation schemas.
    """
    return {
        language_code_enum: translation_schema
        for language_code_enum in LanguageCode
        if (
            translation_schema := parse_yaml_translation_file_for_language(
                language_code=language_code_enum
            )
        )
        is not None
    }


def traverse_nested_translation_attributes(
    *, translation_object: Any, key_path: str
) -> str:
    """Traverse nested translation attributes using dot-separated key path.

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
    """Load localization strings for specified language with English fallback.

    Args:
        language_code: Language code string (e.g., 'en', 'pt').

    Returns:
        TranslationSchema instance with validated translations for the language.
        Falls back to English if the requested language is invalid.
    """
    # Validate and fallback to English if invalid
    try:
        validated_language_code = LanguageCode(language_code)
    except ValueError:
        validated_language_code = LanguageCode.ENGLISH

    # Load requested language or fallback to English
    return (
        parse_yaml_translation_file_for_language(language_code=validated_language_code)
        or parse_yaml_translation_file_for_language(language_code=LanguageCode.ENGLISH)
        or TranslationSchema(**{})  # Empty schema as last resort
    )
