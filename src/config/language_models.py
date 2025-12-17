"""Language model configurations and translations management."""

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
    export: str
    analysis: str


class ResultSummaryTranslations(BaseModel):
    """Result summary translations."""

    objective: str
    status: str
    optimal: str


class ResultExportTranslations(BaseModel):
    """Result export translations."""

    download_json: str
    download_csv: str


class ResultAnalysisTranslations(BaseModel):
    """Result analysis translations."""

    coming_soon: str


class ResultsTranslations(BaseModel):
    """Results section translations."""

    objective_value: str
    solve_time: str
    variables: str
    constraints: str
    solution_variables: str
    tabs: ResultTabsTranslations
    summary: ResultSummaryTranslations
    export: ResultExportTranslations
    analysis: ResultAnalysisTranslations


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

    model_config = ConfigDict(frozen=True)  # Immutable after creation


class LanguageManager(BaseModel):
    """Manages language translations with Pydantic validation and immutable state."""

    current_language_code: LanguageCode = LanguageCode.ENGLISH
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
        loaded_translations = _load_all_translation_files()
        return LanguageManager(translation_schemas=loaded_translations)

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
        return LanguageManager(
            current_language_code=language_code, translation_schemas=translation_schemas
        )

    def get_translations(
        self, language_code: LanguageCode | None = None
    ) -> TranslationSchema:
        """Get translation schema for specified or current language.

        Args:
            language_code: Optional language code. Uses current if not specified.

        Returns:
            TranslationSchema for the requested language, falls back to English.
        """
        requested_language = language_code or self.current_language_code

        if requested_language not in self.translation_schemas:
            # Fallback to English if requested language unavailable
            requested_language = LanguageCode.ENGLISH

        return self.translation_schemas[requested_language]

    def get_translation(
        self, key_path: str, language_code: LanguageCode | None = None
    ) -> str:
        """Get specific translation using dot notation key path.

        Args:
            key_path: Dot-separated path to translation (e.g., 'app.title').
            language_code: Optional language code for translation lookup.

        Returns:
            Translation string for the specified key path.
        """
        translation_schema = self.get_translations(language_code)
        return _navigate_translation_path(
            translation_object=translation_schema, key_path=key_path
        )


def _get_translations_directory_path() -> Path:
    """Get path to the translations directory.

    Returns:
        Path to the translations directory relative to project root.
    """
    return Path(__file__).parent.parent.parent / "translations"


def _load_translation_file(*, language_code: LanguageCode) -> TranslationSchema | None:
    """Load translation file for specific language code.

    Args:
        language_code: Language code to load translations for.

    Returns:
        TranslationSchema if file exists and is valid, None otherwise.
    """
    translations_directory = _get_translations_directory_path()
    translation_file_path = translations_directory / f"{language_code.value}.yaml"

    if not translation_file_path.exists():
        return None

    try:
        with open(translation_file_path, encoding="utf-8") as yaml_file:
            translation_data = yaml.safe_load(yaml_file)
            return TranslationSchema(**translation_data)
    except (yaml.YAMLError, OSError):
        return None


def _load_all_translation_files() -> dict[LanguageCode, TranslationSchema]:
    """Load all available translation files.

    Returns:
        Dictionary mapping language codes to their translation schemas.
    """
    loaded_translations = {}

    for language_code in LanguageCode:
        translation_schema = _load_translation_file(language_code=language_code)
        if translation_schema is not None:
            loaded_translations[language_code] = translation_schema

    return loaded_translations


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
    current_object = translation_object

    for key in key_path.split("."):
        current_object = getattr(current_object, key)

    return current_object
