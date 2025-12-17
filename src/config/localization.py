"""Localization utilities for multi-language support using Pydantic."""

from pathlib import Path
from typing import Optional

from config import language_models


def _create_language_manager() -> language_models.LanguageManager:
    """Create a new language manager instance with loaded translations.
    
    Returns:
        New LanguageManager instance with translations loaded from files.
    """
    return language_models.LanguageManager.load_translations()


def load_language_translations(*, language_code: str) -> language_models.TranslationSchema:
    """Load localization strings for specified language using Pydantic validation.
    
    Args:
        language_code: Language code string (e.g., 'en', 'pt').
        
    Returns:
        TranslationSchema instance with validated translations for the language.
        Falls back to English if the requested language is invalid.
    """
    manager = _create_language_manager()
    try:
        validated_language_code = language_models.LanguageCode(language_code)
        return manager.get_translations(validated_language_code)
    except ValueError:
        # Fallback to English for invalid language codes
        return manager.get_translations(language_models.LanguageCode.ENGLISH)


def get_translation_by_key(
    *, 
    key_path: str, 
    language_code: Optional[str] = None
) -> str:
    """Get a specific translation using dot notation key path.
    
    Args:
        key_path: Dot-separated path to translation key (e.g., 'app.title').
        language_code: Optional language code. Defaults to English if not provided.
        
    Returns:
        Translated string for the specified key path.
        Falls back to English if language code is invalid.
    """
    manager = _create_language_manager()
    validated_language_code = language_models.LanguageCode.ENGLISH
    
    if language_code:
        try:
            validated_language_code = language_models.LanguageCode(language_code)
        except ValueError:
            # Keep English fallback for invalid language codes
            pass
    
    return manager.get_translation(key_path, validated_language_code)


# Backward compatibility alias - prefer load_language_translations
def load_language(language_code: str) -> language_models.TranslationSchema:
    """Load language translations (backward compatibility alias).
    
    DEPRECATED: Use load_language_translations with keyword arguments instead.
    
    Args:
        language_code: Language code string.
        
    Returns:
        TranslationSchema instance with loaded translations.
    """
    return load_language_translations(language_code=language_code)