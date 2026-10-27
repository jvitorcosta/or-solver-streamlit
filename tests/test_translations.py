import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from solvedor.config import language, settings
from solvedor.solver.parser import parse_lp_problem

_PACKAGE = Path(__file__).parents[1] / "src" / "solvedor"


def _read(code: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(
        (_PACKAGE / "translations" / f"{code}.yaml").read_text(encoding="utf-8")
    )
    return data


def _leaves(data: dict[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        if isinstance(value, dict):
            keys |= _leaves(value, f"{prefix}{key}.")
        else:
            keys.add(f"{prefix}{key}")
    return keys


def test_languages_have_the_same_keys() -> None:
    assert _leaves(_read("en")) == _leaves(_read("pt"))


def test_every_key_is_used() -> None:
    source = "".join(path.read_text(encoding="utf-8") for path in _PACKAGE.rglob("*.py"))
    # gallery.examples is looked up by example key, checked in the next test.
    keys = {key for key in _leaves(_read("en")) if not key.startswith("gallery.examples.")}
    unused = sorted(key for key in keys if not re.search(rf"\.{re.escape(key)}\b", source))
    assert unused == []


def test_every_example_has_a_name_in_every_language() -> None:
    examples = set(settings.extract_examples_from_resources_directory())
    for code in language.LanguageCode:
        names = language.load_language_translations(language_code=code.value).gallery.examples
        assert set(names) == examples, code


def test_values_differ_where_portuguese_needs_its_own_words() -> None:
    en, pt = (language.load_language_translations(language_code=c) for c in ("en", "pt"))
    assert pt.interface.workspace_title != en.interface.workspace_title
    assert pt.tabs.readme != en.tabs.readme
    assert pt.app.subtitle.startswith("Uma ")


def test_missing_key_falls_back_to_english() -> None:
    portuguese = _read("pt")
    del portuguese["app"]["title"]
    del portuguese["results"]["solve_time"]
    merged = language.build_translations(_read("en"), portuguese)
    assert merged.app.title == _read("en")["app"]["title"]
    assert merged.results.solve_time == _read("en")["results"]["solve_time"]
    assert merged.app.subtitle == portuguese["app"]["subtitle"]


def test_broken_translation_falls_back_to_english() -> None:
    broken = _read("pt")
    broken["results"] = "not a section"
    merged = language.build_translations(_read("en"), broken)
    assert merged.results.solution == _read("en")["results"]["solution"]


def test_unknown_language_code_is_english() -> None:
    english = language.load_language_translations(language_code="en")
    assert language.load_language_translations(language_code="xx") == english


@pytest.mark.parametrize("code", [c.value for c in language.LanguageCode])
def test_guide_exists_for_every_language(code: str) -> None:
    guide = (_PACKAGE / "resources" / f"guide.{code}.md").read_text(encoding="utf-8")
    assert "subject to" in guide  # it documents the syntax
    assert "free" in guide


@pytest.mark.parametrize("code", [c.value for c in language.LanguageCode])
def test_guide_example_parses(code: str) -> None:
    guide = (_PACKAGE / "resources" / f"guide.{code}.md").read_text(encoding="utf-8")
    example = guide.split("```text", 1)[1].split("```", 1)[0]
    problem = parse_lp_problem(example)
    assert problem.variables["z"].lower_bound is None
    assert problem.is_mip
