from __future__ import annotations

from typing import Any, Sequence


def format_candidate_skills(skills: Sequence[Any] | None) -> str:
    if not skills:
        return "none"

    formatted = [_format_skill(skill) for skill in skills]
    return "; ".join(skill for skill in formatted if skill) or "none"


def format_string_list(values: Any) -> str:
    if values is None:
        return "none"
    if isinstance(values, str):
        return values if values.strip() else "none"
    if not isinstance(values, Sequence):
        return str(values)

    formatted = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(formatted) or "none"


def _format_skill(skill: Any) -> str:
    if isinstance(skill, str):
        return skill

    if isinstance(skill, dict):
        name = skill.get("name")
        years = skill.get("years_of_experience")
        context = skill.get("context")
        last_used_year = skill.get("last_used_year")
    else:
        name = getattr(skill, "name", None)
        years = getattr(skill, "years_of_experience", None)
        context = getattr(skill, "context", None)
        last_used_year = getattr(skill, "last_used_year", None)

    if not name:
        return ""

    details = []
    if years is not None:
        details.append(f"{years}y")
    if context:
        details.append(str(context))
    if last_used_year is not None:
        details.append(f"last used {last_used_year}")

    if not details:
        return str(name)
    return f"{name} ({', '.join(details)})"
