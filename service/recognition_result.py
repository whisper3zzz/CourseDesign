from __future__ import annotations


def select_display_name(result: str | dict | None) -> str:
    if isinstance(result, str):
        return result

    if not isinstance(result, dict):
        return "未知"

    name = result.get("name")
    if isinstance(name, str) and name and name != "未知":
        return name

    candidate_name = result.get("candidate_name")
    if isinstance(candidate_name, str) and candidate_name and candidate_name != "未知":
        return candidate_name

    return "未知"
