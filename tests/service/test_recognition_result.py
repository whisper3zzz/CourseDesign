from service.recognition_result import select_display_name


def test_select_display_name_prefers_confirmed_name() -> None:
    payload = {
        "name": "B23051614杨凯",
        "candidate_name": "B23051626黄鸿",
    }

    assert select_display_name(payload) == "B23051614杨凯"


def test_select_display_name_uses_candidate_for_unknown_result() -> None:
    payload = {
        "name": "未知",
        "candidate_name": "B23051626黄鸿",
    }

    assert select_display_name(payload) == "B23051626黄鸿"


def test_select_display_name_falls_back_to_unknown_without_candidate() -> None:
    payload = {
        "name": "未知",
        "candidate_name": "未知",
    }

    assert select_display_name(payload) == "未知"
