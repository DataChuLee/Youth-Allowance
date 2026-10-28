from collections import Counter
from pathlib import Path

from benchmarks.build_policy_safety_v2 import make_cases


def test_make_cases_creates_balanced_150_case_dataset() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cases = make_cases(project_root / "backend" / "evaluation_results" / "testset.csv")

    assert len(cases) == 150
    assert Counter(case["scenario"] for case in cases) == {
        "answerable": 60,
        "conditional": 40,
        "policy_blocked": 25,
        "insufficient": 25,
    }
    assert len({case["id"] for case in cases}) == 150
    assert len({case["question"] for case in cases}) == 150
