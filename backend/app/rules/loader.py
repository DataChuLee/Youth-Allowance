from functools import cache
from pathlib import Path
from typing import Any

import yaml

RULES_DIR = Path(__file__).parent


@cache
def load_rules(name: str) -> dict[str, Any]:
    """선언된 판정 규칙 파일(spending, eligibility)을 읽는다."""
    with (RULES_DIR / f"{name}.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)
