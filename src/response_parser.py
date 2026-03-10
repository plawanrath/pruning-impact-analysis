"""Robust extraction of answer letters from model output."""
import re


def parse_response(raw: str, num_choices: int = 3) -> str | None:
    text = raw.strip()
    valid = set("ABC"[:num_choices])
    if text.upper() in valid:
        return text.upper()
    m = re.match(r"^\(?([A-C])\)?[.)\s]", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(
        r"(?:answer|choose|select|pick)\s*(?:is\s*)?\(?([A-C])\)?",
        text, re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    for char in text:
        if char.upper() in valid:
            return char.upper()
    return None
