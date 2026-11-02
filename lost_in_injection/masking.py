"""Protect literal identifiers from translation.

Emails, URLs, quoted spans, format placeholders and any token with a digit, underscore, slash
or tilde are replaced by numbered placeholders such as [1] before translation and restored
afterwards. A translation is only usable if every placeholder comes back exactly once.
"""

import re

IDENTIFIER = re.compile(
    r"\S+@\S+\.\w+"  # email
    r"|https?://\S+"  # URL
    r"|(?<!\w)'[^']+'(?!\w)|\"[^\"]+\""  # quoted span, not an apostrophe in "don't"
    r"|\{\w+\}"  # format placeholder such as {injected_prompt}
    r"|[^\s,.;:!?()]*[\d_/~][^\s,;:!?()]*"  # token with a digit, underscore, slash or tilde
)
PLACEHOLDER = re.compile(r"\[(\d+)\]")


def mask(text: str) -> tuple[str, list[str]]:
    """Return the text with identifiers replaced by [1], [2], ... and the identifiers."""
    identifiers: list[str] = []

    def replace(match: re.Match) -> str:
        identifiers.append(match.group(0).rstrip("."))
        trailing = match.group(0)[len(identifiers[-1]) :]
        return f"[{len(identifiers)}]{trailing}"

    return IDENTIFIER.sub(replace, text), identifiers


def unmask(text: str, identifiers: list[str]) -> str | None:
    """Restore identifiers, or return None unless every placeholder appears exactly once."""
    found = [int(number) for number in PLACEHOLDER.findall(text)]
    if sorted(found) != list(range(1, len(identifiers) + 1)):
        return None
    return PLACEHOLDER.sub(lambda match: identifiers[int(match.group(1)) - 1], text)
