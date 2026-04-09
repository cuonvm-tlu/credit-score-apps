"""Shared utility helpers for anonymization."""


def cmp_str(element1, element2):
    """Compare integer-like strings."""
    a = int(element1)
    b = int(element2)
    return (a > b) - (a < b)

