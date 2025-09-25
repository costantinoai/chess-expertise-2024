"""Text/label helpers for MVPA figures and tables."""

def format_contrast(s: str) -> str:
    """Normalize contrast labels for display.

    - Replace underscores with spaces
    - Replace 'vs' with '-'
    - Capitalize words
    """
    s = s.replace("_", " ")
    s = s.replace("vs", "-")
    return " ".join(word.capitalize() for word in s.split())

