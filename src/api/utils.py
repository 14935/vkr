import re

def sanitize_group(name: str, default: str = "trash") -> str:
    if not name:
        return default
    cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_')
    return cleaned[:128] or default
