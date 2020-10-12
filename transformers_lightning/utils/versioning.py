"""
Standard way of versioning a name
"""

def set_version(version: int):
    return f"version_{version}"

def get_version(name: str):
    return int(name.split("_")[-1])

def is_version(name: str):
    return name.startswith("version")