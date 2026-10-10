"""Load API keys from `.env.local` at the repository root without printing them."""

import os

from reqcore.paths import repo_root


def load_env(filename: str = ".env.local") -> list[str]:
    """Set variables from `filename` that are not already in the environment.

    Returns the names that were loaded (never the values).
    """
    path = repo_root() / filename
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded
