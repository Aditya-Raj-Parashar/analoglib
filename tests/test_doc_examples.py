"""Test runner for code snippets embedded in AnalogLib documentation.

Ensures that all executable code snippets in README.md and docs/ run cleanly
without errors against AnalogLib v0.1.0.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest
import numpy as np
import analoglib as al


def extract_python_snippets(file_path: Path):
    """Extract python code snippets from a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    pattern = r"```python\s*(.*?)\s*```"
    snippets = []
    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1).strip()
        # Skip snippets explicitly marked as pseudocode or planned/unsupported
        if "# pseudocode" in code or "# Status: Planned" in code or "# Not Implemented" in code:
            continue
        snippets.append(code)
    return snippets


# Discover markdown files
DOCS_DIR = Path(__file__).parent.parent / "docs"
README_PATH = Path(__file__).parent.parent / "README.md"

md_files = list(DOCS_DIR.glob("**/*.md")) + ([README_PATH] if README_PATH.exists() else [])


@pytest.mark.parametrize("md_file", md_files, ids=lambda p: str(p.relative_to(p.parent.parent)))
def test_markdown_code_snippets(md_file: Path):
    snippets = extract_python_snippets(md_file)
    for idx, code in enumerate(snippets):
        # Create execution environment
        exec_globals = {
            "al": al,
            "analoglib": al,
            "np": np,
            "__name__": "__main__",
        }
        try:
            exec(code, exec_globals)
        except ModuleNotFoundError as e:
            if "torch" in str(e):
                pytest.skip(f"Skipping PyTorch example in {md_file.name} (torch not installed)")
            raise e
        except Exception as e:
            pytest.fail(f"Failed executing snippet #{idx+1} in {md_file.name}:\n{code}\n\nError: {e}")

