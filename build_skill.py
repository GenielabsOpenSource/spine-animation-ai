#!/usr/bin/env python3
"""
build_skill.py — Builds SKILL.md by injecting script contents into SKILL.template.md.

Run by GitHub Actions on every push to scripts/ or the template.
Can also be run locally: python build_skill.py

How it works:
  1. Reads SKILL.template.md
  2. Finds all <!-- EMBED:scripts/filename.py --> markers
  3. Replaces each marker with the actual file contents in a code block
  4. Writes the final SKILL.md

This ensures SKILL.md is always a self-contained document that Claude can use
without needing access to the repo — just paste it into Claude Projects.
"""

import re
import os
import sys
from pathlib import Path

TEMPLATE = "SKILL.template.md"
OUTPUT = "SKILL.md"

# Header comment added to generated file
GENERATED_HEADER = """\
<!-- ⚠️  AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY
     Edit SKILL.template.md instead, then push to trigger rebuild.
     Scripts are embedded automatically by build_skill.py via GitHub Actions. -->

"""


def build():
    repo_root = Path(__file__).parent

    template_path = repo_root / TEMPLATE
    if not template_path.exists():
        print(f"ERROR: {TEMPLATE} not found at {template_path}")
        sys.exit(1)

    template = template_path.read_text(encoding="utf-8")

    # Find all <!-- EMBED:path/to/file.py --> markers
    pattern = re.compile(r"<!-- EMBED:(\S+?) -->")
    matches = list(pattern.finditer(template))

    if not matches:
        print("WARNING: No <!-- EMBED:... --> markers found in template")

    result = template
    embedded_count = 0

    for match in reversed(matches):  # reverse so indices stay valid
        filepath = match.group(1)
        full_path = repo_root / filepath

        if not full_path.exists():
            print(f"WARNING: Referenced file not found: {filepath}")
            continue

        content = full_path.read_text(encoding="utf-8").rstrip()

        # Detect language from extension
        ext = full_path.suffix.lstrip(".")
        lang = {"py": "python", "js": "javascript", "sh": "bash"}.get(ext, ext)

        # Build the replacement block
        replacement = (
            f"<!-- EMBED:{filepath} -->\n"
            f"<details>\n"
            f"<summary>📄 <code>{filepath}</code> ({_line_count(content)} lines)</summary>\n\n"
            f"```{lang}\n"
            f"{content}\n"
            f"```\n\n"
            f"</details>"
        )

        result = result[:match.start()] + replacement + result[match.end():]
        embedded_count += 1
        print(f"  Embedded: {filepath} ({_line_count(content)} lines)")

    # Add generated header after the frontmatter
    # Find end of YAML frontmatter (second ---)
    fm_end = _find_frontmatter_end(result)
    if fm_end >= 0:
        result = result[:fm_end] + "\n" + GENERATED_HEADER + result[fm_end:]
    else:
        result = GENERATED_HEADER + result

    # Write output
    output_path = repo_root / OUTPUT
    output_path.write_text(result, encoding="utf-8")

    print(f"\n✅ Built {OUTPUT}: {embedded_count} scripts embedded, "
          f"{len(result)} chars, {_line_count(result)} lines")


def _line_count(text: str) -> int:
    return len(text.strip().split("\n"))


def _find_frontmatter_end(text: str) -> int:
    """Find the position after the closing --- of YAML frontmatter."""
    if not text.startswith("---"):
        return -1
    second = text.find("---", 3)
    if second < 0:
        return -1
    return second + 3


if __name__ == "__main__":
    build()
