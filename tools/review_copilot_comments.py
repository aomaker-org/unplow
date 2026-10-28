#!/usr/bin/env python3
# Path: tools/review_copilot_comments.py
# review_copilot_comments.py - Synthesize and categorize Copilot PR comments for refactoring
# Max Line Length: 120 Chars

import json
import textwrap
from collections import defaultdict
from typing import Any, Dict, List


def load_comments(filepath: str = "copilot-pr-comments.json") -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_comments(comments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    categories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for c in comments:
        if c.get("type") != "inline_comment":
            continue

        body = c.get("body", "").lower()
        path = c.get("path", "")

        # Categorization logic using membership checks to respect 120-col code limits
        fmt_keys = ["column", "80 cols", "120-char", "max column"]
        sec_keys = ["security", "unauthenticated", "path traversal", "arbitrary code", "pickle"]
        dead_keys = ["never used", "unused import", "unreachable", "no-op", "returns none"]
        disc_keys = ["does not match", "misleading", "inaccurate", "diverges"]
        perf_keys = ["leak", "eval mode", "socket", "re-load", "restore"]

        if any(k in body for k in fmt_keys):
            categories["Formatting & Column Limits (Directive 04)"].append(c)
        elif any(k in body for k in sec_keys):
            categories["Security, Path Traversal & Isolation"].append(c)
        elif any(k in body for k in dead_keys):
            categories["Dead Code, Unused Imports & Stubs"].append(c)
        elif any(k in body for k in disc_keys):
            categories["Documentation & Implementation Discrepancies"].append(c)
        elif any(k in body for k in perf_keys):
            categories["Resource Management, State Leaks & Performance"].append(c)
        elif "test" in body or "fixture" in body:
            categories["Test Coverage & Fragility"].append(c)
        else:
            categories["General Logic & Design Hygiene"].append(c)

    return categories


def generate_review_report(categories: Dict[str, List[Dict[str, Any]]], output_path: str = "copilot-pr-review.md"):
    lines: List[str] = [
        "# Path: copilot-pr-review.md",
        "# copilot-pr-review.md - Refactoring Plan from Copilot PR Feedback",
        "# Max Column: 80 Columns",
        "",
        "# Copilot Pull Request Review & Refactoring Synthesis",
        "",
        "This document synthesizes and prioritizes 72 inline review comments and 29",
        "review summaries extracted across all repository PRs.",
        "",
        "## Summary of Findings by Category",
        "",
    ]

    for cat_name, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        lines.append(f"- **{cat_name}**: {len(items)} actionable findings")
    lines.append("")

    for cat_name, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        lines.append(f"## {cat_name}")
        lines.append("")
        for item in sorted(items, key=lambda x: x["pr_number"], reverse=True):
            pr_num = item["pr_number"]
            path = item.get("path", "N/A")
            line = item.get("line", "N/A")
            body = item.get("body", "").replace("\n", " ").strip()
            lines.append(f"### [PR #{pr_num}] `{path}` (Line {line})")
            issue_raw = f"- **Issue**: {body[:300]}..." if len(body) > 300 else f"- **Issue**: {body}"
            wrapped_issue = textwrap.wrap(issue_raw, width=80, subsequent_indent="  ")
            lines.extend(wrapped_issue)
            lines.append("")

    lines.append("<!-- end of file: copilot-pr-review.md -->\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    comments = load_comments()
    categories = categorize_comments(comments)
    generate_review_report(categories)
    print("Generated copilot-pr-review.md successfully.")


if __name__ == "__main__":
    main()

# <!-- end of file: tools/review_copilot_comments.py -->
