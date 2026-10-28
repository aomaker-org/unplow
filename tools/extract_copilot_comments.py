#!/usr/bin/env python3
# Path: tools/extract_copilot_comments.py
# extract_copilot_comments.py - Extract Copilot PR comments, reviews, and findings
# Max Line Length: 120 Chars

import argparse
import json
import subprocess
import sys
import textwrap
from typing import Any, Dict, List


def run_gh_json(cmd: List[str]) -> Any:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(res.stdout)


def fetch_copilot_comments(repo: str = "aomaker-org/unplow", limit: int = 5) -> List[Dict[str, Any]]:
    cmd = [
        "gh", "pr", "list",
        "--repo", repo,
        "--state", "all",
        "--limit", str(limit),
        "--json", "number,title,state,url",
    ]
    prs = run_gh_json(cmd)
    copilot_records: List[Dict[str, Any]] = []

    for pr in prs:
        num = pr["number"]
        title = pr["title"]
        pr_url = pr["url"]
        sys.stderr.write(f"Fetching comments for PR #{num}...\n")

        # 1. Inline review comments
        try:
            comments = run_gh_json(["gh", "api", f"repos/{repo}/pulls/{num}/comments"])
            for c in comments:
                user = c.get("user", {}).get("login", "")
                if "copilot" in user.lower():
                    copilot_records.append({
                        "pr_number": num,
                        "pr_title": title,
                        "pr_url": pr_url,
                        "type": "inline_comment",
                        "user": user,
                        "path": c.get("path"),
                        "line": c.get("line"),
                        "original_line": c.get("original_line"),
                        "diff_hunk": c.get("diff_hunk"),
                        "created_at": c.get("created_at"),
                        "body": c.get("body"),
                        "html_url": c.get("html_url"),
                    })
        except Exception as e:
            sys.stderr.write(f"  Failed inline comments on PR #{num}: {e}\n")

        # 2. General issue/PR comments
        try:
            comments = run_gh_json(["gh", "api", f"repos/{repo}/issues/{num}/comments"])
            for c in comments:
                user = c.get("user", {}).get("login", "")
                if "copilot" in user.lower():
                    copilot_records.append({
                        "pr_number": num,
                        "pr_title": title,
                        "pr_url": pr_url,
                        "type": "issue_comment",
                        "user": user,
                        "created_at": c.get("created_at"),
                        "body": c.get("body"),
                        "html_url": c.get("html_url"),
                    })
        except Exception as e:
            sys.stderr.write(f"  Failed issue comments on PR #{num}: {e}\n")

        # 3. PR reviews
        try:
            reviews = run_gh_json(["gh", "api", f"repos/{repo}/pulls/{num}/reviews"])
            for r in reviews:
                user = r.get("user", {}).get("login", "")
                if "copilot" in user.lower() and r.get("body"):
                    copilot_records.append({
                        "pr_number": num,
                        "pr_title": title,
                        "pr_url": pr_url,
                        "type": "review_summary",
                        "user": user,
                        "state": r.get("state"),
                        "created_at": r.get("submitted_at"),
                        "body": r.get("body"),
                        "html_url": r.get("html_url"),
                    })
        except Exception as e:
            sys.stderr.write(f"  Failed reviews on PR #{num}: {e}\n")

    return copilot_records


def generate_markdown_report(records: List[Dict[str, Any]], output_path: str) -> None:
    lines: List[str] = [
        "# Path: copilot-pr-comments.md",
        "# copilot-pr-comments.md - Extracted GitHub Copilot PR Review Comments",
        "",
        "# Copilot Pull Request Review Comments & Findings",
        "",
        f"Total Extracted Copilot Artifacts: {len(records)}",
        "",
    ]

    for rec in sorted(records, key=lambda x: (x["pr_number"], x["created_at"]), reverse=True):
        title = rec["pr_title"].replace("\n", " ").strip()
        lines.append(f"## PR #{rec['pr_number']}: {title}")
        lines.append(f"- **URL**: {rec['pr_url']}")
        lines.append(f"- **Type**: `{rec['type']}`")
        lines.append(f"- **User**: `{rec['user']}`")
        lines.append(f"- **Created At**: {rec['created_at']}")
        if rec.get("path"):
            lines.append(f"- **File**: `{rec['path']}` (Line: {rec.get('line')})")
        if rec.get("state"):
            lines.append(f"- **Review State**: `{rec['state']}`")
        if rec.get("html_url"):
            lines.append(f"- **Comment URL**: {rec['html_url']}")
        lines.append("")
        lines.append("### Content")
        lines.append("```markdown")
        body_text = rec["body"].strip()
        for p in body_text.split("\n"):
            if len(p) <= 80:
                lines.append(p)
            else:
                lines.extend(textwrap.wrap(p, width=80))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("<!-- end of file: copilot-pr-comments.md -->\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Copilot comments from PRs.")
    parser.add_argument("--repo", default="aomaker-org/unplow", help="GitHub repo")
    parser.add_argument("--limit", type=int, default=5, help="Number of PRs to inspect")
    parser.add_argument("--output-json", default="copilot-pr-comments.json", help="Path for JSON output")
    parser.add_argument("--output-md", default="copilot-pr-comments.md", help="Path for Markdown output")
    args = parser.parse_args()

    records = fetch_copilot_comments(repo=args.repo, limit=args.limit)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    sys.stderr.write(f"Saved {len(records)} records to {args.output_json}\n")

    generate_markdown_report(records, args.output_md)
    sys.stderr.write(f"Generated markdown report at {args.output_md}\n")


if __name__ == "__main__":
    main()

# <!-- end of file: tools/extract_copilot_comments.py -->
