#!/usr/bin/env python3
"""Compact feedback manifests, prompt context, and semantic completion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


MANIFEST_VERSION = 1
STATES = {"triage", "ready", "in_progress", "review_pending", "blocked", "rejected", "addressed"}


def fail(message: str) -> "None":
    print(f"workflow: {message}", file=sys.stderr)
    raise SystemExit(1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_feedback(path: Path) -> bool:
    name = path.name
    return (
        path.is_file()
        and not name.endswith(".addressed.md")
        and (name.startswith("human_feedback") or "REVIEW" in name)
    )


def checklist_items(raw: str) -> list[str]:
    """Split on blank lines and list starts without changing request characters."""
    items: list[str] = []
    current: list[str] = []
    list_start = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
    for line in raw.splitlines():
        if not line.strip():
            if current:
                items.append("\n".join(current))
                current = []
            continue
        if list_start.match(line) and current:
            items.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        items.append("\n".join(current))
    return items or [raw]


def manifest_path(direction: Path, source: Path) -> Path:
    return direction / ".tasks" / f"{source.name}.manifest.json"


def create_manifest(direction: Path, source: Path) -> Path:
    target = manifest_path(direction, source)
    if target.exists():
        return target
    raw = source.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": MANIFEST_VERSION,
        "state": "triage",
        "source_feedback": source.name,
        "source_sha256": digest(source),
        "required_outputs": None,
        "may_modify": None,
        "must_remain_unchanged": None,
        "unresolved_ambiguities": [
            "Classify output routing from the exact feedback before doing research."
        ],
        "checklist": [
            {
                "request": item,
                "status": "pending",
                "completion": {"what_was_done": "", "evidence": "", "ambiguity": ""},
            }
            for item in checklist_items(raw)
        ],
    }
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {path}: {exc}")


def direction_for(path: Path) -> Path:
    if path.parent.name != ".tasks":
        fail(f"manifest must live in DIR/.tasks: {path}")
    return path.parent.parent


def validate(path: Path, require_review: bool = False) -> dict:
    data = load_manifest(path)
    direction = direction_for(path)
    if data.get("version") != MANIFEST_VERSION:
        fail(f"unsupported manifest version in {path}")
    if data.get("state") not in STATES:
        fail(f"invalid state in {path}: {data.get('state')!r}")
    source_rel = data.get("source_feedback")
    if not isinstance(source_rel, str) or not source_rel:
        fail(f"missing source_feedback in {path}")
    source = direction / source_rel
    if data["state"] != "addressed":
        if not source.is_file():
            fail(f"source feedback is missing: {source_rel}")
        if digest(source) != data.get("source_sha256"):
            fail(f"source feedback changed after manifest creation: {source_rel}")
        expected = checklist_items(source.read_text(encoding="utf-8"))
        actual = [item.get("request") for item in data.get("checklist", [])]
        if actual != expected:
            fail(f"verbatim checklist differs from source feedback: {path}")
    if data["state"] in {"ready", "in_progress", "review_pending", "rejected"}:
        for field in ("required_outputs", "may_modify", "must_remain_unchanged"):
            value = data.get(field)
            if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
                fail(f"{field} must be an explicit list before research: {path}")
        if data.get("unresolved_ambiguities"):
            fail(f"ready task still has unresolved ambiguities: {path}")
    if data["state"] == "blocked" and not data.get("unresolved_ambiguities"):
        fail(f"blocked task must record its ambiguity: {path}")
    if require_review:
        if data["state"] != "review_pending":
            fail(f"manifest is not awaiting review: {path}")
        baseline = data.get("baseline_sha256")
        if not isinstance(baseline, dict):
            fail("task was not sealed before research")
        for declared in data.get("must_remain_unchanged", []):
            candidate = resolve_declared(direction, declared)
            current = digest(candidate) if candidate.is_file() else None
            if current != baseline.get(declared):
                fail(f"file declared unchanged was modified: {declared}")
        for item in data.get("checklist", []):
            completion = item.get("completion", {})
            if item.get("status") != "done":
                fail("every checklist item must be done before review")
            for field in ("what_was_done", "evidence", "ambiguity"):
                if not isinstance(completion.get(field), str) or not completion[field].strip():
                    fail(f"checklist completion field {field!r} is empty")
    return data


def guard_source(path: Path) -> None:
    """Undo a worker's premature addressed rename before validating the gate."""
    data = load_manifest(path)
    if data.get("state") == "addressed":
        return
    direction = direction_for(path)
    source = direction / data["source_feedback"]
    premature = source.with_name(source.name + ".addressed.md")
    if not source.exists() and premature.is_file():
        premature.rename(source)
        print(f"workflow: restored premature feedback rename: {source.name}")


def active_manifests(direction: Path) -> list[Path]:
    task_dir = direction / ".tasks"
    if not task_dir.is_dir():
        return []
    active = []
    for path in sorted(task_dir.glob("*.manifest.json")):
        if load_manifest(path).get("state") != "addressed":
            active.append(path)
    return active


def section(text: str, title: str) -> str:
    match = re.search(
        rf"(?ms)^##\s+{re.escape(title)}\s*$.*?(?=^##\s+|\Z)", text
    )
    return match.group(0).strip() if match else ""

def clip_excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return text[:head] + f"\n\n[... {omitted} characters omitted from this excerpt ...]\n\n" + text[-tail:]



def compact_context(direction: Path) -> str:
    manifests = active_manifests(direction)
    chunks = ["# Compact iteration context"]
    if manifests:
        manifest = manifests[0]
        chunks.append("\n## Active task manifest\n" + manifest.read_text(encoding="utf-8").strip())
    else:
        chunks.append("\n## Active task manifest\nNo unaddressed feedback task.")

    plan_path = direction / "PLAN.md"
    if plan_path.is_file():
        plan = plan_path.read_text(encoding="utf-8")
        selected = [
            section(plan, name)
            for name in ("Success criterion (definition of \"done\")",
                         "Out of scope (do NOT)", "Current status", "Next step")
        ]
        chunks.append("\n## Relevant PLAN.md sections\n" + "\n\n".join(clip_excerpt(x, 6000) for x in selected if x))

    journal_path = direction / "JOURNAL.md"
    if journal_path.is_file():
        journal = journal_path.read_text(encoding="utf-8")
        starts = list(re.finditer(r"(?m)^##\s+", journal))
        latest = journal[starts[-1].start():] if starts else journal[-4000:]
        chunks.append("\n## Latest JOURNAL.md entry only\n" + clip_excerpt(latest.strip(), 8000))
    return "\n".join(chunks) + "\n"


WORKER_RULES = """You are one iteration of an unattended research workflow. Use the compact context below; do not reread whole journals, changelogs, results, or reports unless the current task truly requires a specific section. CLAUDE.md remains the general operator reference and WRITING.md is a high-priority system instruction.

Feedback tasks use the active JSON manifest. In state triage, do no research: classify exact required outputs, files that may be modified, files that must remain unchanged, and ambiguities. Preserve every checklist request byte-for-byte. Set state to ready only when routing and meaning are clear; otherwise set state to blocked and name the ambiguity. A literal character such as & is data, not permission to reinterpret it as a word. Never choose a different research question or output filename to escape ambiguity.

For ready/in_progress/rejected tasks, answer or repair the verbatim checklist and write only to declared paths. Read an existing report section only when editing it. For every checklist item, fill completion.what_was_done, completion.evidence (exact figure/table/experiment/section/file), and completion.ambiguity (use "None" when none remains), then set status to done. Set state to review_pending only when every item and every required output is complete. Never rename feedback or create STOP yourself; the wrapper does that only after format checks and an independent content review.

Your final response must enumerate every checklist item with what was done, its exact evidence location, and any remaining ambiguity. For ordinary plan work with no active feedback, take one focused step from Current status/Next step and load only the files needed for that step.

Before ordinary plan work, compare Current status against the success criterion. If the criterion is already satisfied, do not run another experiment or follow a speculative "next step if reopened." Finalize the concise deliverables and create STOP.

Do not add every completed experiment to REPORT*.md. Preserve detailed evidence in RESULTS.md and keep only the shortest result chain needed for the report's core question.
"""


def resolve_declared(direction: Path, declared: str) -> Path:
    path = Path(declared)
    return path if path.is_absolute() else direction / path


def snapshot_files(direction: Path, data: dict) -> dict[str, str | None]:
    declared = (
        set(data.get("required_outputs") or [])
        | set(data.get("may_modify") or [])
        | set(data.get("must_remain_unchanged") or [])
    )
    for report in direction.glob("REPORT*.md"):
        declared.add(report.name)
    snapshots: dict[str, str | None] = {}
    for item in sorted(declared):
        path = resolve_declared(direction, item)
        snapshots[item] = digest(path) if path.is_file() else None
    return snapshots


def files_for_review(path: Path, data: dict) -> list[str]:
    direction = direction_for(path)
    outputs = list(data.get("required_outputs") or [])
    baseline = data.get("baseline_sha256", {})
    for declared, before in baseline.items():
        candidate = resolve_declared(direction, declared)
        after = digest(candidate) if candidate.is_file() else None
        if after != before and Path(declared).name.startswith("REPORT") and declared.endswith(".md"):
            outputs.append(declared)
    for report in sorted(direction.glob("REPORT*.md")):
        if report.name not in baseline:
            outputs.append(report.name)
    return list(dict.fromkeys(outputs))


REVIEW_RULES = 
"""Act as a fresh completion reviewer. Inspect only the bundled original feedback, task checklist, and every declared output. Reject if any request was omitted or reinterpreted; the work answers a different research question; unrequested analysis displaced requested analysis; jargon is important but undefined; correlation is called a mechanism or causal explanation; a claim is stronger than its evidence; or technically formatted prose is difficult for a new researcher. Rendering success is necessary but never sufficient. Return pass=false with concrete failures whenever in doubt. List every exact declared output path in inspected_outputs after inspecting it.

Reject if a report exceeds its declared/default word or figure budget; duplicates the
detailed result archive; includes secondary experiments unnecessary to answer the core
question; or has grown by appending one experiment per iteration instead of being curated.
"""


def make_bundle(manifest: Path, bundle: Path) -> None:
    data = validate(manifest, require_review=True)
    direction = direction_for(manifest)
    bundle.mkdir(parents=True, exist_ok=True)
    source = direction / data["source_feedback"]
    shutil.copy2(source, bundle / "original_feedback")
    shutil.copy2(manifest, bundle / "task_manifest.json")
    mapping = []
    for index, declared in enumerate(files_for_review(manifest, data), 1):
        source_path = resolve_declared(direction, declared)
        if not source_path.is_file():
            fail(f"required review output is missing: {declared}")
        bundled = f"output_{index:03d}{source_path.suffix}"
        shutil.copy2(source_path, bundle / bundled)
        mapping.append({"declared_path": declared, "bundled_file": bundled})
    (bundle / "review_map.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    prompt = (
        "Read original_feedback, task_manifest.json, review_map.json, and every bundled_file listed "
        "in review_map.json. Those are the only task materials. Evaluate the exact feedback against "
        "every output. In inspected_outputs, return the declared_path values exactly.\n"
    )
    (bundle / "review_prompt.txt").write_text(prompt, encoding="utf-8")


def parse_review(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw.get("structured_output"), dict):
        return raw["structured_output"]
    if isinstance(raw.get("result"), str):
        try:
            parsed = json.loads(raw["result"])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return raw


def record_review(manifest: Path, review_path: Path, format_passed: bool) -> bool:
    data = validate(manifest, require_review=True)
    review = parse_review(review_path)
    expected = files_for_review(manifest, data)
    inspected = review.get("inspected_outputs")
    passed = (
        format_passed
        and review.get("pass") is True
        and isinstance(inspected, list)
        and len(inspected) == len(expected)
        and set(inspected) == set(expected)
        and review.get("failures") == []
    )
    data["review"] = {
        "format_passed": format_passed,
        "content": review,
        "expected_outputs": expected,
    }
    direction = direction_for(manifest)
    if passed:
        source = direction / data["source_feedback"]
        addressed = source.with_name(source.name + ".addressed.md")
        source.rename(addressed)
        data["addressed_feedback"] = addressed.name
        data["state"] = "addressed"
    else:
        data["state"] = "rejected"
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "active", "context"):
        p = sub.add_parser(name)
        p.add_argument("direction", type=Path)
    sub.add_parser("worker-rules")
    sub.add_parser("review-rules")
    p = sub.add_parser("validate")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("seal")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("state")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("guard-source")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("format-files")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("make-review-bundle")
    p.add_argument("manifest", type=Path)
    p.add_argument("bundle", type=Path)
    p = sub.add_parser("record-review")
    p.add_argument("manifest", type=Path)
    p.add_argument("review", type=Path)
    p.add_argument("--format-passed", choices=("yes", "no"), required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        direction = args.direction.resolve()
        created = [create_manifest(direction, source) for source in sorted(direction.iterdir()) if is_feedback(source)]
        for path in created:
            validate(path)
            print(path)
    elif args.command == "active":
        paths = active_manifests(args.direction.resolve())
        if paths:
            print(paths[0].resolve())
    elif args.command == "context":
        print(compact_context(args.direction.resolve()), end="")
    elif args.command == "worker-rules":
        print(WORKER_RULES)
    elif args.command == "review-rules":
        print(REVIEW_RULES)
    elif args.command == "validate":
        validate(args.manifest.resolve())
        print("OK")
    elif args.command == "seal":
        path = args.manifest.resolve()
        data = validate(path)
        if data["state"] == "ready" and "baseline_sha256" not in data:
            data["baseline_sha256"] = snapshot_files(direction_for(path), data)
            data["state"] = "in_progress"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(data["state"])
    elif args.command == "state":
        print(load_manifest(args.manifest.resolve()).get("state", ""))
    elif args.command == "guard-source":
        guard_source(args.manifest.resolve())
        validate(args.manifest.resolve())
    elif args.command == "format-files":
        path = args.manifest.resolve()
        data = validate(path, require_review=True)
        direction = direction_for(path)
        files = files_for_review(path, data)
        for report in sorted(direction.glob("REPORT*.md")):
            files.append(report.name)
        if (direction / "RESULTS.md").is_file():
            files.append("RESULTS.md")
        for item in dict.fromkeys(files):
            candidate = resolve_declared(direction, item)
            if candidate.suffix.lower() == ".md":
                print(candidate.resolve())
    elif args.command == "make-review-bundle":
        make_bundle(args.manifest.resolve(), args.bundle.resolve())
    elif args.command == "record-review":
        ok = record_review(args.manifest.resolve(), args.review.resolve(), args.format_passed == "yes")
        print("ADDRESSED" if ok else "REJECTED")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
